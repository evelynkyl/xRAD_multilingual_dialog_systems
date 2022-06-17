import argparse
import logging
import os
import sys
import time
import torch
from typing import Optional
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.torch_agent import Batch, TorchAgent, Output
from parlai.core.torch_generator_agent import TorchGeneratorAgent, TorchGeneratorModel
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BatchEncoding,
    MT5ForConditionalGeneration
)
from transformers import logging as transformers_logging
try:
    from transformers.models.t5.modeling_t5 import T5Stack
except ModuleNotFoundError:
    # Prior versions of transformers package do not have T5Stack
    T5Stack = object
try:
    import transformers
except ImportError:
    raise ImportError('Please run `pip install transformer.')

def set_device(func):
    """
    Decorator for setting device.

    HF's model parallel uses `torch.cuda.set_device`, which does not vibe well with
    ParlAI.
    """

    def wrap(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.set_device('cuda:0')
        ret = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.set_device('cuda:0')
        return ret

    return wrap
    
    
##############
# mT5 Modules #
##############
class ParlaimGENEncoder(torch.nn.Module):
    def __init__(self, opt: Opt, encoder: T5Stack, padding_idx: Optional[int] = None):
        # must call super on all nn.Modules.
        super().__init__()
        
        self.stack = encoder
        self.padding_idx = padding_idx
        self.paralleled = not opt[
            'mt5_model_parallel'
        ]  # need to parallel in forward; bug in HF

    @set_device
    def forward(
        self,
        input: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        """
        Forward pass for the encoder.

        :param LongTensor[batch,seqlen] input:
            The input IDs
        :param LongTensor[batch,seqlen] positions:
            Positions for input IDs
        :param LongTensor[batch,seqlen] segments:
            If provided, additionally adds ``segments`` as extra embedding features.
        """
        if not self.paralleled:
            self.stack.parallelize()
        mask = input != self.padding_idx
        outputs = self.stack(input, attention_mask=mask, output_hidden_states=False)
        for k in outputs:
            if torch.is_tensor(outputs[k]):
                outputs[k] = outputs[k].to(input.device)
        return outputs[0], mask


class ParlaimGENDecoder(torch.nn.Module):
    def __init__(self, opt: Opt, decoder: T5Stack, padding_idx: Optional[int] = None):
        super().__init__()
        self.stack = decoder
        self.padding_idx = padding_idx
        self.paralleled = not opt[
            'mt5_model_parallel'
        ]  # need to parallel in forward; bug in HF

    @set_device
    def forward(
        self, input: torch.LongTensor, encoder_state: Tuple[Any], incr_state=None
    ):
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The decoder inputs (partial or full decoded token IDs).
        :param encoder_state:
            Output from the encoder module forward pass.
        :param incr_state:
            The incremental state: a dictionary whose keys index the layers and whose
            values contain the incremental state for each layer.
        """
        if not self.paralleled:
            self.stack.parallelize()
        encoder_output, encoder_mask = encoder_state

        mask = input != self.padding_idx
        mask[:, 0] = True  # first token is pad

        outputs = self.stack(
            input_ids=input,
            attention_mask=mask,
            encoder_hidden_states=encoder_output.to(input.device),
            encoder_attention_mask=encoder_mask.to(input.device),
        )
        return outputs[0].to(input.device), incr_state
    
    
class ParlaimGENModel(TorchGeneratorModel):
    """
    Wrap mGEN in ParlAI.
    It also instantiates the embedding table, encoder, and decoder, and defines the
    final output layer.
    """

    def __init__(self, opt, dictionary):
        self.pad_idx = dictionary[dictionary.null_token]
        self.start_idx = self.pad_idx
        self.end_idx = dictionary[dictionary.end_token]
        
        super().__init__(self.pad_idx, self.start_idx, self.end_idx)
        
        self.mGEN = build_t5(opt)
        #  instantiate the encoder and decoder
        self.encoder = ParlaimGENEncoder(opt, self.mGEN.get_encoder(), self.pad_idx)
        self.decoder = ParlaimGENDecoder(opt, self.mGEN.get_decoder(), self.pad_idx)
        
        
    @set_device
    def output(self, tensor):
        """
        Compute output logits. 
        # to project the output of the decoder back into the token space
        """
        # Taken directly from HuggingFace
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        tensor = tensor * (self.mGEN.model_dim ** -0.5)
        lm_logits = self.mGEN.lm_head(tensor)
        return lm_logits
    
    @set_device
    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder the encoder states to select only the given batch indices.
        
        Since encoder_state can be arbitrary, you must implement this yourself.
        Typically you will just want to index select on the batch dimension.
        
        See ``TorchGeneratorModel.reorder_encoder_states`` for a description.
        """
        enc, mask = encoder_states
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(enc.device)
        enc = torch.index_select(enc, 0, indices)
        mask = torch.index_select(mask, 0, indices)
        return enc, mask


    def reorder_decoder_incremental_state(
        self, incremental_state: Dict[int, dict], inds: torch.Tensor
    ) -> Dict[int, dict]:
        """
        Reorder the decoder states to select only the given batch indices.
        This method can be a stub which always returns None; this will result in the
        decoder doing a complete forward pass for every single token, making generation
        O(n^2). However, if any state can be cached, then this method should be
        implemented to reduce the generation complexity to O(n).

        Not *quite* sure how to reconcile this with HF.
        """
        return {}
    

class CORAAgent(TorchGeneratorAgent):
    """
    mGEN(mT5) Agent.
    """
    
    @classmethod
    def add_cmdline_args(cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        group = parser.add_argument_group('mGEN Args')
        group.add_argument(
            '--mGEN-dropout', type=float, default=0.1, help='Dropout for mGEN'
        )

        group.add_argument(
            '--mGEN-epochs', type=int, default=50, help='Number of training epochs for mGEN'
        )
        group.add_argument(
            '--mGEN-learning-rate',type=float,default=3e-05,help='Learning rate for mGEN'
        )
        group.add_argument(
            '--mGEN-batch-size',type=float,default=4,help='Training batch size for mGEN'
        )

        return parser


    def train_step(self, batch):
        pass

    def eval_step(self, batch):
        # for each row in batch, convert tensor to back to text strings
        return Output([self.dict.vec2txt(row) for row in batch.text_vec])

    def build_model(self, batch):
        # Our agent doesn't have a real model, so we will return a placeholder
        # here.
        model = ParlaimGENModel(self.opt, self.dict)
        if self.opt['mt5_model_parallel']:
            model.mGEN.parallelize()
        return model