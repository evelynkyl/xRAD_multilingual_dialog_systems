#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer

See <https://aclanthology.org/2021.naacl-main.41/>

The mT5 agent can be instantiated as simply `-m mgen`
"""
import random
import numpy as np
import logging
import torch
from parlai.core.opt import Opt
from typing import Optional, Dict, Any, Tuple
from parlai.utils.distributed import is_distributed, sync_parameters
from transformers import (
    AutoConfig,
    T5ForConditionalGeneration,
    MT5ForConditionalGeneration,
       # AutoTokenizer,
  #  MT5EncoderModel
)
from parlai_internal.agents.hugging_face.t5 import T5Agent, ParlaiT5Model
import parlai.utils.fsdp as fsdp_utils

#from transformers.models.mt5.modeling_mt5 import MT5Model
from transformers.training_args import ParallelMode

try:
    from transformers.models.t5.modeling_t5 import * #T5Stack
    from transformers.models.mt5.modeling_mt5 import *
except ModuleNotFoundError:
    # Prior versions of transformers package do not have T5Stack
    T5Stack = object

from parlai_internal.agents.hugging_face.hugging_face import HF_VERSION
from parlai_internal.agents.hugging_face.dict import mGENDictionaryAgent
from parlai_internal.agents.mgen.utils import (  # noqa: E402 # isort:skip
    calculate_exact_match,
    flatten_list,
    get_git_info,
    is_rag_model,
    lmap,
    pickle_save,
    save_git_info,
    save_json,
    set_extra_model_params,
    Seq2SeqDataset,
)
from parlai.core.params import ParlaiParser
from parlai.core.torch_agent import Batch, TorchAgent, Output
from parlai.core.torch_generator_agent import TorchGeneratorAgent, TorchGeneratorModel

# for tpu training
#import torch_xla
#import torch_xla.core.xla_model as xm

# deepspeed integration
#from transformers.deepspeed import HfDeepSpeedConfig
#from transformers import AutoModel, AutoConfig, TrainingArguments, Trainer, MT5Tokenizer, MT5ForConditionalGeneration
#import deepspeed
from torch import nn

#ds_config = "/home/evelyn/thesis/CORA/models/train_with_deepseed/ds_config_mgen_custom.json" 
#dschf = HfDeepSpeedConfig(ds_config) 

def check_hf_version(v: Tuple[int, int]) -> bool:
    """
    Check that HF version is greater than 4.3.
    """
    main, sub = v
    return main > 4 or (main == 4 and sub >= 3)


def build_mGEN(opt: Opt) -> MT5ForConditionalGeneration:
    """ to update:""" 
    mGEN_model_dir = '/home/evelyn/parley/models/mGEN_model/' #/home/evelyn/thesis/CORA/models/mGEN_model/
    config_class = AutoConfig
    config = config_class.from_pretrained(mGEN_model_dir)
    
    if not check_hf_version(HF_VERSION):
        raise RuntimeError('Must use transformers package >= 4.3 to use t5')
    return MT5ForConditionalGeneration.from_pretrained(
        mGEN_model_dir, 
        local_files_only=True, 
        config=config
       # model_max_length=512,
        # extra (optional)_model params for generator configs and load_model
       # encoder_layerdrop=opt['mgen_encoder_layerdrop'],
       # decoder_layerdrop=opt['mgen_decoder_layerdrop'],
      #  attention_dropout = opt['mgen_attention_dropout'],
      #  dropout_rate = opt['mgen_dropout'],
    )


def set_device(func):
    """
    Decorator for setting device.

    HF's model parallel uses `torch.cuda.set_device`, which does not vibe well with
    ParlAI.
    """
   # def tpu():
    #    device = xm.xla_device()
    #    return device

    def wrap(*args, **kwargs):
        self = args[0]
        # self.paralleled implies whether the model has been paralleled.
        # it is set to the opposite of `opt['t5_model_parallel]`
        parallel = hasattr(self, 'paralleled') and not self.paralleled
        if torch.cuda.is_available() and parallel:
            torch.cuda.set_device('cuda:0')
        ret = func(*args, **kwargs)
        if torch.cuda.is_available() and parallel:
            torch.cuda.set_device('cuda:0')
        return ret

    return wrap # for GPU!!!


##############
# mGEN Modules #
##############

class ParlaimGENEncoder(torch.nn.Module):
    def __init__(self, opt: Opt, encoder: T5Stack, padding_idx: Optional[int] = None):
        super().__init__()
        self.stack = encoder
        self.padding_idx = padding_idx
        self.paralleled = not opt[
            'mGEN_model_parallel'
        ]  # need to parallel in forward; bug in HF

    @set_device
    def forward(
        self,
        input: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        """
        Forward pass.

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
   #     print("encoder output: ", outputs)
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
            'mGEN_model_parallel'
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
  #      print("decoder output: ", outputs)
        return outputs[0].to(input.device), incr_state


class ParlaimGENModel(TorchGeneratorModel):
    """
    Wrap mGEN in ParlAI.
    """

    def __init__(self, opt, dictionary):
        self.pad_idx = dictionary[dictionary.null_token]
        self.start_idx = self.pad_idx
        self.end_idx = dictionary[dictionary.end_token]
        super().__init__(self.pad_idx, self.start_idx, self.end_idx)
        self.mGEN = build_mGEN(opt)
        self.paralleled = not opt['mGEN_model_parallel']
        
        """ deepspeed test code """
      #  engine, _, _, _ = deepspeed.initialize(model=self.mGEN, config_params=ds_config)
        
        self.encoder = ParlaimGENEncoder(opt, self.mGEN.get_encoder(), self.pad_idx)
#        print(f"this is self.encoder: {self.encoder}")
        self.decoder = ParlaimGENDecoder(opt, self.mGEN.get_decoder(), self.pad_idx)
 #       print(f"this is self.encoder: {self.decoder}")

    @set_device
    def _get_initial_forced_decoder_input(self, bsz: int, inputs: torch.LongTensor):
        """
        Return initial input to the decoder.

        :param bsz:
            batchsize
        :param inputs:
            inputs to decode

        :return initial_input:
            initial input for the decoder.
        """
        inputs = torch.cat([self.START.detach().expand(bsz, 1), inputs], 1)
        return inputs
    
    @set_device
    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder the encoder states.

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
        Not *quite* sure how to reconcile this with HF.
        """
        return {}
    
    @set_device
    def output(self, tensor):
        """
        Compute output logits.
        """
        tensor = tensor * (self.mGEN.model_dim ** -0.5)
        lm_logits = self.mGEN.lm_head(tensor)
        return lm_logits
    
    
class MgenAgent(TorchGeneratorAgent):
    """
    mGEN Agent.
    
    Relies on the mT5 model implemented in huggingface
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        group = parser.add_argument_group('mGEN Args')
        group.add_argument(
            '--mGEN_gpus',
            type=int,
          #  default=1,
        )
        group.add_argument(
            '--mGEN_model_parallel', 
            type='bool',
            default=False,
            help='use HF model parallel',
        )
     #   group.add_argument(
     #       '--mGEN_deepspeed', type=float, default=0.0, help='deepspeed config for mGEN'
      #  )
      #  group.add_argument(
       #     '--mgen-dropout', type=float, default=0.0, help='Dropout for mGEN'
       # )
        return parser

    def build_model(self) -> 'ParlaimGENModel':
        """
        Build and return model.
        """
        model = ParlaimGENModel(self.opt, self.dict)
        if self.opt['mGEN_model_parallel']:
            model.mGEN.parallelize()
        return model

    def build_dictionary(self):
        """
        Overrides TorchAgent.build_dictionary to use mGEN dict.
        """
        return mGENDictionaryAgent(self.opt)
    
    def vectorize(self, *args, **kwargs):
        """
        Override vectorize for mT5.

        mT5 dict already adds the end token.
        """
        kwargs['add_start'] = False  # model does this in module code
        kwargs['add_end'] = False  # T5 tokenizer takes care of this
        return TorchAgent.vectorize(self, *args, **kwargs)
    
    def _generate(
        self,
        batch: Batch,
        beam_size: int,
        max_ts: int,
        prefix_tokens: Optional[torch.LongTensor] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ):
        """
        Generate an output with beam search.

        Use HF's built-in generation to perform beam search.
        """
        bad_words_ids = None
        if self.beam_block_list is not None:
            bad_words_ids = [
                gram for _, ngram in self.beam_block_list.items() for gram in ngram
            ]

        method = self.opt.get('inference', 'greedy')
    
        mGEN_model_dir = '/home/evelyn/parley/models/mGEN_model/' #/home/evelyn/thesis/CORA/models/mGEN_model/
        config_class = AutoConfig
        config = config_class.from_pretrained(mGEN_model_dir)
        
        generation_params = {
            'input_ids': batch.text_vec,
            'max_length': max_ts,
            'min_length': self.beam_min_length,
            'do_sample': self.opt['inference'] in ['topk', 'topp'],
            'early_stopping': None,
            'num_beams': beam_size,
            'temperature': self.temperature,
            'top_k': self.opt['topk'] if method in ['topk', 'delayedbeam'] else None,
            'top_p': self.opt['topp'] if method == 'nucleus' else None,
            'repetition_penalty': None,
            'bad_words_ids': bad_words_ids if bad_words_ids else None,
            'bos_token_id': self.START_IDX,
            'pad_token_id': self.NULL_IDX,
            'eos_token_id': self.END_IDX,
            'length_penalty': self.opt['beam_length_penalty'],
            'no_repeat_ngram_size': self.beam_block_ngram,
            'num_return_sequences': None,
            'attention_mask': batch.text_vec != self.NULL_IDX,
            'decoder_start_token_id': self.NULL_IDX,
        }
        
        if overrides:
            generation_params.update(overrides)

        outputs = self.model.mGEN.generate(**generation_params)
    #    print("this is generation outputs: ", outputs)
        outputs = [(outputs[i], 0, None) for i in range(outputs.size(0))]
        return outputs, []
    