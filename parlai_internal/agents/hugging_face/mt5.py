#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer

See <https://aclanthology.org/2021.naacl-main.41/>

The mT5 agent can be instantiated as simply `-m hugging_face/mt5`
"""
import torch
from typing import Optional, Dict, Any, Tuple
from transformers import (
    AutoConfig,
    AutoTokenizer,
    T5ForConditionalGeneration,
    MT5ForConditionalGeneration,
    MT5EncoderModel
)
from parlai.agents.hugging_face.t5 import T5Agent, ParlaiT5Model

#from transformers.models.mt5.modeling_mt5 import MT5Model

try:
    from transformers.models.t5.modeling_t5 import T5Stack
except ModuleNotFoundError:
    # Prior versions of transformers package do not have T5Stack
    T5Stack = object

from parlai.agents.hugging_face.hugging_face import HF_VERSION
from parlai.agents.hugging_face.dict import mT5DictionaryAgent

from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.torch_agent import Batch, TorchAgent
from parlai.core.torch_generator_agent import TorchGeneratorAgent, TorchGeneratorModel


def check_hf_version(v: Tuple[int, int]) -> bool:
    """
    Check that HF version is greater than 4.3.
    """
    main, sub = v
    return main > 4 or (main == 4 and sub >= 3)


def build_mt5(opt: Opt) -> MT5ForConditionalGeneration:
    if not check_hf_version(HF_VERSION):
        raise RuntimeError('Must use transformers package >= 4.3 to use t5')
    return MT5ForConditionalGeneration.from_pretrained(
        opt['mt5_model_arch'], dropout_rate=opt['mt5_dropout']
    )


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


class ParlaimT5Encoder(torch.nn.Module):
    def __init__(self, opt: Opt, encoder: T5Stack, padding_idx: Optional[int] = None):
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
        for k in outputs:
            if torch.is_tensor(outputs[k]):
                outputs[k] = outputs[k].to(input.device)
        return outputs[0], mask


class ParlaimT5Decoder(torch.nn.Module):
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


class ParlaimT5Model(ParlaiT5Model):
    """
    Wrap mT5 in ParlAI.
    """

    def __init__(self, opt, dictionary):
        self.pad_idx = dictionary[dictionary.null_token]
        self.start_idx = self.pad_idx
        self.end_idx = dictionary[dictionary.end_token]
        super().__init__(self.pad_idx, self.start_idx, self.end_idx)
        self.mt5 = build_mt5(opt)
        self.encoder = ParlaimT5Encoder(opt, self.mt5.get_encoder(), self.pad_idx)
        self.decoder = ParlaimT5Decoder(opt, self.mt5.get_decoder(), self.pad_idx)


    @set_device
    def output(self, tensor):
        """
        Compute output logits.
        """
        tensor = tensor * (self.mt5.model_dim ** -0.5)
        lm_logits = self.mt5.lm_head(tensor)
        return lm_logits
    

class mT5Agent(T5Agent):
    """
    mT5 Agent.

    Relies on the mT5 model implemented in huggingface
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        group = parser.add_argument_group('mT5 Args')
        group.add_argument(
            '--mt5-model-arch',
            type=str,
            default='mt5-base',
            choices=["mt5-small", "mt5-base", "mt5-large", "mt5-xxl"],
        )
        group.add_argument(
            '--mt5-model-parallel',
            type='bool',
            default=False,
            help='use HF model parallel',
        )
        group.add_argument(
            '--mt5-dropout', type=float, default=0.0, help='Dropout for mT5'
        )
        return parser

    def build_model(self) -> 'ParlaimT5Model':
        """
        Build and return model.
        """
        model = ParlaimT5Model(self.opt, self.dict)
        if self.opt['mt5_model_parallel']:
            model.mt5.parallelize()
        return model

    def build_dictionary(self):
        """
        Overrides TorchAgent.build_dictionary to use mt5 dict.
        """
        return mT5DictionaryAgent(self.opt)