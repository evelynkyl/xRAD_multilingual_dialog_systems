#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Integration with Hugging Face Transformers.

Please see <https://huggingface.co/transformers/>. Currently, the only implementations
are GPT2 and DialoGPT. To use these models, run with `-m hugging_face/gpt2` or `-m
hugging_face/dialogpt`.
"""
try:
    import transformers
except ImportError:
    raise ImportError('Please run `pip install transformers`.')


HF_VERSION = (
    int(transformers.__version__.split('.')[0]),
    int(transformers.__version__.split('.')[1]),
)


class HuggingFaceAgent:
    def __init__(self, opt, shared=None):
        raise RuntimeError(
            '`-m hugging_face` is not a valid choice. Please run with '
            '`-m hugging_face/gpt2`, `-m hugging_face/dialogpt`, '
            '`-m hugging_face/t5`', 'or `-m hugging_face/mt5`'
        )
