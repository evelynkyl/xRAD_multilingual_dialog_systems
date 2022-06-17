"""
Utilities for working with the local dataset cache. 
script adopted from huggingface file_util.py for transformers
https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/file_utils.py
"""

from urllib.parse import urlparse
import importlib.util
import io
import json
import os
import re
from parlai.core import logging

#from transformers.utils.logging 
import tqdm
from parlai.core.utils_versions  import importlib_metadata
# transformers.utils.versions

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
USE_TF = os.environ.get("USE_TF", "AUTO").upper()

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    _torch_available = importlib.util.find_spec("torch") is not None
    if _torch_available:
        try:
            _torch_version = importlib_metadata.version("torch")
            logger.info(f"PyTorch version {_torch_version} available.")
        except importlib_metadata.PackageNotFoundError:
            _torch_available = False
else:
    logger.info("Disabling PyTorch because USE_TF is set")
    _torch_available = False


def is_torch_available():
    return _torch_available


# gpu
def is_torch_cuda_available():
    if is_torch_available():
        import torch

        return torch.cuda.is_available()
    else:
        return False

# tpu
def is_torch_tpu_available():
    if not _torch_available:
        return False
    # This test is probably enough, but just in case, we unpack a bit.
    if importlib.util.find_spec("torch_xla") is None:
        return False
    if importlib.util.find_spec("torch_xla.core") is None:
        return False
    return importlib.util.find_spec("torch_xla.core.xla_model") is not None