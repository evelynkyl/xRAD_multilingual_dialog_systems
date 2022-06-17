#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Pre-trained DPR Model.
"""
import gzip
import os
import os.path
from parlai.core.build_data import built, download_models, get_model_dir
import parlai.utils.logging as logger

path = 'https://nlp.cs.washington.edu/xorqa/cora/models/'


def download(datapath):
    ddir = os.path.join(get_model_dir(datapath), 'cora/wiki_passages')
    model_type = 'mdpr'
    version = 'v1.0'
    if not built(os.path.join(ddir, model_type), version):
        opt = {'datapath': datapath, 'model_type': model_type}
        fnames = ['all_w100.tsv']
        download_models(
            opt,
            fnames,
            'cora',
            version=version,
            use_model_type=True,
            path=path,
        )
