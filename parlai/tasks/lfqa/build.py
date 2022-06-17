#!/usr/bin/env python3

from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
import subprocess
from os.path import join as pjoin
from os.path import isfile, isdir


RESOURCES = [
    # wet.paths.gz is false because the archive format is not recognized
    # It gets unzipped with subprocess after RESOURCES are downloaded.
    DownloadableFile(
        'https://huggingface.co/datasets/vblagoje/lfqa/resolve/main/test.json',
        'test.json',
        '',
        zipped=False,
    ),
    DownloadableFile(
        'https://huggingface.co/datasets/vblagoje/lfqa/resolve/main/train.json',
        'train.json',
        '',
        zipped=False,
    ),
    DownloadableFile(
        'https://huggingface.co/datasets/vblagoje/lfqa/resolve/main/validation.json',
        'valid.json',
        '',
        zipped=False,
    ),
]

def build(opt):
    dpath = pjoin(opt['datapath'], 'lfqa')
    version = '1.0'

    #if not build_data.built(dpath, version_string=version):
       # print('[building data: ' + dpath + ']')
        #if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
     #       build_data.remove_dir(dpath)
      #  build_data.make_dir(dpath)

        # Download the data.
       # for downloadable_file in RESOURCES[:2]:
        #    downloadable_file.download_file(dpath)

        # Mark the data as built.
    build_data.mark_done(dpath, version_string=version)
