import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile
from parlai.utils.io import PathManager

RESOURCES = [
    DownloadableFile(
        'https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_dev_full_v1_1.jsonl',
        'dev-v1.1.json',
        '<checksum for this file>',
        zipped=False,
    ),
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'xorqa')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES[:2]:
            downloadable_file.download_file(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)