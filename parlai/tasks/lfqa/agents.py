#!/usr/bin/env python3

from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.teachers import FixedDialogTeacher
from parlai.utils.io import PathManager
from .build import build
import os
import json


class LFQATeacher(FixedDialogTeacher):
    """
    LFQA Teacher, based on ELI5 Teacher, taken from https://github.com/facebookresearch/ELI5.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        build(opt)
        self.id = 'lfqa'
        self.messages = self.load_lfqa(opt)
        self.reset()

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        group = parser.add_argument_group('LFQA Knowledge arguments')
        group.add_argument(
            '--knowledge',
            type='bool',
            default=True,
            help='Whether to include supporting document knowledge',
        )
        return parser

    def load_lfqa(self, opt):
        """
        Load data based on data split.
        """
        dp = opt['datapath']
        dt = opt['datatype'].split(':')[0]
        eli_path = "/home/evelyn/parley/data/lfqa/"
        fname = os.path.join(dp, eli_path + dt + ".json")
        if not PathManager.exists(fname):
            raise FileNotFoundError(
                f"{fname} not found. Please follow the instructions found at "
                "https://github.com/facebookresearch/ParlAI/tree/main/parlai/tasks/eli5/README.md"
                " to construct the dataset."
            )
        opt['datafile'] = fname
    #    with PathManager.open(fname) as json_file:
     #       data = json.load(json_file)
        data = []
        for line in open(fname, 'r'):
            data.append(json.loads(line))
            
        ds = []
        for d in data:
            act = {
                'id': 'lfqa',
                'text': d['title'],
                'labels': [d['answers']['text'][0]],
                'episode_done': True,
            }
            ds.append(act)
        return ds

    def get(self, episode_idx, entry_idx=0):
        return self.messages[episode_idx]

    def num_examples(self):
        return len(self.messages)

    def num_episodes(self):
        return len(self.messages)


class DefaultTeacher(LFQATeacher):
    pass
