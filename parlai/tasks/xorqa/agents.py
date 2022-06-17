from parlai.core.params import ParlaiParser
from .build import build
import copy
import json
import os
from parlai.core.opt import Opt
from parlai.core.agents import create_agents_from_shared
from parlai.core.agents import create_agent
from parlai.core.teachers import ParlAIDialogTeacher

class DefaultTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        suffix = 'train' if opt['datatype'].startswith('train') else 'dev'
        opt = copy.deepcopy(opt)

        # get datafile
        take_file = f"/home/evelyn/parley/data/xorqa/tgt_language/{suffix}.txt"
        # for eval in En (baseline):  f"/home/evelyn/ParlAI/data/xorqa/preprocessed/translated_to_en/{suffix}.txt" 
        opt['parlaidialogteacher_datafile'] = take_file

        super().__init__(opt, shared)
        

# "/home/evelyn/ParlAI/parlai/tasks/xorqa" # 
def _path(opt, filtered):
    # build the data if it does not exist
    build(opt)

    # set up path to data (specific to each dataset)
    dt = opt['datatype'].split(':')[0]
    return "/home/evelyn/ParlAI/parlai/tasks/xorqa/train.txt"
  #  return os.path.join(opt['datapath'], 'xorqa', dt + '.txt')
