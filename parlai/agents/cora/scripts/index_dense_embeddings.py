#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Create a FAISS Index with a series of dense embeddings.
"""
import os
import random
import torch
from typing import List
import pickle
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript
import parlai.utils.logging as logging

from parlai.agents.cora.cora import CoraAgent
from parlai.agents.cora.indexers import indexer_factory, CompressedIndexer


class Indexer(ParlaiScript):
    """
    Index Dense Embeddings.
    """

    @classmethod
    def setup_args(cls):
        """
        Setup args.
        """
        parser = ParlaiParser(True, True, 'Index Dense Embs')
        parser.add_argument(
            '--embeddings-dir', type=str, help='directory of embeddings'
        )
        parser.add_argument(
            '--embeddings-name', type=str, default='', help='name of emb part'
        )
        parser.add_argument(
            '--partition-index',
            type='bool',
            default=False,
            help='specify True to partition indexing per file (useful when all files do not fit into memory)',
        )
        parser.add_argument(
            '--save-index-dir',
            type=str,
            help='directory in which to save index',
            default=None,
        )
        parser.add_argument(
            '--num-shards',
            type=int,
            default=1,
            help='how many workers to use to split up the work',
        )
        parser.add_argument(
            '--shard-id',
            type=int,
            help='shard id for this worker. should be between 0 and num_shards',
        )
        parser = CoraAgent.add_cmdline_args(parser)
        parser.set_defaults(compressed_indexer_gpu_train=True)
        return parser

    def run(self):
        """
        Load dense embeddings and index with FAISS.
        """
        # create index
        index_dir = self.opt['embeddings_dir']
        embs_name = (
            f"{self.opt['embeddings_name']}_" if self.opt['embeddings_name'] else ''
        )
        
     #   print(f"index_dir is {index_dir}")
     #   print(f"embs_name is {embs_name}")
        
            
        en_num_parts = len([f for f in os.listdir(index_dir)
                            if 'sample' not in f and 'en' in f])
        
        others_num_parts = len([f for f in os.listdir(index_dir)
                            if 'sample' not in f and 'others' in f]) -4
     #  print(f"en_num_parts is {en_num_parts}")
     #   print(f"others_num_parts is {others_num_parts}")
        
        en_input_files = [
            os.path.join(index_dir, f'{embs_name}en_{i}') for i in range(en_num_parts)
        ]
        
        others_input_files = [
            os.path.join(index_dir, f'{embs_name}others_{i}') for i in range(others_num_parts)
        ]
     #   print("type of en_input_files is ", type(en_input_files))
        input_files = en_input_files + others_input_files
        
        input_files = others_input_files
        
      #  print("this is the input files", input_files)
        
        if self.opt['indexer_type'] == 'compressed':
            index_name = self.opt['compressed_indexer_factory'].replace(',', '__')
        elif self.opt['embeddings_name']:
            index_name = self.opt['embeddings_name']
        else:
            index_name = 'hnsw_flat'
        index_path = os.path.join(index_dir, index_name)

        if self.opt['save_index_dir']:
            index_path, index_name = os.path.split(index_path)
            index_path = os.path.join(self.opt['save_index_dir'], index_name)
            if not os.path.exists(self.opt['save_index_dir']):
                logging.info(f'Creating directory for file {index_path}')
                os.makedirs(self.opt['save_index_dir'])

        print(f'index path: {index_path}')
        self.index_path = index_path

        self.index = indexer_factory(self.opt)
        if self.opt['indexer_type'] != 'exact':
            self.train_then_add(input_files)
        else:
            self.index_data(input_files)
        # save data
        self.index.serialize(index_path)


    def iterate_encoded_files(self, input_files: List[str]):
        for i, file in enumerate(input_files):
            print('Reading file ', file)
            with open(file, "rb") as reader:
                doc_vectors = pickle.load(reader)
                for doc in doc_vectors:
                    db_id, doc_vector = doc
                    yield db_id, doc_vector
                
                
    def index_data(self, input_files: List[str], add_only: bool = False):
        """
        Index data.

        :param input_files:
            files to load.
        """
        all_docs = []
        for i, item in enumerate(self.iterate_encoded_files(input_files)):
            logging.info(f'Loading in file {item}')
            db_id, doc_vector = item
            all_docs.append((db_id, doc_vector))

        self.index.index_data(all_docs)
        logging.info('Data indexing completed.')


    def train_then_add(self, input_files: List[str]):
        """
        First train data, then add it.

        If we're only training... then don't add!!
        """
        assert isinstance(self.index, CompressedIndexer)
        # Train
        random.seed(42)

        tensors = []
        for i, item in enumerate(self.iterate_encoded_files(input_files)):
            logging.info(f'Loading in file {item}')
            db_id, doc_vector = item
            tensor_doc_vector = torch.from_numpy(doc_vector)
          #  tensor_db_id = torch.from_numpy(db_id)
            
            if self.opt['partition_index']:
                self.index.train([tensor_doc_vector])
                self.index.add([tensor_doc_vector])
            else:
                tensors.append(tensor_doc_vector)

        if not self.opt['partition_index']:
            self.index.train([torch.cat(tensors)])
            self.index.add([torch.cat(tensors)])


if __name__ == "__main__":
    Indexer.main()
