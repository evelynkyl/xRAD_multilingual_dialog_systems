#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
Main launch script for single-host, multi-GPU training.

This is a drop-in replacement for [train_model]. This script will
launch N subprocess, each which runs the full training loop
independently.

Uses torch.nn.parallel.DistributedDataParallel for its main uses. Agents
must specifically implement the wrapper of DistributedDatParallel, but
all TorchRankerAgents and TorchGeneratorAgents support this.

## Examples

```shell
parlai tensorprocessing_train -m transformer/generator --batchsize 16 --task convai2 --model-file /tmp/mymodel
```
"""

import torch
import torch_xla.core.xla_model as xm
import os
import signal
import traceback
import parlai.scripts.train_model as single_train
import parlai.utils.distributed as distributed_utils
from parlai.core.script import ParlaiScript, register_script

device = xm.xla_device()

def setup_args():
    parser = single_train.setup_args()
    parser.add_distributed_training_args()
    parser.add_argument('--tpu', type=int, default=61337, help='TCP port number')
  #  parser.add_argument('--port', type=int, default=61337, help='TCP port number')
    return parser


class TPUTrain(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
   #    with distributed_utils.slurm_distributed_context(self.opt) as opt:
        try:
            return single_train.TrainLoop(opt).train()
        except Exception:
            import parlai.utils.logging as logging

            logging.critical(traceback.format_exc())
            logging.critical(
                f"Got the above exception on worker. "
                "This may cause hangs requiring manual killing of processes."
            )
            raise
      #      self.train_loop = single_train.TrainLoop(opt)
         #   return self.train_loop.train()


if __name__ == '__main__':
    TPUTrain.main()

