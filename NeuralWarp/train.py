import argparse

import torch
import numpy as np
import sys
sys.path.append('..')
from spheres.trainer import SpheresTrainer

from training.trainer import Trainer
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/NeuralWarp.conf')
    parser.add_argument('--is_continue', default=False, action="store_true",
                        help='If set, indicates continuing from a previous run.')
    parser.add_argument('--timestamp', default='latest', type=str,
                        help='The timestamp of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--checkpoint', default='latest', type=str,
                        help='The checkpoint iteration number of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--scene', default=None, help='If set, taken to be the scan id.')
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--exp_name", type=str)
    parser.add_argument('--spheres_conf', type=str, default=None, help='Use spheres config')
    
    opt = parser.parse_args()

    set_seed(42)

    if opt.spheres_conf:
        spheres_trainer = SpheresTrainer(opt.spheres_conf, device="cuda")
    else:
        spheres_trainer = None

    set_seed(opt.seed)

    trainrunner = Trainer(spheres_trainer=spheres_trainer,
                          conf=opt.conf,
                          is_continue=opt.is_continue,
                          timestamp=opt.timestamp,
                          checkpoint=opt.checkpoint,
                          scene=opt.scene,
                          debug=opt.debug,
                          exp_name=opt.exp_name,
                          spheres_config=opt.spheres_conf
                          )

    trainrunner.run()
