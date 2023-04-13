import os
import sys
sys.path.append('..')
import logging
import time
import shutil
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.optim as optim
from distutils import version
from torch.utils.tensorboard import SummaryWriter
import dataloading as dl
import model as mdl

from spheres.trainer import SpheresTrainer

logger_py = logging.getLogger(__name__)

# Fix seeds
np.random.seed(42)
torch.manual_seed(42)

# Arguments
parser = argparse.ArgumentParser(
    description='Training of UNISURF model'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of '
                         'seconds with exit code 2.')
parser.add_argument('--spheres-config', type=str, default=None, help='Use spheres config')
parser.add_argument('--exp-name', type=str, default='out', help='Name of experiment')
parser.add_argument('--max-iter', type=int, default=450000, help='Number of iterations')

args = parser.parse_args()
cfg = dl.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

# params
out_dir = cfg['training']['out_dir']
out_dir = os.path.join(args.exp_name, *os.path.normpath(out_dir).split(os.sep)[1:])
backup_every = cfg['training']['backup_every']
exit_after = args.exit_after
batch_size = cfg['training']['batch_size']
n_workers = cfg['dataloading']['n_workers']
lr = cfg['training']['learning_rate']

# init dataloader
train_loader = dl.get_dataloader(cfg, mode='train')
test_loader = dl.get_dataloader(cfg, mode='test')
iter_test = iter(test_loader)
data_test = next(iter_test)

# init spheres
if args.spheres_config is not None:
    spheres_trainer = SpheresTrainer(args.spheres_config, lambda model, x: -model(x, return_logits=True).squeeze(-1) * 10, device)
else:
    spheres_trainer = None

# init network
model_cfg = cfg['model']
model = mdl.NeuralNetwork(model_cfg)
print(model)

# init renderer
rendering_cfg = cfg['rendering']
if spheres_trainer is None:
    renderer = mdl.Renderer(model, rendering_cfg, device=device)
else:
    renderer = mdl.SphereRenderer(model, rendering_cfg, spheres_trainer.spheres, device=device)

# init optimizer
weight_decay = cfg['training']['weight_decay']
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# init training
training_cfg = cfg['training']
trainer = mdl.Trainer(renderer, optimizer, training_cfg, device=device)

# init checkpoints and load
checkpoint_io = mdl.CheckpointIO(out_dir, model=model, optimizer=optimizer)

try:
    load_dict = checkpoint_io.load('model.pt')
    if spheres_trainer is not None:
        spheres_trainer.load_weights(os.path.join(out_dir, 'spheres.pt'))
except FileExistsError:
    load_dict = dict()
epoch_it = load_dict.get('epoch_it', -1)
it = load_dict.get('it', -1)

scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, cfg['training']['scheduler_milestones'],
    gamma=cfg['training']['scheduler_gamma'], last_epoch=epoch_it)

logger = SummaryWriter(os.path.join(out_dir, 'logs'))
    
# init training output
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
visualize_every = cfg['training']['visualize_every']
render_path = os.path.join(out_dir, 'rendering')
if visualize_every > 0:
    visualize_skip = cfg['training']['visualize_skip']
    visualize_path = os.path.join(out_dir, 'visualize')
    if not os.path.exists(visualize_path):
        os.makedirs(visualize_path)


# Print model
nparameters = sum(p.numel() for p in model.parameters())
logger_py.info(model)
logger_py.info('Total number of parameters: %d' % nparameters)
t0b = time.time()


while True:
    epoch_it += 1

    for batch in train_loader:
        it += 1
        loss_dict = trainer.train_step(batch, it, spheres_trainer=spheres_trainer)
        if spheres_trainer is not None:
            spheres_trainer.spheres.update_radius(it)
            loss_dict.update(spheres_trainer.train_step(model, it))
        loss = loss_dict['loss']
        metric_val_best = loss
        # Print output
        if print_every > 0 and (it % print_every) == 0:
            print('[Epoch %02d] it=%03d, loss=%.4f, time=%.4f'
                           % (epoch_it, it, loss, time.time() - t0b))
            logger_py.info('[Epoch %02d] it=%03d, loss=%.4f, time=%.4f'
                           % (epoch_it, it, loss, time.time() - t0b))
            t0b = time.time()
            for l, num in loss_dict.items():
                logger.add_scalar('train/'+l, num.detach().cpu(), it)
        
        if visualize_every > 0 and (it % visualize_every)==0:
            logger_py.info("Rendering")
            out_render_path = os.path.join(render_path, '%04d_vis' % it)
            if not os.path.exists(out_render_path):
                os.makedirs(out_render_path)
            val_rgb = trainer.render_visdata(
                        data_test, 
                        cfg['training']['vis_resolution'], 
                        it, out_render_path)
            #logger.add_image('rgb', val_rgb, it)
        
        # Save checkpoint
        if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
            logger_py.info('Saving checkpoint')
            print('Saving checkpoint')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
            if spheres_trainer is not None:
                spheres_trainer.save_weights(os.path.join(out_dir, 'spheres.pt'))

        # Backup if necessary
        if (backup_every > 0 and (it % backup_every) == 0):
            logger_py.info('Backup checkpoint')
            checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
            if spheres_trainer is not None:
                spheres_trainer.save_weights(os.path.join(out_dir, 'spheres_%d.pt' % it))
        # Backup if necessary
        if it >= args.max_iter:
            break
    if it >= args.max_iter:
        break
    scheduler.step()
