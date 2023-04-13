import os
import sys
from pathlib import Path

sys.path.append('..')
from spheres.trainer import SpheresTrainer
import argparse
import time

import torch
from scipy.spatial.transform import Rotation as R

from dataloading import get_dataloader, load_config
from model.checkpoints import CheckpointIO
from model.network import NeuralNetwork
from model.common import transform_mesh
from model.extracting import Extractor3D, post_filtering


torch.manual_seed(0)

# Config
parser = argparse.ArgumentParser(
    description='Extract meshes from occupancy process.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--upsampling-steps', type=int, default=-1,
                    help='Overrites the default upsampling steps in config')
parser.add_argument('--refinement-step', type=int, default=-1,
                    help='Overrites the default refinement steps in config')
parser.add_argument('--eval-iter', type=int, default=-1,
                    help='Overrites the model file in config')
parser.add_argument('--exp-name', type=str, default='out_null', help='Name of experiment')
parser.add_argument('--spheres-config', type=str, default='configs/sphere.yaml', help='Path to the spheres config')
parser.add_argument('--filter', type=str, choices=['none', 'pre', 'post'], default='pre', help='What kind of filtering to perform')

args = parser.parse_args()
cfg = load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

if args.upsampling_steps != -1:
    cfg['extraction']['upsampling_steps'] = args.upsampling_steps
if args.refinement_step != -1:
    cfg['extraction']['refinement_step'] = args.refinement_step
if args.eval_iter != -1:
    cfg['extraction']['model_file'] = f"model_{args.eval_iter}.pt"

out_dir = cfg['training']['out_dir']
out_dir = os.path.join(args.exp_name, *os.path.normpath(out_dir).split(os.sep)[1:])
generation_dir = os.path.join(out_dir, cfg['extraction']['extraction_dir'])
if args.eval_iter != -1:
    generation_dir = os.path.join(generation_dir, f"{args.eval_iter}")
else:
    generation_dir = os.path.join(generation_dir, "pretrained")
    
mesh_extension = cfg['extraction']['mesh_extension']

# Model
model_cfg = cfg['model']
model = NeuralNetwork(model_cfg)
checkpoint_io = CheckpointIO(out_dir, model=model)
checkpoint_io.load(cfg['extraction']['model_file'])
 
# Generator
generator = Extractor3D(
    model, resolution0=cfg['extraction']['resolution'], 
    upsampling_steps=cfg['extraction']['upsampling_steps'], 
    device=device
)

# Dataloading
test_loader = get_dataloader(cfg, mode='test')
iter_test = iter(test_loader)
data_test = next(iter_test)

filter_choice = args.filter
if filter_choice == 'none':
    test_mask_loader = None
else:
    test_mask_loader = get_dataloader(
        cfg, mode='test', shuffle=False,
        spilt_model_for_images=True, with_mask=True
    )

sphere_ckpt = Path(out_dir) / cfg['extraction']['model_file'].replace('model', 'spheres')
if args.spheres_config and sphere_ckpt.exists():
    spheres_trainer = SpheresTrainer(args.spheres_config, lambda model, x: -model(x, return_logits=True).squeeze(-1) * 10, device)
    spheres_trainer.load_weights(str(sphere_ckpt))
    spheres = spheres_trainer.spheres
    spheres.update_radius(args.eval_iter)
else:
    spheres = None

# Generate
model.eval()
if filter_choice == 'pre':
    mesh_dir = os.path.join(generation_dir, 'meshes_prefiltered')
elif filter_choice == 'post':
    mesh_dir = os.path.join(generation_dir, 'meshes_postfiltered')
else:
    mesh_dir = os.path.join(generation_dir, 'meshes')
if not os.path.exists(mesh_dir):
    os.makedirs(mesh_dir)

try:
    t0 = time.time()
    if filter_choice == 'pre':
        out = generator.generate_mesh(mask_loader=test_mask_loader, spheres=spheres)
    else:
        out = generator.generate_mesh(mask_loader=None, spheres=spheres)

    try:
        mesh, stats_dict = out
    except TypeError:
        mesh, stats_dict = out, {}

    # For DTU transformed-back mesh file
    scale_mat = data_test.get('img.scale_mat')[0]
    mesh_transformed = transform_mesh(mesh, scale_mat)
    if filter_choice == 'post':
        mesh_transformed = post_filtering(mesh_transformed, test_mask_loader)
    mesh_out_file = os.path.join(
        mesh_dir, 'scan_world_scale.%s' %mesh_extension)
    mesh_transformed.export(mesh_out_file)

except RuntimeError:
    print("Error generating mesh")

