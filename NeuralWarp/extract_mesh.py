import argparse
import os
import sys
sys.path.append('..')
from spheres.trainer import SpheresTrainer
sys.path.append('..')
from pyhocon import ConfigFactory
import torch
import numpy as np
from tqdm import tqdm
import utils.general as utils
from skimage import morphology as morph
from evaluation import get_mesh
from datasets import scene_dataset
from model.neuralWarp import NeuralWarp, NeuralWarpSpheres
from evaluation import mesh_filtering
from pathlib import Path


def extract_mesh(args):
    torch.set_default_dtype(torch.float32)

    conf = ConfigFactory.parse_file(args.conf)

    exps_folder_name = Path("exps") / args.exp_name
    evals_folder_name = Path("evals") / args.exp_name
 
    expname = args.conf.split("/")[-1].split(".")[0]
    scene = args.scene
    if scene is not None:
        expname = expname + '_{0}'.format(scene)

    if args.timestamp == 'latest':
        if (exps_folder_name / expname).exists():
            timestamps = list((exps_folder_name / expname).iterdir())
            if (len(timestamps)) == 0:
                print('WRONG EXP FOLDER')
                exit()
            else:
                timestamp = sorted(timestamps)[-1].name
        else:
            print('WRONG EXP FOLDER')
            exit()
    else:
        timestamp = args.timestamp

    utils.mkdir_ifnotexists(evals_folder_name)
    expdir = exps_folder_name / expname
    evaldir = evals_folder_name / expname
    utils.mkdir_ifnotexists(evaldir)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.spheres_conf is not None:
        spheres_trainer = SpheresTrainer(args.spheres_conf, device=device)
        model = NeuralWarpSpheres(conf.get_config('model')).to(device)
        spheres_trainer.run_model = lambda model, x: -model(x)[..., 0].clone()
    else:
        spheres_trainer = None
        model = NeuralWarp(conf=conf.get_config('model')).to(device)

    dataset_conf = conf.get_config('dataset')
    if args.scene is not None:
        dataset_conf['scene'] = args.scene

    old_checkpnts_dir = os.path.join(expdir, timestamp, 'checkpoints')

    saved_model_state = torch.load(os.path.join(old_checkpnts_dir, 'ModelParameters', str(args.checkpoint) + ".pth"))
    model.load_state_dict(saved_model_state["model_state_dict"], strict=False)
    if spheres_trainer is not None:
        spheres_trainer.load_weights(os.path.join(old_checkpnts_dir, 'SpheresParameters', str(args.checkpoint) + ".pth"))

    if 'nerf' in dataset_conf['data_dir']:
        eval_dataset = scene_dataset.NerfDataset(**dataset_conf)
    else:
        eval_dataset = scene_dataset.SceneDataset(**dataset_conf)
    scale_mat = eval_dataset.get_scale_mat()
    num_images = len(eval_dataset)
    K = eval_dataset.intrinsics_all
    pose = eval_dataset.pose_all
    masks = eval_dataset.org_object_masks

    print("dilation...")
    dilated_masks = list()

    for m in tqdm(masks, desc="Mask dilation"):
        if args.no_masks:
            dilated_masks.append(torch.ones_like(m, device="cuda"))
        else:
            struct_elem = morph.disk(args.dilation_radius)
            dilated_masks.append(torch.from_numpy(morph.binary_dilation(m.numpy(), struct_elem)))
    masks = torch.stack(dilated_masks).cuda()

    model.eval()

    with torch.no_grad():
        size = conf.dataset.img_res[::-1]
        pose = pose.cuda()
        cams = [
            K[:, :3, :3].cuda(),
            pose[:, :3, :3].transpose(2, 1),
            - pose[:, :3, :3].transpose(2, 1) @ pose[:, :3, 3:],
            torch.tensor([size for i in range(num_images)]).cuda().float()
        ]

        mesh = get_mesh.get_surface_high_res_mesh(
            sdf=lambda x: model.implicit_network(x)[:, 0], refine_bb=not args.no_refine_bb,
            resolution=args.resolution, cams=cams, masks=masks, bbox_size=args.bbox_size,
            spheres=None if spheres_trainer is None else spheres_trainer.spheres
        )

        mesh_filtering.mesh_filter(args, mesh, masks, cams)  # inplace filtering

    if args.one_cc: # Taking the biggest connected component
        components = mesh.split(only_watertight=False)
        areas = np.array([c.area for c in components], dtype=float)
        mesh = components[areas.argmax()]

    # Transform to world coordinates
    mesh.apply_transform(scale_mat)
    mesh.export(evaldir / f'output_mesh{args.suffix}.ply', 'ply')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str)
    parser.add_argument('--timestamp', default='latest', type=str, help='The experiment timestamp to test.')
    parser.add_argument('--checkpoint', default='latest',type=str,help='The trained model checkpoint to test')
    parser.add_argument('--scene', type=str, default=None, help='If set, taken to be the scan id.')
    parser.add_argument('--resolution', default=512, type=int, help='Grid resolution for marching cube')
    parser.add_argument('--no_refine_bb', action="store_true", help='Skip bounding box refinement')
    parser.add_argument("--bbox_size", default=1., type=float, help="Size of the bounding volume to querry")
    parser.add_argument("--one_cc", action="store_true", default=True,
                        help="Keep only the biggest connected component or all")
    parser.add_argument("--no_one_cc", action="store_false", dest="one_cc")
    parser.add_argument("--filter_visible_triangles", action="store_true",
                        help="Whether to remove triangles that have no projection in images (uses mesh rasterization)")
    parser.add_argument('--min_nb_visible', type=int, default=2, help="Minimum number of images used for visual hull"
                                                                      "filtering and triangle visibility filtering")
    parser.add_argument("--no_masks", action="store_true", help="Ignore the visual hull masks")
    parser.add_argument("--dilation_radius", type=int, default=12)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--spheres_conf", type=str)
    args = parser.parse_args()

    extract_mesh(args)
