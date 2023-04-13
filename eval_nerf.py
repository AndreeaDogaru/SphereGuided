import argparse
import trimesh
import numpy as np
import json
from pathlib import Path
import torch
import sys
sys.path.append("./NeuralWarp")
from evaluation.mesh_filtering import *
from scipy.spatial import KDTree
from utils import utils_3D
import imageio.v2 as imageio
from skimage import morphology as morph
import cv2


def compute_chamfer_meshes(gt:trimesh.Trimesh, pred: trimesh.Trimesh, num_points=1000000, norm=2):
    """
        Computes chamfer distance between uniformly sampled points from both meshes

        Returns:
                chamf_pred_gt: mean chamfer l2 from predicted to ground truth 
                chamf_gt_pred: mean chamfer l2 from ground truth to predictedeturns
    """
    gt_points = np.asarray(gt.as_open3d.sample_points_uniformly(num_points).points)
    pred_points = np.asarray(pred.as_open3d.sample_points_uniformly(num_points).points)

    dist_gt_pred = KDTree(pred_points).query(gt_points, p=norm)[0]
    dist_pred_gt = KDTree(gt_points).query(pred_points, p=norm)[0]

    chamf_gt_pred = np.mean(dist_gt_pred)
    chamf_pred_gt = np.mean(dist_pred_gt)

    return {
        'accuracy': chamf_pred_gt,
        'completeness': chamf_gt_pred,
        'chamfer': (chamf_gt_pred + chamf_pred_gt) / 2
    }


def cut_mesh(vertices, faces=None, normalization_mat=np.eye(4), radius=1):
    normalized_vertices = vertices @ normalization_mat[:3, :3].T + normalization_mat[:3, -1]
    in_sphere = (normalized_vertices ** 2).sum(1) <= radius
    new_vertices = vertices[in_sphere]
    new_indices = np.empty(len(vertices)) 
    if faces is not None:
        new_indices[in_sphere] = np.arange(in_sphere.sum())
        new_faces = new_indices[faces[np.all(in_sphere[faces], 1)]]
        return new_vertices, new_faces
    else:
        return new_vertices


def visual_hull_mask(full_points, cams, masks, nb_visible=1):
    # K, R: N * 3 * 3 // t: N * 3 * 1; sizes N*2; points n*3; masks: N * h * w
    K, R, t, sizes = cams[:4]
    res = list()

    with torch.no_grad():
        for points in torch.split(full_points, 100000):
            n = points.shape[0]

            proj = (K @ (R @ points.view(n, 1, 3, 1) + t)).squeeze(-1) # n * N * 3
            in_cam_mask = proj[..., 2] > 1e-8
            proj = proj[..., :2] / torch.clamp(proj[..., 2:], 1e-8)

            proj = utils_3D.normalize(proj, sizes[0:1, 1], sizes[0:1, 0])
            grid = torch.clamp(proj.transpose(0, 1).unsqueeze(1), -10, 10)
            warped_masks = F.grid_sample(masks.unsqueeze(1).cuda(), grid, align_corners=True).squeeze().t() > 0

            # in cam_mask: n * N
            in_cam_mask = in_cam_mask & (proj <= 1).all(dim=-1) & (proj >= -1).all(dim=-1)
            is_not_obj_mask = in_cam_mask & ~warped_masks

            res.append((in_cam_mask.sum(dim=1) >= nb_visible) & ~(is_not_obj_mask.any(dim=1)))

    return torch.cat(res)


def filter_mesh(mesh, cams, resolution, masks):
    num_faces = len(mesh.faces)
    count = torch.zeros(num_faces, device="cuda")

    K, R, t, sizes = cams[:4]

    n = len(K)
    intr = torch.zeros((1, 4, 4), device="cuda")
    intr[:, :3, :3] = K[:1]
    intr[:, 3, 3] = 1
    vertices = torch.from_numpy(mesh.vertices).cuda().float()
    faces = torch.from_numpy(mesh.faces).cuda().long()
    meshes = Meshes(verts=[vertices],
                    faces=[faces])
    with torch.no_grad():
        for i in tqdm(range(n), desc="Rasterization"):
            cam = corrected_cameras_from_opencv_projection(camera_matrix=intr, R=R[i:i + 1].cuda(),
                                                        tvec=t[i:i + 1].squeeze(2).cuda(),
                                                        image_size=sizes[i:i + 1, [1, 0]].cuda())
            cam = cam.cuda()
            raster_settings = rasterizer.RasterizationSettings(image_size=[resolution, resolution],
                                                            faces_per_pixel=1, cull_backfaces=False, bin_size=0)
            meshRasterizer = rasterizer.MeshRasterizer(cam, raster_settings)

            f = meshRasterizer(meshes)
            pix_to_face, zbuf, bar, pixd = f.pix_to_face, f.zbuf, f.bary_coords, f.dists

            visible_faces = pix_to_face.view(-1).unique()
            count[visible_faces[visible_faces > -1]] += 1       
        valid_faces_mask = (count > 0).cpu().numpy()
        if masks is not None:
            vert_hull_mask = visual_hull_mask(torch.from_numpy(mesh.vertices).float().cuda(), cams, masks, 1)
            hull_mask = vert_hull_mask[mesh.faces].all(dim=-1).cpu().numpy()
            valid_faces_mask &= hull_mask
    return trimesh.Trimesh(mesh.vertices, mesh.faces[valid_faces_mask])



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str)
    parser.add_argument('--resolution', default=800, type=int)
    parser.add_argument('--mesh_path', type=str)
    parser.add_argument('--evaluate', action="store_true", default=True)
    parser.add_argument('--subdivide', action="store_true")
    parser.add_argument('--mask_filter', action="store_true", default=True)
    parser.add_argument("--data_dir", type=str, help='Path to data')

    args = parser.parse_args()
    print(f"Evaluating scene {args.scene} with mesh {args.mesh_path}")

    scene_dir = Path(args.data_dir) / args.scene
    mesh_path = Path(args.mesh_path)
    mesh = trimesh.load(mesh_path)
    if args.subdivide:
        mesh = trimesh.Trimesh(*trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=0.1, max_iter=10))

    save_path = mesh_path.parent / f"{mesh_path.stem}_visible.ply"

    h, w = 800, 800

    with open(scene_dir / "transforms_train.json") as fp:
        camera_dict = json.load(fp)
    pose_all = []
    if args.mask_filter:
        masks = []
        dilation_radius = 12
        struct_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_radius * 2 + 1,) * 2)
    for frame in camera_dict['frames']:
        pose = np.array(frame['transform_matrix'])
        pose[:, 1:3] *= -1
        pose_all.append(pose)
        if args.mask_filter:
            mask = imageio.imread(scene_dir / f"{frame['file_path']}.png")[..., -1] / 256.0 > 0.2
            dilated_mask = torch.from_numpy(morph.dilation(mask, struct_elem)).float()[None]
            masks.append(dilated_mask)
    if args.mask_filter:
        masks = torch.cat(masks, dim=0)
    else:
        masks = None

    camera_angle_x = float(camera_dict['camera_angle_x'])
    focal = .5 * w / np.tan(.5 * camera_angle_x)
    intrinsics = np.eye(4)
    intrinsics[[0, 1], [0, 1]] = focal
    intrinsics[0, 2] = w / 2 
    intrinsics[1, 2] = h / 2 

    scale_mat = np.eye(4, 4)
    scale_mat[:3, -1] += camera_dict['translation']
    scale_mat[:3] *= camera_dict['scale']

    num_images = len(pose_all)
    pose = torch.tensor(np.array(pose_all)).float().cuda()
    K = torch.tensor(intrinsics).float().expand(num_images, -1, -1).cuda()
    size = [w, h]
    cams = [
            K[:, :3, :3].cuda(),
            pose[:, :3, :3].transpose(2, 1),
            - pose[:, :3, :3].transpose(2, 1) @ pose[:, :3, 3:],
            torch.tensor([size for i in range(num_images)]).cuda().float()
        ]

    if not save_path.exists():
        res = filter_mesh(mesh, cams, args.resolution, masks)
        res.export(save_path)      
    else:
        res = trimesh.load(save_path)

    res = trimesh.Trimesh(*cut_mesh(res.vertices, res.faces, scale_mat))
    res.export(mesh_path.parent / f"{mesh_path.stem}_visible_cut.ply")

    if args.evaluate:
        gt_mesh = trimesh.load(scene_dir / f"{args.scene}_visible.ply")
        print(compute_chamfer_meshes(gt_mesh, res))
    