import argparse
from pathlib import Path

import multiprocessing as mp
from typing import List

import trimesh
import sklearn.neighbors as skln
from scipy.io import loadmat

import numpy as np


def eval_geometry(data_dir, pred, scene, norm=2):
    """
        Code adapted from  https://github.com/jzhangbs/DTUeval-python

        Returns:
            mean_d2s: mean chamfer l2 from predicted to ground truth 
            mean_s2d: mean chamfer l2 from ground truth to predicted
    """
    gt = trimesh.load(data_dir / 'Points' / 'stl' / f'stl{str(scene).zfill(3)}_total.ply')
    max_dist = 20
    patch = 60
    thresh = 0.2
    
    tri_vert = pred.vertices[pred.faces]
    v1 = tri_vert[:, 1] - tri_vert[:, 0]
    v2 = tri_vert[:, 2] - tri_vert[:, 0]
    l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
    non_zero_area = (area2 > 0)[:, 0]
    l1, l2, area2, v1, v2, tri_vert = [
        arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]
    ]
    thr = thresh * np.sqrt(l1 * l2 / area2)
    n1 = np.floor(l1 / thr)
    n2 = np.floor(l2 / thr)

    with mp.Pool() as mp_pool:
        new_pts = mp_pool.map(_sample_single_tri,
                            ((n1[i, 0], n2[i, 0], v1[i:i + 1], v2[i:i + 1], tri_vert[i:i + 1, 0]) for i in
                            range(len(n1))), chunksize=1024)
        new_pts = np.concatenate(new_pts, axis=0)
        data_pcd = np.concatenate([pred.vertices, new_pts], axis=0)

    shuffle_rng = np.random.default_rng()
    shuffle_rng.shuffle(data_pcd, axis=0)
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1, p=norm)
    nn_engine.fit(data_pcd)
    rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=thresh, return_distance=False)
    mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1
    data_down = data_pcd[mask]
    obs_mask_file = loadmat(data_dir / 'ObsMask' / f'ObsMask{scene}_10.mat')
    ObsMask, BB, Res = [obs_mask_file[attr] for attr in ['ObsMask', 'BB', 'Res']]
    BB = BB.astype(np.float32)
    inbound = ((data_down >= BB[:1] - patch) & (data_down < BB[1:] + patch * 2)).sum(axis=-1) == 3
    data_in = data_down[inbound]
    
    data_grid = np.around((data_in - BB[:1]) / Res).astype(np.int32)
    grid_inbound = ((data_grid >= 0) & (data_grid < np.expand_dims(ObsMask.shape, 0))).sum(axis=-1) == 3
    data_grid_in = data_grid[grid_inbound]
    in_obs = ObsMask[data_grid_in[:, 0], data_grid_in[:, 1], data_grid_in[:, 2]].astype(np.bool_)
    data_in_obs = data_in[grid_inbound][in_obs]
    
    nn_engine.fit(gt.vertices)
    dist_d2s, idx_d2s = nn_engine.kneighbors(data_in_obs, n_neighbors=1, return_distance=True)
    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()
    ground_plane = loadmat(data_dir / 'ObsMask' / f'Plane{scene}.mat')['P']

    stl_hom = np.concatenate([gt.vertices, np.ones_like(gt.vertices[:, :1])], -1)
    above = (ground_plane.reshape((1, 4)) * stl_hom).sum(-1) > 0

    stl_above = gt.vertices[above]
    nn_engine.fit(data_in)
    dist_s2d, idx_s2d = nn_engine.kneighbors(stl_above, n_neighbors=1, return_distance=True)
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

    return mean_d2s, mean_s2d

def _sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:n1 + 1, :n2 + 1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1, 2, 0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:, :1] + v2 * k[:, 1:] + tri_vert
    return q

def get_metrics(data_dir, mesh_path, scene, chamfer_norm=2):
    pred = trimesh.load(mesh_path)
    acc, comp = eval_geometry(data_dir, pred, scene, norm=chamfer_norm)
    return {
        'accuracy': acc,
        'completeness': comp,
        'chamfer': (acc + comp) / 2
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help='Path to data')
    parser.add_argument("--mesh_path", type=str, help='Path to mesh')
    parser.add_argument("--scene", type=str, help='The scene to be evaluated')

    args = parser.parse_args()
    print(f"Evaluating scene {args.scene} with mesh {args.mesh_path}")
    results = get_metrics(Path(args.data_dir), Path(args.mesh_path), args.scene)
    print(results)

