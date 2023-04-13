import contextlib
import numpy as np
import random
import torch
import torch.nn.functional as F


def random_in_sphere(shape, safe_factor=0.99, device=None):
    x = torch.randn((*shape, 3), device=device)
    mag = torch.linalg.norm(x, dim=-1, keepdim=True)
    c = torch.rand_like(mag) ** (1. / 3)
    return x / mag * c * safe_factor


def batchify_query(func, chunk=1000, dim_batchify=0):
    def batchified(*args, **kwargs):
        assert all(x.shape[dim_batchify]==args[0].shape[dim_batchify] for x in args)
        outs = None
        single_output = False
        args = [a.split(chunk, dim_batchify) for a in args]
        for args_chunk in zip(*args):
            outs_chunk = func(*args_chunk, **kwargs)
            if type(outs_chunk) not in [list, tuple]:
                single_output = True
                outs_chunk = [outs_chunk]
            if outs is None:
                outs = [[] for _ in range(len(outs_chunk))]
            for out, out_chunk in zip(outs, outs_chunk):
                out.append(out_chunk)
        outs = [torch.cat(out, dim=dim_batchify) for out in outs]
        if single_output:
            return outs[0]
        return outs
    return batchified


@contextlib.contextmanager
def freeze_gradients(model):
    is_training = model.training
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    yield
    if is_training:
        model.train()
    for p in model.parameters():
        p.requires_grad_(True)


def sample_sphere_rays(pose, intrinsics, H, W, N_rays, spheres):
    device = spheres.origins.device
    pose = pose.to(device)
    intrinsics = intrinsics.to(device)
    extrinsics = torch.inverse(pose)
    cam_loc = pose[..., :3, 3]
    noise = random_in_sphere(spheres.origins.shape[:1], device=spheres.origins.device)
    points_in_sphere = spheres.origins + noise * spheres.radius

    coords = project_to_frame(points_in_sphere, extrinsics, intrinsics)
    spheres_in_frame = filter_in_frame(coords, (H, W))

    visible_points = points_in_sphere[spheres_in_frame]

    N_rays = min(N_rays, visible_points.shape[0])
    select_inds = torch.randperm(visible_points.shape[0], device=device)[:N_rays]
    rays_d = visible_points[select_inds] - cam_loc[..., None, :]
    rays_d /= torch.linalg.norm(rays_d, ord=2, dim=-1, keepdim=True)
    rays_o = cam_loc[..., None, :].expand_as(rays_d)
    return rays_o, rays_d, coords[spheres_in_frame][select_inds]


def sample_2d_grid(grid, coords, align_corners=True):
    assert grid.ndim >= 3
    assert coords.ndim == grid.ndim

    if align_corners:
        coords = coords * 2 / (coords.new_tensor(grid.shape[-2:][::-1]) - 1) - 1
    else:
        coords = (coords * 2 + 1) / coords.new_tensor(grid.shape[-2:][::-1]) - 1
    if grid.ndim == 4:
        return F.grid_sample(grid, coords, align_corners=align_corners)
    else:
        return F.grid_sample(grid.unsqueeze(0), coords.unsqueeze(0), align_corners=align_corners).squeeze(0)


def project_to_frame(pts, extr, intr):
    pts = torch.cat([pts, torch.ones(pts.shape[:-1] + (1, ), device=pts.device)], dim=-1)
    local_frame = (extr @ pts.transpose(-1, -2))[..., :3, :]
    image_coord = (intr[:3, :3] @ local_frame).transpose(-1, -2)
    image_coord = image_coord[..., :-1] / image_coord[..., -1, None]
    return image_coord


def filter_in_frame(coords, shape):
    return (coords[..., 0] >= 0) & \
           (coords[..., 1] >= 0) & \
           (coords[..., 0] <= shape[-1] - 1) & \
           (coords[..., 1] <= shape[-2] - 1)


def fill_tensor(x, mask, c=0):
    if x is not None:
        out = x.new_ones((mask.shape[0], *x.shape[1:])) * c
        out[mask] = x
        return out 
    else:
        return x


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)