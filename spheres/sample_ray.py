import torch 
from collections import namedtuple

from spheres.model import OptimizableSpheres

@torch.no_grad()
def sample_from_spheres(spheres: OptimizableSpheres, rays_o, rays_d, n_steps, near, far, k=256):
    b = rays_o.shape[:-2]
    intersections_t, valid_intersections, _ = spheres.intersect(rays_o.view(-1, 3), rays_d.view(-1, 3), k=k, limits_t=(near.view(-1), far.view(-1)))
    ray_has_intersection = valid_intersections.any(-1)
    N_rays = ray_has_intersection.sum().item()

    if N_rays < 1:
        return rays_o.new_zeros(*b, N_rays, n_steps), \
        rays_o.new_zeros((*b, N_rays, n_steps), dtype=torch.int), \
        ray_has_intersection.view(*b, -1, *ray_has_intersection.shape[1:])

    d_coarse, intersection_samples = sample_from_intersections_linspace(intersections_t, valid_intersections, ray_has_intersection, n_steps)
    d_coarse = d_coarse.view(*b, -1, *d_coarse.shape[1:])
    ray_has_intersection = ray_has_intersection.view(*b, -1, *ray_has_intersection.shape[1:])
    return d_coarse, intersection_samples, ray_has_intersection

@torch.no_grad()
def sample_from_intersections(intersections_t, ray_has_intersection, N_samples):
    interval_size = intersections_t[ray_has_intersection, :, 0] - intersections_t[ray_has_intersection, :, 1]
    intersection_samples = torch.multinomial(torch.abs(interval_size), N_samples, replacement=True)
    d_spheres = torch.rand(ray_has_intersection.sum(), N_samples, device=intersections_t.device)
    d_spheres = d_spheres * torch.gather(interval_size, -1, intersection_samples) + \
                                        torch.gather(intersections_t[ray_has_intersection, :, 1], -1, intersection_samples)
    d_coarse, coarse_sort_indices = torch.sort(d_spheres, dim=-1)
    intersection_samples = intersection_samples.gather(-1, coarse_sort_indices)
    return d_coarse, intersection_samples

@torch.no_grad()
def sample_from_intersections_linspace(sphere_intersections_t, valid_intersections, ray_has_intersection, N_samples, align_corners=True):
    intersections_t, intervals_mask = intervals_union(sphere_intersections_t, valid_intersections)
    N_rays = ray_has_intersection.sum()
    N_spheres = intersections_t.shape[1]
    device = intersections_t.device
    interval_size = intersections_t[ray_has_intersection, :, 1] - intersections_t[ray_has_intersection, :, 0]
    # distribute samples according to weights
    ray_sphere_samples = torch.floor(interval_size / interval_size.sum(-1, keepdims=True) * N_samples).long()
    # distribute leftover samples 
    leftover = N_samples - ray_sphere_samples.sum(-1, keepdim=True)
    ray_sphere_samples[torch.arange(N_spheres, device=device).expand(N_rays, -1) < leftover] += 1
    ray_sphere_samples_cum_shifted = torch.cat(
        [
            ray_sphere_samples.new_zeros([*ray_sphere_samples.shape[:-1], 1]),
            torch.cumsum(ray_sphere_samples, -1)[..., :-1],
        ], dim=-1
    )
    sphere_index = torch.repeat_interleave(
        torch.arange(0, N_spheres, device=device).expand(N_rays, -1).reshape(-1), 
        ray_sphere_samples.view(-1)).view(N_rays, N_samples)
    diff_term = torch.repeat_interleave(ray_sphere_samples_cum_shifted.view(-1), ray_sphere_samples.view(-1)).view(N_rays, N_samples)
    if align_corners:
        div_term = torch.gather(ray_sphere_samples, -1, sphere_index) - 1
        d_coarse = torch.arange(0, N_samples, device=device)
        div_term.masked_fill_(div_term == 0, 1)
    else:
        div_term = torch.gather(ray_sphere_samples, -1, sphere_index) + 1
        d_coarse = torch.arange(1, N_samples + 1, device=device)
    d_coarse = (d_coarse.expand(N_rays, -1) - diff_term) / div_term
    t1 = torch.gather(intersections_t[ray_has_intersection, :, 0], -1, sphere_index)
    d_coarse = d_coarse * torch.gather(interval_size, -1, sphere_index) + t1
    d_coarse, coarse_sort_indices = torch.sort(d_coarse, dim=-1)
    sphere_index = sphere_index.gather(-1, coarse_sort_indices)
    return d_coarse, sphere_index


@torch.no_grad()
def multi_put(indices, values, out=None):
    assert indices.ndim == 2 and values.ndim == 2
    fa = torch.arange(indices.shape[0], device=indices.device).view(-1, 1).expand_as(indices).contiguous()
    if out is None:
        out = torch.empty_like(indices)
    out[fa.view(-1), indices.view(-1)] = values.view(-1)
    return out


def inverse_permutation(permutation):
    identity = torch.arange(permutation.shape[-1], device=permutation.device).expand_as(permutation).contiguous()
    return multi_put(permutation, identity)


@torch.no_grad()
def stable_true_first(tensor: torch.Tensor):
    n = tensor.shape[-1]
    values = torch.arange(n, device=tensor.device).expand_as(tensor)
    values = torch.masked_fill(values, ~tensor, n)
    ind = torch.sort(values).values
    mask = ind != n
    ind.masked_fill_(~mask, 0)
    return mask, ind


@torch.no_grad()
def intervals_union(initial_intervals: torch.Tensor, valid_mask: torch.Tensor):
    """
    Computes the union of input intervals
    :param initial_intervals: torch.Tensor of shape B x N x 2
    :param valid_mask: torch.Tensor of shape B x N
    :return: named tuple of
            new_intervals: torch.Tensor of shape B x N x 2,
            mask: torch.Tensor of shape B x N
    """
    b, n = initial_intervals.shape[:2]
    x_sort, ind = torch.sort(initial_intervals.view(b, -1), dim=-1)  # B x (N*2)
    valid_mask = valid_mask.unsqueeze(-1).expand(-1, -1, 2).reshape(b, -1)
    valid_mask = torch.gather(valid_mask, dim=1, index=ind)
    rev_ind = inverse_permutation(ind)
    par = (1 - (torch.arange(ind.shape[-1], device=ind.device) % 2) * 2).expand_as(ind).contiguous()
    events = multi_put(rev_ind, par)  # +1 for open interval; -1 for closed interval
    events.masked_fill_(~valid_mask, 0)
    collected = torch.cumsum(events, dim=-1)  # collected events 1 corresponds to open, 0 corresponds to closed
    start = (collected == 1) & (events == 1) & valid_mask
    end = (collected == 0) & (events == -1) & valid_mask
    # Compute interval indices
    mask, start_ind = stable_true_first(start)
    _, end_ind = stable_true_first(end)
    # Drop the unused half
    start_ind = start_ind[:, :n]
    end_ind = end_ind[:, :n]
    mask = mask[:, :n]
    # Pair intervals
    intervals = torch.stack([start_ind, end_ind], dim=-1)
    # Gather interval point values
    values = torch.gather(x_sort[..., None].expand(-1, -1, 2), dim=1, index=intervals)

    return namedtuple('ValidIntervals', 'new_intervals mask')(values, mask)
