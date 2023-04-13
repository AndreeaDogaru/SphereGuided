import functools
import torch
from torch import nn 
import math

from spheres.utils import batchify_query, random_in_sphere


class OptimizableSpheres(nn.Module):
    def __init__(self, n_spheres, radius_scheduler, bounding_radius=1.0):
        super().__init__()
        origins = random_in_sphere((n_spheres, )) * bounding_radius
        self.bounding_radius = bounding_radius
        self.origins = nn.Parameter(origins, requires_grad=True)
        self.radius_scheduler_args = {k:torch.tensor([v], dtype=torch.float) for k, v in radius_scheduler.items()}
        self.register_buffer('radius', self.radius_scheduler_args['max'], persistent=True)

    @torch.no_grad()
    def resample_wandering_spheres(self, optimizer_state=None):
        # resample spheres that are outside the scene bounds
        mask = torch.norm(self.origins, dim=-1) > self.bounding_radius
        noise = torch.randn_like(self.origins[mask]) * self.radius_scheduler_args['min'].to(self.origins.device) / 2
        good = (~mask).sum().item()
        selected = torch.randperm(good)
        if good < len(self.origins) - good:
            selected = selected.repeat(math.ceil((len(self.origins) - good) / good))
        self.origins[mask] = self.origins[~mask][selected[:mask.sum()]] + noise

        if optimizer_state:
            optimizer_state['exp_avg'][mask] = 0
            optimizer_state['exp_avg_sq'][mask] = 0
        return mask.sum()

    @torch.no_grad()
    def resample_empty_spheres(self, implicit_f, num_samples=1000, optimizer_state=None):
        batched_f = batchify_query(implicit_f, chunk=10, dim_batchify=0)
        s = self.origins.shape[0]
        noise = random_in_sphere((num_samples, s), device=self.origins.device)
        points = self.origins + noise * self.radius
        out = -batched_f(points)
        max_implicit_value = out.max(dim=0).values
        min_implicit_value = out.min(dim=0).values
        mask = (max_implicit_value < 0) | (min_implicit_value > 0)

        noise = torch.randn_like(self.origins[mask]) * self.radius_scheduler_args['min'].to(self.origins.device) / 2
        good = (~mask).sum().item()
        if good < 1:  # Cancel resampling
            return 0
        selected = torch.randperm(good)
        if good < len(self.origins) - good:
            selected = selected.repeat(math.ceil((len(self.origins) - good) / good))
        self.origins[mask] = self.origins[~mask][selected[:mask.sum()]] + noise
        if optimizer_state:
            optimizer_state['exp_avg'][mask] = 0
            optimizer_state['exp_avg_sq'][mask] = 0
        return mask.sum()

    def update_radius(self, it):
        self.radius = max(
            (-it * self.radius_scheduler_args['beta']).exp() * self.radius_scheduler_args['max'], 
            self.radius_scheduler_args['min']).to(self.radius.device)

    def intersect(self, rays_o, rays_d, k=256, limits_t=None):
        """
            Get all sphere-ray intersections
            rays_o: N_rays x 3
            rays_d: N_rays x 3
            k: number of closest spheres to consider for each ray

            Returns:
                intersections_t: N_rays x k x 2
                ray_has_intersection: N_rays
        """

        N_rays = rays_o.shape[-2]
        # select spheres closest to rays 
        distances = torch.cross(
            self.origins.unsqueeze(-3) - rays_o.unsqueeze(-2), 
            rays_d[:, None].expand(-1, self.origins.shape[0], -1)
            ).norm(dim=-1)
        selected = distances.topk(k, largest=False, sorted=False)
        d = rays_o.unsqueeze(-2) - self.origins[selected.indices.view(-1)].view(-1, k, 3)
        a = (rays_d ** 2).sum(-1, keepdims=True).expand(-1, k)
        b = 2 * (rays_d.unsqueeze(-2) * d).sum(-1)
        c = (d ** 2).sum(-1) - self.radius ** 2
        delta = b ** 2 - 4 * a * c
        intersects = delta > 0
        b_inters = b[intersects]
        q = -0.5 * (b_inters + torch.sign(b_inters) * torch.sqrt(delta[intersects]))
        intersections_t = rays_o.new_zeros(N_rays, k, 2)
        intersections_t[intersects] = torch.sort(torch.stack([q / a[intersects], c[intersects] / q], -1).float(), -1).values        
        if limits_t is not None:
            near, far = limits_t
            intersections_t = torch.max(torch.min(intersections_t, far[:, None, None]), near[:, None, None])
            intersects = intersects & ((intersections_t[..., 1] - intersections_t[..., 0]) > 0)
        return intersections_t, intersects, selected.indices

