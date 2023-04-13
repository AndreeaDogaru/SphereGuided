#  NeuralWarp  All rights reseved to Thales LAS and ENPC.
#
#  This code is freely available for academic use only and Provided “as is” without any warranty.
#
#  Modification are allowed for academic research provided that the following conditions are met :
#    * Redistributions of source code or any format must retain the above copyright notice and this list of conditions.
#    * Neither the name of Thales LAS and ENPC nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

# function specfic to volSDF
import math
import torch
from torch import nn
from NeuralWarp.utils import rend_util

from spheres.sample_ray import sample_from_spheres

def volSDF_sigma(sdf, alpha):
    # eq (2) and (3)
    if len(sdf.shape) == 2:
        tmp = 1 / 2 * torch.exp(-torch.abs(sdf * alpha.view(-1, 1)))
    else:
        tmp = 1 / 2 * torch.exp(-torch.abs(sdf * alpha.view(-1, 1, 1)))
    return alpha.view(-1, 1) * torch.where(sdf <= 0, tmp, 1 - tmp)

def approx_error_intervals(d_star, sigma, delta, beta, valid_interval=None):
    # Implements eq (14) of volSDF

    # eq (9) + add by convention 0 for k = 0
    if valid_interval is not None:
        delta_valid = delta * valid_interval
    else:
        delta_valid = delta
    R_hat = torch.cat((torch.zeros(sigma.shape[:-1], device=sigma.device).unsqueeze(-1), torch.cumsum(delta_valid * sigma[..., :-1], dim=1)), dim=-1)
    # eq (12)
    E_hat = (1 / beta ** 2).view(-1, 1) / 4  * torch.cumsum(delta_valid**2 * torch.exp(-d_star / beta.view(-1, 1)), dim=-1)

    # clamp E_hat --> very high value can go above R_hat even if completely occluded
    E_hat = torch.clamp(E_hat, max=100)
    # eq (14)
    if valid_interval is not None:
        return (torch.exp(E_hat - R_hat[..., :-1]) - torch.exp(-R_hat[..., :-1])).clamp(max=100) * valid_interval
    return torch.exp(E_hat - R_hat[..., :-1]) - torch.exp(-R_hat[..., :-1])

def get_d_star(sdf_values, delta):
    # eq (11)
    abs_d = torch.abs(sdf_values)
    d_star = torch.maximum(torch.zeros_like(delta), (abs_d[..., 1:] + abs_d[..., :-1] - delta) / 2)
    # if sdf varies very quickly, (grad > 1) a sign change may not be set to 0 in above eq (11), set it manually
    d_star[torch.sign(sdf_values[..., 1:]) != torch.sign(sdf_values[..., :-1])] = 0

    return d_star


class OpacityApproximator(nn.Module):
    def __init__(self, n, epsilon, algo_iterations, bisection_iterations, **kwargs):
        super().__init__()
        self.n = n
        self.epsilon = epsilon
        self.nb_iter_bisection = bisection_iterations
        self.n_iterations = algo_iterations

    def upsample_T(self, sdf_vals, delta, intervals_dist, beta, not_valid_beta):
        # add n new samples in the intervals with the most error (line 4 of algo)
        device=sdf_vals.device
        Np, Ns = sdf_vals.shape
        d_star = get_d_star(sdf_vals, delta)
        sigma = volSDF_sigma(sdf_vals, 1 / beta)
        int_error = approx_error_intervals(d_star, sigma, delta, beta)
        int_error = torch.clamp(int_error, max=100) #multiply with dif label
        error_prop = self.n * int_error / (int_error.sum(dim=1, keepdim=True) + 1e-6)  # (float) number of points to add for each interval

        # first add the floor of error_prop then randomly add points up to n
        floor_error_prop = torch.floor(error_prop)
        nb_points_to_add = floor_error_prop.long()
        candidates = torch.topk(error_prop - floor_error_prop, k=min(Ns - 1, self.n), sorted=True).indices
        valid_candidates = torch.arange(min(Ns - 1, self.n), device=device)[None].expand(Np, -1) < (self.n - torch.sum(nb_points_to_add, dim=1, keepdim=True))

        first_idx = torch.arange(Np, device=device)[:, None].expand_as(valid_candidates)[valid_candidates]
        second_idx = candidates[valid_candidates]
        nb_points_to_add[first_idx, second_idx] += 1

        nb_points_to_add[:, 0] += (self.n - nb_points_to_add.sum(dim=1)) # manually add points in the first bin, never needed except when i == 0 and int_error = 0
        nb_points_to_add = nb_points_to_add.view(-1)

        # create new interval dist by adding the right number of points
        # tricky way to add efficiently new points inside the intervals with torch

        nb_points_plus_1 = nb_points_to_add + 1
        nb_tot = nb_points_plus_1.shape[0]
        origin_points = torch.arange(nb_tot, device=device).repeat_interleave(
            nb_points_plus_1, dim=0)

        nb_cumsum = torch.cumsum(nb_points_plus_1, dim=0)
        nb_repeated = nb_points_to_add.repeat_interleave(nb_points_plus_1)

        points_offset = nb_repeated + torch.arange(len(nb_repeated), device=device) - nb_cumsum.repeat_interleave(nb_points_plus_1) + 1
        points_offset = points_offset.reshape(Np, -1)
        origin_points = origin_points.reshape(Np, -1)
        nb_repeated = nb_repeated.reshape(Np, -1)

        new_intervals_dist = intervals_dist[:, :-1].reshape(-1).repeat_interleave(nb_points_plus_1).reshape(Np, -1)
        new_intervals_dist += points_offset * delta.view(-1)[origin_points] / (nb_repeated + 1)

        # fill new sdf values with linear interpolations
        # --> sdf_to_recompute is a mask of points where this approximation is not valid and we must call implicit_network
        new_sdf_vals = sdf_vals[:, :-1].reshape(-1).repeat_interleave(nb_points_plus_1).reshape(Np, -1)
        new_sdf_vals += points_offset * (
                sdf_vals[:, 1:].reshape(-1)[origin_points] - sdf_vals[:, :-1].reshape(-1)[origin_points]) / (nb_repeated + 1)
        
        sdf_to_recompute = (points_offset > 0) & not_valid_beta.unsqueeze(-1)

        # readd the last point of interval
        new_intervals_dist = torch.cat((new_intervals_dist, intervals_dist[:, -1:]), dim=-1)
        new_sdf_vals = torch.cat((new_sdf_vals, sdf_vals[:, -1:]), dim=-1)
        sdf_to_recompute = torch.cat((sdf_to_recompute, torch.zeros_like(intervals_dist[:, -1:], dtype=torch.bool)), dim=-1)

        return new_intervals_dist, new_sdf_vals, sdf_to_recompute

    def find_beta_p(self, sdf_values, delta, beta_p, beta_target):
        # bisection method (line 6 of algo)
        # returns new estimation of beta_p and also checks whether beta_target is valid
        d_star = get_d_star(sdf_values, delta)

        sigma = volSDF_sigma(sdf_values, 1 / beta_target)
        sigma_p = volSDF_sigma(sdf_values, 1 / beta_p)

        error_beta = torch.amax(approx_error_intervals(d_star, sigma, delta, beta_target), dim=1)
        error_beta_p = torch.amax(approx_error_intervals(d_star, sigma_p, delta, beta_p), dim=1)
        already_ok = error_beta <= self.epsilon
        beta_p_not_ok = error_beta_p > self.epsilon
        not_finished = (~already_ok) & (~beta_p_not_ok)

        beta_high = beta_p[not_finished]
        beta_low = beta_target * torch.ones(not_finished.sum(), device=sdf_values.device)

        d_star = d_star[not_finished]
        sdf_values = sdf_values[not_finished]
        delta = delta[not_finished]

        for i in range(self.nb_iter_bisection):
            if len(beta_low) == 0:
                break
            beta_mid = (beta_high + beta_low) / 2
            sigma_mid = volSDF_sigma(sdf_values, 1 / beta_mid)
            error_mid = torch.amax(approx_error_intervals(d_star, sigma_mid, delta, beta_mid), dim=1)

            err_above = error_mid > self.epsilon
            beta_low[err_above] = beta_mid[err_above]
            beta_high[~err_above] = beta_mid[~err_above]

        res = torch.ones_like(beta_p)
        res[not_finished] = beta_high
        res[already_ok] = beta_target
        res[beta_p_not_ok] = beta_p[beta_p_not_ok]

        return res, ~already_ok

    def forward(self, sdf, cam_loc, ray_directions, sampler_min, sampler_max, beta):
        device = cam_loc.device
        Np = ray_directions.shape[1]
        # remove batch dimension never used legacy
        ray_directions = ray_directions.squeeze(0)
        epsilon = self.epsilon

        # eq (16) with sqrt
        beta_p = sampler_max.squeeze(0) / 2 / math.sqrt((self.n - 1) * math.log(1 + epsilon))
        for i in range(self.n_iterations + 1):
            if i == 0:
                max_dist = sampler_max.view(-1, 1)
                min_dist = sampler_min.view(-1, 1)
                intervals_dist = min_dist + torch.linspace(0, 1, steps=self.n, device=device).unsqueeze(0) * (max_dist - min_dist)
                points = cam_loc.unsqueeze(0) + intervals_dist.unsqueeze(-1) * ray_directions.unsqueeze(1)
                sdf_values = sdf(points.reshape(-1, 3)).view(Np, self.n)

            else:
                intervals_dist, sdf_values, sdf_to_recompute = self.upsample_T(sdf_values, delta, intervals_dist, beta_p, not_valid_beta)
                points = cam_loc.unsqueeze(0) + intervals_dist.unsqueeze(-1) * ray_directions.unsqueeze(1)
                sdf_values[sdf_to_recompute] = sdf(points[sdf_to_recompute])

            delta = intervals_dist[:, 1:] - intervals_dist[:, :-1]
            beta_p, not_valid_beta = self.find_beta_p(sdf_values, delta, beta_p, beta)

        return intervals_dist, sdf_values, beta_p

class OpacityApproximatorSpheres(OpacityApproximator):
    def __init__(self, n, epsilon, algo_iterations, bisection_iterations, **kwargs):
        super().__init__(n, epsilon, algo_iterations, bisection_iterations, **kwargs)

    def find_beta_p(self, sdf_values, delta, beta_p, beta_target, valid_intervals):
        # bisection method (line 6 of algo)
        # returns new estimation of beta_p and also checks whether beta_target is valid
        d_star = get_d_star(sdf_values, delta)

        sigma = volSDF_sigma(sdf_values, 1 / beta_target)
        sigma_p = volSDF_sigma(sdf_values, 1 / beta_p)

        error_beta = torch.amax(approx_error_intervals(d_star, sigma, delta, beta_target, valid_intervals), dim=1)
        error_beta_p = torch.amax(approx_error_intervals(d_star, sigma_p, delta, beta_p, valid_intervals), dim=1)
        already_ok = error_beta <= self.epsilon
        beta_p_not_ok = error_beta_p > self.epsilon
        not_finished = (~already_ok) & (~beta_p_not_ok)

        beta_high = beta_p[not_finished]
        beta_low = beta_target * torch.ones(not_finished.sum(), device=sdf_values.device)

        d_star = d_star[not_finished]
        sdf_values = sdf_values[not_finished]
        delta = delta[not_finished]
        valid_intervals = valid_intervals[not_finished]

        for i in range(self.nb_iter_bisection):
            if len(beta_low) == 0:
                break
            beta_mid = (beta_high + beta_low) / 2
            sigma_mid = volSDF_sigma(sdf_values, 1 / beta_mid)
            error_mid = torch.amax(approx_error_intervals(d_star, sigma_mid, delta, beta_mid, valid_intervals), dim=1)

            err_above = error_mid > self.epsilon
            beta_low[err_above] = beta_mid[err_above]
            beta_high[~err_above] = beta_mid[~err_above]

        res = torch.ones_like(beta_p)
        res[not_finished] = beta_high
        res[already_ok] = beta_target
        res[beta_p_not_ok] = beta_p[beta_p_not_ok]

        return res, ~already_ok

    def upsample_T(self, sdf_vals, delta, intervals_dist, beta, not_valid_beta, interval_label):
        # add n new samples in the intervals with the most error (line 4 of algo)
        device=sdf_vals.device
        Np, Ns = sdf_vals.shape
        d_star = get_d_star(sdf_vals, delta)
        sigma = volSDF_sigma(sdf_vals, 1 / beta)
        int_error = approx_error_intervals(d_star, sigma, delta, beta, interval_label[:, :-1] == interval_label[:, 1:])
        int_error = torch.clamp(int_error, max=100) #multiply with dif label
        error_prop = self.n * int_error / (int_error.sum(dim=1, keepdim=True) + 1e-6)  # (float) number of points to add for each interval

        # first add the floor of error_prop then randomly add points up to n
        floor_error_prop = torch.floor(error_prop)
        nb_points_to_add = floor_error_prop.long()
        candidates = torch.topk(error_prop - floor_error_prop, k=min(Ns - 1, self.n), sorted=True).indices
        valid_candidates = torch.arange(min(Ns - 1, self.n), device=device)[None].expand(Np, -1) < (self.n - torch.sum(nb_points_to_add, dim=1, keepdim=True))

        first_idx = torch.arange(Np, device=device)[:, None].expand_as(valid_candidates)[valid_candidates]
        second_idx = candidates[valid_candidates]
        nb_points_to_add[first_idx, second_idx] += 1

        nb_points_to_add[:, 0] += (self.n - nb_points_to_add.sum(dim=1)) # manually add points in the first bin, never needed except when i == 0 and int_error = 0
        nb_points_to_add = nb_points_to_add.view(-1)

        # create new interval dist by adding the right number of points
        # tricky way to add efficiently new points inside the intervals with torch

        nb_points_plus_1 = nb_points_to_add + 1
        nb_tot = nb_points_plus_1.shape[0]
        origin_points = torch.arange(nb_tot, device=device).repeat_interleave(
            nb_points_plus_1, dim=0)

        nb_cumsum = torch.cumsum(nb_points_plus_1, dim=0)
        nb_repeated = nb_points_to_add.repeat_interleave(nb_points_plus_1)

        points_offset = nb_repeated + torch.arange(len(nb_repeated), device=device) - nb_cumsum.repeat_interleave(nb_points_plus_1) + 1
        points_offset = points_offset.reshape(Np, -1)
        origin_points = origin_points.reshape(Np, -1)
        nb_repeated = nb_repeated.reshape(Np, -1)

        new_intervals_dist = intervals_dist[:, :-1].reshape(-1).repeat_interleave(nb_points_plus_1).reshape(Np, -1)
        new_intervals_dist += points_offset * delta.view(-1)[origin_points] / (nb_repeated + 1)

        # fill new sdf values with linear interpolations
        # --> sdf_to_recompute is a mask of points where this approximation is not valid and we must call implicit_network
        new_sdf_vals = sdf_vals[:, :-1].reshape(-1).repeat_interleave(nb_points_plus_1).reshape(Np, -1)
        new_sdf_vals += points_offset * (
                sdf_vals[:, 1:].reshape(-1)[origin_points] - sdf_vals[:, :-1].reshape(-1)[origin_points]) / (nb_repeated + 1)

        new_intervals_label = torch.gather(interval_label[:, :-1].reshape(-1), -1, origin_points.reshape(-1)).reshape_as(origin_points)
        sdf_to_recompute = (points_offset > 0) & not_valid_beta.unsqueeze(-1)

        # readd the last point of interval
        new_intervals_dist = torch.cat((new_intervals_dist, intervals_dist[:, -1:]), dim=-1)
        new_sdf_vals = torch.cat((new_sdf_vals, sdf_vals[:, -1:]), dim=-1)
        sdf_to_recompute = torch.cat((sdf_to_recompute, torch.zeros_like(intervals_dist[:, -1:], dtype=torch.bool)), dim=-1)
        new_intervals_label = torch.cat([new_intervals_label, interval_label[:, -1:]], dim=-1)

        return new_intervals_dist, new_sdf_vals, sdf_to_recompute, new_intervals_label

    def forward(self, sdf, cam_loc, ray_directions, sampler_min, sampler_max, beta, spheres, bounding_radius=4):
        device = cam_loc.device
        Np = ray_directions.shape[1]
        # remove batch dimension never used legacy
        ray_directions = ray_directions.squeeze(0)
        epsilon = self.epsilon

        # eq (16) with sqrt
        beta_p = sampler_max.squeeze(0) / 2 / math.sqrt((self.n - 1) * math.log(1 + epsilon))
        for i in range(self.n_iterations + 1):
            if i == 0:
                max_dist = sampler_max.view(-1, 1)
                min_dist = sampler_min.view(-1, 1)
                bounding_sphere_intersections, intersection_mask = rend_util.get_sphere_intersection(cam_loc, ray_directions[None], r=spheres.bounding_radius)
                min_bound = torch.maximum(bounding_sphere_intersections[..., 0].view(-1, 1), min_dist)
                max_bound = bounding_sphere_intersections[..., 1].view(-1, 1)
                distance_to_center = torch.linalg.norm(cam_loc)
                dist_mean = distance_to_center + bounding_radius
                if bounding_radius > spheres.bounding_radius:
                    n_before = (torch.clamp(distance_to_center - spheres.bounding_radius, min=0) / dist_mean * self.n).floor().int().item()
                    n_after = ((bounding_radius - spheres.bounding_radius) / dist_mean * self.n).floor().int().item()
                else:
                    n_after = 0
                    n_before = 0
                dist = dist_mean / self.n
                samples_before = min_dist + torch.linspace(0, 1, steps=n_before, device=device).unsqueeze(0) * (min_bound - min_dist - dist)
                samples_after = max_bound + dist + torch.linspace(0, 1, steps=n_after, device=device).unsqueeze(0) * (max_dist - max_bound - dist)
                intervals_dist_sph, interval_label_sph, ray_has_intersection = sample_from_spheres(
                    spheres, cam_loc.expand_as(ray_directions), ray_directions, 
                    self.n - n_before - n_after, min_bound, max_bound)
                Np = ray_has_intersection.sum().item()
                intervals_dist = torch.cat([samples_before[ray_has_intersection], intervals_dist_sph, samples_after[ray_has_intersection]], dim=-1)
                interval_label = torch.cat([-interval_label_sph.new_ones(Np, n_before), interval_label_sph, -2 * interval_label_sph.new_ones(Np, n_after)], dim=-1)  
                ray_directions = ray_directions[ray_has_intersection]
                beta_p = beta_p[ray_has_intersection]
                points = cam_loc.unsqueeze(0) + intervals_dist.unsqueeze(-1) * ray_directions.unsqueeze(1)
                sdf_values = sdf(points.reshape(-1, 3)).view(Np, self.n)
                if Np == 0:
                    return intervals_dist, sdf_values, beta_p, ray_has_intersection, interval_label

            else:
                intervals_dist, sdf_values, sdf_to_recompute, interval_label = self.upsample_T(sdf_values, delta, intervals_dist, beta_p, not_valid_beta, interval_label)
                points = cam_loc.unsqueeze(0) + intervals_dist.unsqueeze(-1) * ray_directions.unsqueeze(1)
                sdf_values[sdf_to_recompute] = sdf(points[sdf_to_recompute])

            delta = intervals_dist[:, 1:] - intervals_dist[:, :-1]
            beta_p, not_valid_beta = self.find_beta_p(sdf_values, delta, beta_p, beta, interval_label[:, :-1] == interval_label[:, 1:])

        return intervals_dist, sdf_values, beta_p, ray_has_intersection, interval_label