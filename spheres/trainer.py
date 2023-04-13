from functools import partial
import torch
import torch.nn.functional as F
import yaml
from spheres.model import OptimizableSpheres
from spheres.utils import freeze_gradients
from pytorch3d.ops import ball_query, knn_points

class SpheresTrainer:
    def __init__(self, config_file, run_model=None, device=None) -> None:
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
        self.spheres = OptimizableSpheres(**config['spheres']).to(device)
        self.sphere_optimizer = torch.optim.Adam(self.spheres.parameters(), config['sphere_lr'])
        self.i_warmup_spheres = config.get('i_warmup_spheres', -1)
        self.i_resample_spheres = config.get('i_resample_spheres', -1)
        self.i_resample_wandering_spheres = config.get('i_resample_wandering_spheres', -1)
        self.k_neighbors = config.get('k_neighbors', 10)
        self.repulsion_radius_factor = config.get('repulsion_radius_factor', 1.3)
        self.loss_factors = config['loss_factors']
        if run_model is None:
            self.run_model = lambda model, x: model(x)
        else:
            self.run_model = run_model

    def load_weights(self, path):
        state_dict = torch.load(path, map_location=self.spheres.origins.device)
        self.spheres.load_state_dict(state_dict['spheres'])
        self.sphere_optimizer.load_state_dict(state_dict['sphere_optimizer'])

    def save_weights(self, path):
        state_dict = {
            'spheres': self.spheres.state_dict(),
            'sphere_optimizer': self.sphere_optimizer.state_dict(),
        }
        torch.save(state_dict, path)

    def origin_loss(self, model):
        with freeze_gradients(model):
            out = self.run_model(model, self.spheres.origins)
        target = torch.zeros_like(out)
        mse = F.mse_loss(out, target, reduction="none")
        return mse.mean()

    def repulsion_loss(self):
        dists, idx, _ = knn_points(self.spheres.origins[None], self.spheres.origins[None], K=self.k_neighbors + 1)
        n_spheres = self.spheres.origins.shape[0]

        valid_mask = torch.logical_and(idx != torch.arange(n_spheres, device=dists.device)[:, None], idx != -1)
        valid_dists = dists[valid_mask]
        identical_mask = valid_dists.isclose(valid_dists.new_tensor(0.))
        # remove distances close to 0 from loss computation to prevent nan
        good_dist = (~identical_mask) & (valid_dists < (self.spheres.radius * self.repulsion_radius_factor) ** 2)
        valid_dists = valid_dists[good_dist]

        f = (valid_dists ** 0.5) / self.spheres.radius
        return f.reciprocal().mean()

    def train_step(self, model, it):
        if it < self.i_warmup_spheres:
            return {}
        if self.i_resample_wandering_spheres > 0 and it % self.i_resample_wandering_spheres == 0 and it > self.i_warmup_spheres:
            state = self.sphere_optimizer.state_dict()['state'][0]
            n_resampled = self.spheres.resample_wandering_spheres(optimizer_state=state)
            print("wandering: ", n_resampled)
        if self.i_resample_spheres > 0 and it % self.i_resample_spheres == 0 and it != 0:
            state = self.sphere_optimizer.state_dict()['state'][0]
            n_resampled = self.spheres.resample_empty_spheres(partial(self.run_model, model), optimizer_state=state)
            print("empty: ", n_resampled)
        losses = {
            'origin': self.origin_loss(model),
            'repulsion': self.repulsion_loss()
        }
        self.sphere_optimizer.zero_grad()
        total_loss = sum(loss * self.loss_factors[name] for name, loss in losses.items())
        total_loss.backward()
        self.sphere_optimizer.step()
        return losses
