#  NeuralWarp  All rights reseved to Thales LAS and ENPC.
#
#  This code is freely available for academic use only and Provided “as is” without any warranty.
#
#  Modification are allowed for academic research provided that the following conditions are met :
#    * Redistributions of source code or any format must retain the above copyright notice and this list of conditions.
#    * Neither the name of Thales LAS and ENPC nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

import json
import os
import torch
import numpy as np
from spheres.utils import sample_sphere_rays

import utils.general as utils
from utils import rend_util

class SceneDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""
    def __init__(self,
                 data_dir,
                 img_res,
                 scene=0,
                 nsrc=0,
                 h_patch_size=None,
                 uv_down=None,
                 views=None
                 ):

        if "DTU" in data_dir:
            self.instance_dir = os.path.join(data_dir, 'scan{0}'.format(scene))
            self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
            im_folder_name = "image"
        elif "blended_MVS" in data_dir:
            self.instance_dir = os.path.join(data_dir, 'bmvs_{0}'.format(scene))
            im_folder_name = "image"
            self.cam_file = '{0}/cameras_sphere.npz'.format(self.instance_dir)
        else:
            # epfl
            self.instance_dir = os.path.join(data_dir, scene + "_dense")
            im_folder_name = "urd"
            self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res

        self.generator = torch.Generator()
        self.generator.manual_seed(np.random.randint(1e9))

        assert os.path.exists(self.instance_dir), "Data directory is empty" + str(self.instance_dir)

        self.sampling_idx = None
        self.small_uv = uv_down is not None
        self.uv_down = uv_down
        if self.small_uv:
            self.plot_img_res = img_res[0] // self.uv_down, img_res[1] // self.uv_down
        else:
            self.plot_img_res = img_res

        image_dir = '{0}/{1}'.format(self.instance_dir, im_folder_name)
        image_paths = sorted(utils.glob_imgs(image_dir))
        mask_dir = '{0}/mask'.format(self.instance_dir)
        mask_paths = sorted(utils.glob_imgs(mask_dir))

        self.n_images = len(image_paths)

        camera_dict = np.load(self.cam_file)
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.n_images, image_paths, mask_paths, world_mats, scale_mats = self.filter_views(views, image_paths, mask_paths, world_mats, scale_mats)
        self.intrinsics_all = []
        self.pose_all = []
        for world_mat, scale_mat in zip(world_mats, scale_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.intrinsics_all = torch.stack(self.intrinsics_all)
        self.inv_intrinsics_all = torch.inverse(self.intrinsics_all)
        self.pose_all = torch.stack(self.pose_all)
        self.inv_pose_all = torch.inverse(self.pose_all)

        self.rgb_images = []
        for path in image_paths:
            rgb = rend_util.load_rgb(path)
            self.rgb_images.append(torch.tensor(rgb).float())

        self.rgb_images = torch.stack(self.rgb_images)

        self.object_masks = []
        self.org_object_masks = []
        for path in mask_paths:
            object_mask = rend_util.load_mask(path)
            self.org_object_masks.append(torch.tensor(object_mask).bool())
            object_mask = object_mask.reshape(-1)
            self.object_masks.append(torch.from_numpy(object_mask).bool())

        if nsrc == 'max':
            self.nsrc = self.rgb_images.shape[0] - 1
        else:
            self.nsrc = nsrc - 1

        self.src_idx = []
        if views is None or len(views) == 0:
            with open(os.path.join(self.instance_dir, "pairs.txt")) as f:
                pairs = f.readlines()

            for p in pairs:
                splitted = p.split()[1:]  # drop the first one since it is the ref img
                fun = lambda s: int(s.split(".")[0])
                self.src_idx.append(torch.tensor(list(map(fun, splitted))))
        else:
            for p in range(self.n_images):
                self.src_idx.append(torch.tensor(list(set(range(self.n_images)) - set([p]))))

        self.h_patch_size = h_patch_size

    def filter_views(self, views, image_paths, mask_paths, world_mats, scale_mats):
        if views is None or len(views) == 0:
            return self.n_images, image_paths, mask_paths, world_mats, scale_mats
        image_paths = [image_paths[i] for i in views]
        mask_paths = [mask_paths[i] for i in views]
        world_mats = [world_mats[i] for i in views]
        scale_mats = [scale_mats[i] for i in views]
        return len(views), image_paths, mask_paths, world_mats, scale_mats


    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        if self.sampling_idx is None and self.small_uv:
            uv = uv[:, ::self.uv_down, ::self.uv_down]
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
        }

        src_idx = self.src_idx[idx][torch.randperm(len(self.src_idx[idx]),
                                                   generator=self.generator)][:self.nsrc]

        idx_list = torch.cat([torch.tensor(idx).unsqueeze(0), src_idx], dim=0)

        sample["pose"] = self.pose_all[idx_list]
        sample["inverse_pose"] = self.inv_pose_all[idx_list]
        sample["intrinsics"] = self.intrinsics_all[idx_list]
        sample["inverse_intrinsics"] = self.inv_intrinsics_all[idx_list]
        sample["idx_list"] = idx_list

        if self.sampling_idx is not None:
            if type(self.sampling_idx) is tuple:
                size, spheres = self.sampling_idx
                sample["uv"] = sample_sphere_rays(self.pose_all[idx], self.intrinsics_all[idx], self.img_res[0], self.img_res[1], size, spheres)[-1].cpu()
            else:
                sample["uv"] = uv[self.sampling_idx, :]

        return idx, sample

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size, spheres=None):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            if self.h_patch_size:
                idx_img = torch.arange(self.total_pixels).view(self.img_res[0], self.img_res[1])
                if self.h_patch_size > 0:
                    idx_img = idx_img[self.h_patch_size:-self.h_patch_size, self.h_patch_size:-self.h_patch_size]
                idx_img = idx_img.reshape(-1)
                self.sampling_idx = idx_img[torch.randperm(idx_img.shape[0])[:sampling_size]]
            else:
                if spheres is not None:
                    self.sampling_idx = (sampling_size, spheres)
                else:
                    self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']


class NerfDataset(SceneDataset):
    def __init__(self,
                 data_dir,
                 img_res,
                 scene=0,
                 nsrc=0,
                 h_patch_size=None,
                 uv_down=None,
                 cam_file=None
                 ):

        self.instance_dir = os.path.join(data_dir, scene)

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res

        self.generator = torch.Generator()
        self.generator.manual_seed(np.random.randint(1e9))

        assert os.path.exists(self.instance_dir), "Data directory is empty" + str(self.instance_dir)

        self.sampling_idx = None
        self.small_uv = uv_down is not None
        self.uv_down = uv_down
        if self.small_uv:
            self.plot_img_res = img_res[0] // self.uv_down, img_res[1] // self.uv_down
        else:
            self.plot_img_res = img_res

        if cam_file is None:
            cam_file = 'transforms_train.json'
        with open(os.path.join(self.instance_dir, cam_file)) as fp:
            camera_dict = json.load(fp)
        self.camera_dict = camera_dict
        self.pose_all = []
        self.images_lis = []
        for frame in camera_dict['frames']:
            self.images_lis.append(os.path.join(self.instance_dir, f"{frame['file_path']}.png"))
            pose = np.array(frame['transform_matrix'])
            pose[:, 1:3] *= -1
            pose[:3, -1] *= float(camera_dict['scale'])
            pose[:3, -1] += np.array(camera_dict['translation'])
            self.pose_all.append(torch.from_numpy(pose).float())
        
        self.n_images = len(self.images_lis)
        
        self.scale_mat = np.eye(4, 4)
        self.scale_mat[:3, -1] -= camera_dict['translation']
        self.scale_mat[:3] /= camera_dict['scale']

        images = []
        masks = []
        for path in self.images_lis:
            rgba = rend_util.load_rgb(path)
            alpha = rgba[-1, None]
            rgb = rgba[:3] * alpha + (1.0 - alpha)
            object_mask = (alpha > 0.2)
            images.append(rgb)
            masks.append(object_mask[0])
        images_np = np.stack(images)
        masks_np = np.stack(masks)

        self.intrinsics_all = []

        camera_angle_x = float(camera_dict['camera_angle_x'])
        focal = .5 * self.img_res[1] / np.tan(.5 * camera_angle_x)
        intrinsics = np.eye(4)
        intrinsics[[0, 1], [0, 1]] = focal
        intrinsics[0, 2] = self.img_res[1] / 2 
        intrinsics[1, 2] = self.img_res[0] / 2 
        self.intrinsics_all = [torch.from_numpy(intrinsics).float() for _ in range(self.n_images)]

        self.rgb_images = torch.from_numpy(images_np.astype(np.float32))  # [n_images, H, W, 3]
        self.org_object_masks = torch.from_numpy(masks_np).bool()   # [n_images, H, W, 3]
        self.object_masks = self.org_object_masks.reshape(self.org_object_masks.shape[0], -1)
        self.intrinsics_all = torch.stack(self.intrinsics_all)   # [n_images, 4, 4]
        self.inv_intrinsics_all = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all)  # [n_images, 4, 4]
        self.inv_pose_all = torch.inverse(self.pose_all)

        if nsrc == 'max':
            self.nsrc = self.rgb_images.shape[0] - 1
        else:
            self.nsrc = nsrc - 1
        with open(os.path.join(self.instance_dir, "pairs.txt")) as f:
            pairs = f.readlines()

        self.src_idx = []
        for p in pairs:
            splitted = p.split()[1:]  # drop the first one since it is the ref img
            fun = lambda s: int(s.split(".")[0].split("_")[-1])
            self.src_idx.append(torch.tensor(list(map(fun, splitted))))

        self.h_patch_size = h_patch_size

    def get_scale_mat(self):
        return self.scale_mat