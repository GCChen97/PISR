import os; opj = os.path.join
import json
import math
import numpy as np
from PIL import Image
import cv2
from glob import glob

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

import datasets
from models.ray_utils import get_ray_directions, get_undistorted_ray_directions
from utils.misc import get_rank



def load_K_Rt_from_P(P=None):
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def create_spheric_poses(cameras, n_steps=120):
    center = torch.as_tensor([0.,0.,0.], dtype=cameras.dtype, device=cameras.device)
    cam_center = F.normalize(cameras.mean(0), p=2, dim=-1) * cameras.mean(0).norm(2)
    eigvecs = torch.linalg.eig(cameras.T @ cameras).eigenvectors
    rot_axis = F.normalize(eigvecs[:,1].real.float(), p=2, dim=-1)
    up = rot_axis
    rot_dir = torch.cross(rot_axis, cam_center)
    max_angle = (F.normalize(cameras, p=2, dim=-1) * F.normalize(cam_center, p=2, dim=-1)).sum(-1).acos().max()

    all_c2w = []
    for theta in torch.linspace(-max_angle, max_angle, n_steps):
        cam_pos = cam_center * math.cos(theta) + rot_dir * math.sin(theta)
        l = F.normalize(center - cam_pos, p=2, dim=0)
        s = F.normalize(l.cross(up), p=2, dim=0)
        u = F.normalize(s.cross(l), p=2, dim=0)
        c2w = torch.cat([torch.stack([s, u, -l], dim=1), cam_pos[:,None]], axis=1)
        all_c2w.append(c2w)

    all_c2w = torch.stack(all_c2w, dim=0)
    
    return all_c2w

class DTUDatasetBase():
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        intrinsics = np.load( opj(self.config.root_dir, "intrinsics.npz") )
        camera_matrix = intrinsics["camera_matrix"]
        dist_coeffs = intrinsics["dist_coeffs"]

        cams = np.load(opj(self.config.root_dir, self.config.cameras_file))

        h, w = self.config.img_hw

        self.w, self.h = w, h
        self.img_wh = (w, h)

        self.has_raw = self.config.has_raw
        self.apply_mask = self.config.apply_mask
        self.has_mask = self.config.has_mask
        self.has_normal = self.config.has_normal
        self.has_depth = self.config.has_depth
        self.shared_camera = True
        self.scale_factor = float(self.config.scale_factor)

        # read gamma value
        gamma = float(
            os.path.splitext(
                os.path.basename(
                    glob( opj(self.config.root_dir, "gamma*") )[0]
                )
            )[0][5:]
        )
        self.gamma = gamma

        self.directions = []
        # if self.shared_camera:
        K = torch.from_numpy(camera_matrix).float()
        K[:2] *= 2 # [1024, 1224] -> [2048, 2448]
        dist = torch.from_numpy(dist_coeffs).float()
        directions = get_undistorted_ray_directions(h*2, w*2, K, dist)
        self.directions = directions

        self.all_c2w, self.all_images, self.all_fg_masks, self.all_raws = [], [], [], []

        # choose images for experiments on number of views
        # non-positive number means using all images
        num_all_images = max([int(k.split('_')[-1]) for k in cams.keys()]) + 1
        ## index of used images
        nd_idx = np.arange(num_all_images)
        num_used_images = self.config.num_used_images
        if isinstance(num_used_images, int):
            if num_used_images > 0:
                n = num_all_images
                nn = num_used_images
                nd_idx = np.round(n/nn*(np.arange(nn))).astype(np.int32)
                assert len(nd_idx) == num_used_images
        else:
            nd_idx = np.array(num_used_images).astype(np.int32)
        num_all_images = len(nd_idx)
        print('Number of used images:', len(nd_idx))

        def load_data(idx_image):
        # for i in range(n_images):
            cams = np.load(opj(self.config.root_dir, self.config.cameras_file)) # avoid numpy bug when use parallel
            world_mat, scale_mat = cams[f'world_mat_{idx_image}'], cams[f'scale_mat_{idx_image}']
            scale_mat[:3, :3] *= self.scale_factor
            P = (world_mat @ scale_mat)[:3,:4]
            K, c2w = load_K_Rt_from_P(P)

            c2w = torch.from_numpy(c2w).float()
            # blender follows opengl camera coordinates (right up back)
            # NeuS DTU data coordinate system (right down front) is different from blender
            # https://github.com/Totoro97/NeuS/issues/9
            # for c2w, flip the sign of input camera coordinate yz
            c2w_ = c2w.clone()
            c2w_[:3,1:3] *= -1. # flip input sign

            if self.split in ['train', 'val']:
                # load color images
                img_path = opj(self.config.root_dir, f'image/{idx_image:03d}.png')
                img = Image.open(img_path) # [0, 255]
                img = img.resize(self.img_wh, Image.BICUBIC)
                img = TF.to_tensor(img).permute(1, 2, 0)[...,:3] # [0.0, 1.0]

                # load masks
                if self.has_mask:
                    mask_path = os.path.join(self.config.root_dir, f'mask/{idx_image:03d}.png')
                    mask = Image.open(mask_path).convert('L') # (H, W, 1)
                    mask = mask.resize(self.img_wh, Image.BICUBIC)
                    mask = TF.to_tensor(mask)[0]
                else:
                    mask = torch.ones(img.shape[:2])

                # load raw images or aop
                if self.has_raw:
                    path_raw_dist = opj(self.config.root_dir, f'raw/{idx_image:03d}.npy')
                    raw_dist = torch.from_numpy( 
                            np.load( path_raw_dist ).astype(np.float32)
                        )/(2**12-1)
                    raw_dist = raw_dist**(1/self.gamma) if not np.equal(self.gamma, 1.0) else raw_dist
                else:
                    path_aop = opj(self.config.root_dir, f'aop/{idx_image:03d}.npy')
                    aop = torch.from_numpy(np.load(path_aop).astype(np.float32))

                    path_dop = opj(self.config.root_dir, f'dop/{idx_image:03d}.npy')
                    dop = torch.from_numpy(np.load(path_dop).astype(np.float32))

                if self.has_raw:
                    return c2w_[:3,:4], img, mask, raw_dist
                else:
                    return c2w_[:3,:4], img, mask, aop, dop
            else:
                return c2w_[:3,:4]

        from tqdm import tqdm
        list_data = []
        for i in tqdm(nd_idx):
            list_data.append(load_data(i))
        
        if self.split in ['train', 'val']:
            if self.has_raw:
                self.all_c2w, self.all_images, self.all_fg_masks, self.all_raws = zip(*list_data)
            else:
                self.all_c2w, self.all_images, self.all_fg_masks, self.all_aops, self.all_dops = zip(*list_data)
        else:
            self.all_c2w = list_data

        self.all_c2w = torch.stack(self.all_c2w, dim=0)

        # auto orient
        # translation = poses[..., :3, 3]
        # mean_translation = torch.mean(translation, dim=0)
        # translation_diff = translation - mean_translation

        if self.split == 'test':
            squareLength=640/(640*12+960)*0.4
            pitch = np.pi/180*20

            # distance = 160*squareLength # rabbit3
            # distance = 60*squareLength # loong1
            distance = float(self.config.camera_distance)
            height = distance*np.sin(pitch)
            distance_to_z = distance*np.cos(pitch)

            # yaw = np.linspace(0, np.pi*2, self.config.n_test_traj_steps)
            yaw = np.linspace(0-np.pi/2, np.pi*2-np.pi/2, 450) # fps 20
            # yaw = yaw[::-1]
            translation_w = np.concatenate([  
                distance_to_z*np.cos(yaw).reshape([-1,1]), 
                distance_to_z*np.sin(yaw).reshape([-1,1]),
                height*np.ones_like(yaw.reshape([-1,1])) ], axis=-1)

            rotation_base = np.array([ 
                [0,1,0],
                [np.sin(pitch), 0, -np.cos(pitch)],
                [-np.cos(pitch), 0, -np.sin(pitch)]
                ]).T
            rotations_w = np.array([ cv2.Rodrigues(np.array([0,0,1])*yaw_i)[0] @ rotation_base for yaw_i in yaw])
            rotations_w[:,:3,1:3] *= -1. # flip input sign

            tf = np.concatenate([rotations_w, translation_w.reshape([-1,3,1])], axis=-1)
            self.all_c2w = torch.from_numpy(tf).float().to(self.rank)

            self.all_c2w = self.all_c2w[:1].to(self.rank)
            # self.all_c2w = self.all_c2w.to(self.rank)
            self.all_images = [0]*len(self.all_c2w)
            self.all_raws = [0]*len(self.all_c2w)
            self.all_fg_masks = [0]*len(self.all_c2w)

            # original
            # self.all_c2w = create_spheric_poses(self.all_c2w[:,:,3], n_steps=self.config.n_test_traj_steps).float().to(self.rank)
            # self.all_raws = torch.zeros((self.config.n_test_traj_steps, self.h, self.w, 3), dtype=torch.float32).float().to(self.rank)
            # self.all_fg_masks = torch.zeros((self.config.n_test_traj_steps, self.h, self.w), dtype=torch.float32).float().to(self.rank)

            self.directions = self.directions.float()[::2,::2].to(self.rank) # speed up test
        else:
            self.all_c2w = self.all_c2w.float().to(self.rank)
            self.all_images = torch.stack(self.all_images, dim=0).float().to(self.rank)
            self.all_fg_masks = torch.stack(self.all_fg_masks, dim=0).float().to(self.rank)
            if self.has_raw:
                self.all_raws = torch.stack(self.all_raws, dim=0).float().to(self.rank)
                q9X = torch.quantile(
                    self.all_raws.view(len(self.all_raws), -1),
                    0.997,
                    interpolation="lower", dim=1).median()
                # self.all_raws = self.all_raws.to(self.rank)
                print("---------------- percentile", q9X*4095)
                self.intensity_q9X = float(q9X)
                del q9X # release VRAM
            else:
                self.all_aops = torch.stack(self.all_aops, dim=0).float().to(self.rank)
                self.all_dops = torch.stack(self.all_dops, dim=0).float().to(self.rank)

            self.directions = self.directions.float().to(self.rank)


class DTUDataset(Dataset, DTUDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_c2w)

    def __getitem__(self, index):
        return {
            'index': index
        }


class DTUIterableDataset(IterableDataset, DTUDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('pol-dtu')
class DTUDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = DTUIterableDataset(self.config, 'train')
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = DTUDataset(self.config, self.config.get('val_split', 'train'))
        if stage in [None, 'test']:
            self.test_dataset = DTUDataset(self.config, self.config.get('test_split', 'test'))
        if stage in [None, 'predict']:
            self.predict_dataset = DTUDataset(self.config, 'train')    

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset, 
            num_workers=os.cpu_count(), 
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )
    
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)       
