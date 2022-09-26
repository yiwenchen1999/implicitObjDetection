import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
import cv2
from PIL import Image
from torchvision import transforms as T
from typing import Optional
# from .rays import *
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd
from jax3d.projects.nesf.nerfstatic.datasets import klevr
import json
from epath import Path



class Nesf_Dataset():
    def __init__(self, dataset_dir, split="train", indices=None, scale=1, near=0, far=10):
        self.root_dir = dataset_dir
        self.near = near
        self.far = far
        self.use_viewdir = True
        self.convention = "opencv"
        self.split = split
        self.scale = scale
        self.white_back = False
        self.indices = indices
        self.main()

    def main(self):
        split = "train"
        # if self.split == "val":
        #     split = "test"
        data, self.metadata = klevr.make_examples(data_dir=self.root_dir, split=split, image_idxs=self.indices, scale=self.scale, enable_sqrt2_buffer=False)
        self.imgs = data.target_view

    def __len__(self):
        # print(len(self.imgs))
        return len(self.imgs.image_ids)

    def __getitem__(self, idx):

        if self.split == "train":
            idx -= ((idx // 5) + 1)
        elif self.split == "val":
            idx = idx // 5

        if idx == 0:
            index = idx+1
        elif idx == len(self.imgs.rgb)-1 :
            index = idx -1
        else:
            index = idx
        sample = {}
        
        # Reading Images                     
        sample["image"] = torch.from_numpy(self.imgs.rgb[index])
        sample["prev_image"] = torch.from_numpy(self.imgs.rgb[index-1])
        sample["next_image"] = torch.from_numpy(self.imgs.rgb[index+1])

        # Reading Masks
        sample["mask"] = torch.from_numpy(self.imgs.semantics[index][:, :, 0])
        sample["prev_mask"] = torch.from_numpy(self.imgs.semantics[index-1][:, :, 0])
        sample["next_mask"] = torch.from_numpy(self.imgs.semantics[index+1][:, :, 0])
        '''
        n_prev_mask = cv2.imread(self.imgs[index-1]["mask_path"],0) /255.0
        n_prev_mask = cv2.resize(n_prev_mask, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        sample["prev_mask"] = torch.from_numpy(n_prev_mask)
        n_next_mask = cv2.imread(self.imgs[index+1]["mask_path"],0)/255.0
        n_next_mask = cv2.resize(n_next_mask, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        sample["next_mask"] = torch.from_numpy(n_next_mask)'''

        # Reading Depth       
        sample["depth"] = torch.from_numpy(self.imgs.depth[index][:, :, 0])

        # Reading poses
        n_pose = self.metadata.cameras.px2world_transform[index]
        sample["pose"] = torch.from_numpy(n_pose)
        n_prev_pose = self.metadata.cameras.px2world_transform[index-1]
        sample["prev_pose"] = torch.from_numpy(n_prev_pose)
        n_next_pose = self.metadata.cameras.px2world_transform[index+1]
        sample["next_pose"] = torch.from_numpy(n_next_pose)

        # # Reading rays
        # rays_o = torch.from_numpy(self.imgs.rays.origin[index]).view(-1, 3)
        # rays_d = torch.from_numpy(self.imgs.rays.direction[index]).view(-1, 3)
        # near, far = self.near * torch.ones_like(rays_d[..., :1]), self.far * torch.ones_like(rays_d[..., :1])
        # n_rays = torch.cat([rays_o, rays_d, near, far], -1)
        # sample["rays"] = n_rays

        # rays_o_prev = torch.from_numpy(self.imgs.rays.origin[index]).view(-1, 3)
        # rays_d_prev = torch.from_numpy(self.imgs.rays.direction[index]).view(-1, 3)
        # n_prev_rays = torch.cat([rays_o_prev, rays_d_prev, near, far], -1)
        # sample["prev_rays"] = n_prev_rays

        # rays_o_next = torch.from_numpy(self.imgs.rays.origin[index]).view(-1, 3)
        # rays_d_next = torch.from_numpy(self.imgs.rays.direction[index]).view(-1, 3)
        # n_next_rays = torch.cat([rays_o_next, rays_d_next, near, far], -1)
        # sample["next_rays"] = n_next_rays

        # packing intrinsics
        self.intrinsic_mat = np.array([[self.metadata.cameras.focal_px_length, 0, self.metadata.cameras.resolution[1] / 2],
                                      [0, self.metadata.cameras.focal_px_length, self.metadata.cameras.resolution[0] / 2],
                                      [0, 0, 1]])
        sample["Intrinsics"] = torch.from_numpy(self.intrinsic_mat)



        return sample

if __name__== "__main__":
    pt = Path()

    dataset_dir = "/gpfs/data/ssrinath/ychen485/implicitSearch/implicitObjDetection/toybox-13/0"
    # p = pt.get(dataset_dir)
    dataloader = Nesf_Dataset(dataset_dir)

    print(dataloader[0]["pose"])


    # dset = DataLoader(dataloader, batch_size = 4, shuffle = True)

    # for batch, data_sample in enumerate(dset):

    #     for key in data_sample:

    #         print(key, " ", data_sample[key].shape)

        # print("###########################################")