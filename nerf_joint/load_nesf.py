import torch
from torch.utils.data import Dataset
import sys
sys.path.append('/Users/jfgvl1187/Desktop/CSCI 2980 3D Vision Research/Phrase Localization in 3D Scene/Sementic CLIP Neural Field/Baseline/models')
#from models.slic_vit import SLICViT
from slic_vit import SLICViT
import glob
import numpy as np
import os
import cv2
from torchvision import transforms as T
from typing import Optional
# from .rays import *
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd
from jax3d.projects.nesf.nerfstatic.datasets import klevr
import json
#from epath import Path
from load_blender import pose_spherical
import torch
from PIL import Image
import os.path as osp
import imageio.v2 as imageio





class Nesf_Dataset():
    def __init__(self, dataset_dir, split="train", indices=None, scale=1, near=0, far=30):
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
        # if self.split == "val":
        #     split = "test"
        data, self.metadata = klevr.make_examples(data_dir=self.root_dir, split=self.split, image_idxs=self.indices, scale=self.scale, enable_sqrt2_buffer=False)
        # data, self.metadata = klevr.make_unreal_examples(data_dir=self.root_dir, split=split)
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
        sample["image"] = self.imgs.rgb[index]
        sample["img_ids"] = self.imgs.image_ids[index]
        # sample["prev_image"] = torch.from_numpy(self.imgs.rgb[index-1])
        # sample["next_image"] = torch.from_numpy(self.imgs.rgb[index+1])

        # Reading Masks
        # sample["mask"] = self.imgs.semantics[index][:, :, 0]
        # sample["prev_mask"] = torch.from_numpy(self.imgs.semantics[index-1][:, :, 0])
        # sample["next_mask"] = torch.from_numpy(self.imgs.semantics[index+1][:, :, 0])
        '''
        n_prev_mask = cv2.imread(self.imgs[index-1]["mask_path"],0) /255.0
        n_prev_mask = cv2.resize(n_prev_mask, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        sample["prev_mask"] = torch.from_numpy(n_prev_mask)
        n_next_mask = cv2.imread(self.imgs[index+1]["mask_path"],0)/255.0
        n_next_mask = cv2.resize(n_next_mask, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        sample["next_mask"] = torch.from_numpy(n_next_mask)'''

        # Reading Depth       
        # sample["depth"] = torch.from_numpy(self.imgs.depth[index][:, :, 0])

        # Reading poses
        n_pose = self.metadata.cameras.px2world_transform[index]
        sample["pose"] = n_pose
        # n_prev_pose = self.metadata.cameras.px2world_transform[index-1]
        # sample["prev_pose"] = torch.from_numpy(n_prev_pose)
        # n_next_pose = self.metadata.cameras.px2world_transform[index+1]
        # sample["next_pose"] = torch.from_numpy(n_next_pose)

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



def load_Nesf_data(basedir, half_res=False, testskip=1, use_saliency = False):
    model = SLICViT
    model_args = {
        'model': 'vit14',
        'alpha': 0.75,
        'aggregation': 'mean',
        'n_segments': list(range(100, 200, 50)),
        'temperature': 0.02,
        'upsample': 2,
        'start_block': 0,
        'compactness': 50,
        'sigma': 0,
    }

    model = model(**model_args)#.cuda()
    with open(os.path.join(basedir,"metadata.json"), 'r') as fp: #base_dir is "../data/toybox-13/0"
            file = json.load(fp)
    splits = ['train', 'val', 'test']
    metas = {}
    H = file["metadata"]['height']
    W = file["metadata"]['width']
    focal = file["camera"]['focal_length']

    dataloader = Nesf_Dataset(basedir, split = "test")
    """
    img = dataloader[0]["image"]
    pose = dataloader[0]["pose"]
    index = dataloader[0]["img_ids"]

    print(img)
    print(pose)
    print(index)
    print(len(dataloader))
    exit(0)
    """
    near = dataloader.near
    far = dataloader.far
    K = dataloader[0]["Intrinsics"]
    all_imgs = []
    all_poses = []
    all_saliencies = []
    counts = [0]
    imgs = []
    poses = []
    saliencies = []
    for i in range(len(dataloader)):
        img = dataloader[i]["image"]
        pose = dataloader[i]["pose"]
        imgs.append(img)
        poses.append(pose)
        # real_img = np.uint8((img)*255)
        index = (dataloader[i]["img_ids"])
        # print(index[-5:])
        if use_saliency:
            # fname = "rgba_" + index[-5:] + '_heat.npy'
        #     saliencies.append(np.load(fname))
            fname = "rgba_" + index[-5:] + '_heat.png'
            dir = "/gpfs/data/ssrinath/ychen485/implicitSearch/implicitObjDetection/Nesf0/"
            fname = os.path.join(dir, fname)
            saliencies.append(imageio.imread(fname))
    if use_saliency:
        saliencies = (4* np.array(saliencies) / 255.).astype(np.float32)
        all_saliencies.append(saliencies)
        # # print(real_img.shape)
        # heatmap = getHeatmap(model, real_img , "chair")
        # saliency = heatmap*200
        # print(saliency.shape)
        # o_im = Image.fromarray(real_img)
        # h_im = Image.fromarray(saliency).convert ('RGB')
        # o_im.save("/gpfs/data/ssrinath/ychen485/implicitSearch/implicitObjDetection/dataDemo/"+str(index)+".png")
        # h_im.save("/gpfs/data/ssrinath/ychen485/implicitSearch/implicitObjDetection/dataDemo/"+str(i)+"_heat.png")

    imgs = (np.array(imgs)).astype(np.float32) # keep all 4 channels (RGBA)
    poses = np.array(poses).astype(np.float32)
    print("imgs: ", imgs.shape)
    print("poses: ", poses.shape)
    # print("saliencies: ", saliencies.shape)
    counts.append(counts[-1] + imgs.shape[0])
    all_imgs.append(imgs)
    all_poses.append(poses)

    dataloader = Nesf_Dataset(basedir, split="test")
    imgs = []
    poses = []
    saliencies = []
    for i in range(len(dataloader)):
        img = dataloader[i]["image"]
        pose = dataloader[i]["pose"]
        imgs.append(img)
        poses.append(pose)
        index = (dataloader[i]["img_ids"])
        # print(index[-5:])
        if use_saliency:
            # fname = "rgba_" + index[-5:] + '_heat.npy'
        #     saliencies.append(np.load(fname))
            fname = "rgba_" + index[-5:] + '_heat.png'
            dir = "/gpfs/data/ssrinath/ychen485/implicitSearch/implicitObjDetection/Nesf0/"
            fname = os.path.join(dir, fname)
            saliencies.append(imageio.imread(fname))
    if use_saliency:
        saliencies = (4* np.array(saliencies) / 255.).astype(np.float32)
        all_saliencies.append(saliencies)
    imgs = (np.array(imgs)).astype(np.float32) # keep all 4 channels (RGBA)
    poses = np.array(poses).astype(np.float32)
    print("imgs: ", imgs.shape)
    print("poses: ", poses.shape)
    # print("saliencies: ", saliencies.shape)

    counts.append(counts[-1] + imgs.shape[0])
    all_imgs.append(imgs)
    all_poses.append(poses)

    dataloader = Nesf_Dataset(basedir, split="test")
    imgs = []
    poses = []
    saliencies = []
    for i in range(len(dataloader)):
        img = dataloader[i]["image"]
        pose = dataloader[i]["pose"]
        imgs.append(img)
        poses.append(pose)
        index = (dataloader[i]["img_ids"])
        # print(index[-5:])
        if use_saliency:
            # fname = "rgba_" + index[-5:] + '_heat.npy'
        #     saliencies.append(np.load(fname))
            fname = "rgba_" + index[-5:] + '_heat.png'
            dir = "/gpfs/data/ssrinath/ychen485/implicitSearch/implicitObjDetection/Nesf0/"
            fname = os.path.join(dir, fname)
            saliencies.append(imageio.imread(fname))
    if use_saliency:
        saliencies = (4* np.array(saliencies) / 255.).astype(np.float32)
        all_saliencies.append(saliencies)
    imgs = (np.array(imgs)).astype(np.float32) # keep all 4 channels (RGBA)
    poses = np.array(poses).astype(np.float32)
    print("imgs: ", imgs.shape)
    print("poses: ", poses.shape)
    # print("saliencies: ", saliencies.shape)

    counts.append(counts[-1] + imgs.shape[0])
    all_imgs.append(imgs)
    all_poses.append(poses)

    #-------------------------------loaded all 3 sets of data--------------------------------------
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    if use_saliency:
        saliencies = np.concatenate(all_saliencies, 0)
    print("imgs, poses, i_split: ", imgs.shape, poses.shape, len(i_split))
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    # for i in range(10):
    #     print (poses[i])
    # print(imgs[0])
    if use_saliency:
        return imgs, saliencies, poses, render_poses, [H, W, focal], i_split, near, far, K
    else:
        return imgs, poses, render_poses, [H, W, focal], i_split, near, far, K

if __name__== "__main__":
    load_Nesf_data("../data/toybox-13/0")
    # pt = Path()

    # dataset_dir = "/gpfs/data/ssrinath/ychen485/implicitSearch/implicitObjDetection/toybox-13/0"
    # # p = pt.get(dataset_dir)
    # dataloader = Nesf_Dataset(dataset_dir)

    # print(dataloader[0]["pose"])


    # dset = DataLoader(dataloader, batch_size = 4, shuffle = True)

    # for batch, data_sample in enumerate(dset):

    #     for key in data_sample:

    #         print(key, " ", data_sample[key].shape)

        # print("###########################################")