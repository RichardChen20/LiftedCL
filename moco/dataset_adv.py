from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import torch
from torch.utils.data import Dataset
import os 
from PIL import Image


class Adv_Dataset(Dataset):
    def __init__(self, traindir, geometry_transform=None, app_transform=None, normal_transform=None):
        self.path = traindir
        self.db = []
        self._get_db()
        self.randomresizecrop = geometry_transform[0]
        self.randomflip = geometry_transform[1]
        self.app_transform = app_transform
        self.normal_transform = normal_transform
        self.joints_3d = torch.load('norm_joints.pt')
        self.joints_3d_num = self.joints_3d.shape[0]

    def _get_db(self):
        self.db = os.listdir(self.path)

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_name = db_rec
        image_file = os.path.join(self.path, image_name)

        image_raw = Image.open(image_file)
        
        if self.randomresizecrop:
            image_q, meta_crop_q = self.randomresizecrop(image_raw)
            image_k, meta_crop_k = self.randomresizecrop(image_raw)

        # here is the appearance transform
        if self.app_transform:
            image_q = self.app_transform(image_q)
            image_k = self.app_transform(image_k)
        
        if self.randomflip:
            image_q, meta_flip_q = self.randomflip(image_q)
            image_k, meta_flip_k = self.randomflip(image_k)

        if self.normal_transform:
            image_q = self.normal_transform(image_q)
            image_k = self.normal_transform(image_k)

        aug_q = [meta_crop_q[0], meta_crop_q[1], meta_crop_q[2], meta_crop_q[3], meta_flip_q]
        aug_q = torch.Tensor(aug_q)
        aug_k = [meta_crop_k[0], meta_crop_k[1], meta_crop_k[2], meta_crop_k[3], meta_flip_k]
        aug_k = torch.Tensor(aug_k)

        joints_real = self.joints_3d[np.random.randint(self.joints_3d_num)].to(torch.float32)

        return image_q, image_k, aug_q, aug_k, joints_real
