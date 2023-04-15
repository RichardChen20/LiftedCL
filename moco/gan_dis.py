import math
import random

import torch
from torch import nn
from torch.nn import functional as F


class Res_Block(nn.Module):
    def __init__(self, dim=512):
        super(Res_Block, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(dim, dim), 
            # nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Linear(dim, dim),
            )
        # self.bn = nn.BatchNorm1d(dim)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x

        out = self.model(x)

        out = residual + out
        # out = self.bn(out)
        out = self.relu(out)

        return out

class Lifting_Network(nn.Module):
    def __init__(self, dim=512, num_joints=17, num_xyz=3):
        super(Lifting_Network, self).__init__()

        self.num_joints = num_joints
        self.num_xyz = num_xyz

        self.linear = nn.Sequential(
            nn.Linear(2048, dim), 
            # nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            )
        
        self.res_1 = Res_Block(dim=dim)

        self.final = nn.Linear(dim, self.num_joints*self.num_xyz)

    def forward(self, x):
        x = self.linear(x)
        x = self.res_1(x)
        joints = self.final(x)

        # joints = joints.reshape(joints.shape[0], self.num_joints, self.num_xyz)
        # joints = joints - joints[:, 0:1, :]
        # std = (joints**2).sum(axis=(1, 2), keepdims=True).sqrt()
        # joints = joints / std
        # joints = joints.reshape(joints.shape[0], -1)

        return joints
    

class Joints_Discriminator(nn.Module):
    def __init__(self, dim=256, num_joints=17, num_xyz=3):
        super(Joints_Discriminator, self).__init__()
        self.num_joints = num_joints
        self.num_xyz = num_xyz

        self.kcs = KCS_layer(dim=dim, num_joints=self.num_joints, num_xyz=self.num_xyz)

        self.pose = nn.Sequential(
            nn.Linear(self.num_joints*self.num_xyz, dim), 
            # nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            Res_Block(dim=dim),
            # nn.Linear(dim, dim), 
            )
        
        self.final = nn.Sequential(
            nn.Linear(dim*2, dim//2), 
            # nn.BatchNorm1d(dim//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim//2, 1)
        )

    def forward(self, joints):
        joints = joints.reshape(joints.shape[0], -1)

        pose = self.pose(joints)
        kcs = self.kcs(joints)

        out = torch.cat((pose, kcs), dim=1)
        validity = self.final(out)

        return validity

actual_joints = {
            0: 'root',
            1: 'rhip',
            2: 'rkne',
            3: 'rank',
            4: 'lhip',
            5: 'lkne',
            6: 'lank',
            7: 'belly',
            8: 'neck',
            9: 'nose',
            10: 'head',
            11: 'lsho',
            12: 'lelb',
            13: 'lwri',
            14: 'rsho',
            15: 'relb',
            16: 'rwri'
        }

bone = [[
    [1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0],
    [0,1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,-1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,-1,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,-1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,-1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,-1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,-1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,-1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,-1,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,-1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,-1],
]]

class KCS_layer(nn.Module):
    def __init__(self, dim=256, num_joints=17, num_xyz=3):
        super(KCS_layer, self).__init__()
        self.num_joints = num_joints
        self.num_xyz = num_xyz
        self.bone = torch.Tensor(bone)
        self.num_bone = self.bone.shape[1]

        self.linear = nn.Sequential(
            nn.Linear(self.num_bone**2, dim), 
            # nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            Res_Block(dim=dim),
            # nn.Linear(dim, dim), 
            )
        
    def forward(self, joints):
        bone = self.bone.to(joints.device)

        joints = joints.reshape(joints.shape[0], self.num_joints, self.num_xyz)
        B = torch.matmul(bone, joints) 
        out = torch.matmul(B, B.transpose(1,2)).reshape(joints.shape[0], -1)

        out = self.linear(out)

        return out
