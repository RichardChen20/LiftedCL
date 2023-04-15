# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.nn as nn

BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PoseResNet_CL(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        self.dim = 128

        super(PoseResNet_CL, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.mlp_invariant = nn.Sequential(
            nn.Linear(2048, 2048), 
            #nn.BatchNorm1d(self.dim),
            nn.ReLU(), 
            nn.Linear(2048, self.dim),
            )

        self.conv_variant = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0), 
            #nn.BatchNorm1d(self.dim),
            nn.ReLU(), 
            nn.Conv2d(2048, self.dim, kernel_size=1, stride=1, padding=0),
            )
            # ouput size for 256x256: 8x8x128


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x_pool = self.pool(x).reshape(batch_size, -1)
        inv_x = self.mlp_invariant(x_pool)
        var_x = self.conv_variant(x) # batch, dim, 8, 8

        return inv_x, var_x

    def init_weights(self, pretrained=None, zero_init_residual=True):
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            print('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            print('=> init weights from kaiming distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    #nn.init.normal_(m.weight, std=0.001)
                    # nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)

                # Zero-initialize the last BN in each residual branch,
                # so that the residual branch starts with zeros, and each residual block behaves like an identity.
                # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
                if zero_init_residual:
                    for m in self.modules():
                        if isinstance(m, Bottleneck):
                            nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                        elif isinstance(m, BasicBlock):
                            nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]


def get_pose_net_cl(num_layers=50, is_pretrain=True, pretrained='./resnet50-19c8e357.pth'):

    resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3])
}
    block_class, layers = resnet_spec[num_layers]

    model = PoseResNet_CL(block_class, layers)

    print('before init')
    print(model.conv1.weight[0, 0, 0, 0], model.layer1[0].bn3.weight[0])
    if is_pretrain:
        model.init_weights(pretrained)
    print('after init')
    print(model.conv1.weight[0, 0, 0, 0], model.layer1[0].bn3.weight[0])

    return model

if __name__ == '__main__':
    num_layers = 50
    is_pretrain = True
    pretrained = ''

    resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3])
}
    block_class, layers = resnet_spec[num_layers]

    model = PoseResNet_CL(block_class, layers)
    print(model)
    print(model.conv1.weight[0, 0, 0, 0], model.layer1[0].bn3.weight[0])
    if is_pretrain:
        model.init_weights(pretrained)
    print(model.conv1.weight[0, 0, 0, 0], model.layer1[0].bn3.weight[0])
    input = torch.rand(2, 3, 256, 192)
    out_inv, out_var = model(input)
    print(out_inv.shape, out_var.shape)
