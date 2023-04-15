# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from moco.pose_res_cl import get_pose_net_cl

class CL_Model(nn.Module):
    def __init__(self, dim=128, K=65536, m=0.999, T=0.2):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(CL_Model, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.dim = dim

        # create the encoders
        self.encoder_q = get_pose_net_cl()
        self.encoder_k = get_pose_net_cl()

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue_inv", torch.randn(dim, K))
        self.queue_inv = nn.functional.normalize(self.queue_inv, dim=0)

        self.register_buffer("queue_var", torch.randn(dim, K))
        self.queue_var = nn.functional.normalize(self.queue_var, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_inv, keys_var):
        # gather keys before updating queue
        keys_inv = concat_all_gather(keys_inv)
        keys_var = concat_all_gather(keys_var)

        batch_size = keys_inv.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_inv[:, ptr:ptr + batch_size] = keys_inv.T
        self.queue_var[:, ptr:ptr + batch_size] = keys_var.T

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, x_var, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        x_var_gather = concat_all_gather(x_var)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], x_var_gather[idx_this]

        
    def _cal_pos(self, q, k, meta_q, meta_k):
        upsample_size = 16
        i_q, j_q, h_q, w_q, flip_q = meta_q[:, 0], meta_q[:, 1], meta_q[:, 2], meta_q[:, 3], meta_q[:, 4]
        i_k, j_k, h_k, w_k, flip_k = meta_k[:, 0], meta_k[:, 1], meta_k[:, 2], meta_k[:, 3], meta_k[:, 4]
        # HorizontalFlip
        q = q * (1 - flip_q).reshape(-1, 1, 1, 1) + flip_q.reshape(-1, 1, 1, 1) * torch.flip(q, dims=[-1])
        k = k * (1 - flip_k).reshape(-1, 1, 1, 1) + flip_k.reshape(-1, 1, 1, 1) * torch.flip(k, dims=[-1])

        # cal shared area, like nms
        box_h_left = torch.max(i_q, i_k)
        box_w_left = torch.max(j_q, j_k)
        box_h_right = torch.min(i_q + h_q, i_k + h_k)
        box_w_right = torch.min(j_q + w_q, j_k + w_k)

        q_h_low = (box_h_left - i_q) / h_q
        q_h_high = (box_h_right - i_q) / h_q
        q_w_low = (box_w_left - j_q) / w_q
        q_w_high = (box_w_right - j_q) / w_q

        k_h_low = (box_h_left - i_k) / h_k
        k_h_high = (box_h_right - i_k) / h_k
        k_w_low = (box_w_left - j_k) / w_k
        k_w_high = (box_w_right - j_k) / w_k

        pos = torch.zeros(q.shape[0]*upsample_size*upsample_size).cuda()
        q_all = torch.zeros(q.shape[0]*upsample_size*upsample_size, 128).cuda()
        k_all = torch.zeros(q.shape[0]*upsample_size*upsample_size, 128).cuda()

        for cnt in range(q.shape[0]):
            q_crop = q[cnt, :, int(q.shape[2]*q_h_low[cnt]+0.5):int(q.shape[2]*q_h_high[cnt]+0.5), int(q.shape[3]*q_w_low[cnt]+0.5):int(q.shape[3]*q_w_high[cnt]+0.5)]
            k_crop = k[cnt, :, int(k.shape[2] * k_h_low[cnt]+0.5):int(k.shape[2] * k_h_high[cnt]+0.5), int(k.shape[3] * k_w_low[cnt]+0.5):int(k.shape[3] * k_w_high[cnt]+0.5)]
            if 0 in q_crop.shape or 0 in k_crop.shape:
                pos[cnt] = 0
            else:
                q_crop = nn.functional.upsample(q_crop.unsqueeze(0), size=[upsample_size, upsample_size], mode='bilinear', align_corners=True)
                k_crop = nn.functional.upsample(k_crop.unsqueeze(0), size=[upsample_size, upsample_size], mode='bilinear', align_corners=True)
                
                q_v = nn.functional.normalize(q_crop, dim=1).reshape(-1, upsample_size*upsample_size).transpose(0, 1)
                k_v = nn.functional.normalize(k_crop, dim=1).reshape(-1, upsample_size*upsample_size).transpose(0, 1)

                q_all[cnt*upsample_size*upsample_size:(cnt+1)*upsample_size*upsample_size, :] = q_v
                k_all[cnt*upsample_size*upsample_size:(cnt+1)*upsample_size*upsample_size, :] = k_v

        pos = torch.einsum('nc,nc->n', [q_all, k_all.detach()]).unsqueeze(-1)

        return pos, q_all, k_all

    def forward(self, im_q, im_k, meta_q, meta_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            meta: includes info about data augmentation
        Output:
            logits, targets for corss entropy loss
        """
        batch_size = im_q.shape[0]
        # compute query features
        q_inv, q_var = self.encoder_q(im_q)  # queries: NxC
        q_inv = nn.functional.normalize(q_inv, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k_inv, k_var = self.encoder_k(im_k)  # keys: NxC
            k_inv = nn.functional.normalize(k_inv, dim=1)

            # undo shuffle
            k_inv, k_var = self._batch_unshuffle_ddp(k_inv, k_var, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos_inv = torch.einsum('nc,nc->n', [q_inv, k_inv]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_inv = torch.einsum('nc,ck->nk', [q_inv, self.queue_inv.clone().detach()])
        # logits: Nx(1+K)
        logits_inv = torch.cat([l_pos_inv, l_neg_inv], dim=1)
        # apply temperature
        logits_inv /= self.T
        # labels: positive key indicators
        labels_inv = torch.zeros(logits_inv.shape[0], dtype=torch.long).cuda()
        
        # varient feature
        l_pos_var, q_var_dense, k_var_dense = self._cal_pos(q_var, k_var, meta_q, meta_k)

        k_var_pool = nn.functional.adaptive_avg_pool2d(k_var, output_size=(1, 1)).reshape(k_var.shape[0], 128)
        k_var_pool = nn.functional.normalize(k_var_pool, dim=1).detach()
        l_neg_var = torch.einsum('nc,ck->nk', [q_var_dense, self.queue_var.clone().detach()])# / self.dim
        # logits: Nx(1+K)
        logits_var = torch.cat([l_pos_var, l_neg_var], dim=1)
        # apply temperature
        logits_var /= self.T
        # labels: positive key indicators
        labels_var = torch.zeros(logits_var.shape[0], dtype=torch.long).cuda()
        # dequeue and enqueue
        self._dequeue_and_enqueue(k_inv, k_var_pool)

        return logits_inv, labels_inv, logits_var, labels_var, [l_pos_inv, l_neg_inv, l_pos_var, l_neg_var]


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
