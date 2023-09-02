import sys
sys.path.insert(0,'..')
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from time import time
from architectures.SGPN_utils import *
from architectures.net_utils import *

class SGPN(nn.Module):
    def __init__(self, num_classes = 2, latent_dims = 3, alpha_step = 50, margin = 0.8):
        super(SGPN, self).__init__()
        self.alpha = 2.0
        self.margin = margin
        self.alpha_step = alpha_step
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        #self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 64])

        #self.fp05 = PointNetFeaturePropagation(160, [128, 128, 64])
        #self.fp0 = PointNetFeaturePropagation(67, [128, 128, 64])
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
        ## similarity
        self.conv2_1 = nn.Conv2d(64, latent_dims, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, xyz, target = None, training=False, epoch = None):
        xyz = xyz.permute(0, 2, 1)
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)

        if not training:
            return l0_points.permute(0, 2, 1)

        Fsim = self.conv2_1(l0_points.unsqueeze(2)).squeeze(2)
        r = torch.sum(Fsim*Fsim,dim=1)
        r = r.view((l0_points.shape[0],-1,1)).permute(0,2,1)
        trans = torch.transpose(Fsim ,2, 1)
        mul = 2 * torch.matmul(trans, Fsim)
        sub = r - mul
        D = sub + torch.transpose(r, 2, 1)
        D[D<=0.0] = 0.0
        return self.compute_loss(D, target, epoch)
    
    def compute_loss(self, D, target, epoch = 1):
        ## similarity
        pts_group_label, group_mask = self.convert_groupandcate_to_one_hot(target)
        # alpha=2.0

        if(epoch % self.alpha_step == 0 and epoch != 0):
            self.alpha = self.alpha + 2.0

        ## Similarity loss
        B = pts_group_label.shape[0]
        N = pts_group_label.shape[1]

        group_mat_label = torch.matmul(pts_group_label,torch.transpose(pts_group_label,1,2))
        diag_idx = torch.arange(0,group_mat_label.shape[1], out=torch.LongTensor())
        group_mat_label[:,diag_idx,diag_idx] = 1.0

        samegroup_mat_label = group_mat_label
        diffgroup_mat_label = 1.0 - group_mat_label

        num_samegroup = torch.sum(samegroup_mat_label)

        pos = samegroup_mat_label * D

        ## TODO : Replace with original format as below:
        ## sub = self.margin - D
        ## sub[sub<=0.0] = 0.0

        ## VERY GOOD: sub = 1/(D+1)
        sub = 1/(D+1) * 1/(D+1)

        neg_samesem = self.alpha * (diffgroup_mat_label * sub)
        
        simmat_loss = neg_samesem + pos

        # TODO: Add these lines back
        ## group_mask_weight = torch.matmul(group_mask.unsqueeze(2), torch.transpose(group_mask.unsqueeze(2), 2, 1))
        ## simmat_loss = simmat_loss * group_mask_weight
        # Instead of
        simmat_loss = 100 * simmat_loss
        
        simmat_loss = torch.mean(simmat_loss)

        return simmat_loss
    
    def convert_groupandcate_to_one_hot(self,grouplabels):
        # grouplabels: BxN
        NUM_GROUPS = 50
        group_one_hot = torch.zeros((grouplabels.shape[0], grouplabels.shape[1], NUM_GROUPS)).cuda()
        pts_group_mask = torch.zeros((grouplabels.shape[0], grouplabels.shape[1])).cuda()

        un, cnt = torch.unique(grouplabels, return_counts=True)
        group_count_dictionary = {}
        for v,u in enumerate(un):
            group_count_dictionary[int(u.item())] = cnt[v].item()
        totalnum = 0
        for k_un, v_cnt in group_count_dictionary.items():
            if k_un != -1:
                totalnum += v_cnt

        for idx in range(grouplabels.shape[0]):
            for jdx in range(grouplabels.shape[1]):
                if grouplabels[idx, jdx] != -1:
                    group_one_hot[idx, jdx, int(grouplabels[idx, jdx])] = 1
                    pts_group_mask[idx, jdx] = float(totalnum) / float(group_count_dictionary[int(grouplabels[idx, jdx])]) # 1. - float(group_count_dictionary[grouplabels[idx, jdx]]) / totalnum

        return group_one_hot.float(), grouplabels
