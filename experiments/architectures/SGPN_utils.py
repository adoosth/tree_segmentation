import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

#ths,ths_,cnt = Get_Ths(pts_corr_val, target['semseg'].cpu().numpy()[0], target['ptsgroup'].cpu().numpy()[0], ths, ths_, cnt)
def Get_Ths(pts_corr, seg, ins, ths, ths_, cnt):
    seg = np.transpose(seg,(1,0))
    pts_corr = pts_corr.detach().cpu().numpy()
    pts_in_ins = {}
    for ip, pt in enumerate(pts_corr):
        if ins[ip] in pts_in_ins.keys():
            pts_in_curins_ind = pts_in_ins[ins[ip]]
            pts_notin_curins_ind = (~(pts_in_ins[ins[ip]])) & (seg==seg[ip]).squeeze()
            hist, bin = np.histogram(pt[pts_in_curins_ind], bins=20)

            if seg[ip]==8:
                print ("pouet",bin)

            numpt_in_curins = np.sum(pts_in_curins_ind)
            numpt_notin_curins = np.sum(pts_notin_curins_ind)

            if numpt_notin_curins > 0:

                tp_over_fp = 0
                ib_opt = -2
                for ib, b in enumerate(bin):
                    if b == 0:
                        break
                    tp = float(np.sum(pt[pts_in_curins_ind] < bin[ib])) / float(numpt_in_curins)
                    fp = float(np.sum(pt[pts_notin_curins_ind] < bin[ib])) / float(numpt_notin_curins)

                    if tp <= 0.5:
                        continue

                    if fp == 0. and tp > 0.5:
                        ib_opt = ib
                        break

                    if tp/fp > tp_over_fp:
                        tp_over_fp = tp / fp
                        ib_opt = ib

                if tp_over_fp >  4.:
                    ths[int(seg[ip])] += bin[ib_opt]
                    ths_[int(seg[ip])] += bin[ib_opt]
                    cnt[int(seg[ip])] += 1

        else:
            pts_in_curins_ind = (ins == ins[ip])
            pts_in_ins[ins[ip]] = pts_in_curins_ind
            pts_notin_curins_ind = (~(pts_in_ins[ins[ip]])) & (seg==seg[ip]).squeeze()
            hist, bin = np.histogram(pt[pts_in_curins_ind], bins=20)
            if seg[ip]==8:
                print ("pouet",bin)
            numpt_in_curins = np.sum(pts_in_curins_ind)
            numpt_notin_curins = np.sum(pts_notin_curins_ind)
            if numpt_notin_curins > 0:
                tp_over_fp = 0
                ib_opt = -2
                for ib, b in enumerate(bin):

                    if b == 0:
                        break

                    tp = float(np.sum(pt[pts_in_curins_ind]<bin[ib])) / float(numpt_in_curins)
                    fp = float(np.sum(pt[pts_notin_curins_ind]<bin[ib])) / float(numpt_notin_curins)

                    if tp <= 0.5:
                        continue

                    if fp == 0. and tp > 0.5:
                        ib_opt = ib
                        break

                    if tp / fp > tp_over_fp:
                        tp_over_fp = tp / fp
                        ib_opt = ib

                if tp_over_fp >  4.:
                    ths[int(seg[ip])] += bin[ib_opt]
                    ths_[int(seg[ip])] += bin[ib_opt]
                    cnt[int(seg[ip])] += 1

    return ths, ths_, cnt

def square_distance(src, dst):
    N = src.shape[0]
    M = dst.shape[0]
    dist = -2 * np.matmul(src, np.transpose(dst,[1,0]))
    dist += np.sum(src ** 2, axis=-1)
    dist += np.sum(dst ** 2, axis=-1)
    return dist

def Create_groups(pts_corr,confidence,seg,pts):
    seg = seg.detach().cpu().numpy()
    pts = pts.detach().cpu().numpy()

    pts_corr = pts_corr.detach().cpu().numpy()
    confidence = confidence.detach().cpu().numpy()
    seg[seg>0.5] = 1
    seg[seg<0.5] = 0
    confvalidpts = (confidence>0.4)
    groupid = np.zeros(seg.shape[0])
    groups = {}
    grp_id = 1
    pts_in_seg = (seg==1)## points in segmentation mask with this class
    valid_seg_group = np.where(pts_in_seg & confvalidpts) ## points with this class and a confidence > 0.5
    for p in valid_seg_group[0]:
        if(groupid[p]==0):## if the point doesnt have a group already
            groupid[p] = grp_id
            valid_grp = np.where(  (pts_corr[p]<0.1) & pts_in_seg )[0]

            # valid_grp = np.where(valid_grp & (distances<10.0))[0]
            groupid[valid_grp] = grp_id
            grp_id = grp_id+1

    print(grp_id)
    return groupid


def GroupMerging(pts_corr, confidence, seg,label_bin):
    seg = seg.detach().cpu().numpy()
    pts_corr = pts_corr.detach().cpu().numpy()
    confidence = confidence.detach().cpu().numpy()
    seg[seg>0.5] = 1
    seg[seg<0.5] = 0
    confvalidpts = (confidence>0.4)
    un_seg = np.unique(seg)
    refineseg = np.zeros(pts_corr.shape[0])
    groupid = -1* np.ones(pts_corr.shape[0])
    # print(groupid,refineseg)
    numgroups = 0
    groupseg = {}
    for i_seg in un_seg:
        if i_seg==-1 :
            continue
        pts_in_seg = (seg==i_seg)## points in segmentation mask with this class
        valid_seg_group = np.where(pts_in_seg & confvalidpts) ## poitns with this class and a confidence > 0.5
        proposals = []
        if valid_seg_group[0].shape[0]==0:## if there are no points in this segmentation group (no points of this class with enough confidence)
            proposals += [pts_in_seg]
        else:
            for ip in valid_seg_group[0]:## for all the points in this class and with enough confidence
                validpt = (pts_corr[ip] < label_bin[int(i_seg)]) & pts_in_seg ## take points in correlation matrix with a distance lower than a threshold and that in same class as pts
                if np.sum(validpt)>5:##if there are more than 5 points
                    flag = False
                    for gp in range(len(proposals)):
                        iou = float(np.sum(validpt & proposals[gp])) / np.sum(validpt|proposals[gp])#uniou
                        validpt_in_gp = float(np.sum(validpt & proposals[gp])) / np.sum(validpt)#uniou
                        if iou > 0.8 or validpt_in_gp > 0.8:
                            flag = True
                            if np.sum(validpt)>np.sum(proposals[gp]):
                                proposals[gp] = validpt
                            continue

                    if not flag:
                        proposals += [validpt]

            if len(proposals) == 0:
                proposals += [pts_in_seg]
        for gp in range(len(proposals)):
            if np.sum(proposals[gp])>50:
                groupid[proposals[gp]] = numgroups
                groupseg[numgroups] = i_seg
                numgroups += 1
                refineseg[proposals[gp]] = stats.mode(seg[proposals[gp]])[0]


    un, cnt = np.unique(groupid, return_counts=True)
    for ig, g in enumerate(un):
        if cnt[ig] < 60:
            groupid[groupid==g] = -1

    un, cnt = np.unique(groupid, return_counts=True)
    groupidnew = groupid.copy()
    for ig, g in enumerate(un):
        if g == -1:
            continue
        groupidnew[groupid==g] = (ig-1)
        groupseg[(ig-1)] = groupseg.pop(g)
    groupid = groupidnew


    for ip, gid in enumerate(groupid):
        if gid == -1:
            pts_in_gp_ind = (pts_corr[ip] < label_bin[int(seg[ip])])
            pts_in_gp = groupid[pts_in_gp_ind]
            pts_in_gp_valid = pts_in_gp[pts_in_gp!=-1]
            if len(pts_in_gp_valid) != 0:
                groupid[ip] = stats.mode(pts_in_gp_valid)[0][0]

    print(np.unique(groupid).shape)
    return groupid, refineseg, groupseg

def BlockMerging(volume, volume_seg, pts, grouplabel, groupseg, gap=1e-3):

    overlapgroupcounts = np.zeros([100,300])
    groupcounts = np.ones(100)
    x=(pts[:,0]/gap).astype(np.int32)
    y=(pts[:,1]/gap).astype(np.int32)
    z=(pts[:,2]/gap).astype(np.int32)
    for i in range(pts.shape[0]):
        xx=x[i]
        yy=y[i]
        zz=z[i]
        if grouplabel[i] != -1:
            if volume[xx,yy,zz]!=-1 and volume_seg[xx,yy,zz]==groupseg[grouplabel[i]]:
                overlapgroupcounts[grouplabel[i],volume[xx,yy,zz]] += 1
        groupcounts[grouplabel[i]] += 1

    groupcate = np.argmax(overlapgroupcounts,axis=1)
    maxoverlapgroupcounts = np.max(overlapgroupcounts,axis=1)

    curr_max = np.max(volume)
    for i in range(groupcate.shape[0]):
        if maxoverlapgroupcounts[i]<7 and groupcounts[i]>30:
            curr_max += 1
            groupcate[i] = curr_max


    finalgrouplabel = -1 * np.ones(pts.shape[0])

    for i in range(pts.shape[0]):
        if grouplabel[i] != -1 and volume[x[i],y[i],z[i]]==-1:
            volume[x[i],y[i],z[i]] = groupcate[grouplabel[i]]
            volume_seg[x[i],y[i],z[i]] = groupseg[grouplabel[i]]
            finalgrouplabel[i] = groupcate[grouplabel[i]]
    return finalgrouplabel
    
def convert_groupandcate_to_one_hot(grouplabels):
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

# class SGPNLoss(nn.Module):
#     def __init__(self):
#         super(SGPNLoss, self).__init__()

#     def forward(self, l0_points, Fsim, target, alpha = 2.0, margin = 0.8):
#         r = torch.sum(Fsim*Fsim,dim=1)
#         r = r.view((l0_points.shape[0],-1,1)).permute(0,2,1)
#         trans = torch.transpose(Fsim ,2, 1)
#         mul = 2 * torch.matmul(trans, Fsim)
#         sub = r - mul
#         D = sub + torch.transpose(r, 2, 1)
#         D[D<=0.0] = 0.0

#         ## similarity
#         pts_group_label, group_mask = convert_groupandcate_to_one_hot(target)
#         # alpha=2.0

#         ## Similarity loss
#         B = pts_group_label.shape[0]
#         N = pts_group_label.shape[1]

#         group_mat_label = torch.matmul(pts_group_label,torch.transpose(pts_group_label,1,2))
#         diag_idx = torch.arange(0,group_mat_label.shape[1], out=torch.LongTensor())
#         group_mat_label[:,diag_idx,diag_idx] = 1.0

#         samegroup_mat_label = group_mat_label
#         diffgroup_mat_label = 1.0 - group_mat_label

#         num_samegroup = torch.sum(samegroup_mat_label)

#         pos = samegroup_mat_label * D

#         ## TODO : Replace with original format as below:
#         ## sub = margin - D
#         ## sub[sub<=0.0] = 0.0

#         ## VERY GOOD: sub = 1/(D+1)
#         sub = 1/(D+1) * 1/(D+1)

#         neg_samesem = alpha * (diffgroup_mat_label * sub)
        
#         simmat_loss = neg_samesem + pos

#         # TODO: Add these lines back
#         ## group_mask_weight = torch.matmul(group_mask.unsqueeze(2), torch.transpose(group_mask.unsqueeze(2), 2, 1))
#         ## simmat_loss = simmat_loss * group_mask_weight
#         # Instead of
#         simmat_loss = 100 * simmat_loss
        
#         simmat_loss = torch.mean(simmat_loss)

#         return simmat_loss

class SGPNLoss(nn.Module):
    def __init__(self):
        super(SGPNLoss, self).__init__()

    def forward(self, l0_points, Fsim, target, alpha = 2.0, margin = 0.8):
        r = torch.sum(Fsim*Fsim,dim=1)
        r = r.view((l0_points.shape[0],-1,1)).permute(0,2,1)
        trans = torch.transpose(Fsim ,2, 1)
        mul = 2 * torch.matmul(trans, Fsim)
        sub = r - mul
        D = sub + torch.transpose(r, 2, 1)
        D[D<=0.0] = 0.0

        ## similarity
        pts_group_label, group_mask = convert_groupandcate_to_one_hot(target)
        # alpha=2.0

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
        sub = margin - D
        sub[sub<=0.0] = 0.0

        ## VERY GOOD: sub = 1/(D+1)
        ## sub = 1/(D+1) * 1/(D+1)

        neg_samesem = alpha * (diffgroup_mat_label * sub)
        
        simmat_loss = neg_samesem + pos

        # TODO: Add these lines back
        group_mask_weight = torch.matmul(group_mask.unsqueeze(2), torch.transpose(group_mask.unsqueeze(2), 2, 1))
        simmat_loss = simmat_loss * group_mask_weight
        # Instead of
        ## simmat_loss = 100 * simmat_loss
        
        simmat_loss = torch.mean(simmat_loss)

        return simmat_loss