#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhanghang
# datetime:2023/7/2 18:47
# software: PyCharm
# Description:


import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings("ignore")


def temporal_smooth(arr):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]
    loss = torch.sum((arr2 - arr) ** 2)
    return loss


def temporal_sparsity(arr):
    loss = torch.sum(arr)
    return loss


def Smooth_Sparsity(logits, seq_len, lamda=8e-5):
    smooth_mse = []
    spar_mse = []
    for i in range(logits.shape[0]):
        tmp_logits = logits[i][:seq_len[i]]
        sm_mse = temporal_smooth(tmp_logits)  # 时间平滑项
        sp_mse = torch.sum(tmp_logits)
        smooth_mse.append(sm_mse)
        spar_mse.append(sp_mse)
    smooth_mse = sum(smooth_mse) / len(smooth_mse)
    spar_mse = sum(spar_mse) / len(spar_mse)
    return (smooth_mse + spar_mse) * lamda


def RFTM(vision, seq_len, half_batch_size, is_topk=True):
    n_vision = vision[:half_batch_size, :, :]
    a_vision = vision[half_batch_size:, :, :]
    normal_seq = seq_len[:half_batch_size]
    abnormal_seq = seq_len[half_batch_size:]
    mean_n_vision = torch.mean(torch.abs(n_vision), dim=2)
    mean_a_vision = torch.mean(torch.abs(a_vision), dim=2)

    instance_normal = torch.zeros(0).cuda()
    for i in range(mean_n_vision.shape[0]):
        if is_topk:
            tmp, _ = torch.topk(mean_n_vision[i][:normal_seq[i]], k=int(normal_seq[i] // 16 + 1), largest=True)
            tmp = torch.norm(tmp, p=2, dim=0).view(1)
        else:
            tmp = torch.mean(n_vision[i, :normal_seq[i]]).view(1)
        instance_normal = torch.cat((instance_normal, tmp), 0)

    instance_abnormal = torch.zeros(0).cuda()
    for i in range(mean_a_vision.shape[0]):
        if is_topk:
            tmp, _ = torch.topk(mean_a_vision[i][:abnormal_seq[i]], k=int(abnormal_seq[i] // 16 + 1), largest=True)
            tmp = torch.norm(tmp, p=2, dim=0).view(1)
        else:
            tmp = torch.mean(a_vision[i, :abnormal_seq[i]]).view(1)
        instance_abnormal = torch.cat((instance_abnormal, tmp), 0)
    margin = 1
    loss_abn = torch.abs(margin - instance_abnormal)
    loss_nor = instance_normal
    if loss_abn.shape == loss_nor.shape:
        loss_rtfm = torch.mean((loss_abn + loss_nor) ** 2)
    else:
        if loss_abn.shape > loss_nor.shape:
            m = loss_nor.size(dim=0)
            loss_abn = loss_abn[:m]
            loss_rtfm = torch.mean((loss_abn + loss_nor) ** 2)
        else:
            m = loss_abn.size(dim=0)
            loss_nor = loss_nor[:m]
            loss_rtfm = torch.mean((loss_abn + loss_nor) ** 2)
    return loss_rtfm


def CLAS(logits, label, seq_len, criterion, is_topk=True):
    logits = logits.squeeze()
    instance_logits = torch.zeros(0).cuda()
    for i in range(logits.shape[0]):
        if is_topk:
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
            tmp = torch.mean(tmp).view(1)
        else:
            tmp = torch.mean(logits[i, :seq_len[i]]).view(1)
        instance_logits = torch.cat((instance_logits, tmp))

    instance_logits = torch.sigmoid(instance_logits)
    clsloss = criterion(instance_logits, label)
    return clsloss


def MSE(logits_v, logits_a, logits_f):
    logits_v = logits_v.squeeze()
    logits_a = logits_a.squeeze()
    logits_f = logits_f.squeeze()

    loss = nn.MSELoss()
    loss_v_a = loss(logits_v, logits_a)
    loss_v_f = loss(logits_v, logits_f)
    loss_a_f = loss(logits_a, logits_f)
    total_loss = (loss_v_a + loss_v_f + loss_a_f) / 3
    return total_loss


def compute_kl_loss(p, q):
    p_logits = p.squeeze()
    q_logits = q.squeeze()
    p_loss = F.kl_div(F.log_softmax(p_logits, dim=-1), F.softmax(q_logits, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q_logits, dim=-1), F.softmax(p_logits, dim=-1), reduction='none')
    p_kl_loss = p_loss.sum()
    q_kl_loss = q_loss.sum()

    KL_loss = (p_kl_loss + q_kl_loss) / 2
    return KL_loss


if __name__ == '__main__':
    logits_1 = torch.as_tensor(torch.randn(5, 67))
    logits_2 = torch.as_tensor(torch.randn(5, 67))
    KL_loss = compute_kl_loss(logits_1, logits_2)
    print('output.shape:', KL_loss.shape)