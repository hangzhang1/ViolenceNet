#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhanghang
# datetime:2023/7/2 18:52
# software: PyCharm
# Description:


from sklearn.metrics import auc, precision_recall_curve
import numpy as np
import torch
from modules.loss import CLAS
from tqdm import tqdm
from modules.loss import RFTM
from modules.loss import MSE
from colorama import Fore
import matplotlib.pyplot as plt


def test(dataloader, model, gt, criterion):
    t_loss = []

    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).cuda()
        loop = tqdm(enumerate(dataloader), bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.GREEN, Fore.RESET),
                    total=len(dataloader), ncols=100, desc='Test')
        for i, (inputs, label) in loop:
            inputs = inputs.cuda()
            inputs = inputs.float().cuda(non_blocking=True)
            label = label.float().cuda(non_blocking=True)
            vision = inputs[:, :, :1024]
            audio = inputs[:, :, 1024:1152]
            flow = inputs[:, :, 1152:]

            logits, last_hs, logits_v, logits_a, logits_f = model(vision, audio, flow)
            seq_len = torch.sum(torch.max(torch.abs(inputs), dim=2)[0] > 0, 1)
            MSE_loss = MSE(logits_v, logits_a, logits_f)
            cls = CLAS(logits, label, seq_len, criterion)
            loss = MSE_loss + cls
            logits = torch.mean(logits, 0)
            logits = torch.sigmoid(logits)

            pred = torch.cat((pred, logits))
            t_loss.append(loss)
        loop.close()
        pred = list(pred.cpu().detach().numpy())

        precision, recall, th = precision_recall_curve(list(gt), np.repeat(pred, 16))
        np.save('./precision_recall/precision.npy', precision)
        np.save('./precision_recall/recall.npy', recall)

        pr_auc = auc(recall, precision)
        print("pr_auc:", pr_auc)

        print("test loss:", sum(t_loss) / len(t_loss))
        return pr_auc, sum(t_loss) / len(t_loss)
