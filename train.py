#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhanghang
# datetime:2023/7/2 18:52
# software: PyCharm
# Description:


import torch
from modules.loss import CLAS
from modules.loss import RFTM
from modules.loss import MSE
from modules.loss import compute_kl_loss
from tqdm import tqdm
from colorama import Fore


def train(n_loader, a_loader, model, optimizer, criterion, args, epoch):
    t_loss = []
    with torch.set_grad_enabled(True):
        epoch_loss = 0
        model.train()
        n_loader_iterator = iter(n_loader)
        loop = tqdm(enumerate(a_loader), bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.GREEN, Fore.RESET), total=len(a_loader), ncols=100)
        loop.set_description(f'Train Epoch [{epoch+1}/{args.max_epoch}]')

        for i, (a_inputs, a_label) in loop:
            try:
                (n_inputs, n_label) = next(n_loader_iterator)
            except StopIteration:
                n_loader_iterator = iter(n_loader)
                (n_inputs, n_label) = next(n_loader_iterator)
            inputs = torch.cat((n_inputs, a_inputs), 0)
            labels = torch.cat((n_label, a_label), 0)
            seq_len = torch.sum(torch.max(torch.abs(inputs), dim=2)[0] > 0, 1)

            inputs = inputs[:, :torch.max(seq_len), :]
            inputs = inputs.float().cuda(non_blocking=True)
            labels = labels.float().cuda(non_blocking=True)

            model.zero_grad()

            vision = inputs[:, :, :1024]
            audio = inputs[:, :, 1024:1152]
            flow = inputs[:, :, 1152:]

            logits, last_hs, logits_v, logits_a, logits_f = model(vision, audio, flow)

            index = args.embed_dim * 2
            linear_vision = (last_hs.permute(1, 0, 2))[:, :, :index]
            half_batch_size = int(args.batch_size // 2)

            MSE_loss = MSE(logits_v, logits_a, logits_f)
            rtfm_loss = RFTM(linear_vision, seq_len, half_batch_size, is_topk=True)
            cls_loss = CLAS(logits, labels, seq_len, criterion)

            loss = cls_loss + args.alpha * rtfm_loss + args.beta * MSE_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            t_loss.append(loss)

        loop.close()

    return sum(t_loss) / len(t_loss)
