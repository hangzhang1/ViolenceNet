#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhanghang
# datetime:2023/7/2 18:51
# software: PyCharm
# Description:


from torch.utils.data import DataLoader
import torch
import numpy as np
from modules.models import MULTModel
from dataset import Dataset
from test import test
import option
import time
from base_function import *

if __name__ == '__main__':
    args = option.parser.parse_args()
    set_only(args)
    device = torch.device("cuda")

    test_loader = DataLoader(Dataset(args, test_mode=True),
                             batch_size=5, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    model = MULTModel(args)
    model = model.to(device)
    model_dict = model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load('./ckpt/v7/xd_a2v.pkl_8172').items()}, strict=False)
    model.eval()
    gt = np.load(args.gt)
    st = time.time()
    criterion = torch.nn.BCELoss()
    pr_auc, t_loss = test(test_loader, model, gt, criterion)

    print('Time:{}'.format(time.time() - st))
    print('test AP pr_auc:{0:.4}\n'.format(pr_auc))
