#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhanghang
# datetime:2023/7/2 18:51
# software: PyCharm
# Description:

import option
import copy
import datetime
import torch.optim as optim
from dataset import Dataset
from torch.utils.data import DataLoader
from modules.models import MULTModel
from test import test
from train import train
from base_function import *

if __name__ == '__main__':
    class_loss = []
    AP = []
    t_loss = []
    set_logger()
    args = option.parser.parse_args()
    print_parser(args)
    setup_seed(args.seed)
    set_only(args)

    train_n_data = Dataset(args, test_mode=False, is_normal=True)
    train_n_loader = DataLoader(train_n_data,
                                batch_size=args.batch_size // 2, shuffle=True,
                                num_workers=args.workers, pin_memory=True)

    train_a_data = Dataset(args, test_mode=False, is_normal=False)
    train_a_loader = DataLoader(train_a_data,
                                batch_size=args.batch_size // 2, shuffle=True,
                                num_workers=args.workers, pin_memory=True)

    print(len(train_n_loader))
    print(len(train_a_loader))

    test_data = Dataset(args, test_mode=True)
    test_loader = DataLoader(test_data,
                             batch_size=5, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    model = MULTModel(args)
    if torch.cuda.is_available():
        model.cuda()

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    criterion = torch.nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch, eta_min=0)

    gt = np.load(args.gt)
    random_ap, random_test_loss = test(test_loader, model, gt, criterion)  # 要完善
    print('Random initialized AP:{:.5f}\n'.format(random_ap))
    print('Random initialized test_loss:{:.5f}\n'.format(random_test_loss))

    best_ap = 0.0
    st = time.time()
    for epoch in range(args.max_epoch):
        start = time.time()
        train_loss = train(train_n_loader, train_a_loader, model, optimizer, criterion, args, epoch)
        ap, test_loss = test(test_loader, model, gt, criterion)
        end = time.time()
        duration = end - start
        scheduler.step()

        cls_loss_1 = train_loss.cpu().item()
        ap_1 = torch.tensor(ap).cpu().item()
        test_loss_1 = test_loss.cpu().item()
        class_loss.append(cls_loss_1)
        AP.append(ap_1)
        t_loss.append(test_loss_1)
        print("now learning rate:", optimizer.state_dict()['param_groups'][0]['lr'])
        print("-" * 50)
        print('[Epoch {}/{}]:|Time {:5.4f} sec | cls loss: {} | '
              'epoch AP: {:.4f} | test loss: {:.4f}'.format(epoch + 1,
                                                            args.max_epoch,
                                                            duration,
                                                            train_loss,
                                                            ap,
                                                            test_loss))
        print("-" * 50)
        if ap > best_ap:
            best_ap = ap
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(),
               './ckpt/' + args.model_name + '.pkl' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

    time_elapsed = time.time() - st
    print('Training completes in {:.0f}m {:.0f}s | '
          'best test AP: {:.4f}\n'.format(time_elapsed // 60, time_elapsed % 60, best_ap))

    draw_picture(args, class_loss, AP, t_loss)