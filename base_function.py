#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhanghang
# datetime:2023/7/2 18:50
# software: PyCharm
# Description:


import torch
import numpy as np
import random
import sys
import time
import os
from matplotlib import pyplot as plt


class Logger(object):
    def __init__(self, file_name='Default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def print_parser(args):
    print("learning rate:", args.lr)
    print('--batch-size', args.batch_size)
    print('--clip', args.clip)
    print('--max-seqlen', args.max_seqlen)
    print('--max-epoch', args.max_epoch)
    print('--weight-decay', args.weight_decay)
    print('--layers', args.layers)
    print('--num_heads', args.num_heads)
    print('--attn_mask', args.attn_mask)
    print('--attn_dropout', args.attn_dropout)
    print('--attn_dropout_v', args.attn_dropout_v)
    print('--attn_dropout_a', args.attn_dropout_a)
    print('--attn_dropout_f', args.attn_dropout_f)
    print('--relu_dropout', args.relu_dropout)
    print('--embed_dropout', args.embed_dropout)
    print('--res_dropout', args.res_dropout)
    print('--out_dropout', args.out_dropout)
    print('--alpha', args.alpha)
    print('--beta', args.beta)


def set_only(args):
    valid_partial_mode = args.vonly + args.aonly + args.fonly
    if valid_partial_mode == 0:
        args.vonly = args.aonly = args.fonly = True
    elif valid_partial_mode != 1:
        raise ValueError("You can only choose one of {l/v/a}only.")


def draw_picture(args, class_loss, AP, t_loss):
    x1 = np.arange(args.max_epoch)
    y1 = class_loss
    x2 = np.arange(args.max_epoch)
    y2 = AP
    x3 = np.arange(args.max_epoch)
    y3 = t_loss

    plt.plot(x1, y1, '.-', label="train loss")
    plt.plot(x2, y2, '.-', label="AP")
    plt.plot(x3, y3, '.-', label="test loss")

    plt.legend()  # 显示图例
    plt.xlabel('epoch')
    plt.title('Model loss & AP')
    pic_file_name = 'figs/' + 'savefig_example_dpi_' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.png'
    plt.savefig(pic_file_name, bbox_inches='tight')
    plt.show()


def set_logger():
    log_path = './Logs/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    log_file_name = log_path + 'log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
    sys.stdout = Logger(log_file_name)

