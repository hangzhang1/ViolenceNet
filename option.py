#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhanghang
# datetime:2023/7/2 18:51
# software: PyCharm
# Description:


import argparse

parser = argparse.ArgumentParser(description='xd_violence')

parser.add_argument('--rgb-list', default='./list/feature_file_name_list/rgb.list', help='list of rgb features ')
parser.add_argument('--flow-list', default='./list/feature_file_name_list/flow.list', help='list of flow features')
parser.add_argument('--audio-list', default='./list/feature_file_name_list/audio.list', help='list of audio features')
parser.add_argument('--test-rgb-list', default='./list/feature_file_name_list/rgb_test.list',
                    help='list of test rgb features ')
parser.add_argument('--test-flow-list', default='./list/feature_file_name_list/flow_test.list',
                    help='list of test flow features')
parser.add_argument('--test-audio-list', default='./list/feature_file_name_list/audio_test.list',
                    help='list of test audio features')

parser.add_argument('--predict_rgb-list', default='./feature/feature_list/rgb_feature.list', help='list of rgb  '
                                                                                                  'predict features ')
parser.add_argument('--predict_flow-list', default='./feature/feature_list/flow_feature.list', help='list of flow '
                                                                                                    'predict features')
parser.add_argument('--predict_audio-list', default='./feature/feature_list/audio_feature.list', help='list of audio '
                                                                                                      'predict features')

parser.add_argument('--dataset-name', default='XD-Violence', help='dataset to train on XD-Violence')
parser.add_argument('--gt', default='list/gt.npy', help='file of ground truth ')

parser.add_argument('--modality', default='MIX_ALL', help='the type of the input, AUDIO,RGB,FLOW, MIX1, MIX2, '
                                                          'or MIX3, MIX_ALL')

parser.add_argument('--seed', type=int, default=9, help='Random Initiation (default: 9)')

# 超参数
parser.add_argument('--lr', type=float, default=0.00001, help='learning rate (default: 0.00001)')
parser.add_argument('--batch-size', type=int, default=32, help='number of instances in a batch of data')
parser.add_argument('--clip', type=float, default=1.2, help='gradient clip value (default: 0.8)')
parser.add_argument('--workers', default=0, help='number of workers in dataloader')
parser.add_argument('--feature-size', type=int, default=1024 + 256 + 1024, help='size of feature (default: 2048)')
parser.add_argument('--num-classes', type=int, default=1, help='number of class')
parser.add_argument('--max-seqlen', type=int, default=250, help='maximum sequence length during training')
parser.add_argument('--max-epoch', type=int, default=30, help='maximum iteration to train (default: 50)')
parser.add_argument('--weight-decay', type=float, default=0.01, help='(default: 0.001)')

# Dropouts
parser.add_argument('--attn_dropout', type=float, default=0.2,
                    help='attention dropout(default: 0.1)')
parser.add_argument('--attn_dropout_v', type=float, default=0.25,
                    help='attention dropout(default: 0.1)')
parser.add_argument('--attn_dropout_a', type=float, default=0.25,
                    help='attention dropout (for audio)(default: 0)')
parser.add_argument('--attn_dropout_f', type=float, default=0.25,
                    help='attention dropout (for visual)(default: 0)')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout(default: 0.1)')
parser.add_argument('--embed_dropout', type=float, default=0.25,
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout(default: 0.1)')
parser.add_argument('--out_dropout', type=float, default=0.1,
                    help='output layer dropout(default: 0.1)')
# 0.1-----0.3  0-----0.1

# Architecture
parser.add_argument('--layers', type=int, default=5,
                    help='number of layers in the network (default: 4)')
parser.add_argument('--num_heads', type=int, default=8,
                    help='number of heads for the transformer network (default: 8)')
parser.add_argument('--attn_mask', action='store_true',
                    help='use attention mask for Transformer (default: False)')


parser.add_argument('--model-name', default='xd_a2v', help='name to save model')
parser.add_argument('--pretrained-ckpt', default=None, help='ckpt for pretrained model')


parser.add_argument('--vonly', action='store_true',
                    help='use the crossmodal fusion into v (default: False)')
parser.add_argument('--aonly', action='store_true',
                    help='use the crossmodal fusion into a (default: False)')
parser.add_argument('--fonly', action='store_true',
                    help='use the crossmodal fusion into f (default: False)')


parser.add_argument('--orig_d_v', type=int, default=1024, help='视频特征的维度(default:1024)')
parser.add_argument('--orig_d_f', type=int, default=1024, help='光流特征的维度(default:1024)')
parser.add_argument('--orig_d_a', type=int, default=128, help='音频特征的维度(default:128)')

parser.add_argument('--output_dim', type=int, default=128, help='输出维度，根据数据集的不同来设置')
parser.add_argument('--ffn_dropout', type=float, default=0.5, help='(default: 0.1)')
parser.add_argument('--alpha', type=float, default=0.01, help='(default: 0.01)')
parser.add_argument('--beta', type=float, default=0.05, help='(default: 0.1)')  # 0.05
parser.add_argument('--embed_dim', type=int, default=128, help='(default: 128)')
