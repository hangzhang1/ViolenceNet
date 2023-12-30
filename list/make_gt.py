#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhanghang
# datetime:2023/4/16 11:37
# software: PyCharm
# Description: 使用测试视频集 和 annotation 列表


import numpy as np
import os
import cv2

clip_len = 16

# the dir of testing images
video_root = 'E:/实现/co_attention_violence/list/data/TestClips/videos'  # the path of test videos

feature_list = 'E:/实现/transformer_violence/list/feature_file_name_list/rgb_test.list'

# the ground truth txt
gt_txt = './feature_file_name_list/annotations.txt'  # the path of test annotations

gt_lines = list(open(gt_txt))

import os

print(os.path.exists('E:/实现/co_attention_violence/list/data/TestClips/videos/A.Beautiful.Mind.2001__#00-25-20_00-29'
                     '-20_label_A.mp4'))

gt = []
lists = list(open(feature_list))
tlens = 0
vlens = 0
for idx in range(len(lists)):
    name = lists[idx].strip('\n').split('/')[-1]
    if '__0.npy' not in name:
        continue
    name = name[8:-7]
    vname = name + '.mp4'
    cap = cv2.VideoCapture(os.path.join(video_root, vname))
    lens = int(cap.get(7))
    """
    将异常视频的某些帧标记为异常，1代表异常
    """
    gt_vec = np.zeros(lens).astype(np.float32)
    if '_label_A' not in name:
        for gt_line in gt_lines:
            if name in gt_line:
                gt_content = gt_line.strip('\n').split()
                abnormal_fragment = [[int(gt_content[i]), int(gt_content[j])] for i in range(1, len(gt_content), 2) \
                                     for j in range(2, len(gt_content), 2) if j == i + 1]
                if len(abnormal_fragment) != 0:
                    abnormal_fragment = np.array(abnormal_fragment)
                    for frag in abnormal_fragment:
                        gt_vec[frag[0]:frag[1]] = 1.0
                break
    mod = (lens - 1) % clip_len
    gt_vec = gt_vec[:-1]
    if mod:
        gt_vec = gt_vec[:-mod]
    gt.extend(gt_vec)
    if sum(gt_vec) / len(gt_vec):
        tlens += len(gt_vec)
        vlens += sum(gt_vec)

np.save('gt.npy', gt)
