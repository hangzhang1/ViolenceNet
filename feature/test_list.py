#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhanghang
# datetime:2023/7/7 14:14
# software: PyCharm
# Description:

import os
import glob


import numpy as np
import os
import cv2

clip_len = 16

# the dir of testing images
video_root = 'E:/实现/co_attention_violence/list/data/TestClips/videos'  # the path of test videos

feature_list = 'E:/实现/Trimodal_VioNets/list/feature_file_name_list/rgb_test.list'

# the ground truth txt
gt_txt = './annotations.txt'  # the path of test annotations

gt_lines = list(open(gt_txt))

gt = []
lists = list(open(feature_list))
tlens = 0
vlens = 0
for idx in range(len(lists)):
    name = lists[idx].strip('\n').split('/')[-1]
    if '__0.npy' not in name:
        continue
    # name = name[:-7]   # 这里需要修改 因为 还包含了文件名
    name = name[8:-7]
    vname = name + '.mp4'
    cap = cv2.VideoCapture(os.path.join(video_root, vname))
    lens = int(cap.get(7))  # get frame nums   # 获取视频文件中的帧数
    """
    将异常视频的某些帧标记为异常，1代表异常
    """
    gt_vec = np.zeros(lens).astype(np.float32)


