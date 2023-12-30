#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhanghang
# datetime:2023/7/4 15:02
# software: PyCharm
# Description:


import os
import glob

base_path = 'E:/实现/Trimodal_VioNets/feature/demo4/'
root_path = base_path + "audio"
files = sorted(glob.glob(os.path.join(root_path, "*.npy")))

print("音频特征列表长度：", files.__len__())
if files.__len__() == 0:
    raise Exception("没有获取到文件")

with open('./feature_list/audio_feature.list', 'w+') as f:
    for file in files:
        newline = file +'\n'
        f.write(newline)


root_path = base_path + "flow"

files = sorted(glob.glob(os.path.join(root_path, "*.npy")))

print("光流特征列表长度：", files.__len__())
if files.__len__() == 0:
    raise Exception("没有获取到文件")

with open('./feature_list/flow_feature.list', 'w+') as f:
    for file in files:
        newline = file +'\n'
        f.write(newline)

# ===============================================
root_path = base_path + "rgb"

files = sorted(glob.glob(os.path.join(root_path, "*.npy")))

print("rgb特征列表长度：", files.__len__())
if files.__len__() == 0:
    raise Exception("没有获取到文件")

with open('./feature_list/rgb_feature.list', 'w+') as f:
    for file in files:
        newline = file +'\n'
        f.write(newline)
