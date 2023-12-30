#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhanghang
# datetime:2023/4/13 20:13
# software: PyCharm
# Description:获取特征文件的列表 保存到list文件中


import os
import glob

root_path = 'E:/实现/co_attention_violence/list/data/dataset/xd-feature/audio/train'    # the path of features
files = sorted(glob.glob(os.path.join(root_path, "*.npy")))

print("音频训练集长度：", files.__len__())
if files.__len__() == 0:
    raise Exception("没有获取到文件")

violents = []
normal = []

with open('E:/实现/transformer_violence/list/feature_file_name_list/audio.list', 'w+') as f:
    for file in files:
        if '_label_A' in file:
            normal.append(file)
        else:
            newline = file +'\n'
            f.write(newline)
    for file in normal:
        newline = file+'\n'
        f.write(newline)


# =============================================================
root_path = 'E:/实现/co_attention_violence/list/data/dataset/xd-feature/audio/test'    ## the path of features
files = sorted(glob.glob(os.path.join(root_path, "*.npy")))

print("音频测试集长度： ", files.__len__())
if files.__len__() == 0:
    raise Exception("没有获取到文件")


violents = []
normal = []
with open('E:/实现/transformer_violence/list/feature_file_name_list/audio_test.list', 'w+') as f:  ## the name of feature list
    for file in files:
        if '_label_A' in file:
            normal.append(file)
        else:
            newline = file+'\n'
            f.write(newline)
    for file in normal:
        newline = file+'\n'
        f.write(newline)


# =============================================================
root_path = 'E:/实现/co_attention_violence/list/data/dataset/xd-feature/visual/Flow'    ## the path of features
files = sorted(glob.glob(os.path.join(root_path, "*.npy")))

print("光流训练集长度： ",files.__len__())
if files.__len__() == 0:
    raise Exception("没有获取到文件")


violents = []
normal = []
with open('E:/实现/transformer_violence/list/feature_file_name_list/flow.list', 'w+') as f:  ## the name of feature list
    for file in files:
        if '_label_A' in file:
            normal.append(file)
        else:
            newline = file+'\n'
            f.write(newline)
    for file in normal:
        newline = file+'\n'
        f.write(newline)


# ====================================================
root_path = 'E:/实现/co_attention_violence/list/data/dataset/xd-feature/visual/FlowTest'    ## the path of features
files = sorted(glob.glob(os.path.join(root_path, "*.npy")))

print("光流测试集长度：", files.__len__())
if files.__len__() == 0:
    raise Exception("没有获取到文件")


violents = []
normal = []
with open('E:/实现/transformer_violence/list/feature_file_name_list/flow_test.list', 'w+') as f:  ## the name of feature list
    for file in files:
        if '_label_A' in file:
            normal.append(file)
        else:
            newline = file+'\n'
            f.write(newline)
    for file in normal:
        newline = file+'\n'
        f.write(newline)

print('finish.')

# ====================================================
root_path = 'E:/实现/co_attention_violence/list/data/dataset/xd-feature/visual/RGB'
files = sorted(glob.glob(os.path.join(root_path, "*.npy")))

print("RGB 训练集长度：", files.__len__())
if files.__len__() == 0:
    raise Exception("没有获取到文件")


violents = []
normal = []
with open('E:/实现/transformer_violence/list/feature_file_name_list/rgb.list', 'w+') as f:
    for file in files:
        if '_label_A' in file:
            normal.append(file)
        else:
            newline = file+'\n'
            f.write(newline)
    for file in normal:
        newline = file+'\n'
        f.write(newline)


# ====================================================
root_path = 'E:/实现/co_attention_violence/list/data/dataset/xd-feature/visual/RGBTest'
files = sorted(glob.glob(os.path.join(root_path, "*.npy")))

print("RGB 测试集长度：", files.__len__())
if files.__len__() == 0:
    raise Exception("没有获取到文件")


violents = []
normal = []
with open('E:/实现/transformer_violence/list/feature_file_name_list/rgb_test.list', 'w+') as f:  ## the name of feature list
    for file in files:
        if '_label_A' in file:
            normal.append(file)
        else:
            newline = file+'\n'
            f.write(newline)
    for file in normal:
        newline = file+'\n'
        f.write(newline)

print('finish.')

print("RGB FLOW特征训练集和测试集的长度是音频特征训练集测试集长度的五倍\n")

print("这是 因为 图片是经过了采样的   "
      "当我们使用“5 裁剪”增强对每个视频帧进行过采样时，"
      "“5 裁剪”意味着将图像裁剪到中心和四个角。_0.npy 是中心，_1~ _4.npy 是角。")