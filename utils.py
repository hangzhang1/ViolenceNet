#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhanghang
# datetime:2023/7/2 18:53
# software: PyCharm
# Description:

import numpy as np


def random_extract(feat, t_max):
    """
    随机抽取特征长度的最大值
    :param feat:传入的features
    :param t_max:features最大的长度length  feature(length, dv)
    :return:返回抽取的最大长度的特征features
    """
    r = np.random.randint(len(feat) - t_max)
    return feat[r:r + t_max]


def uniform_extract(feat, t_max):
    """
    和上面的函数相似 ，只不过不是随机抽取 而是服从某个分布
    :param feat: 传入的features
    :param t_max: features最大的长度length  feature(length, dv)
    :return: 返回抽取的最大长度的特征features
    """
    r = np.linspace(0, len(feat) - 1, t_max, dtype=np.uint16)
    return feat[r, :]


def pad(feat, min_len):
    """
    特征填充 防止特征length小于min_len
    :param feat: 传入的features
    :param min_len: 特征length的最小长度
    :return: 返回填充好的features
    """
    if np.shape(feat)[0] <= min_len:
        return np.pad(feat, ((0, min_len - np.shape(feat)[0]), (0, 0)), mode='constant', constant_values=0)
    else:
        return feat


def process_feat(feat, length, is_random=True):
    """
    对特征进行处理，是否随机抽取 还有判断长度是否合理
    :param feat: 传入的features
    :param length: 传入的特征最长length max_seqlen
    :param is_random: 是否随机抽取
    :return: 返回处理好的特征  如果特征长度大于200就截断 随机抽取200个特征
            如果特征长度小于200 就对其进行扩充 使其变为长度为200的features
            返回的每一个feature都是特征数量为200的 矩阵
    """
    if len(feat) > length:  # 如果特征向量的长度大于max_length
        if is_random:
            return random_extract(feat, length)  # 随机选取长度为max_length的特征
        else:
            return uniform_extract(feat, length)
    else:
        return pad(feat, length)
