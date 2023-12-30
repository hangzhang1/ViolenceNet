#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhanghang
# datetime:2023/7/2 18:50
# software: PyCharm
# Description:
import torch.utils.data as data
import numpy as np
import torch

from utils import process_feat


class Dataset(data.DataLoader):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        self.modality = args.modality
        self.is_normal = is_normal
        if test_mode:
            self.rgb_list_file = args.test_rgb_list
            self.flow_list_file = args.test_flow_list
            self.audio_list_file = args.test_audio_list
        else:
            self.rgb_list_file = args.rgb_list
            self.flow_list_file = args.flow_list
            self.audio_list_file = args.audio_list

        self.max_seqlen = args.max_seqlen
        self.transform = transform
        self.test_mode = test_mode
        self.normal_flag = '_label_A'
        self.labels = None
        self.__parse_list()

    def __parse_list(self):
        if self.test_mode:
            if self.modality == 'AUDIO':
                self.list = list(open(self.audio_list_file))
            elif self.modality == 'RGB':
                self.list = list(open(self.rgb_list_file))
            elif self.modality == 'FLOW':
                self.list = list(open(self.flow_list_file))
            elif self.modality == 'MIX1':
                self.list = list(open(self.rgb_list_file))
                self.flow_list = list(open(self.flow_list_file))
            elif self.modality == 'MIX2':
                self.list = list(open(self.rgb_list_file))
                self.audio_list = list(open(self.audio_list_file))
            elif self.modality == 'MIX3':
                self.list = list(open(self.flow_list_file))
                self.audio_list = list(open(self.audio_list_file))
            elif self.modality == 'MIX_ALL':
                self.list = list(open(self.rgb_list_file))
                self.flow_list = list(open(self.flow_list_file))
                self.audio_list = list(open(self.audio_list_file))
            else:
                assert 1 > 2, 'Modality is wrong!'
        else:
            if self.modality == 'AUDIO':
                self.list = list(open(self.audio_list_file))
            elif self.modality == 'RGB':
                self.list = list(open(self.rgb_list_file))
            elif self.modality == 'FLOW':
                self.list = list(open(self.flow_list_file))
            elif self.modality == 'MIX1':
                self.list = list(open(self.rgb_list_file))
                self.flow_list = list(open(self.flow_list_file))
                self.abnormal_list = self.list[:9525]
                self.normal_list = self.list[9525:]
                self.abnormal_flow_list = self.list[:9525]
                self.normal_flow_list = self.list[9525:]
            elif self.modality == 'MIX2':  # 混合两个模态 一个 RGB 一个 audio
                self.list = list(open(self.rgb_list_file))
                self.audio_list = list(open(self.audio_list_file))
                self.abnormal_list = self.list[:9525]
                self.normal_list = self.list[9525:]
                self.abnormal_audio_list = self.audio_list[:1905]
                self.normal_audio_list = self.audio_list[1905:]
            elif self.modality == 'MIX3':
                self.list = list(open(self.flow_list_file))
                self.audio_list = list(open(self.audio_list_file))
            elif self.modality == 'MIX_ALL':
                self.list = list(open(self.rgb_list_file))
                self.flow_list = list(open(self.flow_list_file))
                self.audio_list = list(open(self.audio_list_file))

                self.abnormal_list = self.list[:9525]
                self.normal_list = self.list[9525:]

                self.abnormal_flow_list = self.flow_list[:9525]
                self.normal_flow_list = self.flow_list[9525:]

                self.abnormal_audio_list = self.audio_list[:1905]
                self.normal_audio_list = self.audio_list[1905:]
            else:
                assert 1 > 2, 'Modality is wrong!'

    def __getitem__(self, index):
        if self.test_mode:
            if self.normal_flag in self.list[index]:
                label = 0.0
            else:
                label = 1.0

            if self.modality == 'AUDIO':
                features = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
            elif self.modality == 'RGB':
                features = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
            elif self.modality == 'FLOW':
                features = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
            elif self.modality == 'MIX1':
                features1 = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
                features2 = np.array(np.load(self.flow_list[index].strip('\n')), dtype=np.float32)
                if features1.shape[0] == features2.shape[0]:
                    features = np.concatenate((features1, features2), axis=1)
                else:
                    features = np.concatenate((features1[:-1], features2), axis=1)
            elif self.modality == 'MIX2':
                features1 = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)  # 加载rgb特征向量
                features2 = np.array(np.load(self.audio_list[index // 5].strip('\n')),dtype=np.float32)  # 加载audio特征向量
                if features1.shape[0] == features2.shape[0]:  #
                    features = np.concatenate((features1, features2), axis=1)

                else:
                    features = np.concatenate((features1[:-1], features2), axis=1)
            elif self.modality == 'MIX3':
                features1 = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
                features2 = np.array(np.load(self.audio_list[index // 5].strip('\n')), dtype=np.float32)
                if features1.shape[0] == features2.shape[0]:
                    features = np.concatenate((features1, features2), axis=1)
                else:
                    features = np.concatenate((features1[:-1], features2), axis=1)
            elif self.modality == 'MIX_ALL':
                features1 = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
                features2 = np.array(+ np.load(self.flow_list[index].strip('\n')), dtype=np.float32)
                features3 = np.array(+ np.load(self.audio_list[index // 5].strip('\n')), dtype=np.float32)
                if features1.shape[0] == features2.shape[0]:
                    features = np.concatenate((features1, features3, features2), axis=1)
                else:
                    features = np.concatenate((features1[:-1], features2, features3[:-1]), axis=1)
            else:
                assert 1 > 2, 'Modality is wrong!'
        else:
            label = self.get_label()
            if self.modality == 'AUDIO':
                features = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
            elif self.modality == 'RGB':
                features = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
            elif self.modality == 'FLOW':
                features = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
            elif self.modality == 'MIX1':
                features1 = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
                features2 = np.array(np.load(self.flow_list[index].strip('\n')), dtype=np.float32)
                if features1.shape[0] == features2.shape[0]:
                    features = np.concatenate((features1, features2), axis=1)
                else:
                    features = np.concatenate((features1[:-1], features2), axis=1)
            elif self.modality == 'MIX2':
                if self.is_normal:
                    features_rgb_normal = np.array(np.load(self.normal_list[index].strip('\n')), dtype=np.float32)
                    features_audio_normal = np.array(np.load(self.normal_audio_list[index // 5].strip('\n')),
                                                     dtype=np.float32)
                    features1 = features_rgb_normal
                    features2 = features_audio_normal
                else:
                    features_rgb_abnormal = np.array(np.load(self.abnormal_list[index].strip('\n')), dtype=np.float32)
                    features_audio_abnormal = np.array(np.load(self.abnormal_audio_list[index // 5].strip('\n')),
                                                       dtype=np.float32)
                    features1 = features_rgb_abnormal
                    features2 = features_audio_abnormal

                if features1.shape[0] == features2.shape[0]:  #
                    features = np.concatenate((features1, features2), axis=1)

                else:
                    features = np.concatenate((features1[:-1], features2), axis=1)
            elif self.modality == 'MIX3':
                features1 = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
                features2 = np.array(np.load(self.audio_list[index // 5].strip('\n')), dtype=np.float32)
                if features1.shape[0] == features2.shape[0]:
                    features = np.concatenate((features1, features2), axis=1)
                else:
                    features = np.concatenate((features1[:-1], features2), axis=1)
            elif self.modality == 'MIX_ALL':
                if self.is_normal:
                    features_rgb_normal = np.array(np.load(self.normal_list[index].strip('\n')), dtype=np.float32)
                    features_flow_normal = np.array(np.load(self.normal_flow_list[index].strip('\n')), dtype=np.float32)
                    features_audio_normal = np.array(np.load(self.normal_audio_list[index // 5].strip('\n')),
                                                     dtype=np.float32)
                    features1 = features_rgb_normal
                    features2 = features_audio_normal
                    features3 = features_flow_normal
                else:
                    features_rgb_abnormal = np.array(np.load(self.abnormal_list[index].strip('\n')), dtype=np.float32)
                    features_flow_abnormal = np.array(np.load(self.abnormal_flow_list[index].strip('\n')), dtype=np.float32)
                    features_audio_abnormal = np.array(np.load(self.abnormal_audio_list[index // 5].strip('\n')),
                                                       dtype=np.float32)
                    features1 = features_rgb_abnormal
                    features2 = features_audio_abnormal
                    features3 = features_flow_abnormal
                if features1.shape[0] == features2.shape[0]:  #
                    features = np.concatenate((features1, features2, features3), axis=1)
                else:
                    features = np.concatenate((features1[:-1], features2, features3), axis=1)

            else:
                assert 1 > 2, 'Modality is wrong!'

        if self.transform is not None:  # transform is None
            features = self.transform(features)

        if self.test_mode:
            return features, label
        else:
            features = process_feat(features, self.max_seqlen, is_random=False)
            return features, label

    def get_label(self):

        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)
        return label

    def __len__(self):
        # 这个需要修改
        if self.test_mode:
            return len(self.list)
        else:
            if self.is_normal:
                return len(self.normal_list)
            else:
                return len(self.abnormal_list)