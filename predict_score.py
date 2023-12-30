#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhanghang
# datetime:2023/7/4 12:22
# software: PyCharm
# Description:
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch
import numpy as np
import copy
from modules.models import MULTModel
import option
import time
from base_function import *
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


class Dataset(data.DataLoader):
    def __init__(self, args, is_normal=True, predict_mode=False):
        self.modality = args.modality
        self.is_normal = is_normal
        if predict_mode:
            self.rgb_list_file = args.predict_rgb_list
            self.flow_list_file = args.predict_flow_list
            self.audio_list_file = args.predict_audio_list

        self.predict_mode = predict_mode
        self.__parse_list()

    def __parse_list(self):
        """
            该函数的作用是 通过视频特征、音频特征列表文件
            获取视频特征 和音频特征 真正的路径
            :return:
        """
        if self.predict_mode:
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

    def __getitem__(self, index):
        """
        创建了该方法后，可以使用对象[]来获得返回值
        :param index:
        :return:
        """
        if self.predict_mode:
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
                features1 = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
                features2 = np.array(np.load(self.audio_list[index // 5].strip('\n')),dtype=np.float32)
                if features1.shape[0] == features2.shape[0]:
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

        if self.predict_mode:
            return features

    def __len__(self):
        if self.predict_mode:
            return len(self.list)


if __name__ == '__main__':
    args = option.parser.parse_args()
    set_only(args)
    device = torch.device("cuda")

    predict_loader = DataLoader(Dataset(args, predict_mode=True),
                                batch_size=5, shuffle=False,
                                num_workers=args.workers, pin_memory=True)

    print(predict_loader)

    model = MULTModel(args)
    model = model.to(device)
    model_dict = model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load('./ckpt/xd_a2v.pkl8121').items()}, strict=False)
    model.eval()
    gt = np.load(args.gt)
    st = time.time()
    criterion = torch.nn.BCELoss()
    file_path = "./feature/demo4/audio"
    name = os.listdir(file_path)
    file_name = name[0][:-12]

    with torch.no_grad():
        pred = torch.zeros(0).cuda()
        for i, inputs in enumerate(predict_loader):

            print("inputs.shape", inputs.shape)
            inputs = inputs.cuda()

            vision = inputs[:, :, :1024]
            audio = inputs[:, :, 1024:1152]
            flow = inputs[:, :, 1152:]
            logits, last_hs, logits_v, logits_a, logits_f = model(vision, audio, flow)
            logits = torch.mean(logits, 0)
            pred = torch.cat((pred, logits))
            pred = torch.sigmoid(pred)
            pred = pred.cpu()
            pred = np.repeat(pred, 16)
            print(pred)
            x = np.arange(len(pred))
            ground_true = torch.zeros_like(pred)
            index = [(154, 647)]
            length = len(index)
            for i in range(length):
                tmp = index[i]
                print("tmp:", tmp)
                ground_true[tmp[0]:tmp[1]] = 1

            plt.title(file_name)
            plt.xlabel("clip")
            plt.ylabel("score")
            plt.plot(x, pred, color='b', label='Anomaly scores', linewidth=1)
            plt.fill_between(x, ground_true, where=ground_true > 0.5, facecolor="red", alpha=0.3)

            plt.grid()
            pic_file_name = 'feature/feature_figs/' + file_name + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.png'
            plt.savefig(pic_file_name, bbox_inches='tight')
            plt.show()

    print('Time:{}'.format(time.time() - st))