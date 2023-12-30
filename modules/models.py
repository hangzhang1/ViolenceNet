#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhanghang
# datetime:2023/7/2 18:48
# software: PyCharm
# Description:
import option
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as torch_init
from torchviz import make_dot
from modules.transformer import TransformerEncoder
import hiddenlayer as h


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.kaiming_uniform_(m.weight)


class Ffn(nn.Module):
    def __init__(self, modal_a, d_ff, dropout_rate):
        super(Ffn, self).__init__()
        self.dropout = dropout_rate
        self.ffn = nn.Sequential(
            nn.Conv1d(modal_a, d_ff, kernel_size=1),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Conv1d(d_ff, 128, kernel_size=1),
            nn.Dropout(self.dropout),
        )
        self.norm = nn.LayerNorm(modal_a)

    def forward(self, x):
        new_x = x
        new_x = self.norm(new_x)
        new_x = new_x.permute(0, 2, 1)
        new_x = self.ffn(new_x)
        return new_x


class Ffn_add(nn.Module):
    def __init__(self, modal_a, dropout_rate):
        super(Ffn_add, self).__init__()
        self.dropout = dropout_rate
        self.ffn = nn.Sequential(
            nn.Conv1d(modal_a, 128, kernel_size=1),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.norm = nn.LayerNorm(modal_a)

    def forward(self, x):
        new_x = x
        new_x = self.norm(new_x)
        new_x = new_x.permute(0, 2, 1)
        new_x = self.ffn(new_x)
        return new_x


class MULTModel(nn.Module):
    def __init__(self, args):
        """
        Construct a Mult model.
        :param args:
        """
        super(MULTModel, self).__init__()
        self.orig_d_v, self.orig_d_f, self.orig_d_a = args.orig_d_v, args.orig_d_f, args.orig_d_a
        self.d_v, self.d_f, self.d_a = args.embed_dim, args.embed_dim, args.embed_dim  # 特征维度  128
        self.vonly = args.vonly
        self.aonly = args.aonly
        self.fonly = args.fonly

        self.num_heads = args.num_heads
        self.layers = args.layers
        self.attn_dropout = args.attn_dropout

        self.attn_dropout_v = args.attn_dropout_v
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_f = args.attn_dropout_f
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.out_dropout = args.out_dropout
        self.embed_dropout = args.embed_dropout
        self.attn_mask = args.attn_mask
        self.ffn_dropout = args.ffn_dropout
        combined_dim = self.d_v + self.d_a + self.d_f

        self.partial_mode = self.vonly + self.aonly + self.fonly

        if self.partial_mode == 1:
            combined_dim = 2 * self.d_v
        else:
            combined_dim = 2 * (self.d_v + self.d_a + self.d_f)
        output_dim = args.output_dim
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=5, padding=2, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=3, padding=1, bias=False)
        self.proj_f = nn.Conv1d(self.orig_d_f, self.d_f, kernel_size=5, padding=2, bias=False)
        if self.fonly:
            self.trans_f_with_a = self.get_network(self_type='fa', layers=self.layers)
            self.trans_f_with_v = self.get_network(self_type='fv', layers=self.layers)
        if self.aonly:
            self.trans_a_with_f = self.get_network(self_type='af', layers=self.layers)
            self.trans_a_with_v = self.get_network(self_type='av', layers=self.layers)
        if self.vonly:
            self.trans_v_with_f = self.get_network(self_type='vf', layers=self.layers)
            self.trans_v_with_a = self.get_network(self_type='va', layers=self.layers)

        self.trans_v_mem = self.get_network(self_type='v_mem', layers=1)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=1)
        self.trans_f_mem = self.get_network(self_type='f_mem', layers=1)
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

        self.proj_add = nn.Linear(self.d_v * 2, self.d_v * 2)
        self.ffn_add = Ffn_add(modal_a=self.d_v * 2, dropout_rate=self.ffn_dropout)

        self.ffn = Ffn(modal_a=self.d_v * 2 * 3, d_ff=256, dropout_rate=self.ffn_dropout)
        self.classifier = nn.Conv1d(128, 1, 7, padding=0)

    def get_network(self, self_type='v', layers=-1):
        if self_type in ['v', 'fv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type in ['a', 'fa', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['f', 'af', 'vf']:
            embed_dim, attn_dropout = self.d_f, self.attn_dropout_f
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.d_v, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.d_a, self.attn_dropout
        elif self_type == 'f_mem':
            embed_dim, attn_dropout = 2*self.d_f, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=layers,
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout)

    def forward(self, x_v, x_a, x_f):
        x_v = x_v.transpose(1, 2)
        x_a = x_a.transpose(1, 2)
        x_f = x_f.transpose(1, 2)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_f = x_f if self.orig_d_f == self.d_f else self.proj_f(x_f)

        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_f = proj_x_f.permute(2, 0, 1)

        if self.vonly:
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            h_f_with_vs = self.trans_f_with_v(proj_x_f, proj_x_v, proj_x_v)
            h_vs = torch.cat([h_a_with_vs, h_f_with_vs], dim=2)
            h_vs = self.trans_v_mem(h_vs)
            last_h_v = last_hs = h_vs

        if self.aonly:
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            h_f_with_as = self.trans_f_with_a(proj_x_f, proj_x_a, proj_x_a)
            h_as = torch.cat([h_v_with_as, h_f_with_as], dim=2)
            h_as = self.trans_a_mem(h_as)
            last_h_a = last_hs = h_as

        if self.fonly:
            h_v_with_fs = self.trans_v_with_f(proj_x_v, proj_x_f, proj_x_f)
            h_a_with_fs = self.trans_a_with_f(proj_x_a, proj_x_f, proj_x_f)
            h_fs = torch.cat([h_v_with_fs, h_a_with_fs], dim=2)
            h_fs = self.trans_f_mem(h_fs)
            last_h_f = last_hs = h_fs

        if self.partial_mode == 3:
            last_hs = torch.cat([last_h_v, last_h_a, last_h_f], dim=2)  # 256 * 3

        last_h_v_proj = self.proj_add(
            F.dropout(F.relu(self.proj_add(last_h_v)), p=self.out_dropout, training=self.training))
        last_h_v_proj += last_h_v
        last_h_v_proj = last_h_v_proj.permute(1, 0, 2)

        last_h_a_proj = self.proj_add(
            F.dropout(F.relu(self.proj_add(last_h_a)), p=self.out_dropout, training=self.training))
        last_h_a_proj += last_h_a
        last_h_a_proj = last_h_a_proj.permute(1, 0, 2)

        last_h_f_proj = self.proj_add(
            F.dropout(F.relu(self.proj_add(last_h_f)), p=self.out_dropout, training=self.training))
        last_h_f_proj += last_h_f
        last_h_f_proj = last_h_f_proj.permute(1, 0, 2)

        final_output_v = self.ffn_add(last_h_v_proj)
        final_output_v = F.pad(final_output_v, (6, 0))
        logits_v = self.classifier(final_output_v)
        logits_v = logits_v.squeeze(dim=1)

        final_output_a = self.ffn_add(last_h_a_proj)
        final_output_a = F.pad(final_output_a, (6, 0))
        logits_a = self.classifier(final_output_a)
        logits_a = logits_a.squeeze(dim=1)

        final_output_f = self.ffn_add(last_h_f_proj)
        final_output_f = F.pad(final_output_f, (6, 0))
        logits_f = self.classifier(final_output_f)
        logits_f = logits_f.squeeze(dim=1)

        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        last_hs_proj = last_hs_proj.permute(1, 0, 2)

        final_output = self.ffn(last_hs_proj)
        final_output = F.pad(final_output, (6, 0))
        logits = self.classifier(final_output)
        logits = logits.squeeze(dim=1)
        return logits, last_hs, logits_v, logits_a, logits_f


if __name__ == '__main__':
    print("ok")
    args = option.parser.parse_args()  # 获取各类参数
    valid_partial_mode = args.vonly + args.aonly + args.fonly
    if valid_partial_mode == 0:
        args.vonly = args.aonly = args.fonly = True
    elif valid_partial_mode != 1:
        raise ValueError("You can only choose one of {l/v/a}only.")

    vision = torch.as_tensor(torch.rand(5, 67, 1024))
    audio = torch.as_tensor(torch.rand(5, 67, 128))
    flow = torch.as_tensor(torch.rand(5, 67, 1024))
    model = MULTModel(args)
    output, last_hs, logits_v, logits_a, logits_f = model(vision, audio, flow)
    linear_vision = (last_hs.permute(1, 0, 2))[:, :, :256]

    print('output.shape:', output.shape)
    print("OK")



