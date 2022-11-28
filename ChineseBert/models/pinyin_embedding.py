#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : pinyin.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/8/16 14:45
@version: 1.0
@desc  : pinyin embedding
"""
import json
import os

from torch import nn
from torch.nn import functional as F


class PinyinEmbedding(nn.Module):
    def __init__(self, embedding_size: int, pinyin_out_dim: int, config_path):
        """
            Pinyin Embedding Module
        Args:
            embedding_size: the size of each embedding vector
            pinyin_out_dim: kernel number of conv
        """
        super(PinyinEmbedding, self).__init__()
        with open(os.path.join(config_path, 'pinyin_map.json')) as fin:
            pinyin_dict = json.load(fin)
        self.pinyin_out_dim = pinyin_out_dim    # 要将token编码成的向量维度，例如768。
        # Embedding(32, 128)。其中32为6+26：6种音调, 26个英文字母。128为将一个拼音中字母（或音调）编码成128维的向量
        self.embedding = nn.Embedding(len(pinyin_dict['idx2char']), embedding_size)
        # 卷积层，输入通道数为128，输出通道数为768。
        self.conv = nn.Conv1d(in_channels=embedding_size, out_channels=self.pinyin_out_dim, kernel_size=2,
                              stride=1, padding=0)

    def forward(self, pinyin_ids):
        """
        Args:
            pinyin_ids: (batch_size, sentence_length, 8), sentence_length包含101和102, 8是固定长度（拼音+音调+不足补0）。

        Returns:
            pinyin_embed: (bs,sentence_length,pinyin_out_dim)
        """
        # 将每个字母（或音调或[PAD]）编码成128维的向量。embed.shape为[bs,sentence_length,8,embed_size]，例如(1, 6, 8, 128)。
        embed = self.embedding(pinyin_ids)  # [bs,sentence_length,pinyin_locs,embed_size]
        bs, sentence_length, pinyin_locs, embed_size = embed.shape
        # 为了进行后续卷积，将batch_size和sentence_length合并。然后embed_size提前。
        view_embed = embed.view(-1, pinyin_locs, embed_size)  # [(bs*sentence_length),pinyin_locs,embed_size]
        input_embed = view_embed.permute(0, 2, 1)  # [(bs*sentence_length), embed_size, pinyin_locs]
        # conv + max_pooling    # 卷积+max_pooling操作
        pinyin_conv = self.conv(input_embed)  # [(bs*sentence_length),pinyin_out_dim,H]
        pinyin_embed = F.max_pool1d(pinyin_conv, pinyin_conv.shape[-1])  # [(bs*sentence_length),pinyin_out_dim,1]
        return pinyin_embed.view(bs, sentence_length, self.pinyin_out_dim)  # [bs,sentence_length,pinyin_out_dim]
