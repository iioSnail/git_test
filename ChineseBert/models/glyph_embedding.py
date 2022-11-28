# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: glyph_embedding
@time: 2020/8/4 15:04

"""

from typing import List

import numpy as np
import torch
from torch import nn


class GlyphEmbedding(nn.Module):
    """Glyph2Image Embedding"""

    def __init__(self, font_npy_files: List[str]):
        super(GlyphEmbedding, self).__init__()
        # font_arrays[i]存储了这个字的字形。 font_arrays[i].shape为(23236, 24, 24)，其中23236是字典大小。数字范围为[0,255]
        font_arrays = [
            np.load(np_file).astype(np.float32) for np_file in font_npy_files
        ]
        self.vocab_size = font_arrays[0].shape[0]   # 字典大小，也就是23236
        self.font_num = len(font_arrays)    # 字体数量，三种字体：“仿宋”、“行楷”和“隶书”
        self.font_size = font_arrays[0].shape[-1]   # 图片大小，24.
        # N, C, H, W
        font_array = np.stack(font_arrays, axis=1)  # 将三种字体组合到一起，font_array.shape为(23236, 3, 24, 24)
        self.embedding = nn.Embedding(  # 定义全连接层（Embedding和Linear本质是一样的）
            num_embeddings=self.vocab_size,
            embedding_dim=self.font_size ** 2 * self.font_num,  # 将字编码成24*24*3大小的token
            _weight=torch.from_numpy(font_array.reshape([self.vocab_size, -1]))
        )

    def forward(self, input_ids):
        """
            get glyph images for batch inputs
        Args:
            input_ids: [batch, sentence_length]
        Returns:
            images: [batch, sentence_length, self.font_num*self.font_size*self.font_size]
        """
        # return self.embedding(input_ids).view([-1, self.font_num, self.font_size, self.font_size])
        return self.embedding(input_ids)
