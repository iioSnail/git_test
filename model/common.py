"""
通用Model，例如LayerNorm等
"""
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class LayerNorm(nn.Module):
    """
    Norm层，其实该层的作用就是BatchNorm。与`torch.nn.BatchNorm2d`的作用一致。
    torch.nn.BatchNorm2d的官方文档地址：https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html

    该LayerNorm就对应原图中 “Add & Norm”中“Norm”的部分
    """

    def __init__(self, features, eps=1e-6):
        """
        features: int类型，含义为特征数。也就是一个词向量的维度，例如128。该值一般和d_model一致。
        """
        super(LayerNorm, self).__init__()
        """
        这两个参数是BatchNorm的参数，a_2对应gamma(γ), b_2对应beta(β)。
        而nn.Parameter的作用就是将这个两个参数作为模型参数，之后要进行梯度下降。
        """
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        # epsilon，一个很小的数，防止分母为0
        self.eps = eps

    def forward(self, x):
        """
        x： 为Attention层或者Feed Forward层的输出。Shape和Encoder的输入一样。（其实整个过程中，x的shape都不会发生改变）。
            例如，x的shape为(1, 7, 128)，即batch_size为1，7个单词，每个单词是128维度的向量。
        """

        # 按最后一个维度求均值。mean的shape为 (1, 7, 1)
        mean = x.mean(-1, keepdim=True)
        # 按最后一个维度求方差。std的shape为 (1, 7, 1)
        std = x.std(-1, keepdim=True)
        # 进行归一化，详情可查阅BatchNorm相关资料。
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class BERT(nn.Module):
    tokenizer = None

    def __init__(self, model_path="hfl/chinese-roberta-wwm-ext"):
        super(BERT, self).__init__()
        self.bert = AutoModel.from_pretrained(model_path)

    @staticmethod
    def get_tokenizer(model_path="hfl/chinese-roberta-wwm-ext"):
        if BERT.tokenizer is None:
            BERT.tokenizer = AutoTokenizer.from_pretrained(model_path)

        return BERT.tokenizer

    @staticmethod
    def get_bert_inputs(sentences, max_length=128, model_path="hfl/chinese-roberta-wwm-ext"):
        """
        The model instance of Hugging Face takes in special input parameters. Therefore,
        we need to construct the input parameters in the function.
        :param sentences: a set of sentences. e.g. ["机器雪习", "练习时长两年半"]
        :param max_length: The max length of a sentence. The sentence will be padded or truncated if it's
                           length is greater than or less than the max length.
        """
        tokenizer = BERT.get_tokenizer()

        inputs = tokenizer(sentences,
                           padding='max_length',
                           max_length=max_length,
                           return_tensors='pt',
                           truncation=True)
        return inputs