import torch
from torch import nn

from model.common import LayerNorm, BERT


class BertDetectionModel(nn.Module):

    def __init__(self, args):
        super(BertDetectionModel, self).__init__()
        self.args = args
        self.bert = BERT()

    def forward(self):
        pass


