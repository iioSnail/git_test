"""
基础模块，包括通用
"""
from torch import nn


class CSCBaseModel(nn.Module):

    def __init__(self):
        super(CSCBaseModel, self).__init__()

    def forward(self, *args, **kwargs):
        pass

    def compute_loss(self, *args, **kwargs):
        pass

    def predict(self, text):
        """
        Predict a sentence, which will revise the wrong characters in the sentence and return the correct sentence.
        :param text: A sentence, e.g. 机器血习
        :return: The sentence that is corrected. e.g. 机器学习
        """
        pass