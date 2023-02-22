import math
import random

import numpy as np
import torch
from torch import nn


class BinaryFocalLoss(nn.Module):
    """
    参考 https://github.com/lonePatient/TorchBlocks
    """

    def __init__(self, gamma=2.0, alpha=0.25, epsilon=1.e-9):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, input, target):
        """
        Args:
            input: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        multi_hot_key = target
        logits = input
        # 如果模型没有做sigmoid的话，这里需要加上
        # logits = torch.sigmoid(logits)
        zero_hot_key = 1 - multi_hot_key
        loss = -self.alpha * multi_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
        loss += -(1 - self.alpha) * zero_hot_key * torch.pow(logits, self.gamma) * (1 - logits + self.epsilon).log()
        return loss.mean()


class FocalLoss(nn.Module):
    """
    参考 https://github.com/lonePatient/TorchBlocks
    """

    def __init__(self, gamma=2.0, alpha=1, epsilon=1.e-9, device=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha, device=device)
        else:
            self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, input, target):
        """
        Args:
            input: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        num_labels = input.size(-1)
        idx = target.view(-1, 1).long()
        one_hot_key = torch.zeros(idx.size(0), num_labels, dtype=torch.float32, device=idx.device)
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        logits = torch.softmax(input, dim=-1)
        loss = -self.alpha * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
        loss = loss.sum(1)
        return loss.mean()


class CscFocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0, epsilon=1.e-9, label_smooth=0.0):
        super(CscFocalLoss, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.label_smooth = label_smooth

        self.criteria = nn.CrossEntropyLoss(ignore_index=0)
        self.soft_criteria = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, outputs, targets, inputs):
        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets['input_ids']

        copy_loss = self.focal_loss(outputs, targets)

        inputs = inputs['input_ids']
        targets = targets.clone()
        loss = self.focal_loss(outputs, targets)

        return self.alpha * copy_loss + (1 - self.alpha) * loss

    def focal_loss(self, outputs, targets):
        num_labels = outputs.size(-1)
        targets = targets.view(-1)
        idx = targets.view(-1, 1).long()
        one_hot_key = torch.zeros(idx.size(0), num_labels, dtype=torch.float32, device=idx.device)
        one_hot_key = one_hot_key.scatter_(1, idx, 1 - self.label_smooth)
        one_hot_key[:, 0] = 0 # ignore 0 index.
        logits = torch.softmax(outputs, dim=-1)
        loss = - one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
        return loss.sum(1).mean()


if __name__ == '__main__':
    loss = FocalLoss(alpha=[0.1, 0.2, 0.3, 0.15, 0.25])
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    output = loss(input, target)
    print(output)
    output.backward()
