"""
Use to test this code framework
"""
import random
import time

import torch
from torch import nn
import lightning.pytorch as pl

from models.common import BERT


class SimpleModel(pl.LightningModule):

    def __init__(self, args):
        super(SimpleModel, self).__init__()

        self.predict_layer = nn.Linear(1, 1)

    def training_step(self, batch, batch_idx, *args, **kwargs):
        inputs, targets, d_targets, _ = batch

        inputs = inputs['input_ids'].view(-1, 1).float()
        targets = targets['input_ids'].view(-1, 1).float()
        outputs = self.predict_layer(inputs)
        loss = nn.functional.mse_loss(outputs, targets) / 1e+10

        time.sleep(0.2)

        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        self.log("val_f1", random.random() * 100)
        time.sleep(0.2)

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=0.001)
