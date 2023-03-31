"""
Use to test this code framework
"""
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
        loss = nn.functional.mse_loss(outputs, targets)

        time.sleep(0.1)

        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        inputs, targets, d_targets, _ = batch
        outputs = self.model(inputs)
        loss = nn.functional.mse_loss(outputs, targets)

        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
