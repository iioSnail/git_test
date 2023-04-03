from pathlib import Path

import torch
from torch import nn

from models.common import BERT
from utils import utils

import lightning.pytorch as pl

"""
纯BERT进行Correction
"""


class BertCSCModel(pl.LightningModule):

    def __init__(self, args: object):
        super(BertCSCModel, self).__init__()
        self.args = args
        self.bert = BERT()
        self.tokenizer = BERT.get_tokenizer()
        self.cls = nn.Sequential(
            nn.Linear(768, len(self.tokenizer)),
        )

        self.loss_fnt = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, inputs, targets):
        outputs = self.bert(inputs).last_hidden_state
        outputs = self.cls(outputs)

        targets = targets['input_ids']
        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        loss = self.loss_fnt(outputs, targets)

        return loss, outputs

    def training_step(self, batch, batch_idx, *args, **kwargs):
        inputs, targets, d_targets, _ = batch

        loss, outputs = self.forward(inputs, targets)

        return {
            'loss': loss,
            'outputs': outputs.argmax(-1),
            'targets': targets['input_ids'],
            'd_targets': d_targets,
            'attention_mask': inputs['attention_mask']
        }

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        inputs, targets, d_targets, _ = batch

        loss, outputs = self.forward(inputs, targets)

        return {
            'loss': loss,
            'outputs': outputs.argmax(-1),
            'targets': targets['input_ids'],
            'd_targets': d_targets,
            'attention_mask': inputs['attention_mask']
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2e-5)
