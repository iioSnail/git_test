import torch
from torch import nn

from model.common import BERT


class BertDetectionModel(nn.Module):

    def __init__(self, args):
        super(BertDetectionModel, self).__init__()
        self.args = args
        self.bert = BERT()
        self.cls = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )

        self.criteria = nn.BCELoss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)

    def forward(self, inputs):
        outputs = self.bert(inputs).last_hidden_state
        d_outputs = self.cls(outputs).squeeze()
        return d_outputs * inputs.attention_mask

    def compute_loss(self, d_outputs, d_targets):
        return self.criteria(d_outputs, d_targets)

    def get_optimizer(self):
        return self.optimizer



