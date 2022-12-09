import torch
from torch import nn

from model.common import BERT

"""
"""


class BertCLDetectionModel(nn.Module):

    def __init__(self, args):
        super(BertCLDetectionModel, self).__init__()
        self.args = args
        self.bert = BERT()
        self.cls = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )

        self.criteria = nn.BCELoss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)

    def forward(self, inputs):
        hidden_states = self.bert(inputs).last_hidden_state
        d_outputs = self.cls(hidden_states).squeeze()
        return d_outputs * inputs.attention_mask, hidden_states

    def forward_and_computer_loss(self, inputs, targets, d_targets):
        d_outputs, inputs_hidden_states = self.forward(inputs)
        cls_loss = self.criteria(d_outputs, d_targets)

        # 计算Cl loss，相同的token越近越好，相反的token越远越好
        targets_hidden_states = self.bert(inputs).last_hidden_state
        inputs_hidden_states = inputs_hidden_states * inputs.attention_mask
        targets_hidden_states = targets_hidden_states * targets.attention_mask

        sims = torch.cosine_similarity(inputs_hidden_states, targets_hidden_states)
        print(sims)

        return cls_loss


    def get_optimizer(self):
        return self.optimizer

    def predict(self, src, tgt):
        inputs = self.bert.get_bert_inputs(src).to(self.args.device)
        tgt_ids = self.bert.get_bert_inputs(tgt).input_ids.to(self.args.device)

        d_outputs = self.forward(inputs) >= self.args.error_threshold
        d_outputs = d_outputs.int().squeeze()[1:-1]
        d_targets = (inputs.input_ids != tgt_ids).int().squeeze()[1:-1]

        return d_outputs, d_targets
