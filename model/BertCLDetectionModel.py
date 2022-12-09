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

        self.bce_loss_func = nn.BCELoss()
        self.mse_lose_func = nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)

    def forward(self, inputs):
        hidden_states = self.bert(inputs).last_hidden_state
        d_outputs = self.cls(hidden_states).squeeze()
        return d_outputs * inputs.attention_mask

    def forward_and_computer_loss(self, inputs, targets, d_targets):
        inputs_hidden_states = self.bert(inputs).last_hidden_state
        d_outputs = self.cls(inputs_hidden_states).squeeze()
        d_outputs = d_outputs * inputs.attention_mask

        cls_loss = self.bce_loss_func(d_outputs, d_targets)

        # 计算Cl loss，相同的token越近越好，相反的token越远越好
        targets_hidden_states = self.bert(inputs).last_hidden_state
        inputs_hidden_states = inputs_hidden_states * torch.broadcast_to(inputs.attention_mask.unsqueeze(2),
                                                                         inputs_hidden_states.shape)
        targets_hidden_states = targets_hidden_states * torch.broadcast_to(targets.attention_mask.unsqueeze(2),
                                                                           targets_hidden_states.shape)

        sims = torch.cosine_similarity(inputs_hidden_states, targets_hidden_states, dim=2)

        cl_labels = targets.attention_mask.clone().float()
        cl_labels[d_targets.bool()] = -1

        cl_loss = self.mse_lose_func(sims, cl_labels)

        return d_outputs, 0.8 * cls_loss + 0.2 * cl_loss

    def get_optimizer(self):
        return self.optimizer

    def predict(self, src, tgt):
        inputs = self.bert.get_bert_inputs(src).to(self.args.device)
        tgt_ids = self.bert.get_bert_inputs(tgt).input_ids.to(self.args.device)

        d_outputs = self.forward(inputs) >= self.args.error_threshold
        d_outputs = d_outputs.int().squeeze()[1:-1]
        d_targets = (inputs.input_ids != tgt_ids).int().squeeze()[1:-1]

        return d_outputs, d_targets
