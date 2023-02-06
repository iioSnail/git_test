import copy

import torch
from torch import nn

from model.common import BERT, LayerNorm
from utils.utils import render_color_for_text, restore_special_tokens


class MyDetectionModel(nn.Module):

    def __init__(self, args):
        super(MyDetectionModel, self).__init__()

        self.args = args
        self.bert = BERT().bert
        self.word_embeddings = self.bert.get_input_embeddings()

        # 使用bert的transformer_encoder来初始化transformer
        self.transformer_blocks = copy.deepcopy(self.bert.encoder.layer[:1])

        self.fusion_layer = nn.Sequential(
            nn.Linear(768 * 3, 768),
            nn.Tanh(),

        )

        self.output_layer = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )

        self.norm = LayerNorm(768)

        # self.init_fusion_layer()

        self.criteria = nn.BCELoss()

    def init_fusion_layer(self):
        for param in self.fusion_layer.parameters():
            param.data = torch.full_like(param.data, 1 / 3)

    def forward(self, inputs):
        with torch.no_grad():
            bert_outputs = self.bert(**inputs)
            hidden_states, pooler_output = bert_outputs.last_hidden_state, bert_outputs.pooler_output

        token_num = inputs['input_ids'].size(1)
        outputs = hidden_states
        word_embeddings = self.word_embeddings(inputs['input_ids'])
        cls_outputs = pooler_output.unsqueeze(1).repeat(1, token_num, 1)
        outputs = torch.concat([outputs, word_embeddings, cls_outputs], dim=2)
        fusion_outputs = self.fusion_layer(outputs)

        x = fusion_outputs
        for transformer_layer in self.transformer_blocks:
            x = transformer_layer(x)[0]
        outputs = x

        # outputs = self.transformer(fusion_outputs)
        outputs = outputs + fusion_outputs
        outputs = self.norm(outputs)
        return self.output_layer(outputs).squeeze(2) * inputs['attention_mask']

    def compute_loss(self, d_outputs, d_targets):
        return self.criteria(d_outputs, d_targets)

    # def get_optimized_params(self):
    #     params = []
    #     for key, value in self.named_parameters():
    #         if not key.startswith("bert."):
    #             params.append(value)
    #
    #     return params
