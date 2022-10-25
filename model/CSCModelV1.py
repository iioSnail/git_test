import torch
from torch import nn

from model.base import CSCBaseModel
from model.common import BERT, LayerNorm


class CSCModel(CSCBaseModel):

    def __init__(self):
        super(CSCModel, self).__init__()

        self.bert = BERT().bert
        self.word_embeddings = self.bert.get_input_embeddings()

        # transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=6, dim_feedforward=1024,
        #                                                        activation='gelu',
        #                                                        norm_first=True,
        #                                                        batch_first=True)
        # self.transformer = nn.TransformerEncoder(transformer_encoder_layer, num_layers=1)

        # 使用bert的transformer_encoder来初始化transformer
        self.transformer_blocks = self.bert.encoder.layer[:2]

        self.fusion_layer = nn.Sequential(
            nn.Linear(768 * 3, 768),
            nn.Sigmoid() # TODO 这里应该用什么激活函数好？
        )

        self.output_layer = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )

        self.norm = LayerNorm(768)

    def forward(self, inputs):
        token_num = inputs['input_ids'].size(1)
        outputs = self.bert(**inputs)
        word_embeddings = self.word_embeddings(inputs['input_ids'])
        cls_outputs = outputs.last_hidden_state[:, 0:1, :].repeat(1, token_num, 1)
        outputs = torch.concat([outputs.last_hidden_state, word_embeddings, cls_outputs], dim=2)
        fusion_outputs = self.fusion_layer(outputs)

        x = fusion_outputs
        for transformer_layer in self.transformer_blocks:
            x = transformer_layer(x)[0]
        outputs = x

        # outputs = self.transformer(fusion_outputs)
        outputs = outputs + fusion_outputs
        outputs = self.norm(outputs)
        return self.output_layer(outputs).squeeze(2) * inputs['attention_mask']