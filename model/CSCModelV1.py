import torch
from torch import nn

from model.base import CSCBaseModel
from model.common import BERT, LayerNorm


class DetectionModel(nn.Module):

    def __init__(self, bert):
        super(DetectionModel, self).__init__()

        self.bert = bert
        self.word_embeddings = self.bert.get_input_embeddings()

        # 使用bert的transformer_encoder来初始化transformer
        self.transformer_blocks = self.bert.encoder.layer[:2]

        self.fusion_layer = nn.Sequential(
            nn.Linear(768 * 3, 768),
            nn.Sigmoid()  # TODO 这里应该用什么激活函数好？
        )

        self.output_layer = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )

        self.norm = LayerNorm(768)

    def forward(self, inputs, bert_outputs):
        token_num = inputs['input_ids'].size(1)
        outputs = bert_outputs
        with torch.no_grad():
            word_embeddings = self.word_embeddings(inputs['input_ids'])
        cls_outputs = outputs[:, 0:1, :].repeat(1, token_num, 1)
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

    def get_optimized_params(self):
        return list(set(self.parameters()) - set(self.bert.parameters()))


class CorrectionModel(nn.Module):

    def __init__(self, bert):
        super(CorrectionModel, self).__init__()

        self.bert = bert
        self.word_embeddings = self.bert.get_input_embeddings()
        self.predict_layer = nn.Linear(768, len(BERT.get_tokenizer()))

    def forward(self, inputs, detection_outputs, bert_outputs):
        outputs = bert_outputs
        word_embeddings = self.word_embeddings(inputs['input_ids'])
        outputs = detection_outputs.unsqueeze(2) * word_embeddings + (1 - detection_outputs).unsqueeze(2) * outputs
        return self.predict_layer(outputs)

    def get_optimized_params(self):
        return list(set(self.parameters()) - set(self.bert.parameters()))

class CSCModel(CSCBaseModel):

    def __init__(self):
        super(CSCModel, self).__init__()

        self.bert = BERT().bert
        self.detection_model = DetectionModel(self.bert)
        self.correction_model = CorrectionModel(self.bert)

        self.detection_criteria = nn.BCELoss()
        self.correction_criteria = nn.CrossEntropyLoss()

    def forward(self, inputs):
        with torch.no_grad():
            bert_outputs = self.bert(**inputs).last_hidden_state

        detection_outputs = self.detection_model(inputs, bert_outputs)
        outputs = self.correction_model(inputs, detection_outputs, bert_outputs)
        return outputs, detection_outputs

    def compute_loss(self, outputs, targets, detection_outputs, detection_targets):
        detection_loss = self.detection_criteria(detection_outputs, detection_targets)
        outputs = outputs.view(outputs.size(0) * outputs.size(1), -1)
        targets = targets.view(-1)
        correction_loss = self.correction_criteria(outputs, targets)

        return detection_loss, correction_loss
