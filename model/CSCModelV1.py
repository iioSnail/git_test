import torch
from torch import nn

from model.base import CSCBaseModel
from model.common import BERT, LayerNorm
from utils.utils import render_color_for_text, restore_special_tokens


class DetectionModel(nn.Module):

    def __init__(self, bert):
        super(DetectionModel, self).__init__()

        self.bert = bert
        self.word_embeddings = self.bert.get_input_embeddings()

        # 使用bert的transformer_encoder来初始化transformer
        self.transformer_blocks = self.bert.encoder.layer[:1]

        self.fusion_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(768 * 3, 768),
            nn.Tanh(),

        )

        self.output_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(768, 1),
            nn.Sigmoid()
        )

        self.norm = LayerNorm(768)

        self.init_fusion_layer()

    def init_fusion_layer(self):
        for param in self.fusion_layer.parameters():
            param.data = torch.full_like(param.data, 1/3)

    def forward(self, inputs, hidden_states, pooler_output):
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

    def get_optimized_params(self):
        params = []
        for key, value in self.named_parameters():
            if not key.startswith("bert."):
                params.append(value)

        return params


class CorrectionModel(nn.Module):

    def __init__(self, bert):
        super(CorrectionModel, self).__init__()

        self.bert = bert
        self.word_embeddings = self.bert.get_input_embeddings()
        self.predict_layer = nn.Sequential(
            nn.Dropout(),
            nn.Linear(768, 2048),
            nn.ReLU(),
            nn.Linear(2048, len(BERT.get_tokenizer()))
        )

    def forward(self, inputs, detection_outputs, bert_outputs):
        outputs = bert_outputs
        word_embeddings = self.word_embeddings(inputs['input_ids'])
        outputs = detection_outputs.unsqueeze(2) * word_embeddings + (1 - detection_outputs).unsqueeze(2) * outputs
        return self.predict_layer(outputs)

    def get_optimized_params(self):
        params = []
        for key, value in self.named_parameters():
            if not key.startswith("bert."):
                params.append(value)

        return params


class CSCModel(CSCBaseModel):

    def __init__(self):
        super(CSCModel, self).__init__()

        self.bert = BERT().bert
        self.detection_model = DetectionModel(self.bert)
        self.correction_model = CorrectionModel(self.bert)

        self.detection_criteria = nn.BCELoss()
        self.correction_criteria = nn.CrossEntropyLoss()

        # self.ignored_tokens =

    def forward(self, inputs):
        with torch.no_grad():
            bert_outputs = self.bert(**inputs)
            hidden_states, pooler_output = bert_outputs.last_hidden_state, bert_outputs.pooler_output

        detection_outputs = self.detection_model(inputs, hidden_states, pooler_output)
        outputs = self.correction_model(inputs, detection_outputs.clone().detach(), hidden_states)
        return outputs, detection_outputs

    def compute_loss(self, outputs, targets, detection_outputs, detection_targets):
        detection_loss = self.detection_criteria(detection_outputs, detection_targets)
        outputs = outputs.view(outputs.size(0) * outputs.size(1), -1)
        targets = targets.view(-1)
        correction_loss = self.correction_criteria(outputs, targets)

        return detection_loss, correction_loss

    def predict(self, text):
        tokenizer = BERT.get_tokenizer()
        inputs = tokenizer(text, return_tensors='pt')
        outputs, detection_outputs = self.forward(inputs)
        outputs, detection_outputs = outputs.squeeze(0).argmax(1)[1:-1], detection_outputs.squeeze(0)[1:-1]

        correct_indices = outputs != inputs['input_ids'][0][1:-1]
        outputs = tokenizer.convert_ids_to_tokens(outputs)
        outputs = restore_special_tokens(text, outputs)
        outputs = ''.join(outputs)
        outputs_rendered = render_color_for_text(outputs, correct_indices, 'green')

        detection_outputs = detection_outputs >= 0.5
        src_char_list = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])[1:-1]
        detection_rendered = render_color_for_text(src_char_list, detection_outputs, 'red')
        return outputs, outputs_rendered, detection_rendered
