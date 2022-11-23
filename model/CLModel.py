import torch
from torch import nn
from torch.nn import functional as F

from model.common import BERT


class DetectionCLModel(nn.Module):

    def __init__(self):
        super(DetectionCLModel, self).__init__()

        self.bert = BERT().bert

        # 学习一个临界点，小于这个临界点就认为是正确字，否则则是错字
        self.discriminative_layer = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

        # 分类损失
        self.d_criteria = nn.BCELoss()
        self.cl_criteria = nn.CrossEntropyLoss()

    def forward(self, inputs):
        outputs = self.bert(**inputs)
        last_hidden_state, pooler_output = outputs.last_hidden_state, outputs.pooler_output
        # 计算token和pooler_output的相似度
        last_hidden_state = F.normalize(last_hidden_state, dim=-1)
        pooler_output = F.normalize(pooler_output, dim=-1)
        sim = torch.einsum("nqv,nv->nq", last_hidden_state, pooler_output)
        detection_outputs = self.discriminative_layer(sim.unsqueeze(-1)).squeeze(-1)
        return detection_outputs * inputs.attention_mask

    def compute_loss(self, d_outputs, d_targets):
        return self.d_criteria(d_outputs, d_targets.float())


if __name__ == '__main__':
    d_model = DetectionCLModel()
    tokenizer = BERT.get_tokenizer()
    inputs = tokenizer(["及你太美", "哎呦，你干嘛"], return_tensors="pt", padding=True)
    d_outputs, sim, last_hidden_state, pooler_output = d_model(inputs)
    d_targets = torch.LongTensor([
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0],
    ])

    d_model.compute_loss(d_outputs, d_targets)

    # 让pooler_output和正确的token求相似度，错字为负样本。
    # 预测的时候也用token与pooler_output的相似度去预测。
