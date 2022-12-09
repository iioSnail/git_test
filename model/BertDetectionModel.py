import torch
from torch import nn

from model.common import BERT

"""
实验结果：

1. 实验结果：P:0.952, R:0.9527, F1:952。 测试集（Sighan15） P:0.611, R:0.703, f1:0.654.
2. 实验过程：使用SIGHAN所有训练集，大概4800个句子，8:2拆分验证集，按f1进行early-stop，训练了17个Epoch，效果也是越来越好，到最后阶段，训练集基本已经到100%了, 但是验证集的表现一直不稳定。
3. 模型情况：纯bert，Adam2e-5的学习率，就一层linear层，bert用的是hfl/chinese-roberta-wwm-ext
"""


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

    def predict(self, src, tgt):
        inputs = self.bert.get_bert_inputs(src).to(self.args.device)
        tgt_ids = self.bert.get_bert_inputs(tgt).input_ids.to(self.args.device)

        d_outputs = self.forward(inputs) >= self.args.error_threshold
        d_outputs = d_outputs.int().squeeze()[1:-1]
        d_targets = (inputs.input_ids != tgt_ids).int().squeeze()[1:-1]

        return d_outputs, d_targets
