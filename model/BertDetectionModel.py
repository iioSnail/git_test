import torch
from torch import nn

from model.common import BERT

"""
实验结果：

1. 验证集：0.986,0.981,0.983,0,0,0,
2. 实验过程：使用5000*0.8个句子，训练了20多个Epoch，效果也是越来越好，到最后阶段，训练集基本已经到100%了,
3. 模型情况：纯bert，Adam2e-5的学习率，5000个训练数据，就一层linear层，bert用的是hfl/chinese-roberta-wwm-ext，没有刻意设置dropout
4. 测试集效果： Precision 0.42264477095901243, Recall 0.694602272726286, F1_Score 0.5255239114043971
5. 总结：该模型再Wang271K上的表现可以很轻松的拿到高分，但再Sighan上就不行了。

猜测：可能是因为Sighan和Wang271K的模型分布区别太大了？
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
        inputs = self.bert.get_bert_inputs(src)
        tgt_ids = self.bert.get_bert_inputs(tgt).input_ids

        d_outputs = self.forward(inputs) >= self.args.error_threshold
        d_outputs = d_outputs.int().squeeze()[1:-1]
        d_targets = (inputs.input_ids != tgt_ids).int().squeeze()[1:-1]

        return d_outputs, d_targets
