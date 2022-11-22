from torch import nn

from model.common import BERT


class DetectionCLModel(nn.Module):

    def __init__(self):
        super(DetectionCLModel, self).__init__()

        self.bert = BERT().bert

        self.output_layer = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        outputs = self.bert(**inputs)
        last_hidden_state, pooler_output = outputs.last_hidden_state, outputs.pooler_output
        return last_hidden_state, pooler_output


if __name__ == '__main__':
    d_model = DetectionCLModel()
    tokenizer = BERT.get_tokenizer()
    inputs = tokenizer(["及你太美", "哎呦，你干嘛"], return_tensors="pt", padding=True)
    d_model(inputs)
    print("a")
    # 让pooler_output和正确的token求相似度，错字为负样本。
    # 预测的时候也用token与pooler_output的相似度去预测。