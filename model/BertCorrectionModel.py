from pathlib import Path

import torch
from torch import nn

from model.common import BERT
from utils import utils

"""
纯BERT进行Correction
"""


class BertCorrectionModel(nn.Module):

    def __init__(self, args):
        super(BertCorrectionModel, self).__init__()
        self.args = args
        self.bert = BERT()
        self.tokenizer = BERT.get_tokenizer()
        self.cls = nn.Sequential(
            nn.Linear(768, len(self.tokenizer)),
        )

        self.criteria = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)

    def forward(self, inputs):
        outputs = self.bert(inputs).last_hidden_state
        return self.cls(outputs)

    def compute_loss(self, outputs, targets):
        targets = targets['input_ids']
        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        return self.criteria(outputs, targets)

    def get_optimizer(self):
        return self.optimizer

    def predict(self, src, tgt=None):
        inputs = self.bert.get_bert_inputs(src).to(self.args.device)
        outputs = self.forward(inputs)
        outputs = outputs.argmax(-1)
        outputs = self.tokenizer.convert_ids_to_tokens(outputs[0][1:-1])
        inputs = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][1:-1])
        outputs = [outputs[i] if len(outputs[i]) == 1 else inputs[i] for i in range(len(outputs))]
        return ''.join(outputs)


if __name__ == '__main__':
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[1]
    model = BertCorrectionModel(utils.mock_args(device='cpu', error_threshold=0.5))
    # model.load_state_dict(torch.load(ROOT / 'output/csc-best-model.pt', map_location='cpu'))

    sentence = " ".join("我起床的时候，他在吃早菜。")

    d_outputs = model.predict(sentence)
    print(utils.render_color_for_text(sentence, d_outputs))

    inputs = model.bert.get_bert_inputs(sentence)
    outputs = model.bert(inputs)
    embeddings, pooler_output = outputs.last_hidden_state, outputs.pooler_output
    embeddings = embeddings.squeeze()[1:-1]
    utils.token_embeddings_visualise(embeddings, sentence)
