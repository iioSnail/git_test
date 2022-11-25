import copy

import torch
import transformers
from torch import nn
from transformers import AutoTokenizer, AutoModel

from model.common import BERT


class CSCRevertModel(nn.Module):

    def __init__(self, args):
        super(CSCRevertModel, self).__init__()
        self.args = args
        self.bert = BERT().bert
        self.tokenizer = BERT.get_tokenizer()

        self.revert_bert = copy.deepcopy(self.bert)
        self.cls_layer = nn.Linear(768, len(self.tokenizer))

    def forward(self, inputs):
        embeddings = self.bert(**inputs).last_hidden_state
        # transformers.models.bert.modeling_bert.BertModel

        outputs = self.bert.encoder(embeddings,
                                    attention_mask=self.bert.get_extended_attention_mask(inputs['attention_mask'],
                                                                                         inputs['input_ids'].shape,
                                                                                         device=None)
                                    ).last_hidden_state

        return embeddings

    def _init_correction_dense_layer(self):
        pass

    def compute_loss(self, correction_outputs, correction_targets, detection_outputs, detection_targets):
        pass

    def predict(self, sentence):
        pass


if __name__ == '__main__':
    model = CSCRevertModel(None)
    tokenizer = BERT.get_tokenizer()
    inputs = tokenizer(["及你太美", "哎呦，你干嘛"], return_tensors="pt", padding=True)
    model(inputs)
