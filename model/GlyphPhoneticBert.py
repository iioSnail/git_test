import torch
from torch import nn

from model.MultiModalBert import MultiModalBertModel
from model.common import BERT



class GlyphPhoneticBertModel(nn.Module):

    def __init__(self, args):
        super(GlyphPhoneticBertModel, self).__init__()
        self.args = args

        # self.bert = BERT().bert
        self.bert = MultiModalBertModel(args)

        self.tokenizer = BERT.get_tokenizer()
        self.cls = nn.Sequential(
            nn.Linear(832 * 2, 768),
            nn.ReLU(),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def bert_embeddings(self, ids):
        input_ids = ids.unsqueeze(1)
        return self.bert(input_ids=input_ids).last_hidden_state.squeeze()

    def forward(self, inputs):
        pair_i = self.tokenizer.convert_tokens_to_ids(inputs[0])
        pair_j = self.tokenizer.convert_tokens_to_ids(inputs[1])

        pair_i = torch.tensor(pair_i, device=self.args.device)
        pair_j = torch.tensor(pair_j, device=self.args.device)

        embeddings_i = self.bert_embeddings(pair_i)
        embeddings_j = self.bert_embeddings(pair_j)

        outputs = self.cls(torch.concat([embeddings_i, embeddings_j], dim=1))
        return outputs.squeeze()

    def parameters(self, recurse: bool = True):
        for name, param in self.bert.glyph_embeddings.named_parameters(recurse=recurse):
            yield param

        for name, param in self.bert.pinyin_embeddings.named_parameters(recurse=recurse):
            yield param

        for name, param in self.cls.named_parameters(recurse=recurse):
            yield param

