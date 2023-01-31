import argparse

import torch
from torch import nn

from model.MultiModalBert import MultiModalBertModel
from model.common import BERT
from utils.utils import mock_args, mkdir


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

    def bert_embeddings(self, ids, characters):
        input_ids = ids.unsqueeze(1)
        return self.bert(input_ids=input_ids, characters=characters).last_hidden_state.squeeze()

    def forward(self, inputs):
        pair_i = self.tokenizer.convert_tokens_to_ids(inputs[0])
        pair_j = self.tokenizer.convert_tokens_to_ids(inputs[1])

        pair_i = torch.tensor(pair_i, device=self.args.device)
        pair_j = torch.tensor(pair_j, device=self.args.device)

        embeddings_i = self.bert_embeddings(pair_i, inputs[0])
        embeddings_j = self.bert_embeddings(pair_j, inputs[1])

        outputs = self.cls(torch.concat([embeddings_i, embeddings_j], dim=1))
        return outputs.squeeze()

    def load_glyph_param(self):
        model_state = torch.load(self.args.glyph_model_path, map_location='cpu')
        for name, param in self.bert.glyph_embeddings.named_parameters():
            param.data = model_state['bert.glyph_embeddings.' + name]

    def load_phonetic_param(self):
        model_state = torch.load(self.args.phonetic_model_path, map_location='cpu')
        for name, param in self.bert.pinyin_embeddings.named_parameters():
            param.data = model_state['bert.pinyin_embeddings.' + name]

    def parameters(self, recurse: bool = True):
        if self.args.train_type == 'glyph':
            for name, param in self.bert.glyph_embeddings.named_parameters(recurse=recurse):
                yield param

        if self.args.train_type in ['pinyin', 'phonetic']:
            for name, param in self.bert.pinyin_embeddings.named_parameters(recurse=recurse):
                yield param

        for name, param in self.cls.named_parameters(recurse=recurse):
            yield param

