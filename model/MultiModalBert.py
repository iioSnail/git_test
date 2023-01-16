import pypinyin
import torch
from torch import nn
from torch.nn import functional as F

from model.common import BERT
from utils.str_utils import is_chinese


class MultiModalBertModel(nn.Module):

    def __init__(self, args):
        super(MultiModalBertModel, self).__init__()
        self.args = args
        self.bert = BERT().bert
        self.tokenizer = BERT.get_tokenizer()
        self.pinyin_feature_size = 8
        self.pinyin_embeddings = nn.GRU(input_size=26, hidden_size=self.pinyin_feature_size, num_layers=2, bias=True, batch_first=True,
                                        dropout=0.15)
        # self.pinyin_embeddings = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=), num_layers=2)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        batch_size = input_ids.size(0)
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids.view(-1))
        input_pinyins = []
        for token in input_tokens:
            if not is_chinese(token):
                input_pinyins.append(torch.zeros(6, 26, dtype=torch.long))
                continue

            pinyin = pypinyin.pinyin(token, style=pypinyin.NORMAL)[0][0]
            embeddings = F.one_hot(torch.tensor([ord(letter) - 97 for letter in pinyin]), 26)

            if embeddings.size(0) <= 6:
                embeddings = torch.concat([embeddings, torch.zeros(6-embeddings.size(0), 26, dtype=torch.long)])
                input_pinyins.append(embeddings)
            else:
                raise Exception("难道还有超过6个字母的拼音？")

        input_pinyins = torch.stack(input_pinyins)
        pinyin_embeddings = self.pinyin_embeddings(input_pinyins.view(-1, 6, 26).float())[1][-1]
        pinyin_embeddings = pinyin_embeddings.view(batch_size, -1, self.pinyin_feature_size)

        bert_outputs.last_hidden_state = torch.concat([bert_outputs.last_hidden_state, pinyin_embeddings], dim=-1)
        bert_outputs.pooler_output = torch.concat([bert_outputs.pooler_output, pinyin_embeddings.sum(dim=1)], dim=-1)

        return bert_outputs
