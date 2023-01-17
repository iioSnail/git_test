import numpy as np
import pypinyin
import torch
from PIL import ImageFont
from torch import nn
from torch.nn import functional as F

from model.common import BERT
from utils.str_utils import is_chinese


class GlyphEmbedding(nn.Module):
    font = None

    def __init__(self, args):
        super(GlyphEmbedding, self).__init__()
        self.args = args
        self.font_size = 32
        self.embeddings = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 56)
        )

    @staticmethod
    def convert_char_to_image(character, font_size=32):
        if GlyphEmbedding.font is None:
            GlyphEmbedding.font = ImageFont.truetype("./assets/font/ms_yahei.ttf", size=font_size)

        image = GlyphEmbedding.font.getmask(character)
        image = np.asarray(image).astype(np.float32).reshape(image.size[::-1])

        image = image[:font_size, :font_size]

        if image.size != (font_size, font_size):
            back_image = np.zeros((font_size, font_size)).astype(np.float32)
            offset0 = (font_size - image.shape[0]) // 2
            offset1 = (font_size - image.shape[1]) // 2
            back_image[offset0:offset0 + image.shape[0], offset1:offset1 + image.shape[1]] = image
            image = back_image

        return torch.tensor(image)

    def forward(self, characters):
        images = [GlyphEmbedding.convert_char_to_image(char_, self.font_size) for char_ in characters]
        images = torch.stack(images).to(self.args.device)
        return self.embeddings(images)


class MultiModalBertModel(nn.Module):

    def __init__(self, args):
        super(MultiModalBertModel, self).__init__()
        self.args = args
        self.bert = BERT().bert
        self.tokenizer = BERT.get_tokenizer()
        self.pinyin_feature_size = 8
        self.pinyin_embeddings = nn.GRU(input_size=26, hidden_size=self.pinyin_feature_size, num_layers=2, bias=True,
                                        batch_first=True, dropout=0.15)
        self.glyph_embeddings = GlyphEmbedding(args)

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
                embeddings = torch.concat([embeddings, torch.zeros(6 - embeddings.size(0), 26, dtype=torch.long)])
                input_pinyins.append(embeddings)
            else:
                raise Exception("难道还有超过6个字母的拼音？")

        input_pinyins = torch.stack(input_pinyins).to(self.args.device)
        pinyin_embeddings = self.pinyin_embeddings(input_pinyins.view(-1, 6, 26).float())[1][-1]
        pinyin_embeddings = pinyin_embeddings.view(batch_size, -1, self.pinyin_feature_size)
        glyph_embeddings = self.glyph_embeddings(input_tokens)
        glyph_embeddings = glyph_embeddings.view(batch_size, -1, 56)

        bert_outputs.last_hidden_state = torch.concat([bert_outputs.last_hidden_state,
                                                       pinyin_embeddings,
                                                       glyph_embeddings], dim=-1)
        bert_outputs.pooler_output = torch.concat([bert_outputs.pooler_output,
                                                   pinyin_embeddings.sum(dim=1),
                                                   glyph_embeddings.sum(dim=1)], dim=-1)

        return bert_outputs
