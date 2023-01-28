import argparse

import numpy as np
import pypinyin
import torch
from PIL import ImageFont
from torch import nn
from torch.nn import functional as F

from model.BertCorrectionModel import BertCorrectionModel
from model.char_cnn import CharResNet
from model.common import BERT
from utils.str_utils import is_chinese
from utils.utils import mock_args, mkdir


class GlyphEmbedding(nn.Module):
    font = None

    def __init__(self, args):
        super(GlyphEmbedding, self).__init__()
        self.args = args
        self.font_size = 32
        self.embeddings = CharResNet()

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

        self.load_model(self.args.bert_path)

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

    def load_model(self, model_path):
        print("Load MultiModalBert From %s" % model_path)
        self.load_state_dict(torch.load(model_path))


class MultiModalBertCorrectionModel(nn.Module):

    def __init__(self, args):
        super(MultiModalBertCorrectionModel, self).__init__()
        self.args = args
        self.bert = MultiModalBertModel(args)
        self.tokenizer = BERT.get_tokenizer()
        self.cls = nn.Sequential(
            nn.Linear(768 + 8 + 56, len(self.tokenizer)),
        )

        self.criteria = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)

    def forward(self, inputs):
        outputs = self.bert(**inputs).last_hidden_state
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
        outputs = [outputs[i] if len(outputs[i]) == 1 else src[i] for i in range(len(outputs))]
        return ''.join(outputs)


def merge_multi_modal_bert():
    parser = argparse.ArgumentParser()
    parser.add_argument('--glyph-model-path', type=str, default='./drive/MyDrive/Glyph/probe-best-model.pt')
    parser.add_argument('--phonetic-model-path', type=str, default='./drive/MyDrive/Phonetic/probe-best-model.pt')
    parser.add_argument('--output-path', type=str, default='./drive/MyDrive/MultiModalBertModel/')
    args = parser.parse_known_args()[0]

    bert = MultiModalBertModel(mock_args(device='cpu'))

    # Merge Glyph params and Phonetic params.
    model_state = torch.load(args.glyph_model_path, map_location='cpu')
    for name, param in bert.glyph_embeddings.named_parameters():
        param.data = model_state['bert.glyph_embeddings.' + name]

    model_state = torch.load(args.phonetic_model_path, map_location='cpu')
    for name, param in bert.pinyin_embeddings.named_parameters():
        param.data = model_state['bert.pinyin_embeddings.' + name]

    mkdir(args.output_path)
    torch.save(bert.state_dict(), args.output_path + 'multi-modal-bert.pt')


if __name__ == '__main__':
    merge_multi_modal_bert()
