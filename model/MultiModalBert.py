import argparse
import os.path

import numpy as np
import pypinyin
import torch
from PIL import ImageFont
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

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


class PinyinGRUEmbeddings(nn.Module):

    def __init__(self, pinyin_feature_size=8):
        super(PinyinGRUEmbeddings, self).__init__()

        self.embeddings = nn.Sequential(
            nn.Embedding(num_embeddings=27, embedding_dim=pinyin_feature_size, padding_idx=0),
            nn.GRU(input_size=8, hidden_size=pinyin_feature_size, num_layers=2, bias=True,
                   batch_first=True, dropout=0.15)
        )

    def forward(self, inputs):
        return self.embeddings(inputs)[1][-1]


class PinyinRGRUEmbeddings(nn.Module):

    def __init__(self, pinyin_feature_size=8):
        super(PinyinRGRUEmbeddings, self).__init__()

        self.embeddings = nn.GRU(input_size=26, hidden_size=pinyin_feature_size, num_layers=2, bias=True,
                                 batch_first=True, dropout=0.15)

    def forward(self, inputs):
        return self.embeddings(inputs.flip(1))


class PinyinTransformerEmbeddings(nn.Module):

    def __init__(self, pinyin_feature_size=8):
        super(PinyinTransformerEmbeddings, self).__init__()

        self.embeddings = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=26, nhead=1, dim_feedforward=256, batch_first=True),
            num_layers=1)

    def forward(self, inputs):
        return self.embeddings(inputs)


class MultiModalBertModel(nn.Module):

    def __init__(self, args):
        super(MultiModalBertModel, self).__init__()
        self.args = args
        self.bert = BERT().bert
        self.tokenizer = BERT.get_tokenizer()
        self.pinyin_feature_size = 8
        if self.args.pinyin_embeddings == 'gru':
            self.pinyin_embeddings = PinyinGRUEmbeddings(self.pinyin_feature_size)
        elif self.args.pinyin_embeddings == 'rgru':
            self.pinyin_embeddings = PinyinRGRUEmbeddings(self.pinyin_feature_size)
        elif self.args.pinyin_embeddings == 'transformer':
            self.pinyin_embeddings = PinyinTransformerEmbeddings(self.pinyin_feature_size)

        self.glyph_embeddings = GlyphEmbedding(args)

        if 'bert_path' in dir(self.args):
            self.load_model(self.args.bert_path)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        batch_size = input_ids.size(0)
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids.view(-1))
        input_pinyins = []
        for token in input_tokens:
            if not is_chinese(token):
                input_pinyins.append(torch.LongTensor([0]))
                continue

            pinyin = pypinyin.pinyin(token, style=pypinyin.NORMAL)[0][0]
            embeddings = torch.tensor([ord(letter) - 96 for letter in pinyin])

            if embeddings.size(0) <= 6:
                input_pinyins.append(embeddings)
            else:
                raise Exception("难道还有超过6个字母的拼音？")

        input_pinyins = pad_sequence(input_pinyins, batch_first=True).to(self.args.device)
        pinyin_embeddings = self.pinyin_embeddings(input_pinyins)
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
        if not os.path.exists(model_path):
            print("\033[31mERROR: 找不到%s文件\033[0m" % model_path)
            return
        print("Load MultiModalBert From %s" % model_path)
        try:
            self.load_state_dict(torch.load(model_path))
        except Exception as e:
            print(e)
            print("\033[31mERROR: 加载%s文件出错\033[0m" % model_path)


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
        self.soft_criteria = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)

    def forward(self, inputs):
        outputs = self.bert(**inputs).last_hidden_state
        return self.cls(outputs)

    def compute_loss(self, outputs, targets):
        targets = targets['input_ids']
        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        return self.criteria(outputs, targets)

    # def compute_loss(self, outputs, targets, inputs, *args, **kwargs):
    #     """
    #     只计算错字的loss，正确字的loss只给一点点。
    #     有潜力，但是会慢一点，最终训练的时候可以用这个
    #     """
    #     targets = targets['input_ids']
    #     targets_bak = targets.clone()
    #     inputs = inputs['input_ids']
    #     outputs = outputs.view(-1, outputs.size(-1))
    #     targets[targets == inputs] = 0
    #     targets = targets.view(-1)
    #     targets_bak = targets_bak.view(-1)
    #     loss = self.criteria(outputs, targets)
    #     soft_loss = self.soft_criteria(outputs, targets_bak)
    #     return 0.7 * loss + 0.3 * soft_loss

    def get_optimizer(self):
        return self.optimizer

    def predict(self, src, tgt=None):
        inputs = self.tokenizer(src, return_tensors='pt').to(self.args.device)
        outputs = self.forward(inputs)
        outputs = outputs.argmax(-1)
        outputs = self.tokenizer.convert_ids_to_tokens(outputs[0][1:-1])
        outputs = [outputs[i] if len(outputs[i]) == 1 else src[i] for i in range(len(outputs))]
        # if ''.join(outputs) != tgt:   # 最后配合Detector，让softmax前5，用Detector来确定用哪一个
        #     # self.tokenizer.convert_ids_to_tokens(prob[0][3].argsort(descending=True)[:5])
        #     print()
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
