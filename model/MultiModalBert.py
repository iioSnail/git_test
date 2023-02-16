import argparse
import os.path

import numpy as np
import pypinyin
import torch
from PIL import ImageFont
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from model.BertCorrectionModel import BertCorrectionModel
from model.char_cnn import CharResNet
from model.common import BERT
from utils.scheduler import PlateauScheduler
from utils.str_utils import is_chinese
from utils.utils import mock_args, mkdir

font = None

bert_path = "hfl/chinese-macbert-base"
# bert_path = "hfl/chinese-roberta-wwm-ext"


def convert_char_to_image(character, font_size=32):
    global font
    if font is None:
        font = ImageFont.truetype("./assets/font/ms_yahei.ttf", size=font_size)

    image = font.getmask(character)
    image = np.asarray(image).astype(np.float32).reshape(image.size[::-1])

    image = image[:font_size, :font_size]

    if image.size != (font_size, font_size):
        back_image = np.zeros((font_size, font_size)).astype(np.float32)
        offset0 = (font_size - image.shape[0]) // 2
        offset1 = (font_size - image.shape[1]) // 2
        back_image[offset0:offset0 + image.shape[0], offset1:offset1 + image.shape[1]] = image
        image = back_image

    return torch.tensor(image)


class GlyphResnetEmbedding(nn.Module):

    def __init__(self, args, font_size=32):
        super(GlyphResnetEmbedding, self).__init__()
        self.args = args
        self.font_size = font_size
        self.embeddings = CharResNet()

    def forward(self, characters):
        images = [convert_char_to_image(char_, self.font_size) for char_ in characters]
        images = torch.stack(images).to(self.args.device)
        return self.embeddings(images)


class GlyphDenseEmbedding(nn.Module):

    def __init__(self, args, font_size=32):
        super(GlyphDenseEmbedding, self).__init__()
        self.args = args
        self.font_size = font_size
        self.embeddings = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 56),
            nn.Tanh()
        )

    def forward(self, images):
        batch_size = len(images)
        images = images.view(batch_size, -1) / 255.
        return self.embeddings(images)


class GlyphConvEmbedding(nn.Module):

    def __init__(self, args, font_size=32):
        super(GlyphConvEmbedding, self).__init__()
        self.args = args
        self.font_size = font_size
        self.embeddings = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Flatten(),
            nn.Linear(2700, 512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, 56),
            nn.Tanh()
        )

    def forward(self, characters):
        images = [convert_char_to_image(char_, self.font_size) for char_ in characters]
        images = torch.stack(images).to(self.args.device)
        images = images.unsqueeze(1) / 255.
        return self.embeddings(images)


class GlyphPCAEmbedding(nn.Module):

    def __init__(self, args, font_size=32):
        super(GlyphPCAEmbedding, self).__init__()
        self.args = args
        self.font_size = font_size

    def PCA_svd(self, X, k, center=True):
        with torch.no_grad():
            n = X.size()[0]
            ones = torch.ones(n).view([n, 1])
            h = ((1 / n) * torch.mm(ones, ones.t())) if center else torch.zeros(n * n).view([n, n])
            H = torch.eye(n) - h
            X_center = torch.mm(H.to(self.args.device), X)
            u, s, v = torch.svd(X_center)
            components = v[:k].t()
            return components

    def forward(self, characters):
        batch_size = len(characters)
        images = [convert_char_to_image(char_, self.font_size) for char_ in characters]
        images = torch.stack(images).to(self.args.device)
        images = images.view(batch_size, -1)  # / 255.
        return self.PCA_svd(images, 56)


class PinyinManualEmbeddings(nn.Module):

    def __init__(self, args, pinyin_feature_size=8):
        super(PinyinManualEmbeddings, self).__init__()
        self.args = args
        self.pinyin_feature_size = pinyin_feature_size

    def forward(self, inputs):
        fill = self.pinyin_feature_size - inputs.size(1)
        if fill > 0:
            inputs = torch.concat([inputs, torch.zeros((len(inputs), fill)).to(self.args.device)], dim=1).long()
        return inputs.float()


class PinyinDenseEmbeddings(nn.Module):

    def __init__(self, args, pinyin_feature_size=8):
        super(PinyinDenseEmbeddings, self).__init__()
        self.args = args

        self.embeddings = nn.Sequential(
            nn.Embedding(num_embeddings=27, embedding_dim=pinyin_feature_size, padding_idx=0),
            nn.Flatten(),
            nn.Linear(pinyin_feature_size * 6, pinyin_feature_size * 3),
            nn.ReLU(),
            nn.Linear(pinyin_feature_size * 3, pinyin_feature_size),
            nn.Tanh()
        )

    def forward(self, inputs):
        fill = 6 - inputs.size(1)
        if fill <= 0:
            return self.embeddings(inputs)
        inputs = torch.concat([inputs, torch.zeros((len(inputs), fill)).to(self.args.device)], dim=1).long()
        return self.embeddings(inputs)


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

    def __init__(self, args, pinyin_feature_size=8):
        super(PinyinTransformerEmbeddings, self).__init__()
        self.args = args

        self.embeddings = nn.Embedding(num_embeddings=28, embedding_dim=128, padding_idx=0)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=1, dim_feedforward=1024, batch_first=True),
            num_layers=1)

        self.pooler = nn.Sequential(
            nn.Linear(128, pinyin_feature_size),
            nn.Tanh()
        )

    def forward(self, inputs):
        inputs = torch.concat([torch.full((len(inputs), 1), 27).to(self.args.device), inputs], dim=1)  # 最前面增加特殊token
        outputs = self.embeddings(inputs)
        outputs = self.transformer(outputs)
        return self.pooler(outputs[:, 0, :])


class MultiModalBertModel(nn.Module):

    def __init__(self, args):
        super(MultiModalBertModel, self).__init__()
        self.args = args
        self.bert = BERT(bert_path).bert
        self.tokenizer = BERT.get_tokenizer(bert_path)
        self.pinyin_feature_size = 8
        if 'pinyin_embeddings' not in dir(self.args):
            self.args.pinyin_embeddings = 'manual'
        if self.args.pinyin_embeddings == 'gru':
            self.pinyin_embeddings = PinyinGRUEmbeddings(self.pinyin_feature_size)
        elif self.args.pinyin_embeddings == 'rgru':
            self.pinyin_embeddings = PinyinRGRUEmbeddings(self.pinyin_feature_size)
        elif self.args.pinyin_embeddings == 'transformer':
            self.pinyin_embeddings = PinyinTransformerEmbeddings(self.args, self.pinyin_feature_size)
        elif self.args.pinyin_embeddings == 'dense':
            self.pinyin_embeddings = PinyinDenseEmbeddings(self.args, self.pinyin_feature_size)
        elif self.args.pinyin_embeddings == 'manual':
            self.pinyin_embeddings = PinyinManualEmbeddings(self.args, self.pinyin_feature_size)

        if 'glyph_embeddings' not in dir(self.args):
            self.args.glyph_embeddings = 'dense'
        if self.args.glyph_embeddings == 'resnet':
            self.glyph_embeddings = GlyphResnetEmbedding(args)
        elif self.args.glyph_embeddings == 'dense':
            self.glyph_embeddings = GlyphDenseEmbedding(args)
        elif self.args.glyph_embeddings == 'conv':
            self.glyph_embeddings = GlyphConvEmbedding(args)
        elif self.args.glyph_embeddings == 'pca':
            self.glyph_embeddings = GlyphPCAEmbedding(args)

        if 'bert_path' in dir(self.args):
            self.load_model(self.args.bert_path)

        self.hidden_size = self.bert.config.hidden_size + self.pinyin_feature_size + 56

        self.pinyin_embedding_cache = None
        self.init_pinyin_embedding_cache()

        self.token_images_cache = None
        self.init_token_images_cache()

    def convert_tokens_to_pinyin_embeddings(self, input_ids):
        input_pinyins = [self.pinyin_embedding_cache.get(input_id.item(), torch.LongTensor([0])) for input_id in input_ids]
        return pad_sequence(input_pinyins, batch_first=True).to(self.args.device)

    def init_pinyin_embedding_cache(self):
        self.pinyin_embedding_cache = {}
        for token, id in self.tokenizer.get_vocab().items():
            if not is_chinese(token):
                continue

            pinyin = pypinyin.pinyin(token, style=pypinyin.NORMAL)[0][0]
            embeddings = torch.tensor([ord(letter) - 96 for letter in pinyin])
            self.pinyin_embedding_cache[id] = embeddings

    def init_token_images_cache(self):
        self.token_images_cache = {}
        for token, id in self.tokenizer.get_vocab().items():
            # FIXME，这个不能加，就算不是中文也需要有glyph信息，否则peformance就会很差
            # 我也不知道啥原因，很奇怪。
            # if not is_chinese(token):
            #     continue

            self.token_images_cache[id] = convert_char_to_image(token, 32)

    def convert_tokens_to_images(self, input_ids):
        images = [self.token_images_cache.get(input_id.item(), torch.zeros(32, 32)) for input_id in input_ids]
        return torch.stack(images).to(self.args.device)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, characters=None, inputs_embeds=None):
        batch_size = input_ids.size(0)
        if inputs_embeds is not None:
            bert_outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        input_pinyins = self.convert_tokens_to_pinyin_embeddings(input_ids.view(-1))
        pinyin_embeddings = self.pinyin_embeddings(input_pinyins)
        pinyin_embeddings = pinyin_embeddings.view(batch_size, -1, self.pinyin_feature_size)

        images = self.convert_tokens_to_images(input_ids.view(-1))
        glyph_embeddings = self.glyph_embeddings(images)
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
        self.tokenizer = BERT.get_tokenizer(bert_path)
        self.cls = nn.Sequential(
            nn.Linear(768 + 8 + 56, 768 + 8 + 56),
            nn.LayerNorm(768 + 8 + 56),
            nn.Linear(768 + 8 + 56, len(self.tokenizer)),
        )

        # self.cls = nn.Sequential(
        #     nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=768 + 8 + 56, nhead=16, batch_first=True), num_layers=2),
        #     nn.Linear(768 + 8 + 56, len(self.tokenizer)),
        # )

        self.criteria = nn.CrossEntropyLoss(ignore_index=0)
        self.soft_criteria = nn.CrossEntropyLoss(ignore_index=0)
        self.bce_criteria = nn.BCELoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=2e-4)
        self.scheduler = PlateauScheduler(self.optimizer)

    def forward(self, inputs):
        outputs = self.bert(**inputs).last_hidden_state
        return self.cls(outputs)

    # def compute_loss(self, outputs, targets, *args, **kwargs):
    #     targets = targets['input_ids']
    #     outputs = outputs.view(-1, outputs.size(-1))
    #     targets = targets.view(-1)
    #     return self.criteria(outputs, targets)

    def get_lr_scheduler(self):
        return self.scheduler

    # def compute_loss(self, outputs, targets, *args, **kwargs):
    #     """
    #     使用bce_loss。FIXME：不知道为什么不work
    #     """
    #     outputs = outputs.sigmoid()
    #     outputs = outputs * targets['attention_mask'].unsqueeze(-1)
    #     targets_ = F.one_hot(targets['input_ids'], num_classes=len(self.tokenizer))
    #     targets_ = targets_ * targets['attention_mask'].unsqueeze(-1)
    #     loss = self.bce_criteria(outputs, targets_.float())
    #     return loss

    def compute_loss(self, outputs, targets, inputs, *args, **kwargs):
        """
        只计算错字的loss，正确字的loss只给一点点。
        有潜力，但是会慢一点，最终训练的时候可以用这个
        """
        outputs = outputs.view(-1, outputs.size(-1))

        targets = targets['input_ids']
        targets_ = targets.clone()
        soft_loss = self.soft_criteria(outputs, targets_.view(-1))

        inputs = inputs['input_ids']
        targets_ = targets.clone()
        targets_[targets_ == inputs] = 0
        loss = self.criteria(outputs, targets_.view(-1))

        return 0.3 * loss + 0.7 * soft_loss

    def get_optimizer(self):
        return self.optimizer

    def predict(self, src, tgt=None):
        src = src.replace(" ", "")
        src = " ".join(src)
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
    parser.add_argument('--pinyin-embeddings', type=str, default='manual')
    args = parser.parse_known_args()[0]

    bert = MultiModalBertModel(args)

    # Merge Glyph params and Phonetic params.
    model_state = torch.load(args.glyph_model_path, map_location='cpu')
    for name, param in bert.glyph_embeddings.named_parameters():
        param.data = model_state['bert.glyph_embeddings.' + name]

    model_state = torch.load(args.phonetic_model_path, map_location='cpu')
    for name, param in bert.pinyin_embeddings.named_parameters():
        param.data = model_state['bert.pinyin_embeddings.' + name]

    mkdir(args.output_path)
    torch.save(bert.state_dict(), args.output_path + 'multi-modal-bert.pt')
    print("Merge success, the model saved to " + str(args.output_path) + 'multi-modal-bert.pt')


if __name__ == '__main__':
    merge_multi_modal_bert()
