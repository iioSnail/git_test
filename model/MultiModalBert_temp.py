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
from model.common import BERT, BertOnlyMLMHead
from utils.dataloader import get_word_segment_collate_fn
from utils.loss import CscFocalLoss, FocalLoss
from utils.scheduler import PlateauScheduler, WarmupExponentialLR
from utils.str_utils import is_chinese, get_common_hanzi, get_common_words, word_segment_targets, word_segment_labels, \
    word_segment
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

        # 未初始化
        self.token_forget_gate = nn.Linear(768, 768)
        nn.init.orthogonal_(self.token_forget_gate.weight, gain=1)
        # self.hidden_forget_gate = nn.Linear(768, 768)

        self.pinyin_embedding_cache = None
        self.init_pinyin_embedding_cache()

        self.token_images_cache = None
        self.init_token_images_cache()

    def convert_tokens_to_pinyin_embeddings(self, input_ids):
        input_pinyins = [self.pinyin_embedding_cache.get(input_id.item(), torch.LongTensor([0])) for input_id in
                         input_ids]
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
            bert_outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                     token_type_ids=token_type_ids)
        else:
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        input_pinyins = self.convert_tokens_to_pinyin_embeddings(input_ids.view(-1))
        pinyin_embeddings = self.pinyin_embeddings(input_pinyins)
        pinyin_embeddings = pinyin_embeddings.view(batch_size, -1, self.pinyin_feature_size)

        images = self.convert_tokens_to_images(input_ids.view(-1))
        glyph_embeddings = self.glyph_embeddings(images)
        glyph_embeddings = glyph_embeddings.view(batch_size, -1, 56)

        # 把经过bert前的embedding加到输出上
        if inputs_embeds is not None:
            token_embeddings = inputs_embeds.clone()
        else:
            token_embeddings = self.bert.embeddings(input_ids)
        # 使用遗忘门过滤token_embeddings
        token_embeddings = token_embeddings * self.token_forget_gate(token_embeddings).sigmoid()
        # 初步试验显示：last_hidden_state加遗忘门不好
        # bert_outputs.last_hidden_state = bert_outputs.last_hidden_state * self.hidden_forget_gate(
        #     bert_outputs.last_hidden_state).sigmoid()

        bert_outputs.last_hidden_state += token_embeddings

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
        self.tokenizer = BERT.get_tokenizer(bert_path)

        self.hanzi_list = self._init_hanzi_list()
        # self.words_list = get_common_words()
        self.token_list = self.hanzi_list  # + self.words_list

        self.bert = MultiModalBertModel(args)
        self.cls = BertOnlyMLMHead(768 + 8 + 56, len(self.token_list) + 2)

        # alpha = [1] * (len(self.token_list) + 2)
        # alpha[0] = 0
        # self.loss_fnt = FocalLoss(alpha=alpha, device=self.args.device)

        self.loss_fnt = FocalLoss(device=self.args.device)

        self.optimizer = self.make_optimizer()
        self.scheduler = PlateauScheduler(self.optimizer)
        # self.scheduler = self.build_lr_scheduler(self.optimizer)
        self.args.multi_forward_args = True

        for layer in self.cls.predictions:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1)

    def make_optimizer(self):
        params = []
        for key, value in self.bert.named_parameters():
            if not value.requires_grad:
                continue
            lr = 2e-6
            weight_decay = 0.01
            # 感觉用处不是很大
            # if "bias" in key:
            #     lr = 4e-6
            #     weight_decay = 0

            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        for key, value in self.cls.named_parameters():
            if not value.requires_grad:
                continue
            lr = 2e-4
            weight_decay = 0.01
            # if "bias" in key:
            #     lr = 4e-4
            #     weight_decay = 0
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        optimizer = torch.optim.AdamW(params)
        return optimizer

    def forward(self, inputs, *args, **kwargs):
        outputs = self.bert(**inputs).last_hidden_state
        # 把该字是否正确这个特征加到里面去。
        return self.cls(outputs), inputs['input_ids']

    def get_lr_scheduler(self):
        if self.args.data_type == 'sighan':
            return None

        return self.scheduler

    def build_lr_scheduler(self, optimizer):
        scheduler_args = {
            "optimizer": optimizer,
        }
        scheduler_args.update({'warmup_factor': 0.01,
                               'warmup_epochs': 1024,
                               'warmup_method': 'linear',
                               'milestones': (10,),
                               'gamma': 0.9999,
                               'max_iters': 10,
                               'delay_iters': 0,
                               'eta_min_lr': 3e-07})
        scheduler = WarmupExponentialLR(**scheduler_args)
        return scheduler

    def compute_loss(self, outputs, _, inputs, detection_targets, targets, *args, **kwargs):
        outputs, _ = outputs
        targets = targets.view(-1)

        targets = self._convert_ids_to_hids(targets)

        return self.loss_fnt(outputs.view(-1, outputs.size(-1)), targets)

    def extract_outputs(self, outputs):
        outputs, input_ids = outputs
        outputs = outputs.argmax(-1)

        outputs = self._convert_hids_to_ids(outputs)

        for i in range(len(outputs)):
            for j in range(len(outputs[i])):
                if outputs[i][j] == 1 or outputs[i][j] == 0:
                    outputs[i][j] = input_ids[i][j]

        return outputs

    def _init_hanzi_list(self):
        # 初始化汉字列表
        return list(get_common_hanzi(6000))

    def _convert_ids_to_hids(self, ids):
        if not hasattr(self, 'hids_map'):
            hanzi_ids = self.tokenizer.convert_tokens_to_ids(self.token_list)
            self.hids_map = dict(zip(hanzi_ids, range(2, len(hanzi_ids) + 2)))

        # 把targets的input_ids转成hanzi_list中对应的“index”
        for i in range(len(ids)):
            tid = int(ids[i])  # token id
            if tid == 0 or tid == 1:
                continue

            if tid in self.hids_map:
                ids[i] = self.hids_map[tid]
                continue

            # 若targets的input_id为错字，但又不是汉字（通常是数据出了问题），则不计算loss
            ids[i] = 0

        return ids

    def _convert_hids_to_ids(self, hids):
        if not hasattr(self, 'hids_map'):
            hanzi_ids = self.tokenizer.convert_tokens_to_ids(self.token_list)
            self.hids_map = dict(zip(hanzi_ids, range(2, len(hanzi_ids) + 2)))

        if not hasattr(self, 'ids_map'):
            self.ids_map = {value: key for key, value in self.hids_map.items()}

        batch_size, token_num = None, None
        if len(hids.shape) == 2:
            batch_size, token_num = hids.shape
            hids = hids.view(-1)

        for i in range(len(hids)):
            hid = int(hids[i])
            if hid in self.ids_map:
                hids[i] = self.ids_map[hid]

        if batch_size and token_num:
            hids = hids.view(batch_size, token_num)

        return hids

    def get_optimizer(self):
        if self.args.data_type == 'sighan':
            print("Fine-tune model with SGD")
            self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

        return self.optimizer

    # def observe_train_performance(self, precision, recall, f1):
    #     alpha = self.loss_fnt.get_alpha()
    #     if recall > precision:
    #         alpha[1] = 1.
    #     else:
    #         alpha[1] = 1 - precision
    #
    #     self.loss_fnt.set_alpha(alpha)

    def predict(self, src):
        src = src.replace(" ", "")
        src = " ".join(src)
        inputs = self.tokenizer(src, return_tensors='pt').to(self.args.device)
        outputs = self.forward(inputs)
        outputs = self.extract_outputs(outputs)
        outputs = self.tokenizer.convert_ids_to_tokens(outputs[0][1:-1])
        outputs = [outputs[i] if len(outputs[i]) == 1 else src[i] for i in range(len(outputs))]
        # if ''.join(outputs) != tgt:   # 最后配合Detector，让softmax前5，用Detector来确定用哪一个
        #     # self.tokenizer.convert_ids_to_tokens(prob[0][3].argsort(descending=True)[:5])
        #     print()
        return ''.join(outputs)

    def get_collate_fn(self):
        def word_segment_collate_fn(batch):
            src, tgt = zip(*batch)
            src, tgt = list(src), list(tgt)

            tgt_sents = [sent.replace(" ", "") for sent in tgt]

            src = BERT.get_bert_inputs(src, tokenizer=self.tokenizer)
            tgt = BERT.get_bert_inputs(tgt, tokenizer=self.tokenizer)

            batch_size, length = src['input_ids'].shape

            tgt_ws_labels = word_segment_targets(tgt_sents)

            targets = tgt.input_ids.clone()

            d_targets = (src['input_ids'] != tgt['input_ids']).bool()

            # 将没有出错的单个字变为1
            # targets[(~detection_targets) & (targets != 0) & (tgt_ws_labels == 1)] = 1

            # 逐个遍历每个字
            for i in range(batch_size):
                for j in range(length):
                    if targets[i, j] == 0:
                        break

                    # 单字token
                    if not d_targets[i, j] and (tgt_ws_labels[i, j] == 0 or tgt_ws_labels[i, j] == 1):
                        targets[i, j] = 1
                        continue

                    if tgt_ws_labels[i, j] == 2:
                        # 找一下词尾在哪
                        for k in range(j + 1, length):
                            if tgt_ws_labels[i, k] == 4:
                                break

                        # 该词没有错
                        if not d_targets[i, j:k + 1].any():
                            targets[i, j:k + 1] = 1

                    # # 双字词
                    # if tgt_ws_labels[i, j] == 2 and tgt_ws_labels[i, j+1] == 4:
                    #     # 该词没有错误
                    #     if not d_targets[i, j] and not d_targets[i, j+1]:
                    #         targets[i, j] = 1
                    #         targets[i, j+1] = 1
                    #     else: # 该词存在错字
                    #         print()

            device = self.args.device
            return src.to(device), tgt.to(device), d_targets.float().to(device), targets.to(device)

        return word_segment_collate_fn


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
