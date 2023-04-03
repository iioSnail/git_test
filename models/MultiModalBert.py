import argparse
import os.path

import numpy as np
import torch
import lightning.pytorch as pl
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from models.common import BERT, BertOnlyMLMHead
from utils.log_utils import log
from utils.loss import FocalLoss
from utils.metrics import CSCMetrics
from utils.scheduler import PlateauScheduler, WarmupExponentialLR
from utils.str_utils import is_chinese, get_common_hanzi, get_common_words, word_segment_targets, word_segment_labels, \
    word_segment
from utils.utils import mock_args, mkdir, convert_char_to_image, convert_char_to_pinyin, restore_special_tokens

font = None

bert_path = "hfl/chinese-macbert-base"


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


class MultiModalBertModel(nn.Module):

    def __init__(self, args):
        super(MultiModalBertModel, self).__init__()
        self.args = args
        self.bert = BERT(bert_path).bert
        self.tokenizer = BERT.get_tokenizer(bert_path)
        self.pinyin_feature_size = 8
        self.glyph_feature_size = 56
        self.pinyin_embeddings = PinyinManualEmbeddings(self.args, self.pinyin_feature_size)
        self.glyph_embeddings = GlyphDenseEmbedding(args)

        if 'bert_path' in dir(self.args):
            self.load_model(self.args.bert_path)

        self.hidden_size = self.bert.config.hidden_size + self.pinyin_feature_size + self.glyph_feature_size

        # 未初始化
        self.token_forget_gate = nn.Linear(768, 768)
        nn.init.orthogonal_(self.token_forget_gate.weight, gain=1)

        self.pinyin_embedding_cache = None
        self.init_pinyin_embedding_cache()

        self.token_images_cache = None
        self.init_token_images_cache()

    def convert_tokens_to_pinyin_embeddings(self, input_ids, characters):
        input_pinyins = []
        for i, input_id in enumerate(input_ids):
            if input_id == 100:
                # 如果这个字不在tokenizer里，那么用原始的字获取图片。
                if characters and i - 1 > 0 and i - 1 < len(characters):
                    input_pinyins.append(convert_char_to_pinyin(characters[i - 1]))
                    continue

            input_pinyins.append(self.pinyin_embedding_cache.get(input_id.item(), torch.LongTensor([0])))

        return pad_sequence(input_pinyins, batch_first=True).to(self.args.device)

    def init_pinyin_embedding_cache(self):
        self.pinyin_embedding_cache = {}
        for token, id in self.tokenizer.get_vocab().items():
            self.pinyin_embedding_cache[id] = convert_char_to_pinyin(token)

    def init_token_images_cache(self):
        self.token_images_cache = {}
        for token, id in self.tokenizer.get_vocab().items():
            # FIXME，这个不能加，就算不是中文也需要有glyph信息，否则peformance就会很差
            # 我也不知道啥原因，很奇怪。
            # if not is_chinese(token):
            #     continue

            self.token_images_cache[id] = convert_char_to_image(token, 32)

    def convert_tokens_to_images(self, input_ids, characters):
        images = []
        for i, input_id in enumerate(input_ids):
            if input_id == 100:
                # 如果这个字不在tokenizer里，那么用原始的字获取图片。
                if characters and i - 1 > 0 and i - 1 < len(characters):
                    images.append(convert_char_to_image(characters[i - 1], 32))
                    continue

            images.append(self.token_images_cache.get(input_id.item(), torch.zeros(32, 32)))
        return torch.stack(images).to(self.args.device)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, characters=None, inputs_embeds=None):
        batch_size = input_ids.size(0)
        if inputs_embeds is not None:
            bert_outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                     token_type_ids=token_type_ids)
        else:
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        input_pinyins = self.convert_tokens_to_pinyin_embeddings(input_ids.view(-1), characters)
        pinyin_embeddings = self.pinyin_embeddings(input_pinyins)
        pinyin_embeddings = pinyin_embeddings.view(batch_size, -1, self.pinyin_feature_size)

        images = self.convert_tokens_to_images(input_ids.view(-1), characters)
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

        # self.scheduler = PlateauScheduler(self.optimizer)
        # self.scheduler = self.build_lr_scheduler(self.optimizer)
        self.args.multi_forward_args = True

        for layer in self.cls.predictions:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1)

    def forward(self, inputs, *args, **kwargs):
        outputs = self.bert(**inputs).last_hidden_state
        # 把该字是否正确这个特征加到里面去。
        return self.cls(outputs), inputs['input_ids']

    def get_lr_scheduler(self):
        if self.args.finetune:
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

    def extract_outputs(self, outputs):
        outputs, input_ids = outputs
        # outputs = outputs.softmax(-1)
        # probs, hids = outputs.topk(2, dim=-1)

        # for batch in range(probs.size(0)):
        #     for i in range(probs.size(1)):
        #         # 若模型认为该字正确的概率小于0.8(TODO, 超参数需要调一下)，则采用正常token
        #         # 第二个字占比要大于0.9(TODO, 超参数需要调一下)，才选第二个字。
        #         """
        #         例如：对于预测结果为hids为[1, 123], prob为[0.6, 0.1]
        #         则0.6<0.8，这个置信度比较低，然后再看0.1/(1-0.6)=0.25，候选字在剩下的占比也不算高，所以不采纳。
        #         """
        #         if hids[batch, i, 0] == 1 and probs[batch, i, 0] < float(self.args.threshold):
        #             # and probs[batch, i, 1] / (1 - probs[batch, i, 0]) > 0.5:
        #             if hids[batch, i, 1] in [5, 6, 7]:  # 忽略“他她它“
        #                 continue
        #
        #             hids[batch, i, 0] = int(hids[batch, i, 1])
        # outputs = hids[:, :, 0]

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

    def convert_ids_to_hids(self, ids):
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

    def _predict(self, src):
        src = " ".join(src)
        inputs = self.tokenizer(src, return_tensors='pt').to(self.args.device)
        inputs['characters'] = src.split(" ")
        outputs = self.forward(inputs)
        outputs = self.extract_outputs(outputs)
        outputs = self.tokenizer.convert_ids_to_tokens(outputs[0][1:-1])
        outputs = [outputs[i] if len(outputs[i]) == 1 else src[i] for i in range(len(outputs))]

        pred = ''.join(outputs).replace(" ", "?")
        return pred

    def predict(self, src):
        src = src.replace(" ", "")

        past_pred = [src]

        for i in range(5):
            pred = self._predict(past_pred[-1])
            if pred in past_pred:
                return pred
            else:
                past_pred.append(pred)

        return pred


class MultiModalBertCscModel(pl.LightningModule):
    tokenizer = None
    device = None

    def __init__(self, args: object):
        super(MultiModalBertCscModel, self).__init__()

        self.args = args
        self.loss_fnt = FocalLoss(device=self.args.device)
        self.model = MultiModalBertCorrectionModel(args)
        self.tokenizer = self.model.tokenizer

        self.train_matrix = np.zeros([4])

        self.csc_metrics = CSCMetrics()

        MultiModalBertCscModel.tokenizer = self.tokenizer
        MultiModalBertCscModel.device = self.args.device

    def training_step(self, batch, batch_idx, *args, **kwargs):
        inputs, targets, d_targets, loss_targets = batch
        outputs = self.model(inputs)
        loss = self.compute_loss(outputs, loss_targets)

        outputs = self.model.extract_outputs(outputs)

        return {
            'loss': loss,
            'outputs': outputs,
            'targets': targets['input_ids'],
            'd_targets': d_targets,
            'attention_mask': inputs['attention_mask']
        }

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        inputs, targets, d_targets, loss_targets = batch
        outputs = self.model(inputs)
        loss = self.compute_loss(outputs, loss_targets)

        outputs = self.model.extract_outputs(outputs)

        return {
            'loss': loss,
            'outputs': outputs,
            'targets': targets['input_ids'],
            'd_targets': d_targets,
            'attention_mask': inputs['attention_mask']
        }

    def test_step(self, batch, batch_idx):
        for src, tgt in zip(*batch):
            c_output = self.model.predict(src)
            c_output = restore_special_tokens(src, c_output)

            self.csc_metrics.add_sentence(src, tgt, c_output)

    def compute_loss(self, outputs, targets):
        outputs, _ = outputs
        targets = targets.view(-1)
        targets = self.model.convert_ids_to_hids(targets)
        return self.loss_fnt(outputs.view(-1, outputs.size(-1)), targets)

    def configure_optimizers(self):
        if self.args.finetune:
            log.info("Fine-tune model with SGD")
            return torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

        return self.make_optimizer()

    def make_optimizer(self):
        params = []
        for key, value in self.model.bert.named_parameters():
            if not value.requires_grad:
                continue

            lr = 2e-6
            weight_decay = 0.01
            if "bias" in key:
                lr = 4e-6
                weight_decay = 0

            if 'token_forget_gate' in key:
                lr = 2e-4

            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        for key, value in self.model.cls.named_parameters():
            if not value.requires_grad:
                continue
            lr = 2e-4
            weight_decay = 0.01
            if "bias" in key:
                lr = 4e-4
                weight_decay = 0
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        optimizer = torch.optim.AdamW(params)
        return optimizer

    @staticmethod
    def collate_fn(batch):
        src, tgt = zip(*batch)
        src, tgt = list(src), list(tgt)

        # 这个数据增强作用不是特别大
        # src, tgt = special_hanzi_augment(src, tgt)

        tgt_sents = [sent.replace(" ", "") for sent in tgt]

        src = BERT.get_bert_inputs(src, tokenizer=MultiModalBertCscModel.tokenizer)
        tgt = BERT.get_bert_inputs(tgt, tokenizer=MultiModalBertCscModel.tokenizer)

        batch_size, length = src['input_ids'].shape

        # tgt_ws_labels = word_segment_targets(tgt_sents)

        targets = tgt.input_ids.clone()

        d_targets = (src['input_ids'] != tgt['input_ids']).bool()

        # 将没有出错的单个字变为1
        # targets[(~d_targets) & (targets != 0) & (tgt_ws_labels == 1)] = 1
        targets[(~d_targets) & (targets != 0)] = 1

        # # 逐个遍历每个字
        # for i in range(batch_size):
        #     for j in range(length):
        #         if targets[i, j] == 0:
        #             break
        #
        #         # 单字token
        #         if not d_targets[i, j] and (tgt_ws_labels[i, j] == 0 or tgt_ws_labels[i, j] == 1):
        #             targets[i, j] = 1
        #             continue
        #
        #         if tgt_ws_labels[i, j] == 2:
        #             # 找一下词尾在哪
        #             for k in range(j + 1, length):
        #                 if tgt_ws_labels[i, k] == 4:
        #                     break
        #
        #             # 该词没有错
        #             if not d_targets[i, j:k + 1].any():
        #                 targets[i, j:k + 1] = 1

        device = MultiModalBertCscModel.device
        return src.to(device), tgt.to(device), (src['input_ids'] != tgt['input_ids']).float().to(
            device), targets.to(device)
