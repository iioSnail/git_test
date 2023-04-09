import argparse

import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer

from models.common import BertOnlyMLMHead, BERT
from utils.loss import FocalLoss
from utils.scheduler import PlateauScheduler
from utils.str_utils import get_common_hanzi
from utils.utils import predict_process, convert_char_to_pinyin


class PinyinManualEmbeddings(nn.Module):

    def __init__(self, args):
        super(PinyinManualEmbeddings, self).__init__()
        self.args = args
        self.pinyin_feature_size = 6

    def forward(self, inputs):
        fill = self.pinyin_feature_size - inputs.size(1)
        if fill > 0:
            inputs = torch.concat([inputs, torch.zeros((len(inputs), fill)).to(self.args.device)], dim=1).long()
        return inputs.float()


class MyModel(pl.LightningModule):
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")

    def __init__(self, args: argparse.Namespace):
        super().__init__()

        self.args = args

        self.bert = AutoModel.from_pretrained("hfl/chinese-macbert-base")
        self.tokenizer = MyModel.tokenizer

        self.hanzi_list = list(get_common_hanzi(6000))
        self.token_list = self.hanzi_list

        self.token_forget_gate = nn.Linear(768, 768)

        self.pinyin_feature_size = 6
        self.pinyin_embeddings = PinyinManualEmbeddings(self.args)

        self.cls = BertOnlyMLMHead(768 + self.pinyin_feature_size, len(self.token_list) + 2, layer_num=1)

        self.loss_fnt = FocalLoss(device=self.args.device)

        self.pinyin_embedding_cache = None
        self.init_pinyin_embedding_cache()

    def init_pinyin_embedding_cache(self):
        self.pinyin_embedding_cache = {}
        for token, id in self.tokenizer.get_vocab().items():
            self.pinyin_embedding_cache[id] = convert_char_to_pinyin(token)

    def _init_parameters(self):
        for layer in self.cls.predictions:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1)

        nn.init.orthogonal_(self.token_forget_gate.weight, gain=1)

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

    def convert_tokens_to_pinyin_embeddings(self, input_ids):
        input_pinyins = []
        for i, input_id in enumerate(input_ids):
            input_pinyins.append(self.pinyin_embedding_cache.get(input_id.item(), torch.LongTensor([0])))

        return pad_sequence(input_pinyins, batch_first=True).to(self.args.device)

    def forward(self, inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        batch_size = input_ids.size(0)

        bert_outputs = self.bert(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)

        token_embeddings = self.bert.embeddings(input_ids)
        token_embeddings = token_embeddings * self.token_forget_gate(token_embeddings).sigmoid()
        bert_outputs.last_hidden_state += token_embeddings

        input_pinyins = self.convert_tokens_to_pinyin_embeddings(input_ids.view(-1))
        pinyin_embeddings = self.pinyin_embeddings(input_pinyins)
        pinyin_embeddings = pinyin_embeddings.view(batch_size, -1, self.pinyin_feature_size)

        cls_inputs = torch.concat([bert_outputs.last_hidden_state, pinyin_embeddings], dim=-1)

        return self.cls(cls_inputs)

    def compute_loss(self, outputs, targets):
        targets = targets.view(-1)
        targets = self.convert_ids_to_hids(targets)
        return self.loss_fnt(outputs.view(-1, outputs.size(-1)), targets)

    def extract_outputs(self, outputs, input_ids):
        outputs = outputs.argmax(-1)

        outputs = self._convert_hids_to_ids(outputs)

        for i in range(len(outputs)):
            for j in range(len(outputs[i])):
                if outputs[i][j] == 1 or outputs[i][j] == 0:
                    outputs[i][j] = input_ids[i][j]

        return outputs

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

    def training_step(self, batch, batch_idx, *args, **kwargs):
        inputs, targets, d_targets, loss_targets = batch
        outputs = self.forward(inputs)

        loss = self.compute_loss(outputs, loss_targets)

        outputs = self.extract_outputs(outputs, inputs['input_ids'])

        return {
            'loss': loss,
            'outputs': outputs,
            'targets': targets['input_ids'],
            'd_targets': d_targets,
            'attention_mask': inputs['attention_mask']
        }

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        inputs, targets, d_targets, loss_targets = batch
        outputs = self.forward(inputs)
        loss = self.compute_loss(outputs, loss_targets)

        outputs = self.extract_outputs(outputs, inputs['input_ids'])

        return {
            'loss': loss,
            'outputs': outputs,
            'targets': targets['input_ids'],
            'd_targets': d_targets,
            'attention_mask': inputs['attention_mask']
        }

    def test_step(self, batch, batch_idx, *args, **kwargs):
        src, tgt = batch
        inputs = BERT.get_bert_inputs(src, tokenizer=MyModel.tokenizer, max_length=9999).to(self.args.device)
        mask = inputs['attention_mask'].bool()

        outputs = self.forward(inputs)
        ids_list = self.extract_outputs(outputs, inputs['input_ids'])

        pred = []

        for i in range(len(ids_list)):
            pred_tokens = self.tokenizer.convert_ids_to_tokens(ids_list[i][mask[i]][1:-1])
            pred.append(predict_process(list(src[i].replace(" ", "")), pred_tokens))

        return pred

    def configure_optimizers(self):
        optimizer = self.make_optimizer()

        scheduler = PlateauScheduler(optimizer)

        return [optimizer], [scheduler]

    def make_optimizer(self):
        params = []
        for key, value in self.bert.named_parameters():
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

        for key, value in self.cls.named_parameters():
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

        src = BERT.get_bert_inputs(src, tokenizer=MyModel.tokenizer)
        tgt = BERT.get_bert_inputs(tgt, tokenizer=MyModel.tokenizer)

        loss_targets = tgt.input_ids.clone()

        d_targets = (src['input_ids'] != tgt['input_ids']).bool()

        # 将没有出错的单个字变为1
        loss_targets[(~d_targets) & (loss_targets != 0)] = 1

        return src, tgt, (src['input_ids'] != tgt['input_ids']).float(), loss_targets
