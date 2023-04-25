"""
将BERT中不用的token变成拼音，然后将一句话连续的三个字替换成相同或相似的拼音，然后对这三个字进行预测。
预测时同样，对每个字都将前后的三个字全部变成拼音，然后对这三个字进行预测。

TODO: 还可以给相似的拼音加一个对比学习，让他们之间的embedding越近越好
"""
import random

import pypinyin
import torch
from torch import nn

from models.common import BERT
from utils import utils

import lightning.pytorch as pl

from transformers import AutoTokenizer, AutoModelForMaskedLM, BertTokenizerFast, BertTokenizer, BertForMaskedLM

from utils.str_utils import is_chinese, get_common_hanzi, to_pinyin
from utils.utils import mock_args


def _init_pinyin():
    pinyins = set()
    hanzi_list = get_common_hanzi()
    for token in hanzi_list:
        if not is_chinese(token):
            continue

        pinyin = pypinyin.pinyin(token, style=pypinyin.Style.TONE3)[0][0]
        pinyin = pinyin.strip("1234567890")
        pinyins.add(pinyin)
    return list(pinyins)


# def get_tokenizer(bert_path):
#     pinyins = _init_pinyin()
#     tokenizer = AutoTokenizer.from_pretrained(bert_path)
#     vocab_dict = tokenizer.get_vocab()
#
#     for i, token in enumerate(tokenizer.convert_ids_to_tokens(range(10001, 10001 + len(pinyins)))):
#         vocab_dict[pinyins[i]] = vocab_dict.pop(token)
#
#     new_tokenizer = AutoTokenizer.from_pretrained(bert_path,
#                                                   vocab=vocab_dict,
#                                                   unk_token=tokenizer.unk_token,
#                                                   pad_token=tokenizer.pad_token,
#                                                   mask_token=tokenizer.mask_token)
#
#     tokenizer.set_vocab(vocab_dict)
#
#     return tokenizer


class BertCSCModel(pl.LightningModule):
    # bert_path = "hfl/chinese-roberta-wwm-ext"
    bert_path = "assets/chinese-roberta-wwm-ext"
    tokenizer = AutoTokenizer.from_pretrained(bert_path)

    def __init__(self, args: object):
        super(BertCSCModel, self).__init__()
        self.args = args

        self.tokenizer = BertCSCModel.tokenizer
        self.model: BertForMaskedLM = AutoModelForMaskedLM.from_pretrained(BertCSCModel.bert_path)

    def training_step(self, batch, batch_idx, *args, **kwargs):
        inputs, labels = batch

        outputs = self.model(**inputs, labels=labels)

        return {
            'loss': outputs.loss,
            'outputs': outputs.logits.argmax(-1),
            'targets': labels,
            'd_targets': (inputs['input_ids'] != labels).int(),
            'attention_mask': inputs['attention_mask']
        }

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        inputs, labels = batch

        outputs = self.model(**inputs, labels=labels)

        return {
            'loss': outputs.loss,
            'outputs': outputs.logits.argmax(-1),
            'targets': labels,
            'd_targets': (inputs['input_ids'] != labels).int(),
            'attention_mask': inputs['attention_mask']
        }

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=3e-4)

    @staticmethod
    def collate_fn(batch):
        _, tgt = zip(*batch)
        tgt = list(tgt)

        src = []
        for sentence in tgt:
            tokens = sentence.split(" ")
            i = random.randint(0, len(tokens) - 1)
            if is_chinese(tokens[i]):
                tokens[i] = "[%s]" % to_pinyin(tokens[i])

            if i + 1 < len(tokens) and is_chinese(tokens[i + 1]):
                tokens[i + 1] = "[%s]" % to_pinyin(tokens[i + 1])

            if i - 1 >= 0 and is_chinese(tokens[i - 1]):
                tokens[i - 1] = "[%s]" % to_pinyin(tokens[i - 1])

            src.append(' '.join(tokens))

        inputs = BERT.get_bert_inputs(src, tokenizer=BertCSCModel.tokenizer)
        labels = BERT.get_bert_inputs(tgt, tokenizer=BertCSCModel.tokenizer)['input_ids']

        # 查看src中的第一个句子
        # ''.join(BertCSCModel.tokenizer.convert_ids_to_tokens(inputs['input_ids'][1]))

        return inputs, labels


if __name__ == '__main__':
    args = mock_args(device='cpu')
    BertCSCModel(args).forward(["昨晚天气真的很冷！"])
