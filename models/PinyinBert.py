"""
将BERT中不用的token变成拼音，然后将一句话连续的三个字替换成相同或相似的拼音，然后对这三个字进行预测。
预测时同样，对每个字都将前后的三个字全部变成拼音，然后对这三个字进行预测。

TODO: 还可以给相似的拼音加一个对比学习，让他们之间的embedding越近越好
"""
import random
from collections import Counter

import numpy as np
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

    def forward(self, inputs, labels, pinyin_masks):
        labels = labels * pinyin_masks  # 只计算pinyin部分的loss
        return self.model(**inputs, labels=labels)

    def training_step(self, batch, batch_idx, *args, **kwargs):
        inputs, labels, pinyin_masks = batch

        outputs = self.forward(inputs, labels, pinyin_masks)

        return {
            'loss': outputs.loss,
            'outputs': outputs.logits.argmax(-1),
            'targets': labels,
            'd_targets': (inputs['input_ids'] != labels).int(),
            'attention_mask': inputs['attention_mask']
        }

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        inputs, labels, pinyin_masks = batch

        outputs = self.forward(inputs, labels, pinyin_masks)

        return {
            'loss': outputs.loss,
            'outputs': outputs.logits.argmax(-1),
            'targets': labels,
            'd_targets': (inputs['input_ids'] != labels).int(),
            'attention_mask': inputs['attention_mask']
        }

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=2e-5)

    def predict(self, sentence):
        """
        和训练时思路类似，逐个字去处理，对于每个字，将其前后的字都改成pinyin，然后进行预测。
        之后忽略前后的预测结果，只取当前字的预测结果。
        结论：效果极差。
        """
        sentence = sentence.replace(" ", "")
        pred_tokens = list(sentence)
        for i in range(len(sentence)):
            tokens = list(sentence)
            if not is_chinese(tokens[i]):
                continue

            tokens[i] = BertCSCModel.to_pinyin(tokens[i])

            if i + 1 < len(tokens) and is_chinese(tokens[i + 1]):
                tokens[i + 1] = BertCSCModel.to_pinyin(tokens[i + 1])

            if i - 1 >= 0 and is_chinese(tokens[i - 1]):
                tokens[i - 1] = BertCSCModel.to_pinyin(tokens[i - 1])

            inputs = self.tokenizer(' '.join(tokens), return_tensors='pt').to(self.args.device)
            outputs = self.model(**inputs).logits
            pred_token = self.tokenizer.convert_ids_to_tokens([outputs[0, i + 1].argmax(-1)])[0]
            pred_tokens[i] = pred_token

        pred = ''.join(pred_tokens)
        return pred

    def predict2(self, sentence):
        """
        和训练时思路类似，逐个字去处理，对于每个字，将其前后的字都改成pinyin，然后进行预测。
        对于每个字，会被预测3次，记录下来，取结果最多的。若一个字的三次预测结果都不一致，那
        么就去置信度最大的那个。
        """
        sentence = sentence.replace(" ", "")
        sent_tokens = list(sentence)
        pred_tokens = list(sentence)
        pred_results = [[] for i in range(len(pred_tokens))]
        for i in range(-1, len(sentence) + 1):
            tokens = list(sentence)
            if 0 <= i < len(tokens) and is_chinese(tokens[i]):
                tokens[i] = BertCSCModel.to_pinyin(tokens[i])

            if i + 1 < len(tokens) and is_chinese(tokens[i + 1]):
                tokens[i + 1] = BertCSCModel.to_pinyin(tokens[i + 1])

            if i - 1 >= 0 and is_chinese(tokens[i - 1]):
                tokens[i - 1] = BertCSCModel.to_pinyin(tokens[i - 1])

            inputs = self.tokenizer(' '.join(tokens), return_tensors='pt').to(self.args.device)
            outputs = self.model(**inputs).logits[0, 1:-1]

            # 统计每个字的三次预测结果
            for j in [i-1, i, i+1]:
                if 0 <= j < len(tokens):
                    if is_chinese(sent_tokens[j]):
                        index = outputs[j].argmax(-1)
                        logit = outputs[j, index].item()
                        token = self.tokenizer.convert_ids_to_tokens([index])[0]
                        pred_results[j].append((token, logit))
                    else:
                        pred_results[j].append((sent_tokens[j], 1))

        for i in range(len(pred_results)):
            tokens, logits = zip(*pred_results[i])
            # 少数服从多数好像不如取概率最大的好用，若一致，则取概率最大的。
            # token, count = Counter(tokens).most_common(1)[0]
            # if count >= 2:
            #     pred_tokens[i] = token
            # else:
            #     pred_tokens[i] = tokens[np.argmax(logits)]
            pred_tokens[i] = tokens[np.argmax(logits)]

        pred = ''.join(pred_tokens)
        return pred

    def predict3(self, sentence, target_sentence):
        """
        和训练时思路类似，逐个字去处理，对于每个字，将其前后的字都改成pinyin，然后进行预测。
        对于每个字，会被预测3次，记录下来，取置信度最大的那个。

        告诉模型哪个是错的，看看最终的结果
        """
        sentence = sentence.replace(" ", "")
        target_sentence = target_sentence.replace(" ", "")
        sent_tokens = list(sentence)
        pred_tokens = list(sentence)
        target_tokens = list(target_sentence)
        pred_results = [[] for i in range(len(pred_tokens))]
        for i in range(-1, len(sentence) + 1):
            tokens = list(sentence)
            if 0 <= i < len(tokens) and is_chinese(tokens[i]):
                tokens[i] = BertCSCModel.to_pinyin(tokens[i])

            if i + 1 < len(tokens) and is_chinese(tokens[i + 1]):
                tokens[i + 1] = BertCSCModel.to_pinyin(tokens[i + 1])

            if i - 1 >= 0 and is_chinese(tokens[i - 1]):
                tokens[i - 1] = BertCSCModel.to_pinyin(tokens[i - 1])

            inputs = self.tokenizer(' '.join(tokens), return_tensors='pt').to(self.args.device)
            outputs = self.model(**inputs).logits[0, 1:-1]

            # 统计每个字的三次预测结果
            for j in [i-1, i, i+1]:
                if 0 <= j < len(tokens):
                    if is_chinese(sent_tokens[j]):
                        index = outputs[j].argmax(-1)
                        logit = outputs[j, index].item()
                        token = self.tokenizer.convert_ids_to_tokens([index])[0]
                        pred_results[j].append((token, logit))
                    else:
                        pred_results[j].append((sent_tokens[j], 1))

        for i in range(len(pred_results)):
            if target_tokens[i] == sent_tokens[i]:
                # 如果这个字没有错，则不预测
                continue

            tokens, logits = zip(*pred_results[i])
            # 少数服从多数好像不如取概率最大的好用，若一致，则取概率最大的。
            # token, count = Counter(tokens).most_common(1)[0]
            # if count >= 2:
            #     pred_tokens[i] = token
            # else:
            #     pred_tokens[i] = tokens[np.argmax(logits)]
            pred_tokens[i] = tokens[np.argmax(logits)]

        pred = ''.join(pred_tokens)
        return pred

    def test_step(self, batch, batch_idx, *args, **kwargs):
        src, tgt = batch

        pred = []
        for i, sentence in enumerate(src):
            pred.append(self.predict3(sentence, tgt[i]))

        return pred

    @staticmethod
    def to_pinyin(token):
        return "[%s]" % to_pinyin(token)

    @staticmethod
    def collate_fn(batch):
        _, tgt = zip(*batch)
        tgt = list(tgt)

        src = []
        masks = []
        for sentence in tgt:
            tokens = sentence.split(" ")
            mask = []
            i = random.randint(0, len(tokens) - 1)
            if is_chinese(tokens[i]):
                tokens[i] = BertCSCModel.to_pinyin(tokens[i])
                mask.append(i + 1)

            if i + 1 < len(tokens) and is_chinese(tokens[i + 1]):
                tokens[i + 1] = BertCSCModel.to_pinyin(tokens[i + 1])
                mask.append(i + 2)

            if i - 1 >= 0 and is_chinese(tokens[i - 1]):
                tokens[i - 1] = BertCSCModel.to_pinyin(tokens[i - 1])
                mask.append(i)

            src.append(' '.join(tokens))
            masks.append(mask)

        inputs = BERT.get_bert_inputs(src, tokenizer=BertCSCModel.tokenizer)
        labels = BERT.get_bert_inputs(tgt, tokenizer=BertCSCModel.tokenizer)['input_ids']

        # 查看src中的第一个句子
        # ''.join(BertCSCModel.tokenizer.convert_ids_to_tokens(inputs['input_ids'][1]))

        pinyin_masks = torch.zeros(labels.size()).int()
        for i, mask in enumerate(masks):
            pinyin_masks[i, mask] = 1

        return inputs, labels, pinyin_masks


if __name__ == '__main__':
    args = mock_args(device='cpu')
    # BertCSCModel(args).forward(["昨晚天气真的很冷！"])
