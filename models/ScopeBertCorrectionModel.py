from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer

from models.common import BERT
from utils import utils

import lightning.pytorch as pl

from ChineseBert.datasets.bert_dataset import BertDataset
from ChineseBert.models.modeling_glycebert import GlyceBertForMaskedLM

"""
纯BERT进行Correction
"""


class BertCSCModel(pl.LightningModule):

    def __init__(self, args: object):
        super(BertCSCModel, self).__init__()
        self.args = args

        bert_path = "./ChineseBert/model/ChineseBERT-base"
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.input_builder = BertDataset(bert_path)
        self.chinese_bert = GlyceBertForMaskedLM.from_pretrained(bert_path)

    def test_step(self, batch, batch_idx: int, *args, **kwargs):
        src, tgt = batch
        pred = []
        for sent in src:
            input_ids, pinyin_ids = self.input_builder.tokenize_sentence(sent)
            input_ids = input_ids.to(self.args.device)
            pinyin_ids = pinyin_ids.to(self.args.device)
            length = input_ids.shape[0]
            input_ids = input_ids.view(1, length)
            pinyin_ids = pinyin_ids.view(1, length, 8)
            output_hidden = self.chinese_bert.forward(input_ids, pinyin_ids).logits
            pred.append(''.join(self.tokenizer.convert_ids_to_tokens(output_hidden.argmax(-1)[0][1:-1])))
        return pred

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        src, tgt = batch
        return tgt

