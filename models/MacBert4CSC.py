"""
github: https://github.com/shibing624/pycorrector
huggingface: shibing624/macbert4csc-base-chinese
"""
import argparse

import lightning.pytorch as pl
from transformers import BertTokenizer, BertForMaskedLM


class MacBert4CSC_Model(pl.LightningModule):

    def __init__(self, args: argparse.Namespace):
        super(MacBert4CSC_Model, self).__init__()

        self.args = args

        self.tokenizer = BertTokenizer.from_pretrained("shibing624/macbert4csc-base-chinese")
        self.model = BertForMaskedLM.from_pretrained("shibing624/macbert4csc-base-chinese")

    def test_step(self, batch, batch_idx, *args, **kwargs):
        src, tgt = batch

        pred = []
        for sentence in src:
            pred.append(self.predict(sentence))

        return pred

    def predict(self, sentence):
        sentence = ' '.join(sentence.replace(" ", ""))

        texts = [sentence]
        outputs = self.model(**self.tokenizer(texts, return_tensors='pt').to(self.args.device)).logits

        return self.tokenizer.decode(outputs.argmax(-1)[0], skip_special_tokens=True).replace(' ', '')
