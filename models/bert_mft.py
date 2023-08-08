import argparse

import lightning.pytorch as pl
from transformers import BertTokenizer, BertForMaskedLM, AutoTokenizer, AutoModel

from utils.utils import predict_process


class BertMFT(pl.LightningModule):

    def __init__(self, args: argparse.Namespace):
        super(BertMFT, self).__init__()

        self.args = args

        self.tokenizer = AutoTokenizer.from_pretrained(r"iioSnail\bert-mft-for-csc", trust_remote_code=True)
        self.model = AutoModel.from_pretrained(r"iioSnail\bert-mft-for-csc", trust_remote_code=True)

    def test_step(self, batch, batch_idx, *args, **kwargs):
        src, tgt = batch

        pred = []
        for sentence in src:
            pred.append(self.predict(sentence))

        return pred

    def predict(self, sentence):
        sentence = sentence.replace(" ", "")
        src_tokens = list(sentence)
        sentence = ' '.join(sentence)

        texts = [sentence]
        outputs = self.model(**self.tokenizer(texts, return_tensors='pt').to(self.args.device)).logits

        pred_tokens = self.tokenizer.convert_ids_to_tokens(outputs.argmax(-1)[0, 1:-1])
        return predict_process(src_tokens, pred_tokens)
