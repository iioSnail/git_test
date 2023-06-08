import argparse

import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer, AutoConfig

from models.common import BertOnlyMLMHead, BERT
from utils.log_utils import log
from utils.loss import FocalLoss
from utils.scheduler import WarmupExponentialLR
from utils.utils import predict_process, pred_token_process

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

default_params = {
    "dropout": 0.1,
    "bert_base_lr": 2e-5,
    "lr_decay_factor": 0.95,
    "weight_decay": 0.01,
    "cls_lr": 2e-4,
}

class MyModel(pl.LightningModule):
    bert_path = "hfl/chinese-macbert-base"
    tokenizer = AutoTokenizer.from_pretrained(bert_path)

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        for key, value in default_params.items():
            if key in self.args.hyper_params:
                continue
            self.args.hyper_params[key] = value

        log.info("Hyper-parameters:" + str(self.args.hyper_params))

        self.bert_config = AutoConfig.from_pretrained(MyModel.bert_path)
        dropout = self.args.hyper_params['dropout']
        self.bert_config.attention_probs_dropout_prob = dropout
        self.bert_config.hidden_dropout_prob = dropout

        self.bert = AutoModel.from_pretrained(MyModel.bert_path, config=self.bert_config)
        self._tokenizer = AutoTokenizer.from_pretrained(MyModel.bert_path)

        self.cls = BertOnlyMLMHead(768, len(self._tokenizer),
                                   layer_num=1)

        self.loss_fnt = FocalLoss(device=self.args.device)

    def _init_parameters(self):
        for layer in self.cls.predictions:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1)

    def forward(self, inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        bert_outputs = self.bert(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)

        return self.cls(bert_outputs.last_hidden_state)

    def compute_loss(self, outputs, targets):
        targets = targets.view(-1)
        return self.loss_fnt(outputs.view(-1, outputs.size(-1)), targets)

    def extract_outputs(self, outputs, input_ids):
        outputs = outputs.argmax(-1)

        for i in range(len(outputs)):
            for j in range(len(outputs[i])):
                if outputs[i][j] == 1 or outputs[i][j] == 0:
                    outputs[i][j] = input_ids[i][j]

        return outputs

    def training_step(self, batch, batch_idx, *args, **kwargs):
        inputs, targets, d_targets, loss_targets = batch
        outputs = self.forward(inputs)

        loss = self.compute_loss(outputs, loss_targets)

        outputs = outputs.argmax(-1)

        return {
            'loss': loss,
            'outputs': outputs,
            'targets': loss_targets,
            'd_targets': d_targets,
            'attention_mask': inputs['attention_mask']
        }

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        inputs, targets, d_targets, loss_targets = batch
        outputs = self.forward(inputs)
        loss = self.compute_loss(outputs, loss_targets)

        outputs = outputs.argmax(-1)

        return {
            'loss': loss,
            'outputs': outputs,
            'targets': loss_targets,
            'd_targets': d_targets,
            'attention_mask': inputs['attention_mask']
        }

    def _predict(self, sentence):
        src_tokens = list(sentence)
        sentence = ' '.join(list(sentence))
        inputs = BERT.get_bert_inputs(sentence, tokenizer=self._tokenizer, max_length=9999).to(self.args.device)
        outputs = self.forward(inputs)
        ids_list = self.extract_outputs(outputs, inputs['input_ids'])
        pred_tokens = self._tokenizer.convert_ids_to_tokens(ids_list[0, 1:-1])
        pred_tokens = pred_token_process(src_tokens, pred_tokens)
        return pred_tokens

    def predict(self, sentence):
        sentence = sentence.replace(" ", "")
        sent_tokens = list(sentence)

        pred_tokens = self._predict(sentence)
        pred_sentence = ''.join(pred_tokens)

        pred_tokens2 = self._predict(pred_sentence)

        for i in range(len(sent_tokens)):
            if sent_tokens[i] != pred_tokens[i] \
                    and pred_tokens[i] != pred_tokens2[i] \
                    and sent_tokens[i] != pred_tokens2[i]:
                pred_tokens2[i] = sent_tokens[i]

            if sent_tokens[i] == pred_tokens[i] and pred_tokens[i] != pred_tokens2[i]:
                if i == 0 and sent_tokens[i + 1] == pred_tokens[i + 1]:
                    pred_tokens2[i] = sent_tokens[i]

                if i == len(sent_tokens) - 1 and sent_tokens[i + 1] == pred_tokens[i + 1]:
                    pred_tokens2[i] = sent_tokens[i]

                if i > 0 and i < len(sent_tokens) - 1 \
                        and sent_tokens[i + 1] == pred_tokens[i + 1] \
                        and sent_tokens[i - 1] == pred_tokens[i - 1]:
                    pred_tokens2[i] = sent_tokens[i]

        return ''.join(pred_tokens2)

    def test_step(self, batch, batch_idx, *args, **kwargs):
        src, tgt = batch

        pred = []
        for sentence in src:
            pred.append(self.predict(sentence))

        return pred

    def configure_optimizers(self):
        optimizer = self.make_optimizer()

        scheduler_args = {
            "optimizer": optimizer,
            'warmup_factor': 0.01,
            'warmup_epochs': 10240,
            'warmup_method': 'linear',
            'milestones': (10,),
            'gamma': 0.99997,
            'max_iters': 10,
            'delay_iters': 0,
            'eta_min_lr': 2e-6
        }
        scheduler = WarmupExponentialLR(**scheduler_args)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def make_optimizer(self):
        params = []
        # BERT的学习率逐层降低
        bert_base_lr = self.args.hyper_params['bert_base_lr']
        decay_factor = self.args.hyper_params['lr_decay_factor']
        for key, value in self.bert.named_parameters():
            if not value.requires_grad:
                continue

            lr, weight_decay = 0, 0
            if key.startswith("embeddings."):
                lr = bert_base_lr * (decay_factor ** 12)
                weight_decay = self.args.hyper_params['weight_decay']

            if key.startswith("encoder.layer."):
                layer = int(key.split('.')[2])
                lr = bert_base_lr * (decay_factor ** (11 - layer))
                weight_decay = self.args.hyper_params['weight_decay']

            if "bias" in key:
                lr *= 2
                weight_decay = 0

            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        for key, value in self.cls.named_parameters():
            if not value.requires_grad:
                continue

            lr = self.args.hyper_params['cls_lr']
            weight_decay = self.args.hyper_params['weight_decay']
            if "bias" in key:
                lr *= 2
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
