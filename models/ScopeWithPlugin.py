import copy
import os
import argparse

import lightning.pytorch as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel


class ScopeWithPlugin(pl.LightningModule):
    tokenizer = None
    max_length = None
    dataset_helper = None

    def __init__(self, args: argparse.Namespace):
        super(ScopeWithPlugin, self).__init__()

        self.args = args
        self.hyper_params = args.hyper_params
        self.tokenizer = AutoTokenizer.from_pretrained("iioSnail\ChineseBERT-for-csc", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("iioSnail\ChineseBERT-for-csc", trust_remote_code=True)

        # 在论文中提到，BERT中Transformer的激活函数使用的是GELU
        transformer_encoder = nn.TransformerEncoderLayer(d_model=768, nhead=12,
                                                         dim_feedforward=768 * 4, dropout=0.1,
                                                         activation=F.gelu, batch_first=True)

        # 多层TransformerEncoder堆叠
        self.plugin_model = nn.ModuleList([copy.deepcopy(transformer_encoder) for _ in range(6)])

    def forward(self, batch):
        inputs, tgt_inputs, d_targets, _ = batch
        embeddings = self.model.model.bert.embeddings(input_ids=inputs['input_ids'], pinyin_ids=inputs['pinyin_ids'],
                                                      token_type_ids=inputs['token_type_ids'])

        bert_hidden_state = self.model.model.bert(**inputs).last_hidden_state

        plugin_hidden_state = embeddings
        for transformer in self.plugin_model:
            plugin_hidden_state = transformer.forward(plugin_hidden_state,
                                                      src_key_padding_mask=~inputs['attention_mask'].bool())

        hidden_state = plugin_hidden_state + bert_hidden_state

        logits = self.model.model.cls(hidden_state)

        return logits

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        # model = self.model
        # no_decay = ["bias", "LayerNorm.weight"]
        # optimizer_grouped_parameters = [
        #     {
        #         "params": [
        #             p
        #             for n, p in model.named_parameters()
        #             if not any(nd in n for nd in no_decay)
        #         ],
        #         "weight_decay": self.hyper_params['weight_decay'],
        #     },
        #     {
        #         "params": [
        #             p
        #             for n, p in model.named_parameters()
        #             if any(nd in n for nd in no_decay)
        #         ],
        #         "weight_decay": 0.0,
        #     },
        # ]
        # optimizer = AdamW(
        #     optimizer_grouped_parameters,
        #     betas=(0.9, 0.98),  # according to RoBERTa paper
        #     lr=self.hyper_params['lr'],
        #     eps=1e-8,
        # )
        # t_total = (
        #         len(self.args.train_loader)
        #         // self.args.accumulate_grad_batches
        #         * self.args.epochs
        # )
        # warmup_steps = int(self.hyper_params['warmup_proporation'] * t_total)
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        # )

        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)

        # return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, tgt_inputs, d_targets, _ = batch
        logits = self.forward(batch)

        vocab_size = logits.size(2)

        targets = tgt_inputs['input_ids']
        targets[targets == 101] = 0
        targets[targets == 102] = 0
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=0)

        return {
            "loss": loss,
            "outputs": logits.argmax(-1),
            "targets": targets,
            "d_targets": d_targets.int(),
            "attention_mask": inputs['attention_mask'],
        }

    def validation_step(self, batch, batch_idx):
        inputs, tgt_inputs, d_targets, _ = batch
        logits = self.forward(batch)

        targets = tgt_inputs['input_ids']
        targets[targets == 101] = 0
        targets[targets == 102] = 0

        return {
            "outputs": logits.argmax(-1),
            "targets": targets,
            "d_targets": d_targets.int(),
            "attention_mask": inputs['attention_mask'],
        }

    def test_step(self, batch, batch_idx, *args, **kwargs):
        src, tgt = batch

        pred = []
        for sentence in src:
            pred.append(self.predict(sentence))

        return pred

    def _predict(self, sentence):
        src_tokens = list(sentence)
        sentence = " ".join(src_tokens)
        encoded = SCOPE_CSC_Model.tokenizer.encode(sentence)
        if len(encoded) - 2 != len(src_tokens):
            print("Can't correctly encode the sentence: %s" % ("".join(src_tokens)))
            return src_tokens

        pinyin_ids = SCOPE_CSC_Model.dataset_helper.convert_sentence_to_pinyin_ids(sentence, encoded)

        input_ids = torch.LongTensor(encoded.ids).unsqueeze(0).to(self.args.device)
        pinyin_ids = torch.LongTensor(pinyin_ids).unsqueeze(0).to(self.args.device)

        outputs = self.forward(
            input_ids, pinyin_ids
        )

        pred_ids = outputs.logits.argmax(-1)[0, 1:-1].tolist()
        pred_tokens = [self.tokenizer.id_to_token(id) for id in pred_ids]
        return pred_tokens
