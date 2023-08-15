import copy
import os
import argparse

import lightning.pytorch as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel

class PluginModel(nn.Module):

    def __init__(self, args):
        super(PluginModel, self).__init__()

        self.args = args

        # 在论文中提到，BERT中Transformer的激活函数使用的是GELU
        transformer_encoder = nn.TransformerEncoderLayer(d_model=768, nhead=12,
                                                         dim_feedforward=768 * 4, dropout=0.1,
                                                         activation=F.gelu, batch_first=True)

        # 多层TransformerEncoder堆叠
        self.plugin_transformer = nn.ModuleList([copy.deepcopy(transformer_encoder) for _ in range(6)])
        # self.head = nn.Sequential(
        #     nn.Linear(768, 768),
        #     nn.GELU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(768, 23236)
        # )

    def forward(self, embeddings, attention_mask):
        plugin_hidden_state = embeddings
        for transformer in self.plugin_transformer:
            plugin_hidden_state = transformer.forward(plugin_hidden_state,
                                                      src_key_padding_mask=~attention_mask.bool())

        # return self.head(plugin_hidden_state)
        return plugin_hidden_state


class ScopeWithPlugin(pl.LightningModule):
    tokenizer = None
    max_length = None
    dataset_helper = None

    def __init__(self, args: argparse.Namespace):
        super(ScopeWithPlugin, self).__init__()

        print("Version: 15:54")

        self.args = args
        self.hyper_params = args.hyper_params
        self.tokenizer = AutoTokenizer.from_pretrained("iioSnail/ChineseBERT-for-csc", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("iioSnail/ChineseBERT-for-csc", trust_remote_code=True)

        self.plugin_model = PluginModel(args)

    def forward(self, inputs):
        embeddings = self.model.model.bert.embeddings(input_ids=inputs['input_ids'], pinyin_ids=inputs['pinyin_ids'],
                                                      token_type_ids=inputs['token_type_ids'])

        bert_hidden_state = self.model.model.bert(**inputs).last_hidden_state
        plugin_hidden_state = self.plugin_model(embeddings, inputs['attention_mask'])

        logits = self.model.model.cls(bert_hidden_state + plugin_hidden_state)

        return logits

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        optimizer = torch.optim.Adam(self.plugin_model.parameters(), lr=3e-4)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.9 ** (epoch))

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def training_step(self, batch, batch_idx):
        inputs, tgt_inputs, d_targets, _ = batch
        logits = self.forward(inputs)

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
        logits = self.forward(inputs)

        targets = tgt_inputs['input_ids']
        targets[targets == 101] = 0
        targets[targets == 102] = 0

        return {
            "loss": torch.tensor(0),
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

    def predict(self, sentence):
        src_tokens = list(sentence.replace(" ", ""))
        sentence = " ".join(src_tokens)
        inputs = self.tokenizer([sentence], return_tensors='pt').to(self.args.device)
        if len(inputs['input_ids'][0]) - 2 != len(src_tokens):
            print("Can't correctly encode the sentence: %s" % ("".join(src_tokens)))
            return ''.join(src_tokens)

        outputs = self.forward(inputs)[0]

        pred_ids = outputs.argmax(-1)[1:-1]
        pred_tokens = self.tokenizer.convert_ids_to_tokens(pred_ids)
        return ''.join(pred_tokens)
