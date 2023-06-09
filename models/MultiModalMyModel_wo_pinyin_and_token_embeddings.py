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
from utils.utils import predict_process, convert_char_to_pinyin, convert_char_to_image, pred_token_process

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

default_params = {
    "dropout": 0.1,
    "bert_base_lr": 2e-5,
    "lr_decay_factor": 0.95,
    "weight_decay": 0.01,
    "cls_lr": 2e-4,
}


class InputHelper:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        self.pinyin_embedding_cache = None
        self._init_pinyin_embedding_cache()

        self.token_images_cache = None
        self._init_token_images_cache()

    def _init_pinyin_embedding_cache(self):
        self.pinyin_embedding_cache = {}
        for token, id in self.tokenizer.get_vocab().items():
            self.pinyin_embedding_cache[id] = convert_char_to_pinyin(token)

    def _init_token_images_cache(self):
        self.token_images_cache = {}
        for token, id in self.tokenizer.get_vocab().items():
            # FIXME，这个不能加，就算不是中文也需要有glyph信息，否则peformance就会很差
            # 我也不知道啥原因，很奇怪。
            # if not is_chinese(token):
            #     continue

            self.token_images_cache[id] = convert_char_to_image(token, 32)

    def convert_tokens_to_pinyin_embeddings(self, input_ids):
        input_pinyins = []
        for i, input_id in enumerate(input_ids):
            input_pinyins.append(self.pinyin_embedding_cache.get(input_id.item(), torch.LongTensor([0])))

        return pad_sequence(input_pinyins, batch_first=True)

    def convert_tokens_to_images(self, input_ids, characters):
        images = []
        for i, input_id in enumerate(input_ids):
            if input_id == 100:
                # 如果这个字不在tokenizer里，那么用原始的字获取图片。
                if characters and i - 1 > 0 and i - 1 < len(characters):
                    images.append(convert_char_to_image(characters[i - 1], 32))
                    continue

            images.append(self.token_images_cache.get(input_id.item(), torch.zeros(32, 32)))
        return torch.stack(images)

class GlyphDenseEmbedding(nn.Module):

    def __init__(self, font_size=32):
        super(GlyphDenseEmbedding, self).__init__()
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

    @staticmethod
    def from_pretrained(pretrained_model_path):
        state_dict = torch.load(pretrained_model_path)
        glyph_embedding = GlyphDenseEmbedding()
        glyph_embedding.load_state_dict(state_dict)
        return glyph_embedding


class MyModel(pl.LightningModule):
    bert_path = "hfl/chinese-macbert-base"
    tokenizer = AutoTokenizer.from_pretrained(bert_path)

    input_helper = InputHelper(tokenizer)

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

        self.token_forget_gate = nn.Linear(768, 768, bias=False)

        self.glyph_feature_size = 56
        self.glyph_embeddings = GlyphDenseEmbedding.from_pretrained('./ptm/hanzi_glyph_embedding.pt')

        self.cls = BertOnlyMLMHead(768 + self.glyph_feature_size, len(self._tokenizer),
                                   layer_num=1)

        self.loss_fnt = FocalLoss(device=self.args.device)

    def _init_parameters(self):
        for layer in self.cls.predictions:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1)

        nn.init.orthogonal_(self.token_forget_gate.weight, gain=1)

    def forward(self, inputs, input_pinyins, images, output_hidden_states=False):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        batch_size = input_ids.size(0)

        bert_outputs = self.bert(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)

        glyph_embeddings = self.glyph_embeddings(images)
        glyph_embeddings = glyph_embeddings.view(batch_size, -1, self.glyph_feature_size)

        hidden_states = torch.concat([bert_outputs.last_hidden_state,
                                      glyph_embeddings], dim=-1)
        if output_hidden_states:
            return self.cls(hidden_states), hidden_states
        else:
            return self.cls(hidden_states)

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
        inputs, targets, d_targets, loss_targets, input_pinyins, images = batch
        outputs = self.forward(inputs, input_pinyins, images)

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
        inputs, targets, d_targets, loss_targets, input_pinyins, images = batch
        outputs = self.forward(inputs, input_pinyins, images)
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
        input_pinyins = MyModel.input_helper.convert_tokens_to_pinyin_embeddings(inputs['input_ids'].view(-1))
        images = MyModel.input_helper.convert_tokens_to_images(inputs['input_ids'].view(-1), None)  # TODO
        input_pinyins, images = input_pinyins.to(self.args.device), images.to(self.args.device)
        outputs = self.forward(inputs, input_pinyins, images)
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

        for key, value in self.token_forget_gate.named_parameters():
            if not value.requires_grad:
                continue

            lr = bert_base_lr
            weight_decay = self.args.hyper_params['weight_decay']
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

        input_pinyins = MyModel.input_helper.convert_tokens_to_pinyin_embeddings(src['input_ids'].view(-1))
        images = MyModel.input_helper.convert_tokens_to_images(src['input_ids'].view(-1), None)  # TODO

        return src, tgt, (src['input_ids'] != tgt['input_ids']).float(), loss_targets, input_pinyins, images
