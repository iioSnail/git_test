import argparse
from collections import Counter

import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer, AutoConfig

from models.common import BertOnlyMLMHead, BERT
from utils.log_utils import log
from utils.loss import FocalLoss
from utils.scheduler import WarmupExponentialLR
from utils.str_utils import get_common_hanzi, word_segment, word_segment_labels
from utils.utils import predict_process, convert_char_to_pinyin, convert_char_to_image, pred_token_process, load_obj

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


class PinyinManualEmbeddings(nn.Module):

    def __init__(self, args):
        super(PinyinManualEmbeddings, self).__init__()
        self.args = args
        self.pinyin_feature_size = 6

    def forward(self, inputs):
        fill = self.pinyin_feature_size - inputs.size(1)
        if fill > 0:
            inputs = torch.concat([inputs, torch.zeros((len(inputs), fill), device=self.args.device)], dim=1).long()
        return inputs.float()


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


class GlyphConvEncoder(nn.Module):

    def __init__(self, font_size=32):
        super(GlyphConvEncoder, self).__init__()
        self.font_size = font_size
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1),
            nn.Sigmoid(),
            nn.Flatten(),
        )

    def forward(self, images):
        batch_size = len(images)
        images = images.view(batch_size, 1, 32, 32) / 255.
        return self.encoder(images)

    @staticmethod
    def from_pretrained(pretrained_model_path):
        state_dict = torch.load(pretrained_model_path)
        glyph_encoder = GlyphConvEncoder()
        glyph_encoder.encoder.load_state_dict(state_dict.state_dict())
        return glyph_encoder


class GlyphDenseEncoder(nn.Module):

    def __init__(self, font_size=32):
        super(GlyphDenseEncoder, self).__init__()
        self.font_size = font_size
        self.encoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 56),
            nn.Tanh()
        )

    def forward(self, images):
        batch_size = len(images)
        images = images.view(batch_size, -1) / 255.
        return self.encoder(images)

    @staticmethod
    def from_pretrained(pretrained_model_path):
        state_dict = torch.load(pretrained_model_path)
        glyph_encoder = GlyphDenseEncoder()
        glyph_encoder.encoder.load_state_dict(state_dict.state_dict())
        return glyph_encoder


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

        self.pinyin_feature_size = 6
        self.pinyin_embeddings = PinyinManualEmbeddings(self.args)

        self.glyph_feature_size = 56
        # self.glyph_embeddings = GlyphDenseEmbedding.from_pretrained('./ptm/hanzi_glyph_embedding.pt')
        # self.glyph_embeddings = GlyphConvEncoder.from_pretrained('./ptm/glyph_conv_encoder.pt')
        self.glyph_embeddings = GlyphDenseEncoder.from_pretrained('./ptm/glyph_dense_encoder.pt')

        self.cls = BertOnlyMLMHead(768 + self.pinyin_feature_size + self.glyph_feature_size, len(self._tokenizer),
                                   layer_num=1)

        self.loss_fnt = FocalLoss(device=self.args.device)

        self.counter = None

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

        token_embeddings = self.bert.embeddings(input_ids)
        token_embeddings = token_embeddings * self.token_forget_gate(token_embeddings).sigmoid()
        bert_outputs.last_hidden_state += token_embeddings

        pinyin_embeddings = self.pinyin_embeddings(input_pinyins)
        pinyin_embeddings = pinyin_embeddings.view(batch_size, -1, self.pinyin_feature_size)

        glyph_embeddings = self.glyph_embeddings(images)
        glyph_embeddings = glyph_embeddings.view(batch_size, -1, self.glyph_feature_size)

        hidden_states = torch.concat([bert_outputs.last_hidden_state,
                                      pinyin_embeddings,
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
        _src_tokens = list(sentence)
        src_tokens = list(sentence)
        pred_tokens = self._predict(sentence)

        for _ in range(1):
            record_index = []
            # 遍历input和pred，找出修改了的token对应的index
            for i, (a, b) in enumerate(zip(src_tokens, pred_tokens)):
                if a != b:
                    record_index.append(i)

            src_tokens = pred_tokens
            pred_tokens = self._predict(''.join(pred_tokens))
            for i, (a, b) in enumerate(zip(src_tokens, pred_tokens)):
                # 若这个token被修改了，且在窗口范围内，则什么都不做。
                if a != b and any([abs(i - x) <= 1 for x in record_index]):
                    pass
                else:
                    pred_tokens[i] = src_tokens[i]

        return predict_process(_src_tokens, pred_tokens)

    def segment_predict_bak(self, sentence):
        if self.counter is None:
            self.counter = load_obj("./ptm/word_frequency.pkl")

        sentence = sentence.replace(" ", "")
        pred_tokens = self._predict(sentence)
        pred_sentence = ''.join(self._predict(sentence))

        segment_labels = word_segment_labels([pred_sentence])[0]

        word_index = []
        for i, label in enumerate(segment_labels):
            if label == 'S':
                continue

            if label in ['B', 'I', 'E']:
                word_index.append(i)

            if label == 'E':
                old_word = sentence[word_index[0]:word_index[-1] + 1]
                pred_word = pred_sentence[word_index[0]:word_index[-1] + 1]

                if old_word != pred_word \
                        and self.counter.get(old_word, 0) > 0 \
                        and self.counter.get(pred_word, 0) == 0:
                    for wi in word_index:
                        pred_tokens[wi] = sentence[wi]

                word_index.clear()

        return ''.join(pred_tokens)

    def segment_predict(self, sentence):
        """
        FIXME 存在错字导致分词不准的问题
        """
        sentence = sentence.replace(" ", "")
        pred_tokens = self._predict(sentence)
        pred_sentence = ''.join(pred_tokens)

        segment_labels = word_segment_labels([pred_sentence])[0]

        corrected_words = set()
        for i in range(len(pred_sentence)):
            if sentence[i] != pred_sentence[i]:
                if segment_labels[i] == 'B':
                    corrected_words.add((i, i + 1))

                if segment_labels[i] == 'E':
                    corrected_words.add((i - 1, i))

        if len(corrected_words) > 0:
            pred_token2 = self._predict(pred_sentence)

            for words in corrected_words:
                for i in words:
                    pred_tokens[i] = pred_token2[i]

        return ''.join(pred_tokens)

    def continuous_predict_bak(self, sentence, max_num=9999):
        """
        连续预测
        """
        sentence = sentence.replace(" ", "")
        src_tokens = list(sentence)
        sentence_list = [sentence]
        input_sentence = sentence
        for _ in range(max_num):
            pred_tokens = self._predict(input_sentence)

            # 1. 对原文进行分词
            # word_segment([sentence])

            # 2. 对修正后的结果进行分词

            # 3. 若原文是词，修正后变成了字，则恢复原始字

            # 4. 若原文的词在词典中，修正后的字不在词典中，则恢复原始词

            pred_sentence = predict_process(src_tokens, pred_tokens)
            sentence_list.append(pred_sentence)
            input_sentence = pred_sentence
            if pred_sentence in sentence_list[:-1]:
                break

        return sentence_list[-1]

    def continuous_predict(self, sentence):
        """
        连续预测
        """
        sentence = sentence.replace(" ", "")
        sent_tokens = list(sentence)

        pred_tokens = self._predict(sentence)
        pred_sentence = ''.join(pred_tokens)

        pred_tokens2 = self._predict(pred_sentence)
        pred_sentence2 = ''.join(pred_tokens2)

        correct_status = []
        for i in range(len(sent_tokens)):
            if sent_tokens[i] != pred_tokens[i] and pred_tokens[i] != pred_tokens2[i] and sent_tokens[i] != \
                    pred_tokens2[i]:
                correct_status.append("ABC")
            elif sent_tokens[i] != pred_tokens[i] and sent_tokens[i] == pred_tokens2[i]:
                correct_status.append("ABA")
            elif sent_tokens[i] == pred_tokens[i] and sent_tokens[i] == pred_tokens2[i]:
                correct_status.append("AAA")
            elif sent_tokens[i] != pred_tokens[i] and pred_tokens[i] == pred_tokens2[i]:
                correct_status.append("ABB")
            elif sent_tokens[i] == pred_tokens[i] and pred_tokens[i] != pred_tokens2[i]:
                correct_status.append("AAB")
            else:
                correct_status.append("???")

        for i in range(len(sent_tokens)):
            if correct_status[i] in ("ABC", "ABA"):
                pred_tokens2[i] = sent_tokens[i]

            if correct_status[i] == 'AAB':
                if (i - 1 >= 0 and correct_status[i - 1] == 'ABB') \
                        or (i + 1 < len(sent_tokens) and correct_status[i + 1] == 'ABB'):
                    pass
                else:
                    pred_tokens2[i] = sent_tokens[i]

        return ''.join(pred_tokens2)

    def vote_predict(self, sentence):
        sentence = sentence.replace(" ", "")
        vote_tokens = [[token] for token in sentence]

        input_sentence = sentence
        for i in range(4):
            pred_tokens = self._predict(input_sentence)
            for ti, token in enumerate(pred_tokens):
                vote_tokens[ti].append(token)

            input_sentence = ''.join(pred_tokens)

        pred_tokens = []
        for tokens in vote_tokens:
            counter = Counter(tokens)
            pred_tokens.append(counter.most_common(1)[0][0])

        return ''.join(pred_tokens)

    def test_step(self, batch, batch_idx, *args, **kwargs):
        src, tgt = batch

        pred = []
        for sentence in src:
            pred.append(self.continuous_predict(sentence))

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
