"""
Paper: Improving Chinese Spelling Check by Character Pronunciation Prediction
Paper links: https://arxiv.org/pdf/2210.10996.pdf
publish date: 2022-10
github: https://github.com/jiahaozhenbang/SCOPE

Need to do for reproduction:
1. Download the pre-trained model from "https://rec.ustc.edu.cn/share/18549500-4936-11ed-bdbb-75a980e00e16" or "https://pan.baidu.com/s/1xhNkFm1zpO5pPigK-_5sxg?pwd=it2t"
2. Extract it to the root directory of this projects.
3. run the following command:
```
python c_train.py \
       --model SCOPE \
       --datas sighan13train,sighan14train,sighan15train,wang271k \
       --bert-path FPT \
       --seed 2333 \
       --max-length 512 \
       --no-resume \
       --accumulate_grad_batches 2 \
       --eval \
       --epochs 30 \
       --min_epochs 20 \
       --val-data sighan15test \
       --test-data sighan15test \
       --ckpt-dir /root/autodl-tmp/csc_outputs/ \
       --hyper-params weight_decay=0,lr=5e-5,warmup_proporation=0.1
```

> You can use download the well-trained model from 链接：https://pan.baidu.com/s/10Ma2bPXrZHhqQzOV8k1cNw?pwd=4h17

4. For eval, run the following command:
```
python c_eval.py \
       --model SCOPE \
       --bert-path FPT \
       --data sighan15test \
       --batch-size 1 \
       --ckpt-path ./TrainedModels/SCOPE/scope.ckpt \
       --print-errors
```

For debug in local:
--workers 0 --limit-batches 40 --no-resume --batch-size 4 --datas sighan13train,sighan14train,sighan15train,wang271k --model SCOPE --bert-path FPT --seed 2333 --max-length 128 --hyper-params weight_decay=0,lr=5e-5,warmup_proporation=0.1 --accumulate_grad_batches 2
```

Note:
    1. Finetune Namespaces is
       accelerator=None, accumulate_grad_batches=2, adam_epsilon=1e-08, amp_backend=None, amp_level=None,
       auto_lr_find=False, auto_scale_batch_size=False, auto_select_gpus=None, batch_size=1, benchmark=None,
       bert_path='./FPT', check_val_every_n_epoch=1, checkpoint_path=None, ckpt_path=None, data_dir='./data',
       default_root_dir=None, detect_anomaly=False, devices=None, enable_checkpointing=True, enable_model_summary=True,
       enable_progress_bar=True, fast_dev_run=False, gamma=1, gpus=0, gradient_clip_algorithm=None, gradient_clip_val=None,
       inference_mode=True, ipus=None, label_file='data/test.sighan15.lbl.tsv', limit_predict_batches=None, limit_test_batches=None,
       limit_train_batches=None, limit_val_batches=None, log_every_n_steps=50, logger=True, lr=5e-05, max_epochs=30,
       max_length=512, max_steps=-1, max_time=None, min_epochs=None, min_steps=None, mode='train', move_metrics_to_cpu=False,
       multiple_trainloader_mode='max_size_cycle', num_nodes=1, num_processes=None, num_sanity_val_steps=2, overfit_batches=0.0,
       plugins=None, precision=32, profiler=None, reload_dataloaders_every_n_epochs=1, replace_sampler_ddp=True,
       resume_from_checkpoint=None, save_path='./outputs', save_topk=5, strategy=None, sync_batchnorm=False, tpu_cores=None,
       track_grad_norm=-1, use_memory=False, val_check_interval=None, warmup_proporation=0.1, warmup_steps=0,
       weight_decay=0.0, workers=8)
"""

import os
import argparse

import lightning.pytorch as pl
import numpy as np
import pypinyin
import torch
from tokenizers.implementations import BertWordPieceTokenizer
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup

from models.common import BertOnlyMLMHead
from utils.utils import predict_process


class SCOPE_CSC_Model(pl.LightningModule):
    tokenizer = None
    max_length = None
    dataset_helper = None

    def __init__(self, args: argparse.Namespace):
        super(SCOPE_CSC_Model, self).__init__()

        self.args = args
        self.hyper_params = args.hyper_params
        self.bert_dir = args.bert_path
        self.bert_config = BertConfig.from_pretrained(
            self.bert_dir, output_hidden_states=False
        )

        self.model = Dynamic_GlyceBertForMultiTask.from_pretrained(self.bert_dir)
        self.vocab_size = self.bert_config.vocab_size

        self.tokenizer = BertWordPieceTokenizer(os.path.join(self.bert_dir, 'vocab.txt'))

        SCOPE_CSC_Model.tokenizer = self.tokenizer
        SCOPE_CSC_Model.max_length = self.args.max_length if hasattr(self.args, 'max_length') else 9999999
        SCOPE_CSC_Model.dataset_helper = ChineseBertDataset(self.bert_dir)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hyper_params['weight_decay'],
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            betas=(0.9, 0.98),  # according to RoBERTa paper
            lr=self.hyper_params['lr'],
            eps=1e-8,
        )
        t_total = (
                len(self.args.train_loader)
                // self.args.accumulate_grad_batches
                * self.args.epochs
        )
        warmup_steps = int(self.hyper_params['warmup_proporation'] * t_total)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, pinyin_ids, labels=None, pinyin_labels=None, tgt_pinyin_ids=None):
        """"""
        attention_mask = (input_ids != 0).long()
        return self.model(
            input_ids,
            pinyin_ids,
            attention_mask=attention_mask,
            labels=labels,
            tgt_pinyin_ids=tgt_pinyin_ids,
            pinyin_labels=pinyin_labels,
            gamma=1,
        )

    def compute_loss(self, batch):
        input_ids, pinyin_ids, labels, tgt_pinyin_ids, pinyin_labels = batch
        batch_size, length = input_ids.shape
        pinyin_ids = pinyin_ids.view(batch_size, length, 8)
        tgt_pinyin_ids = tgt_pinyin_ids.view(batch_size, length, 8)
        outputs = self.forward(
            input_ids, pinyin_ids, labels=labels, pinyin_labels=pinyin_labels, tgt_pinyin_ids=tgt_pinyin_ids
        )
        loss = outputs.loss
        return loss, outputs

    def training_step(self, batch, batch_idx):
        input_ids, pinyin_ids, labels, tgt_pinyin_ids, pinyin_labels = batch
        mask = (input_ids != 0) * (input_ids != 101) * (input_ids != 102).long()
        loss, outputs = self.compute_loss(batch)
        return {
            "loss": loss,
            "outputs": outputs.logits.argmax(-1),
            "targets": labels,
            "d_targets": (input_ids != labels).int(),
            "attention_mask": mask,
        }

    def validation_step(self, batch, batch_idx):
        input_ids, pinyin_ids, labels, tgt_pinyin_ids, pinyin_labels = batch
        mask = (input_ids != 0) * (input_ids != 101) * (input_ids != 102).long()
        loss, outputs = self.compute_loss(batch)
        return {
            "loss": loss,
            "outputs": outputs.logits.argmax(-1),
            "targets": labels,
            "d_targets": (input_ids != labels).int(),
            "attention_mask": mask,
        }

    def test_step(self, batch, batch_idx, *args, **kwargs):
        src, tgt = batch

        pred = []
        for sentence in src:
            pred.append(self.predict(sentence))

        return pred

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

    def _predict(self, sentence):
        encoded = SCOPE_CSC_Model.tokenizer.encode(sentence)
        pinyin_ids = SCOPE_CSC_Model.dataset_helper.convert_sentence_to_pinyin_ids(sentence, encoded)

        input_ids = torch.LongTensor(encoded.ids).unsqueeze(0).to(self.args.device)
        pinyin_ids = torch.LongTensor(pinyin_ids).unsqueeze(0).to(self.args.device)

        outputs = self.forward(
            input_ids, pinyin_ids
        )

        pred_tokens = outputs.logits.argmax(-1)[0, 1:-1].tolist()
        pred_tokens = list(self.tokenizer.decode(pred_tokens).replace(" ", ""))
        return pred_tokens

    @staticmethod
    def collate_fn(batch):
        src, tgt = zip(*batch)
        src_sents, tgt_sents = list(src), list(tgt)
        src_sents[0] = "当 然 这 个 家 庭 抱 括 孩 子 们 。 我 非 常 努 力 工 作 赚 钱 ， 照 顾 妻 子 与 父 母 。"
        src_sents[1] = "对 我 来 说 ， 我 父 母 是 最 棒 的 父 母 ， 因 为 他 们 可 以 当 我 的 朋 又 ， 我 的 爱 人 ， 也 当 我 的 老 师 。 虽 然 他 们 有 的 时 候 骂 我 一 顿 ， 有 的 时 候 不 让 我 做 我 喜 欢 的 事 情 ， 但 是 他 们 其 实 很 疼 喔 。"
        src_sents[2] = "他 长 的 很 有 高 又 漂 亮 ， 你 知 道 吗 ？ 他 在 讲 的 时 候 大 家 都 很 专 心 的 在 听 ， 他 说 的 中 问 让 别 人 感 觉 到 他 是 一 个 很 有 知 识 的 人 。"
        src_sents[3] = "我 们 可 以 在 这 里 吃 很 多 台 湾 好 吃 的 东 西 ， 打 球 还 有 奇 脚 踏 车 ， 做 别 的 事 ， 然 后 我 们 可 以 去 宫 殿 博 物 馆 ， 看 待 万 的 历 史 ， 然 后 我 们 会 去 看 别 的 地 方 。"

        tgt_sents[0] = "当 然 这 个 家 庭 包 括 孩 子 们 。 我 非 常 努 力 工 作 赚 钱 ， 照 顾 妻 子 与 父 母 。"
        tgt_sents[1] = "对 我 来 说 ， 我 父 母 是 最 棒 的 父 母 ， 因 为 他 们 可 以 当 我 的 朋 友 ， 我 的 爱 人 ， 也 当 我 的 老 师 。 虽 然 他 们 有 的 时 候 骂 我 一 顿 ， 有 的 时 候 不 让 我 做 我 喜 欢 的 事 情 ， 但 是 他 们 其 实 很 疼 我 。"
        tgt_sents[2] = "他 长 得 很 又 高 又 漂 亮 ， 你 知 道 吗 ？ 他 在 讲 的 时 候 大 家 都 很 专 心 地 在 听 ， 他 说 的 中 间 让 别 人 感 觉 到 他 是 一 个 很 有 知 识 的 人 。"
        tgt_sents[3] = "我 们 可 以 在 这 里 吃 很 多 台 湾 好 吃 的 东 西 ， 打 球 还 有 骑 脚 踏 车 ， 做 别 的 事 ， 然 后 我 们 可 以 去 宫 殿 博 物 馆 ， 看 台 湾 的 历 史 ， 然 后 我 们 会 去 看 别 的 地 方 。"


        input_ids_list = []
        input_pinyin_ids = []
        for sent in src_sents:
            encoded = SCOPE_CSC_Model.tokenizer.encode(sent)
            pinyin_ids = SCOPE_CSC_Model.dataset_helper.convert_sentence_to_pinyin_ids(sent, encoded)

            input_ids_list.append(torch.LongTensor(encoded.ids))
            input_pinyin_ids.append(torch.LongTensor(pinyin_ids))

        label_list = []
        tgt_pinyin_ids = []
        pinyin_label_list = []
        for sent in tgt_sents:
            encoded = SCOPE_CSC_Model.tokenizer.encode(sent)
            pinyin_ids = SCOPE_CSC_Model.dataset_helper.convert_sentence_to_pinyin_ids(sent, encoded)
            pinyin_label = SCOPE_CSC_Model.dataset_helper.convert_sentence_to_shengmu_yunmu_shengdiao_ids(sent, encoded)

            label_list.append(torch.LongTensor(encoded.ids))
            tgt_pinyin_ids.append(torch.LongTensor(pinyin_ids))
            pinyin_label_list.append(torch.LongTensor(pinyin_label))

        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
        labels = pad_sequence(label_list, batch_first=True, padding_value=0)
        input_pinyin_ids = pad_sequence(input_pinyin_ids, batch_first=True)
        input_pinyin_ids = input_pinyin_ids.view(input_pinyin_ids.size(0), -1)
        tgt_pinyin_ids = pad_sequence(tgt_pinyin_ids, batch_first=True)
        tgt_pinyin_ids = input_pinyin_ids.view(tgt_pinyin_ids.size(0), -1)
        pinyin_labels = pad_sequence(pinyin_label_list, batch_first=True)

        return input_ids, input_pinyin_ids, labels, tgt_pinyin_ids, pinyin_labels


########################### ChineseBERT from SCOPE Source #################################
from typing import List

import warnings
import tokenizers

import json
import math
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler, BertOnlyMLMHead, BertPreTrainedModel, \
    BertLMPredictionHead
from transformers.models.bert.modeling_bert import BertModel, BertPredictionHeadTransform
from transformers.modeling_outputs import BaseModelOutputWithPooling, MaskedLMOutput, SequenceClassifierOutput, \
    QuestionAnsweringModelOutput, TokenClassifierOutput

from torch.nn import functional as F
from torch.utils.data import Dataset


class ChineseBertDataset(Dataset):

    def __init__(self, chinese_bert_path):
        """
        Dataset Base class
        Args:
            chinese_bert_path: pretrain model path
        """
        super().__init__()
        self.vocab_file = os.path.join(chinese_bert_path, 'vocab.txt')
        self.config_path = os.path.join(chinese_bert_path, 'config')
        self.tokenizer = BertWordPieceTokenizer(self.vocab_file)
        # load pinyin map dict
        with open(os.path.join(self.config_path, 'pinyin_map.json'), encoding='utf8') as fin:
            self.pinyin_dict = json.load(fin)
        # load char id map tensor
        with open(os.path.join(self.config_path, 'id2pinyin.json'), encoding='utf8') as fin:
            self.id2pinyin = json.load(fin)
        # load pinyin map tensor
        with open(os.path.join(self.config_path, 'pinyin2tensor.json'), encoding='utf8') as fin:
            self.pinyin2tensor = json.load(fin)

        self.pho_convertor = Pinyin()

    @property
    def get_lines(self):
        """read data lines"""
        raise NotImplementedError

    def convert_sentence_to_pinyin_ids(self, sentence: str, tokenizer_output: tokenizers.Encoding) -> List[List[int]]:
        # get pinyin of a sentence
        pinyin_list = pypinyin.pinyin(sentence, style=pypinyin.Style.TONE3, heteronym=True,
                                      errors=lambda x: [['not chinese'] for _ in x])
        pinyin_locs = {}
        # get pinyin of each location
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]
            # not a Chinese character, pass
            if pinyin_string == "not chinese":
                continue
            if pinyin_string in self.pinyin2tensor:
                pinyin_locs[index] = self.pinyin2tensor[pinyin_string]
            else:
                ids = [0] * 8
                for i, p in enumerate(pinyin_string):
                    if p not in self.pinyin_dict["char2idx"]:
                        ids = [0] * 8
                        break
                    ids[i] = self.pinyin_dict["char2idx"][p]
                pinyin_locs[index] = ids

        # find chinese character location, and generate pinyin ids
        pinyin_ids = []
        for idx, (token, offset) in enumerate(zip(tokenizer_output.tokens, tokenizer_output.offsets)):
            if offset[1] - offset[0] != 1:
                pinyin_ids.append([0] * 8)
                continue
            if offset[0] in pinyin_locs:
                pinyin_ids.append(pinyin_locs[offset[0]])
            else:
                pinyin_ids.append([0] * 8)

        return pinyin_ids

    def convert_sentence_to_shengmu_yunmu_shengdiao_ids(self, sentence: str, tokenizer_output: tokenizers.Encoding) -> \
            List[List[int]]:
        # get pinyin of a sentence
        pinyin_list = pypinyin.pinyin(sentence, style=pypinyin.Style.TONE3, neutral_tone_with_five=True, heteronym=True,
                                      errors=lambda x: [['not chinese'] for _ in x])
        pinyin_locs = {}
        # get pinyin of each location
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]
            # not a Chinese character, pass
            if pinyin_string == "not chinese":
                continue
            pinyin_locs[index] = self.pho_convertor.get_sm_ym_sd_labels(pinyin_string)

        # find chinese character location, and generate pinyin ids
        pinyin_labels = []
        for idx, (token, offset) in enumerate(zip(tokenizer_output.tokens, tokenizer_output.offsets)):
            if offset[1] - offset[0] != 1:
                pinyin_labels.append((0, 0, 0))
                continue
            if offset[0] in pinyin_locs:
                pinyin_labels.append(pinyin_locs[offset[0]])
            else:
                pinyin_labels.append((0, 0, 0))

        return pinyin_labels


class BertMLP(nn.Module):
    def __init__(self, config, ):
        super().__init__()
        self.dense_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_to_labels_layer = nn.Linear(config.hidden_size, config.num_labels)
        self.activation = nn.Tanh()

    def forward(self, sequence_hidden_states):
        sequence_output = self.dense_layer(sequence_hidden_states)
        sequence_output = self.activation(sequence_output)
        sequence_output = self.dense_to_labels_layer(sequence_output)
        return sequence_output


class FusionBertEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position, glyph, pinyin and token_type embeddings.
    """

    def __init__(self, config):
        super(FusionBertEmbeddings, self).__init__()
        config_path = os.path.join(config.name_or_path, 'config')
        font_files = []
        for file in os.listdir(config_path):
            if file.endswith(".npy"):
                font_files.append(os.path.join(config_path, file))
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.pinyin_embeddings = PinyinEmbedding(embedding_size=128, pinyin_out_dim=config.hidden_size,
                                                 config_path=config_path)
        self.glyph_embeddings = GlyphEmbedding(font_npy_files=font_files)

        # self.LayerNorm is not snake-cased to stick with TensorFlow models variable name and be able to load
        # any TensorFlow checkpoint file
        self.glyph_map = nn.Linear(1728, config.hidden_size)
        self.map_fc = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, pinyin_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # get char embedding, pinyin embedding and glyph embedding
        word_embeddings = inputs_embeds  # [bs,l,hidden_size]
        pinyin_embeddings = self.pinyin_embeddings(pinyin_ids)  # [bs,l,hidden_size]
        glyph_embeddings = self.glyph_map(self.glyph_embeddings(input_ids))  # [bs,l,hidden_size]
        # fusion layer
        concat_embeddings = torch.cat((word_embeddings, pinyin_embeddings, glyph_embeddings), 2)
        inputs_embeds = self.map_fc(concat_embeddings)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class GlyphEmbedding(nn.Module):
    """Glyph2Image Embedding"""

    def __init__(self, font_npy_files: List[str]):
        super(GlyphEmbedding, self).__init__()
        font_arrays = [
            np.load(np_file).astype(np.float32) for np_file in font_npy_files
        ]
        self.vocab_size = font_arrays[0].shape[0]
        self.font_num = len(font_arrays)
        self.font_size = font_arrays[0].shape[-1]
        # N, C, H, W
        font_array = np.stack(font_arrays, axis=1)
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.font_size ** 2 * self.font_num,
            _weight=torch.from_numpy(font_array.reshape([self.vocab_size, -1]))
        )

    def forward(self, input_ids):
        """
            get glyph images for batch inputs
        Args:
            input_ids: [batch, sentence_length]
        Returns:
            images: [batch, sentence_length, self.font_num*self.font_size*self.font_size]
        """
        # return self.embedding(input_ids).view([-1, self.font_num, self.font_size, self.font_size])
        return self.embedding(input_ids)


class PinyinEmbedding(nn.Module):
    def __init__(self, embedding_size: int, pinyin_out_dim: int, config_path):
        """
            Pinyin Embedding Module
        Args:
            embedding_size: the size of each embedding vector
            pinyin_out_dim: kernel number of conv
        """
        super(PinyinEmbedding, self).__init__()
        with open(os.path.join(config_path, 'pinyin_map.json')) as fin:
            pinyin_dict = json.load(fin)
        self.pinyin_out_dim = pinyin_out_dim
        self.embedding = nn.Embedding(len(pinyin_dict['idx2char']), embedding_size)
        self.conv = nn.Conv1d(in_channels=embedding_size, out_channels=self.pinyin_out_dim, kernel_size=2,
                              stride=1, padding=0)

    def forward(self, pinyin_ids):
        """
        Args:
            pinyin_ids: (bs*sentence_length*pinyin_locs)

        Returns:
            pinyin_embed: (bs,sentence_length,pinyin_out_dim)
        """
        # input pinyin ids for 1-D conv
        embed = self.embedding(pinyin_ids)  # [bs,sentence_length,pinyin_locs,embed_size]
        bs, sentence_length, pinyin_locs, embed_size = embed.shape
        view_embed = embed.view(-1, pinyin_locs, embed_size)  # [(bs*sentence_length),pinyin_locs,embed_size]
        input_embed = view_embed.permute(0, 2, 1)  # [(bs*sentence_length), embed_size, pinyin_locs]
        # conv + max_pooling
        pinyin_conv = self.conv(input_embed)  # [(bs*sentence_length),pinyin_out_dim,H]
        pinyin_embed = F.max_pool1d(pinyin_conv, pinyin_conv.shape[-1])  # [(bs*sentence_length),pinyin_out_dim,1]
        return pinyin_embed.view(bs, sentence_length, self.pinyin_out_dim)  # [bs,sentence_length,pinyin_out_dim]


class GlyceBertModel(BertModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the models.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the models at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        models = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = models(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """

    def __init__(self, config):
        super(GlyceBertModel, self).__init__(config)
        self.config = config

        self.embeddings = FusionBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            pinyin_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the models is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the models is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, pinyin_ids=pinyin_ids, position_ids=position_ids, token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def forward_with_embedding(
            self,
            input_ids=None,
            pinyin_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            embedding=None
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the models is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the models is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        assert embedding is not None
        embedding_output = embedding
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class GlyceBertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super(GlyceBertForMaskedLM, self).__init__(config)

        self.bert = GlyceBertModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
            self,
            input_ids=None,
            pinyin_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert "lm_labels" not in kwargs, "Use `BertWithLMHead` for autoregressive language modeling task."
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            pinyin_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class GlyceBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = GlyceBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            pinyin_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            pinyin_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class GlyceBertForQuestionAnswering(BertPreTrainedModel):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.

    Outputs:
        if `start_positions` and `end_positions` are not `None`:
            Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
        if `start_positions` or `end_positions` is `None`:
            Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
            position tokens of shape [batch_size, sequence_length].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = GlyceBertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            pinyin_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            pinyin_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class GlyceBertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config, mlp=False):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = GlyceBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if mlp:
            self.classifier = BertMLP(config)
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self,
                input_ids=None,
                pinyin_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):
        r"""
                labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                    Labels for computing the token classification loss.
                    Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
                """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            pinyin_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep the active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Pinyin(object):
    """docstring for Pinyin"""

    def __init__(self):
        super(Pinyin, self).__init__()
        self.shengmu = ['zh', 'ch', 'sh', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 'r',
                        'z', 'c', 's', 'y', 'w']
        self.yunmu = ['a', 'ai', 'an', 'ang', 'ao', 'e', 'ei', 'en', 'eng', 'er', 'i', 'ia', 'ian', 'iang', 'iao', 'ie',
                      'in', 'ing', 'iong', 'iu', 'o', 'ong', 'ou', 'u', 'ua', 'uai', 'uan', 'uang', 'ue', 'ui', 'un',
                      'uo', 'v', 've']
        self.shengdiao = ['1', '2', '3', '4', '5']
        self.sm_size = len(self.shengmu) + 1
        self.ym_size = len(self.yunmu) + 1
        self.sd_size = len(self.shengdiao) + 1

    def get_sm_ym_sd(self, pinyin):
        s = pinyin
        assert isinstance(s, str), 'input of function get_sm_ym_sd is not string'
        assert s[-1] in '12345', f'input of function get_sm_ym_sd is not valid,{s}'
        sm, ym, sd = None, None, None
        for c in self.shengmu:
            if s.startswith(c):
                sm = c
                break
        if sm == None:
            ym = s[:-1]
        else:
            ym = s[len(sm):-1]
        sd = s[-1]
        return sm, ym, sd

    def get_sm_ym_sd_labels(self, pinyin):
        sm, ym, sd = self.get_sm_ym_sd(pinyin)
        return self.shengmu.index(sm) + 1 if sm in self.shengmu else 0, \
               self.yunmu.index(ym) + 1 if ym in self.yunmu else 0, \
               self.shengdiao.index(sd) + 1 if sd in self.shengdiao else 0

    def get_pinyinstr(self, sm_ym_sd_label):
        sm, ym, sd = sm_ym_sd_label
        sm -= 1
        ym -= 1
        sd -= 1
        sm = self.shengmu[sm] if sm >= 0 else ''
        ym = self.yunmu[ym] if ym >= 0 else ''
        sd = self.shengdiao[sd] if sd >= 0 else ''
        return sm + ym + sd


class Phonetic_Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pinyin = Pinyin()
        self.transform = BertPredictionHeadTransform(config)
        self.sm_classifier = nn.Linear(config.hidden_size, self.pinyin.sm_size)
        self.ym_classifier = nn.Linear(config.hidden_size, self.pinyin.ym_size)
        self.sd_classifier = nn.Linear(config.hidden_size, self.pinyin.sd_size)

    def forward(self, sequence_output):
        sequence_output = self.transform(sequence_output)
        sm_scores = self.sm_classifier(sequence_output)
        ym_scores = self.ym_classifier(sequence_output)
        sd_scores = self.sd_classifier(sequence_output)
        return sm_scores, ym_scores, sd_scores


class Pinyin_Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.classifier = nn.Linear(config.hidden_size, 1378)

    def forward(self, sequence_output):
        sequence_output = self.transform(sequence_output)
        scores = self.classifier(sequence_output)
        return scores


class MultiTaskHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.Phonetic_relationship = Phonetic_Classifier(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        sm_scores, ym_scores, sd_scores = self.Phonetic_relationship(sequence_output)
        return prediction_scores, sm_scores, ym_scores, sd_scores


class AblationHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.Phonetic_relationship = Pinyin_Classifier(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        pinyin_scores = self.Phonetic_relationship(sequence_output)
        return prediction_scores, pinyin_scores


class GlyceBertForMultiTask(BertPreTrainedModel):
    def __init__(self, config):
        super(GlyceBertForMultiTask, self).__init__(config)

        self.bert = GlyceBertModel(config)
        self.cls = MultiTaskHeads(config)
        self.loss_fct = CrossEntropyLoss()

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
            self,
            input_ids=None,
            pinyin_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            pinyin_labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            gamma=1,
            **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert "lm_labels" not in kwargs, "Use `BertWithLMHead` for autoregressive language modeling task."
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        loss_mask = (input_ids != 0) * (input_ids != 101) * (input_ids != 102).long()
        outputs = self.bert(
            input_ids,
            pinyin_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores, sm_scores, ym_scores, sd_scores = self.cls(sequence_output)

        masked_lm_loss = None
        loss_fct = self.loss_fct  # -100 index = padding token
        if labels is not None:
            active_loss = loss_mask.view(-1) == 1
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), active_labels)

        phonetic_loss = None
        if pinyin_labels is not None:
            active_loss = loss_mask.view(-1) == 1
            active_labels = torch.where(
                active_loss, pinyin_labels[..., 0].view(-1), torch.tensor(loss_fct.ignore_index).type_as(pinyin_labels)
            )
            sm_loss = loss_fct(sm_scores.view(-1, self.cls.Phonetic_relationship.pinyin.sm_size), active_labels)
            active_labels = torch.where(
                active_loss, pinyin_labels[..., 1].view(-1), torch.tensor(loss_fct.ignore_index).type_as(pinyin_labels)
            )
            ym_loss = loss_fct(ym_scores.view(-1, self.cls.Phonetic_relationship.pinyin.ym_size), active_labels)
            active_labels = torch.where(
                active_loss, pinyin_labels[..., 2].view(-1), torch.tensor(loss_fct.ignore_index).type_as(pinyin_labels)
            )
            sd_loss = loss_fct(sd_scores.view(-1, self.cls.Phonetic_relationship.pinyin.sd_size), active_labels)
            phonetic_loss = (sm_loss + ym_loss + sd_loss) / 3

        loss = None
        if masked_lm_loss is not None:
            loss = masked_lm_loss
            if phonetic_loss is not None:
                loss += phonetic_loss * gamma

        if not return_dict:
            output = (prediction_scores, sm_scores, ym_scores, sd_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states_a,
            hidden_states_b,
            attention_mask=None,
            output_attentions=False,
    ):
        query_layer = self.query(hidden_states_a)
        key_layer = self.transpose_for_scores(self.key(hidden_states_b))
        value_layer = self.transpose_for_scores(self.value(hidden_states_b))

        query_layer = self.transpose_for_scores(query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        outputs = context_layer

        return outputs


class Dynamic_GlyceBertForMultiTask(BertPreTrainedModel):
    def __init__(self, config):
        super(Dynamic_GlyceBertForMultiTask, self).__init__(config)

        self.bert = GlyceBertModel(config)
        self.cls = MultiTaskHeads(config)
        self.loss_fct = CrossEntropyLoss(reduction='none')

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
            self,
            input_ids=None,
            pinyin_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            tgt_pinyin_ids=None,
            pinyin_labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            gamma=1,
            var=1,
            **kwargs
    ):

        assert "lm_labels" not in kwargs, "Use `BertWithLMHead` for autoregressive language modeling task."
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        loss_mask = (input_ids != 0) * (input_ids != 101) * (input_ids != 102).long()
        outputs_x = self.bert(
            input_ids,
            pinyin_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        encoded_x = outputs_x[0]
        if tgt_pinyin_ids is not None:
            with torch.no_grad():
                outputs_y = self.bert(
                    labels,
                    tgt_pinyin_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                encoded_y = outputs_y[0]
                pron_x = self.cls.Phonetic_relationship.transform(encoded_x)
                pron_y = self.cls.Phonetic_relationship.transform(encoded_y)  # [bs, seq, hidden_states]
                assert pron_x.shape == pron_y.shape
                sim_xy = F.cosine_similarity(pron_x, pron_y, dim=-1)  # [ns, seq]
                factor = torch.exp(-((sim_xy - 1.0) / var).pow(2)).detach()

        prediction_scores, sm_scores, ym_scores, sd_scores = self.cls(encoded_x)

        masked_lm_loss = None
        phonetic_loss = None
        loss_fct = self.loss_fct  # -100 index = padding token
        if labels is not None and pinyin_labels is not None:
            active_loss = loss_mask.view(-1) == 1

            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), active_labels)

            active_labels = torch.where(
                active_loss, pinyin_labels[..., 0].view(-1), torch.tensor(loss_fct.ignore_index).type_as(pinyin_labels)
            )
            sm_loss = loss_fct(sm_scores.view(-1, self.cls.Phonetic_relationship.pinyin.sm_size), active_labels)

            active_labels = torch.where(
                active_loss, pinyin_labels[..., 1].view(-1), torch.tensor(loss_fct.ignore_index).type_as(pinyin_labels)
            )
            ym_loss = loss_fct(ym_scores.view(-1, self.cls.Phonetic_relationship.pinyin.ym_size), active_labels)

            active_labels = torch.where(
                active_loss, pinyin_labels[..., 2].view(-1), torch.tensor(loss_fct.ignore_index).type_as(pinyin_labels)
            )
            sd_loss = loss_fct(sd_scores.view(-1, self.cls.Phonetic_relationship.pinyin.sd_size), active_labels)
            phonetic_loss = (sm_loss + ym_loss + sd_loss) / 3

            def weighted_mean(weight, input):
                return torch.sum(weight * input) / torch.sum(weight)

            masked_lm_loss = weighted_mean(torch.ones_like(masked_lm_loss), masked_lm_loss)
            phonetic_loss = weighted_mean(factor.view(-1), phonetic_loss)

        loss = None
        if masked_lm_loss is not None and phonetic_loss is not None:
            loss = masked_lm_loss
            loss += phonetic_loss * gamma

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs_x.hidden_states,
            attentions=outputs_x.attentions,
        )
