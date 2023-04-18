import math

import lightning.pytorch as pl
import pypinyin
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from utils.str_utils import is_chinese
from utils.utils import load_obj, mock_args, predict_process


class AdjustProbByPinyin(pl.LightningModule):

    def __init__(self, args, pinyin_distance_filepath='./ptm/pinyin_distances.pkl',
                 bert_path='hfl/chinese-roberta-wwm-ext'):
        super().__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.model = AutoModelForMaskedLM.from_pretrained(bert_path)

        self.vocab_pinyin = self._init_vocab_pinyin()

        # self.pinyin_distance_dict: dict = load_obj(pinyin_distance_filepath)
        # for key in list(self.pinyin_distance_dict.keys()):
        #     self.pinyin_distance_dict[key.replace("0", "")] = self.pinyin_distance_dict.pop(key)

    def _init_vocab_pinyin(self):
        vocab_pinyin = []
        for i in range(len(self.tokenizer)):
            token = self.tokenizer._convert_id_to_token(i)

            if not is_chinese(token):
                vocab_pinyin.append(None)
                continue

            pinyin = pypinyin.pinyin(token, style=pypinyin.Style.TONE3)[0][0]
            vocab_pinyin.append(pinyin)
        return vocab_pinyin

    def get_pinyin_sims(self, pinyin):
        """
        计算当前拼音和vocab中的拼音的相似度。
        """
        sims = []
        for v_pinyin in self.vocab_pinyin:
            key = '%s,%s' % (pinyin, v_pinyin)
            if key not in self.pinyin_distance_dict:
                sims.append(0)
                continue
            else:
                x = self.pinyin_distance_dict[key]
                sim = 2 * (math.e ** (-0.03 * x)) - 1  # 这三个参数可以通过网络进行训练
                sims.append(sim)

        return torch.Tensor(sims).to(self.args.device)

    def get_simple_pinyin_sims(self, pinyin):
        """
        计算当前拼音和vocab中的拼音的相似度。
        """
        sims = []
        for v_pinyin in self.vocab_pinyin:
            if v_pinyin is None:
                sims.append(0)
                continue

            v_pinyin = v_pinyin[:-1]
            length = max(len(pinyin), len(v_pinyin))
            sim_len = 0
            for i in range(min(len(pinyin), len(v_pinyin))):
                if pinyin[i] == v_pinyin[i]:
                    sim_len += 1
            sims.append(5 * sim_len / length)

        return torch.Tensor(sims).to(self.args.device)

    def predict(self, sentence):
        sentence = sentence.replace(" ", "").strip()

        inputs = self.tokenizer(' '.join(list(sentence)), return_tensors='pt').to(self.args.device)
        logits = self.model(**inputs).logits[0][1:-1]

        for i, token in enumerate(sentence):
            std = logits[i].std()
            pinyin = pypinyin.pinyin(token, style=pypinyin.Style.TONE3)[0][0]
            sims = self.get_pinyin_sims(pinyin)

            # 计算两个pinyin的相似度：
            # self.pinyin_distance_dict[key]

            # 获取第i个字的最相似的那些字
            # self.tokenizer.convert_ids_to_tokens(sims.argsort(descending=True)[:15])

            # 获取第i个字的可能的取值
            # self.tokenizer.convert_ids_to_tokens(logits[i].argsort(descending=True)[:15])
            # 获取第i个字的可能取值对应的输出
            # logits[i].sort(descending=True).values[:15]

            # 拼音相同的，logits加1个标准差，拼音相似的，logits加0.x个标准差，拼音完全不相似的，不加标准差
            logits[i] = logits[i] + sims * std

            # token_index = inputs['input_ids'][0][i + 1]
            # logits[i][token_index] = logits[i][token_index] + std  # 本身这个字再加1个标准差，防止把正确的字变成错误的字。

        return ''.join(self.tokenizer.convert_ids_to_tokens(logits.argmax(-1)))

    def predict2(self, sent_tokens, pinyins):
        inputs = self.tokenizer(' '.join(sent_tokens), return_tensors='pt').to(self.args.device)
        logits = self.model(**inputs).logits[0][1:-1]

        for i, token in enumerate(sent_tokens):
            std = logits[i].std()
            # pinyin = pypinyin.pinyin(token, style=pypinyin.Style.TONE3)[0][0]
            pinyin = pinyins[i]
            sims = self.get_simple_pinyin_sims(pinyin)

            # 拼音相同的，logits加1个标准差，拼音相似的，logits加0.x个标准差，拼音完全不相似的，不加标准差
            logits[i] = logits[i] + sims * std

            # token_index = inputs['input_ids'][0][i + 1]
            # logits[i][token_index] = logits[i][token_index] + std  # 本身这个字再加1个标准差，防止把正确的字变成错误的字。

        pred_tokens = self.tokenizer.convert_ids_to_tokens(logits.argmax(-1))
        return predict_process(sent_tokens, pred_tokens)

    def predict3(self, sentence):
        sentence = sentence.replace(" ", "").strip()

        inputs = self.tokenizer(' '.join(list(sentence)), return_tensors='pt').to(self.args.device)
        logits = self.model(**inputs).logits[0][1:-1]

        for i, token in enumerate(sentence):
            std = logits[i].std()
            pinyin = pypinyin.pinyin(token, style=pypinyin.Style.NORMAL)[0][0]
            sims = self.get_simple_pinyin_sims(pinyin)

            # 计算两个pinyin的相似度：
            # self.pinyin_distance_dict[key]

            # 获取第i个字的最相似的那些字
            # self.tokenizer.convert_ids_to_tokens(sims.argsort(descending=True)[:15])

            # 获取第i个字的可能的取值
            # self.tokenizer.convert_ids_to_tokens(logits[i].argsort(descending=True)[:15])
            # 获取第i个字的可能取值对应的输出
            # logits[i].sort(descending=True).values[:15]

            # 拼音相同的，logits加1个标准差，拼音相似的，logits加0.x个标准差，拼音完全不相似的，不加标准差
            logits[i] = logits[i] + sims * std

            # token_index = inputs['input_ids'][0][i + 1]
            # logits[i][token_index] = logits[i][token_index] + std  # 本身这个字再加1个标准差，防止把正确的字变成错误的字。

        return ''.join(self.tokenizer.convert_ids_to_tokens(logits.argmax(-1)))

    def test_step(self, batch, batch_idx: int, *args, **kwargs):
        src, tgt = batch
        pred = []

        for sent, label in zip(src, tgt):
            # If I told the sent which token is wrong.
            sent_tokens = sent.split(" ")
            label_tokens = label.split(" ")
            pinyins = []
            for i in range(len(sent_tokens)):
                pinyin = pypinyin.pinyin(sent_tokens[i], style=pypinyin.Style.TONE3)[0][0]
                pinyins.append(pinyin)

                if sent_tokens[i] != label_tokens[i]:
                    sent_tokens[i] = '[MASK]'

            pred.append(self.predict2(sent_tokens, pinyins))
        return pred

if __name__ == '__main__':
    print(AdjustProbByPinyin(mock_args(device='cpu'), pinyin_distance_filepath='../ptm/pinyin_distances.pkl').predict3(
        "陈先生会说三种语言，西班牙语、华语根日文。"))
