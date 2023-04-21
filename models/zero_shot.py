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

        # self.vocab_pinyin = self._init_vocab_pinyin()
        self.vocab_pinyin = self._init_vocab_pinyin2()

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

    def _init_vocab_pinyin2(self):
        vocab_pinyin = []
        for i in range(len(self.tokenizer)):
            token = self.tokenizer._convert_id_to_token(i)

            if not is_chinese(token):
                vocab_pinyin.append(None)
                continue

            initial = pypinyin.pinyin(token, style=pypinyin.Style.INITIALS, strict=False)[0][0]
            final = pypinyin.pinyin(token, style=pypinyin.Style.FINALS_TONE3, strict=False)[0][0]
            final = final.rstrip("1234567890")
            vocab_pinyin.append((initial, final))
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

    def get_simple_pinyin_sims2(self, pinyin):
        """
        计算当前拼音和vocab中的拼音的相似度。
        """
        sims = []
        for v_pinyin in self.vocab_pinyin:
            if v_pinyin is None:
                sims.append(0)
                continue

            v_initial, v_final = v_pinyin
            initial, final = pinyin
            length = max(len(v_initial) + len(v_final), len(initial) + len(final))

            sim_len = 0

            for i in range(min(len(v_initial), len(initial))):
                if v_initial[i] == initial[i]:
                    sim_len += 1

            for i in range(min(len(v_final), len(final))):
                if v_final[i] == final[i]:
                    sim_len += 1

            sims.append(self.args.hyper_params['sim_times'] * sim_len / length)

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

            sims = self.get_simple_pinyin_sims2(pinyin)
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

    def tell_mask(self, sent_tokens, label):
        # If I told the sent which token is wrong.
        label_tokens = label.split(" ")
        pinyins = []
        for i in range(len(sent_tokens)):
            # pinyin = pypinyin.pinyin(sent_tokens[i], style=pypinyin.Style.TONE3)[0][0]
            initial = pypinyin.pinyin(sent_tokens[i], style=pypinyin.Style.INITIALS, strict=False)[0][0]
            final = pypinyin.pinyin(sent_tokens[i], style=pypinyin.Style.FINALS_TONE3, strict=False)[0][0]
            final = final.rstrip("1234567890")
            pinyins.append((initial, final))

            if sent_tokens[i] != label_tokens[i]:
                sent_tokens[i] = '[MASK]'
        return sent_tokens, pinyins

    def predict_mask(self, sent_tokens):
        inputs = self.tokenizer(' '.join(sent_tokens), return_tensors='pt').to(self.args.device)
        logits = self.model(**inputs).logits[0][1:-1]
        pred_tokens = self.tokenizer.convert_ids_to_tokens(logits.argmax(-1))
        pinyins = []
        for i in range(len(sent_tokens)):
            # pinyin = pypinyin.pinyin(sent_tokens[i], style=pypinyin.Style.TONE3)[0][0]
            initial = pypinyin.pinyin(sent_tokens[i], style=pypinyin.Style.INITIALS, strict=False)[0][0]
            final = pypinyin.pinyin(sent_tokens[i], style=pypinyin.Style.FINALS_TONE3, strict=False)[0][0]
            final = final.rstrip("1234567890")
            pinyins.append((initial, final))

            if sent_tokens[i] != pred_tokens[i]:
                sent_tokens[i] = '[MASK]'

        return sent_tokens, pinyins

    def detect_predict(self, sent_tokens):
        inputs = self.tokenizer(' '.join(sent_tokens), return_tensors='pt').to(self.args.device)
        logits = self.model(**inputs).logits[0][1:-1]

        for i, index in enumerate(inputs['input_ids'][0][1:-1]):
            std = logits[i].std()
            # 把原有字降低一个std的置信度
            logits[i, index] = logits[i, index] - std

        pred_tokens = self.tokenizer.convert_ids_to_tokens(logits.argmax(-1))

        for i in range(len(sent_tokens)):
            if sent_tokens[i] != pred_tokens[i]:
                sent_tokens[i] = '?'

        return ''.join(sent_tokens)

    def detect_predict2(self, sent_tokens):
        inputs = self.tokenizer(' '.join(sent_tokens), return_tensors='pt').to(self.args.device)
        bs, seq_num = inputs['input_ids'].size()
        assert bs == 1, "Sorry, Batch size must be 1"
        attention_mask = (~torch.eye(seq_num, dtype=bool)).int()
        attention_mask[0, 0] = 1
        attention_mask[-1, -1] = 1
        # inputs['attention_mask'] = attention_mask
        inputs['encoder_attention_mask'] = attention_mask
        logits = self.model(**inputs).logits[0][1:-1]

        pred_tokens = self.tokenizer.convert_ids_to_tokens(logits.argmax(-1))
        # for i in range(len(sent_tokens)):
        #     if sent_tokens[i] != pred_tokens[i]:
        #         sent_tokens[i] = '?'

        return ''.join(pred_tokens)

    # def test_step(self, batch, batch_idx: int, *args, **kwargs):
    #     src, tgt = batch
    #     pred = []
    #
    #     for sent, label in zip(src, tgt):
    #         sent_tokens = sent.split(" ")
    #         # sent_tokens, pinyins = self.tell_mask(sent_tokens, label)
    #         sent_tokens, pinyins = self.predict_mask(sent_tokens)
    #
    #         pred.append(self.predict2(sent_tokens, pinyins))
    #     return pred

    def predict4(self, sent_tokens):
        pred_tokens = [item for item in sent_tokens]
        for i in range(len(sent_tokens)):
            input_tokens = [item for item in sent_tokens]
            token = input_tokens[i]
            input_tokens[i] = '[MASK]'

            inputs = self.tokenizer(' '.join(input_tokens), return_tensors='pt').to(self.args.device)
            logits = self.model(**inputs).logits[0][1:-1]

            std = logits[i].std()
            initial = pypinyin.pinyin(token, style=pypinyin.Style.INITIALS, strict=False)[0][0]
            final = pypinyin.pinyin(token, style=pypinyin.Style.FINALS_TONE3, strict=False)[0][0]
            final = final.rstrip("1234567890")
            sims = self.get_simple_pinyin_sims2((initial, final))

            # 查看哪些拼音与其比较相似
            # self.tokenizer.convert_ids_to_tokens(sims.argsort(descending=True)[:15])

            # 获取第i个字的可能的取值
            # self.tokenizer.convert_ids_to_tokens(logits[i].argsort(descending=True)[:15])
            # 获取第i个字的可能取值对应的输出
            # logits[i].sort(descending=True).values[:15]

            logits[i] = logits[i] + sims * std

            """
            # 通过候选值的方式增加精准率
            candidate_tokens = self.tokenizer.convert_ids_to_tokens(logits[i].argsort(descending=True)[:3])
            if token in candidate_tokens:
                # 如果调整过后，发现原来的字在候选值中，则不修改该字
                pred_tokens[i] = token
            else:
                # 否则则使用修改后的字
                pred_tokens[i] = self.tokenizer._convert_id_to_token(logits[i].argmax(-1))
            """

            # 再给原始字增加些概率
            token_index = self.tokenizer.convert_tokens_to_ids(token)
            logits[i, token_index] = logits[i, token_index] + std * self.args.hyper_params['token_times']
            pred_tokens[i] = self.tokenizer._convert_id_to_token(logits[i].argmax(-1))

        return predict_process(sent_tokens, pred_tokens)

    def test_step(self, batch, batch_idx: int, *args, **kwargs):
        """
        test for detection
        """
        src, tgt = batch
        preds = []

        for sent, label in zip(src, tgt):
            sent_tokens = sent.split(" ")
            preds.append(self.predict4(sent_tokens))
        return preds


if __name__ == '__main__':
    sent_tokens = list("吃了早菜以后他去上课。")
    # sent_tokens = "[MASK] 再 也 不 会 [PAD] 扬 。 [PAD]".split(' ')
    # print(
        # AdjustProbByPinyin(mock_args(device='cpu'),
        #                    pinyin_distance_filepath='../ptm/pinyin_distances.pkl').predict4(
        #     sent_tokens))

    print(AdjustProbByPinyin(mock_args(device='cpu')).predict4(sent_tokens))
