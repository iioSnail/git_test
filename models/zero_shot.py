import copy
import math

import lightning.pytorch as pl
import numpy as np
import pypinyin
import torch
from matplotlib import pyplot as plt
from transformers import AutoTokenizer, AutoModelForMaskedLM

from utils.pinyin_utils import pinyin_is_sim
from utils.str_utils import is_chinese, pinyin_distance, to_full_pinyin, to_pinyin
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

        self.c_logits = []
        self.e_logits = []

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

    def get_simple_pinyin_sims2(self, pinyin, sim_times=5.):
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

            sims.append(sim_times * sim_len / length)

        return torch.Tensor(sims).to(self.args.device)

    def get_simple_pinyin_sims3(self, pinyin, sim_times=5.):
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
            score = 0
            # 如果声母完全相同，+2分
            if v_initial == initial:
                score += 2
            # 如果声母只有第一个字母相同，则+1分。例如：zh，z
            elif len(v_initial) > 0 and len(initial) > 0 and v_initial[0] == initial[0]:
                score += 1

            # 如果韵母完全相同，+2分
            if v_final == final:
                score += 2
            # 如果韵母不同，但存在交集，+1分
            elif set(v_final) & set(final):
                score += 1

            # 如果声母和韵母就没相同的，则分数归零
            if v_initial != initial and v_final != final:
                score = 0

            sims.append(sim_times * score / 4)  # 4为总分

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
            if token != '[MASK]':
                token_index = inputs['input_ids'][0][i + 1]
                # FXIME，若不是[MASK]，则不对其进行预测。
                logits[i][token_index] = logits[i][token_index] + 9999
                continue

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

    def predict2_2(self, sent_tokens, pinyins):
        """
        相比predict2.0，增加对logit的筛选，若logit小于10，则不考虑
        这个idea不太行
        """
        inputs = self.tokenizer(' '.join(sent_tokens), return_tensors='pt').to(self.args.device)
        logits = self.model(**inputs).logits[0][1:-1]

        logits[logits < -100] = 10  # logit小于10的直接不考虑

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

    def predict2_3(self, sent_tokens, pinyins):
        """
        更新：对拼音完全不相似的，logit直接归0
        """
        inputs = self.tokenizer(' '.join(sent_tokens), return_tensors='pt').to(self.args.device)
        logits = self.model(**inputs).logits[0][1:-1]

        for i, token in enumerate(sent_tokens):
            if token != '[MASK]':
                token_index = inputs['input_ids'][0][i + 1]
                # FXIME，若不是[MASK]，则不对其进行预测。
                logits[i][token_index] = logits[i][token_index] + 9999
                continue

            std = logits[i].std()
            # pinyin = pypinyin.pinyin(token, style=pypinyin.Style.TONE3)[0][0]
            pinyin = pinyins[i]

            sims = self.get_simple_pinyin_sims2(pinyin, sim_times=3.5)
            # 拼音相同的，logits加1个标准差，拼音相似的，logits加0.x个标准差
            logits[i] = logits[i] + sims * std
            # 拼音完全不相似的，直接归零
            logits[i] = logits[i] * (sims != 0).int()

            # token_index = inputs['input_ids'][0][i + 1]
            # logits[i][token_index] = logits[i][token_index] + std  # 本身这个字再加1个标准差，防止把正确的字变成错误的字。

        pred_tokens = self.tokenizer.convert_ids_to_tokens(logits.argmax(-1))
        return predict_process(sent_tokens, pred_tokens)

    def predict2_4(self, sent_tokens, pinyins):
        """
        更新：使用了新的sims函数
        """
        inputs = self.tokenizer(' '.join(sent_tokens), return_tensors='pt').to(self.args.device)
        logits = self.model(**inputs).logits[0][1:-1]

        for i, token in enumerate(sent_tokens):
            if token != '[MASK]':
                token_index = inputs['input_ids'][0][i + 1]
                # FXIME，若不是[MASK]，则不对其进行预测。
                logits[i][token_index] = logits[i][token_index] + 9999
                continue

            std = logits[i].std()
            # pinyin = pypinyin.pinyin(token, style=pypinyin.Style.TONE3)[0][0]
            pinyin = pinyins[i]

            sims = self.get_simple_pinyin_sims3(pinyin, sim_times=3.5)
            # 拼音相同的，logits加1个标准差，拼音相似的，logits加0.x个标准差
            logits[i] = logits[i] + sims * std
            # 拼音完全不相似的，直接归零
            logits[i] = logits[i] * (sims != 0).int()

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

    def predict5(self, sent_tokens, pinyins):
        """
        “只对[MASK]”部分进行预测
        在候选值中，从大到小选择，选择第一个韵母或声母有其中之一相同的字
        这个idea还行，需要进一步优化
        """
        inputs = self.tokenizer(' '.join(sent_tokens), return_tensors='pt').to(self.args.device)
        logits = self.model(**inputs).logits[0][1:-1]
        pred_tokens = copy.deepcopy(sent_tokens)

        for i, token in enumerate(sent_tokens):
            if token != '[MASK]':
                continue

            initial, final = pinyins[i]
            candidates = self.tokenizer.convert_ids_to_tokens(logits[i].argsort(descending=True)[:100])
            for c_token in candidates:
                if not is_chinese(c_token):
                    continue

                c_initial = pypinyin.pinyin(c_token, style=pypinyin.Style.INITIALS, strict=False)[0][0]
                c_final = pypinyin.pinyin(c_token, style=pypinyin.Style.FINALS_TONE3, strict=False)[0][0]
                c_final = c_final.rstrip("1234567890")

                if c_initial == initial or c_final == final:
                    pred_tokens[i] = c_token
                    break

            if pred_tokens[i] == '[MASK]':
                # There is no anyone for fitting, then use the first candidate.
                pred_tokens[i] = candidates[0]

        pred_sentence = predict_process(sent_tokens, pred_tokens)
        return pred_sentence

    def predict5_2(self, sent_tokens, pinyins, label_tokens=None):
        """
        “只对[MASK]”部分进行预测
        在候选值中，从大到小选择，选择第一个韵母或声母有其中之一相同的字

        更新：在5.0的基础上增加了相似度判断。若相似度小于某个值才算，避免(kan, dan)这样明显不一致的音被误判
        """
        inputs = self.tokenizer(' '.join(sent_tokens), return_tensors='pt').to(self.args.device)
        logits = self.model(**inputs).logits[0][1:-1]
        pred_tokens = copy.deepcopy(sent_tokens)

        for i, token in enumerate(sent_tokens):
            if token != '[MASK]':
                continue

            initial, final, tone = pinyins[i]
            pinyin = initial + final + tone
            candidates = self.tokenizer.convert_ids_to_tokens(logits[i].argsort(descending=True)[:100])
            for c_token in candidates:
                if not is_chinese(c_token):
                    continue

                c_initial, c_final, c_tone = to_full_pinyin(c_token, tone=True)
                c_pinyin = c_initial + c_final + c_tone
                if (c_initial == initial or c_final[:2] == final[:2]) and pinyin_distance(pinyin, c_pinyin) < 100:
                    pred_tokens[i] = c_token
                    break

            if pred_tokens[i] == '[MASK]':
                # There is no anyone for fitting, then use the first candidate.
                pred_tokens[i] = candidates[0]

        pred_sentence = predict_process(sent_tokens, pred_tokens)
        return pred_sentence

    def predict5_3(self, sent_tokens, pinyins, label_tokens=None):
        """
        “只对[MASK]”部分进行预测
        在候选值中，从大到小选择，选择第一个韵母或声母有其中之一相同的字

        更新：在5.2，直接使用拼音是否相似的函数
        """
        inputs = self.tokenizer(' '.join(sent_tokens), return_tensors='pt').to(self.args.device)
        logits = self.model(**inputs).logits[0][1:-1]
        pred_tokens = copy.deepcopy(sent_tokens)

        for i, token in enumerate(sent_tokens):
            if token != '[MASK]':
                continue

            initial, final, tone = pinyins[i]
            pinyin = initial + final + tone
            candidates = self.tokenizer.convert_ids_to_tokens(logits[i].argsort(descending=True)[:100])
            for c_token in candidates:
                if not is_chinese(c_token):
                    continue

                c_pinyin = to_pinyin(c_token)
                if pinyin_is_sim(c_pinyin, pinyin):
                    pred_tokens[i] = c_token
                    break

            if pred_tokens[i] == '[MASK]':
                # There is no anyone for fitting, then use the first candidate.
                pred_tokens[i] = candidates[0]

        pred_sentence = predict_process(sent_tokens, pred_tokens)
        return pred_sentence

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

    def tell_mask_full_pinyin(self, sent_tokens, label):
        # If I told the sent which token is wrong.
        label_tokens = label.split(" ")
        pinyins = []
        for i in range(len(sent_tokens)):
            pinyins.append(to_full_pinyin(sent_tokens[i], tone=True))

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

    def detect_predict3(self, sent_tokens):
        """
        根据logits判断某个字是否是错字。
        当当前字的logit小于某个值时，则认为它是错字
        """
        logit_threshold = 10  # 越低，召回率越低，精准率越高

        pred_tokens = [item for item in sent_tokens]
        for i in range(len(sent_tokens)):
            input_tokens = [item for item in sent_tokens]
            token = input_tokens[i]
            input_tokens[i] = '[MASK]'

            inputs = self.tokenizer(' '.join(input_tokens), return_tensors='pt').to(self.args.device)
            logits = self.model(**inputs).logits[0][1:-1]

            # 获取第i个字的可能的取值
            # self.tokenizer.convert_ids_to_tokens(logits[i].argsort(descending=True)[:15])
            # 获取第i个字的可能取值对应的输出
            # logits[i].sort(descending=True).values[:15]

            token_index = self.tokenizer.convert_tokens_to_ids(token)
            if logits[i][token_index] < logit_threshold:
                pred_tokens[i] = '×'

            # pred_tokens[i] = self.tokenizer._convert_id_to_token(logits[i].argmax(-1))

        return ''.join(pred_tokens)

    def detect_predict4(self, sent_tokens):
        """
        根据候选值来判断是否是错字。
        逐个将每个字进行[MASK]，当该字的前n名不存在该字时，则认为该字时错字。
        """
        n = 3  # n越低，召回率越高，但是精准率越低

        pred_tokens = [item for item in sent_tokens]
        for i in range(len(sent_tokens)):
            input_tokens = [item for item in sent_tokens]
            token = input_tokens[i]
            input_tokens[i] = '[MASK]'

            inputs = self.tokenizer(' '.join(input_tokens), return_tensors='pt').to(self.args.device)
            logits = self.model(**inputs).logits[0][1:-1]

            # 获取第i个字的可能的取值
            # self.tokenizer.convert_ids_to_tokens(logits[i].argsort(descending=True)[:15])
            # 获取第i个字的可能取值对应的输出
            # logits[i].sort(descending=True).values[:15]

            candidates = self.tokenizer.convert_ids_to_tokens(logits[i].argsort(descending=True)[:n])
            if token not in candidates:
                pred_tokens[i] = '×'

        pred = ''.join(pred_tokens)
        return pred

    def predict4(self, sent_tokens):
        pred_tokens = [item for item in sent_tokens]
        for i in range(len(sent_tokens)):
            input_tokens = [item for item in sent_tokens]
            token = input_tokens[i]
            token_index = self.tokenizer.convert_tokens_to_ids(token)
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

            """
            # 通过logit判断当前字是否为错字。如果当前token的logit较大，则不认为该字是错字
            if logits[i][token_index] > 15: # FIXME 超参，需要调
                continue

            """

            logits[i] = logits[i] + sims * std

            # 通过候选值的方式增加精准率
            candidate_tokens = self.tokenizer.convert_ids_to_tokens(logits[i].argsort(descending=True)[:3])
            if token in candidate_tokens:
                # 如果调整过后，发现原来的字在候选值中，则不修改该字
                pred_tokens[i] = token
            else:
                # 否则则使用修改后的字
                pred_tokens[i] = self.tokenizer._convert_id_to_token(logits[i].argmax(-1))

            # 通过给当前字增加一些概率的方式，判断该字是否是错字
            # 再给原始字增加些概率
            logits[i, token_index] = logits[i, token_index] + std * self.args.hyper_params['token_times']

            pred_tokens[i] = self.tokenizer._convert_id_to_token(logits[i].argmax(-1))

        return predict_process(sent_tokens, pred_tokens)

    def logits_probe(self, sent_tokens, tgt_tokens):
        pred_tokens = [item for item in sent_tokens]
        for i in range(len(sent_tokens)):
            input_tokens = [item for item in sent_tokens]
            token = input_tokens[i]
            input_tokens[i] = '[MASK]'

            inputs = self.tokenizer(' '.join(input_tokens), return_tensors='pt').to(self.args.device)
            logits = self.model(**inputs).logits[0][1:-1]

            # 获取第i个字的可能的取值
            # self.tokenizer.convert_ids_to_tokens(logits[i].argsort(descending=True)[:15])
            # 获取第i个字的可能取值对应的输出
            # logits[i].sort(descending=True).values[:15]

            token_index = self.tokenizer.convert_tokens_to_ids(token)

            logit = logits[i][token_index]
            if token == tgt_tokens[i]:
                self.c_logits.append(logit.item())
            else:
                self.e_logits.append(logit.item())

            pred_tokens[i] = token

        return ''.join(pred_tokens)

    # def test_step(self, batch, batch_idx: int, *args, **kwargs):
    #     """
    #     test for detection
    #     """
    #     src, tgt = batch
    #     preds = []
    #
    #     for sent, label in zip(src, tgt):
    #         sent_tokens = sent.split(" ")
    #         tgt_tokens = label.split(" ")
    #         preds.append(self.logits_probe(sent_tokens, tgt_tokens))
    #         # preds.append(self.detect_predict3(sent_tokens))
    #     return preds

    # def test_step(self, batch, batch_idx: int, *args, **kwargs):
    #     src, tgt = batch
    #     pred = []
    #
    #     for sent, label in zip(src, tgt):
    #         sent_tokens = sent.split(" ")
    #         sent_tokens, pinyins = self.tell_mask(sent_tokens, label)
    #         # sent_tokens, pinyins = self.predict_mask(sent_tokens)
    #
    #         pred.append(self.predict2_4(sent_tokens, pinyins))
    #         # pred.append(self.predict5(sent_tokens, pinyins))
    #     return pred

    def test_step(self, batch, batch_idx: int, *args, **kwargs):
        src, tgt = batch
        pred = []

        for sent, label in zip(src, tgt):
            sent_tokens = sent.split(" ")
            label_tokens = label.split(" ")
            sent_tokens, pinyins = self.tell_mask_full_pinyin(sent_tokens, label)
            # sent_tokens, pinyins = self.predict_mask(sent_tokens)

            pred.append(self.predict5_3(sent_tokens, pinyins, label_tokens))
        return pred

    def on_test_end(self) -> None:
        # Calculate mean and standard deviation
        # norm_distribute_plot(self.c_logits)
        # norm_distribute_plot(self.e_logits)
        pass


if __name__ == '__main__':
    args = mock_args(device='cpu',
                     hyper_params={
                         'sim_times': 5,
                         'token_times': 1.2
                     })

    sent_tokens = list("吃了早菜以后他去上课。")
    tgt_tokens = list("吃了早餐以后他去上课。")
    model = AdjustProbByPinyin(args)

    # sent_tokens = "[MASK] 再 也 不 会 [PAD] 扬 。 [PAD]".split(' ')
    # print(
    # AdjustProbByPinyin(mock_args(device='cpu'),
    #                    pinyin_distance_filepath='../ptm/pinyin_distances.pkl').predict4(
    #     sent_tokens))
    # print(AdjustProbByPinyin(args).predict4(sent_tokens))
    # print(AdjustProbByPinyin(args).logits_probe(sent_tokens, tgt_tokens))
    # sent_tokens, pinyins = model.tell_mask(sent_tokens, ' '.join(tgt_tokens))
    # print(model.predict2_3(sent_tokens, pinyins))
    sent_tokens, pinyins = model.tell_mask_full_pinyin(sent_tokens, ' '.join(tgt_tokens))
    # print(model.predict5_2(sent_tokens, pinyins))
    print(model.predict5_3(sent_tokens, pinyins))
