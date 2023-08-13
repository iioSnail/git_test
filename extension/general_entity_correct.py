"""
通用的实体词汇修正。

思路：对于一句话，首先由CSC模型进行修正，然后进行接下来的几步：
1. 使用hanLP进行分词
2. 将每个词使用Chinese_Word_Classifier模型进行分类
3. 到对应的类别下查找相似的词汇进行替换。
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification

from extension.entity_correct import QueryEntity
from utils.ws import word_segment


class GeneralEntityCorrect(object):

    def __init__(self):
        # self.tokenizer = AutoTokenizer.from_pretrained("iioSnail/bert-base-chinese-word-classifier")
        # self.model = AutoModelForSequenceClassification.from_pretrained("iioSnail/bert-base-chinese-word-classifier")

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.model = BertForSequenceClassification.from_pretrained("bert-base-chinese",
                                                                   num_labels=11,
                                                                   problem_type='single_label_classification')

        ckpt_state = torch.load("./ptm/cwc.ckpt", map_location='cpu')
        state_dict = {}
        for key, value in ckpt_state['state_dict'].items():
            state_dict[key[6:]] = value

        self.model.load_state_dict(state_dict, strict=False)

        entity_files = [
            "1_人物.pkl",
            "2_化学.pkl",
            "3_医学.pkl",
            "4_城市信息.pkl",
            "5_工业工程.pkl",
            "6_法律.pkl",
            "7_生物.pkl",
            "8_电子游戏.pkl",
            "9_计算机.pkl",
            "10_诗词.pkl",
        ]

        self.query_entity_list = [None]
        self.labels = {}
        for file in entity_files:
            print("Loading entity file " + file)
            self.query_entity_list.append(QueryEntity("ptm/" + file))
            index, label = file[:-4].split("_")
            self.labels[int(index)] = label

        self.common_words = self.load_common_words("ptm/common_words.txt")

    def load_common_words(self, filepath):
        with open(filepath, encoding='utf-8') as f:
            lines = f.readlines()

        words = set()
        for line in lines:
            line = line.strip()
            if line == "":
                continue

            words.add(line)

        return words

    def word_classify(self, words):
        inputs = self.tokenizer(words, return_tensors='pt', padding=True)
        outputs = self.model(**inputs).logits

        output_max = outputs.softmax(-1).max(-1)

        results = []
        for logit, label in zip(output_max.values, output_max.indices):
            if logit > 0.8:
                results.append(int(label))
            else:
                results.append(0)

        return results

    def correct(self, sentence):
        words = word_segment(sentence)

        # print(words)  # FIXME verbose log

        suspicious_words = []
        for i, word in enumerate(words):
            if len(word) >= 2 and len(words) <= 10 and word not in self.common_words:
                suspicious_words.append([i, word])

        if len(suspicious_words) <= 0:
            return ''.join(words)

        word_labels = self.word_classify([item[1] for item in suspicious_words])

        for i, label in enumerate(word_labels):
            if label == 0:
                continue
            word = self.query_entity_list[label].query_entity(suspicious_words[i][1],
                                                              entity_class=self.labels[label])

            if len(word) == len(suspicious_words[i][1]):
                suspicious_words[i][1] = word

        for i, word in suspicious_words:
            words[i] = word

        return ''.join(words)
