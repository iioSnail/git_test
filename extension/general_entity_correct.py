"""
通用的实体词汇修正。

思路：对于一句话，首先由CSC模型进行修正，然后进行接下来的几步：
1. 使用hanLP进行分词
2. 将每个词使用Chinese_Word_Classifier模型进行分类
3. 到对应的类别下查找相似的词汇进行替换。
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from extension.entity_correct import QueryEntity
from utils.ws import word_segment


class GeneralEntityCorrect(object):

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("iioSnail/bert-base-chinese-word-classifier")
        self.model = AutoModelForSequenceClassification.from_pretrained("iioSnail/bert-base-chinese-word-classifier")

        entity_files = [
            "1_人文科学.pkl",
            "2_农林渔畜.pkl",
            "3_医学.pkl",
            "4_城市信息大全.pkl",
            "5_娱乐.pkl",
            "6_工程与应用科学.pkl",
            "7_生活.pkl",
            "8_电子游戏.pkl",
            "9_社会科学.pkl",
            "10_自然科学.pkl",
            "11_艺术.pkl",
            "12_运动休闲.pkl",
        ]

        self.query_entity_list = [None]
        for file in entity_files:
            print("Loading entity file " + file)
            # self.query_entity_list.append(QueryEntity("ptm/" + file))

    def word_classify(self, words):
        inputs = self.tokenizer(words, return_tensors='pt', padding=True)
        outputs = self.model(**inputs).logits
        outputs = outputs.sigmoid()

        output_max = outputs.max(-1)

        results = []
        for logit, label in zip(output_max.values, output_max.indices):
            if logit > 0.6:
                results.append(int(label))
            else:
                results.append(0)

        return results

    def correct(self, sentence):
        words = word_segment(sentence)

        suspicious_words = []
        for i, word in enumerate(words):
            if len(word) >= 2:
                suspicious_words.append([i, word])

        word_labels = self.word_classify([item[1] for item in suspicious_words])

        for i, label in enumerate(word_labels):
            word = self.query_entity_list[i].query_entity(suspicious_words[i][1])
            suspicious_words[i][1] = word

        for i, word in suspicious_words:
            words[i] = word

        return ''.join(words)
