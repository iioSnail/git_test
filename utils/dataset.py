import os.path
import pickle
import random

import opencc
import pandas as pd
import pypinyin
from hanzi_chaizi import HanziChaizi
from torch.utils.data import Dataset
from tqdm import tqdm

from model.common import BERT
from utils.confusions import confuse_char
from utils.str_utils import is_chinese
from utils.utils import preprocess_text, mkdir, load_obj, save_obj
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]


class CSCDataset(Dataset):

    def __init__(self, dataset, args):
        super(CSCDataset, self).__init__()
        self.dataset = dataset
        self.args = args

    def __getitem__(self, index):
        src = self.dataset[index]['src']
        tgt = self.dataset[index]['tgt']
        return " ".join(src), " ".join(tgt)

    def __len__(self):
        if self.args.limit_data_size and self.args.limit_data_size > 0:
            return min(self.args.limit_data_size, len(self.dataset))
        else:
            return len(self.dataset)


class SighanTrainDataset(Dataset):

    def __init__(self):
        super(SighanTrainDataset, self).__init__()
        """
        Sighan做的预处理有：
        1. 去掉如下特殊字符：[' ', '“', '”', '‘', '’', '琊', '…', '—', '擤']
        """

        sighan_2013 = pd.read_csv(ROOT / 'datasets/sighan_2013_train.csv')
        sighan_2014 = pd.read_csv(ROOT / 'datasets/sighan_2014_train.csv')
        sighan_2015 = pd.read_csv(ROOT / 'datasets/sighan_2015_train.csv')

        self.dataset = pd.concat([sighan_2013, sighan_2014, sighan_2015])

    def __getitem__(self, index):
        src = self.dataset.iloc[index]['src']
        tgt = self.dataset.iloc[index]['tgt']
        src = preprocess_text(src)
        tgt = preprocess_text(tgt)
        return " ".join(src), " ".join(tgt)

    def __len__(self):
        return len(self.dataset)


class CSCTestDataset(Dataset):

    def __init__(self, args):
        super(CSCTestDataset, self).__init__()
        self.dataset = pd.read_csv(ROOT / args.test_data)

    def __getitem__(self, index):
        src = self.dataset.iloc[index]['src']
        tgt = self.dataset.iloc[index]['tgt']
        src = preprocess_text(src)
        tgt = preprocess_text(tgt)
        return " ".join(src), " ".join(tgt)

    def __len__(self):
        return len(self.dataset)


class ConfusionMaskDataset(Dataset):

    def __init__(self, args):
        self.args = args
        self.dataset = pd.read_csv(self.args.train_data, header=None, dtype=str)
        self.dataset.columns = ['sentence', 'indexes']

    def __getitem__(self, index):
        row = self.dataset.iloc[index]
        sentence = row['sentence']
        indexes = row['indexes']
        tgt = ' '.join(sentence)
        src = self.mask_sentence(sentence, indexes)
        return src, tgt

    def mask_sentence(self, sentence, indexes):
        sentence = list(sentence)
        mask_len = max(len(sentence) // 10, 1)
        if pd.isnull(indexes):
            indexes = ','.join(list(map(str, range(len(sentence)))))
        indexes = list(map(int, indexes.split(",")))
        if mask_len > len(indexes):
            mask_len = len(indexes)
        indexes = random.sample(indexes, mask_len)
        for i in indexes:
            sentence[i] = confuse_char(sentence[i], type='same_pinyin')

        return ' '.join(sentence)

    def __len__(self):
        if self.args.limit_data_size and self.args.limit_data_size > 0:
            return min(self.args.limit_data_size, len(self.dataset))
        return len(self.dataset)


class PhoneticProbeDataset(Dataset):

    def __init__(self):
        super(PhoneticProbeDataset, self).__init__()
        tokenizer = BERT.get_tokenizer()

        # Get all chinese characters.
        tw2zh = opencc.OpenCC('t2s.json')
        chinese_chars = set()
        for token in tokenizer.get_vocab().keys():
            if is_chinese(token):
                chinese_chars.add(tw2zh.convert(token))
        chinese_chars = list(chinese_chars)

        # Create chinese pinyin list of chinese characters.
        chinese_chars_pinyin = []
        for char_ in chinese_chars:
            chinese_chars_pinyin.append(pypinyin.pinyin(char_, style=pypinyin.NORMAL)[0])

        # Create Positive samples.
        positive_samples = []
        for i in range(len(chinese_chars)):
            pinyin_i = chinese_chars_pinyin[i]
            for j in range(i + 1, len(chinese_chars)):
                pinyin_j = chinese_chars_pinyin[j]
                if pinyin_i == pinyin_j:
                    positive_samples.append(((chinese_chars[i], chinese_chars[j]), 1))

        # Create negative samples.
        negative_samples = []
        while True:
            i = random.randint(0, len(chinese_chars) - 1)
            j = random.randint(0, len(chinese_chars) - 1)
            if chinese_chars_pinyin[i] != chinese_chars_pinyin[j]:
                negative_samples.append(((chinese_chars[i], chinese_chars[j]), 0))

            if len(negative_samples) == len(positive_samples):
                break

        # Merge positive samples and negative samples and then shuffle them.
        dataset = positive_samples + negative_samples
        random.shuffle(dataset)
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class GlyphProbeDataset(Dataset):

    chinese_chars_components = None

    def __init__(self, cache=True):
        super(GlyphProbeDataset, self).__init__()
        cache_path = './cache/GlyphProbeDataset.pkl'
        if cache and os.path.exists(cache_path):
            self.dataset = load_obj(cache_path)
            print("Load GlyphProbeDataset from cache.")
            return

        tokenizer = BERT.get_tokenizer()

        # Get all chinese characters.
        tw2zh = opencc.OpenCC('t2s.json')
        chinese_chars = set()
        for token in tokenizer.get_vocab().keys():
            if is_chinese(token):
                chinese_chars.add(tw2zh.convert(token))
        chinese_chars = list(chinese_chars)

        chaizi = HanziChaizi()
        chinese_chars_components = GlyphProbeDataset.get_chinese_chars_components()

        positive_samples = set()
        for u in tqdm(chinese_chars_components, desc="Init Glyph Dataset"):
            for w in chinese_chars:
                if w not in chaizi.data:
                    continue

                w_components = chaizi.query(w)
                if u in w_components:
                    positive_samples.add((u, w))

        negative_samples = set()
        while True:
            u = chinese_chars_components[random.randint(0, len(chinese_chars_components)-1)]
            w = chinese_chars[random.randint(0, len(chinese_chars)-1)]

            if (u, w) not in positive_samples:
                negative_samples.add((u, w))

            if len(negative_samples) == len(positive_samples):
                break

        positive_samples = [(item, 1) for item in positive_samples]
        negative_samples = [(item, 0) for item in negative_samples]

        # Merge positive samples and negative samples and then shuffle them.
        dataset = positive_samples + negative_samples
        random.shuffle(dataset)
        self.dataset = dataset
        if cache:
            mkdir('./cache') # FIXME
            save_obj(self.dataset, cache_path)

    @staticmethod
    def get_chinese_chars_components():
        if GlyphProbeDataset.chinese_chars_components is not None:
            return GlyphProbeDataset.chinese_chars_components

        chaizi = HanziChaizi()
        chinese_chars_components = set()
        for values in chaizi.data.values():
            for value in values:
                for component in value:
                    chinese_chars_components.add(component)

        GlyphProbeDataset.chinese_chars_components = list(chinese_chars_components)
        return GlyphProbeDataset.chinese_chars_components

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

