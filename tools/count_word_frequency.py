import sys, os
from collections import Counter

import pandas as pd

from utils.str_utils import word_segment

sys.path.insert(1, os.path.abspath("."))
sys.path.insert(1, os.path.abspath(".."))

os.chdir(os.path.pardir)


def load_sentences_from_csv(filepath):
    with open(filepath, mode='r', encoding='utf-8') as f:
        lines = f.readlines()

    sentences = []
    for line in lines[1:]:
        items = line.split(",")
        src = items[0].strip()

        src = ' '.join(src.replace(" ", "").replace(u"\u3000", ""))
        sentences.append(src)

    return sentences

if __name__ == '__main__':
    sentences = load_sentences_from_csv("./datasets/cscd_ime_2m.csv") \
                + load_sentences_from_csv("./datasets/cscd_ime_dev_bak.csv") \
                + load_sentences_from_csv("./datasets/cscd_ime_test.csv") \
                + load_sentences_from_csv("./datasets/wang271k.csv")

    word_segment(sentences)

    print(df)