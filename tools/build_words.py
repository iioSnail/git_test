import sys, os

import torch
from tqdm import tqdm

from utils.utils import save_obj

sys.path.insert(1, os.path.abspath("."))
sys.path.insert(1, os.path.abspath(".."))

import pickle
from ltp import LTP
from collections import Counter

def load_sentences():
    train_data = '../data/Wang271K_processed.pkl'
    with open(train_data, mode='br') as f:
        train_data = pickle.load(f)

    sentences = [item['tgt'] for item in train_data]

    return sentences[:500]


def load_model():
    ltp = LTP("LTP/base1")

    if torch.cuda.is_available():
        ltp.to("cuda")

    def word_segment(sentences):
        return ltp.pipeline(sentences, tasks=["cws"])

    return word_segment


def build_words():
    batch_size = 64

    sentences_datas = load_sentences()
    model = load_model()

    counter = Counter()

    for i in tqdm(range(0, len(sentences_datas), batch_size), desc='Build words'):
        outputs = model(sentences_datas[i:i+batch_size])

        for words in outputs.cws:
            counter.update(words)

    print("Build End. Words size:", len(counter))

    save_obj(counter, '../words.pkl')

if __name__ == '__main__':
    build_words()