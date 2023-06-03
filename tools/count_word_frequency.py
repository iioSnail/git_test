import sys, os
from collections import Counter

from tqdm import tqdm

from utils.str_utils import word_segment
from utils.utils import save_obj, load_obj

sys.path.insert(1, os.path.abspath("."))
sys.path.insert(1, os.path.abspath(".."))

os.chdir(os.path.pardir)


def load_sentences_from_csv(filepath):
    print("load", filepath)
    with open(filepath, mode='r', encoding='utf-8') as f:
        lines = f.readlines()

    sentences = []
    for line in lines[1:]:
        items = line.split(",")
        src = items[0].strip()

        src = ''.join(src.replace(" ", "").replace(u"\u3000", ""))
        sentences.append(src)

    return sentences

def generate_word_frequency():
    sentences = load_sentences_from_csv("./datasets/cscd_ime_2m.csv") \
                + load_sentences_from_csv("./datasets/cscd_ime_dev_bak.csv") \
                + load_sentences_from_csv("./datasets/cscd_ime_test.csv") \
                + load_sentences_from_csv("./datasets/wang271k.csv")

    counter = Counter()
    batch_size = 32
    for i in tqdm(range(0, len(sentences) // batch_size)):
        start_index = i * batch_size
        end_index = (i + 1) * batch_size
        segments = word_segment(sentences[start_index:end_index])
        for segment in segments:
            counter.update(segment)

        if i > 1000:
            break

    save_obj(counter, "./outputs/counter.pkl")

if __name__ == '__main__':
    counter = load_obj("./outputs/counter.pkl")

    print(counter)


