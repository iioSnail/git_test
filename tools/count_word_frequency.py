import sys, os
from collections import Counter

from utils.ws import word_segment

sys.path.insert(1, os.path.abspath("."))
sys.path.insert(1, os.path.abspath(".."))

os.chdir(os.path.pardir)

from tqdm import tqdm

from utils.utils import save_obj, load_obj


def load_sentences_from_csv(filepath):
    print("load", filepath)
    with open(filepath, mode='r', encoding='utf-8') as f:
        lines = f.readlines()

    sentences = []
    for line in lines[1:]:
        items = line.split(",")
        tgt = items[1].strip()

        tgt = ''.join(tgt.replace(" ", "").replace(u"\u3000", ""))
        sentences.append(tgt)

    return sentences


def generate_word_frequency():
    sentences = load_sentences_from_csv("./datasets/wang271k.csv")
    # + load_sentences_from_csv("./datasets/cscd_ime_train.csv") \
    # + load_sentences_from_csv("./datasets/cscd_ime_dev.csv") \
    # + load_sentences_from_csv("./datasets/cscd_ime_test.csv") \
    # + load_sentences_from_csv("./datasets/cscd_ime_2m.csv") \

    counter = Counter()
    batch_size = 64
    for i in tqdm(range(0, len(sentences) // batch_size)):
        start_index = i * batch_size
        end_index = (i + 1) * batch_size
        segments = word_segment(sentences[start_index:end_index])
        for segment in segments:
            counter.update(segment)

    save_obj(counter, "./outputs/word_frequency.pkl")

    return counter


def generate_common_words(counter):
    common_words = []
    for word, count in counter.most_common():
        if len(word) <= 1:
            continue

        if count < 10:  # 非常见词，过滤掉
            continue

        common_words.append(word)

    with open("outputs/common_words.txt", mode='w', encoding='utf-8') as f:
        f.write('\n'.join(common_words))

    return common_words


def main():
    counter = generate_word_frequency()
    generate_common_words(counter)


if __name__ == '__main__':
    main()
