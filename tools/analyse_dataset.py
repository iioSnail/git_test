import sys, os
from collections import Counter

import pandas as pd
from tqdm import tqdm

from utils.utils import load_obj, preprocess_text

sys.path.insert(1, os.path.abspath("."))
sys.path.insert(1, os.path.abspath(".."))


def continuous_num(lst):
    counts = []
    current_count = 0

    for i in range(len(lst) - 1):
        if lst[i] == lst[i + 1] - 1:
            current_count += 1
        else:
            counts.append(current_count + 1)
            current_count = 0

    counts.append(current_count + 1)

    return counts


def analyse_dataset(datasets):
    max_length = 0
    min_length = 999999
    total_num = 0
    error_sentence_num = 0
    correct_sentence_num = 0

    invalid_sentence_num = 0
    invalid_sentences = []

    total_char_num = 0
    error_char_num = 0

    ta_error_char_num = 0
    ta_error_sentence_num = 0
    ta_chars = list('他她它')

    de_error_char_num = 0
    de_error_sentence_num = 0
    de_chars = list('的地得')

    continuous_error_counter = Counter()

    for item in tqdm(datasets):
        src, tgt = item

        src = src.replace(" ", "")
        tgt = tgt.replace(" ", "")

        if len(src) != len(tgt):
            invalid_sentence_num += 1
            invalid_sentences.append(item)
            continue

        if src == tgt:
            correct_sentence_num += 1
        else:
            error_sentence_num += 1

        error_indexes = []
        has_ta_error = False
        has_de_error = False
        for i in range(len(src)):
            if src[i] != tgt[i]:
                error_char_num += 1

                error_indexes.append(i)

                if tgt[i] in ta_chars:
                    ta_error_char_num += 1
                    has_ta_error = True

                if tgt[i] in de_chars:
                    de_error_char_num += 1
                    has_de_error = True

        continuous_error_counter.update(continuous_num(error_indexes))

        if has_ta_error:
            ta_error_sentence_num += 1

        if has_de_error:
            de_error_sentence_num += 1

        if len(src) > max_length:
            max_length = len(src)

        if len(src) < min_length:
            min_length = len(src)

        total_char_num += len(src)
        total_num += 1

    print("句子数量：", total_num)
    print("异常句子的数量：", invalid_sentence_num)
    print("正确的句子数量：", correct_sentence_num)
    print("含错字的句子数量：", error_sentence_num)
    print("最长句子长度：", max_length)
    print("最短句子长度：", min_length)
    print("平局句子长度：", total_char_num / total_num)
    print("错字数量：", error_char_num)
    print("平均每句错字数量：", error_char_num / total_num)
    print("平均每多少字出现一个错字：", total_char_num / error_char_num)
    print("-" * 20)
    print("含“他她它”错字的句子数量：", ta_error_sentence_num)
    print("含“的地得”错字的句子数量：", de_error_sentence_num)
    print("“他她它”错字数量：", ta_error_char_num)
    print("“的地得”错字数量：", de_error_char_num)
    print("连续错字分布：", continuous_error_counter)

    if invalid_sentence_num > 0:
        print("-" * 20)
        print("异常句子如下：", invalid_sentences)


def analysis_wang271k():
    dataset = load_obj("../data/Wang271K_processed.pkl")
    dataset = [(item['src'], item['tgt']) for item in dataset]

    analyse_dataset(dataset)


def analysis_cscd_ime(filepath):
    with open(filepath, mode='r', encoding='UTF-8') as f:
        lines = f.readlines()

    dataset = []
    for line in lines:
        line = line.strip()
        items = line.split("\t")
        if len(items) < 3:
            continue

        dataset.append((items[1], items[2]))

    analyse_dataset(dataset)


def analysis_sighan(filepath):
    if type(filepath) == list:
        df_list = []
        for path in filepath:
            df_list.append(pd.read_csv(path))
        df = pd.concat(df_list)
    else:
        df = pd.read_csv(filepath)

    dataset = []
    for index in range(len(df)):
        src = str(df.iloc[index]['src'])
        tgt = str(df.iloc[index]['tgt'])

        dataset.append((src, tgt))

    analyse_dataset(dataset)


if __name__ == '__main__':
    # analysis_wang271k()
    # analysis_cscd_ime("../datasets/cscd-ime/all.tsv")
    # analysis_cscd_ime("../datasets/cscd-ime/train.tsv")
    # analysis_cscd_ime("../datasets/cscd-ime/dev.tsv")
    # analysis_cscd_ime("../datasets/cscd-ime/test.tsv")
    # analysis_cscd_ime("../datasets/cscd-ime/lcsts-ime-2m.tsv")
    # analysis_sighan(["../datasets/sighan_2015_train.csv",
    #                  "../datasets/sighan_2014_train.csv",
    #                  "../datasets/sighan_2013_train.csv", ])
    # analysis_sighan("../datasets/sighan_2015_train.csv")
    # analysis_sighan("../datasets/sighan_2014_train.csv")
    # analysis_sighan("../datasets/sighan_2013_train.csv")
    # analysis_sighan("../datasets/sighan_2015_test.csv")
    # analysis_sighan("../datasets/sighan_2014_test.csv")
    # analysis_sighan("../datasets/sighan_2013_test.csv")
    # analysis_sighan("../datasets/sighan_2015_test_revise.csv")
    # analysis_sighan("../datasets/ec_law.csv")
    # analysis_sighan("../datasets/ec_med.csv")
    # analysis_sighan("../datasets/ec_odw.csv")
    # analysis_sighan("../datasets/lemon_car.csv")
    # analysis_sighan("../datasets/lemon_enc.csv")
    # analysis_sighan("../datasets/lemon_gam.csv")
    # analysis_sighan("../datasets/lemon_mec.csv")
    # analysis_sighan("../datasets/lemon_new.csv")
    # analysis_sighan("../datasets/lemon_nov.csv")
    analysis_sighan("../datasets/lemon_cot.csv")
