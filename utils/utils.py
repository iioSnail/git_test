import os
import pickle
import random

import numpy as np
import torch
from sympy.polys.rootisolation import Q2

from utils.str_utils import Q2B

special_tokens = set("`1234567890-=~!！@#$%^&*()_+（）qwertyuiop"
                     "asddfghjklzxcvbnmQWERTYUIOPASDFGHHJKLZXCVBNM"
                     "[]\\;'./{}|:\"<>?，。、？’‘“”；：ＡＢＣＤＥＦＧＨＩ"
                     "ＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ０１２３４５６７８９"
                     "「」＂…")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_obj(obj, filepath):
    with open(filepath, "bw") as f:
        pickle.dump(obj, f)


def load_obj(filepath):
    with open(filepath, "br") as f:
        return pickle.load(f)


def render_color_for_text(text, indices, color='red'):
    color_indices = {
        'black': '30',
        'red': '31',
        'green': '32',
        'yellow': '33'
    }

    char_list = list(text)
    for i in range(len(indices)):
        if indices[i]:
            char_list[i] = "\033[" + color_indices.get(color, '30') + "m" + char_list[i] + "\033[0m"

    return ''.join(char_list)


def compare_text(src: str, tgt: str):
    src_char_list = list(src)
    tgt_char_list = list(tgt)

    result = [False] * max(len(src_char_list), len(tgt_char_list))

    for i in range(min(len(src_char_list), len(tgt_char_list))):
        if src_char_list[i] != tgt_char_list[i]:
            result[i] = True

    return result


def restore_special_tokens(src, output):
    """
    Some special tokens shouldn't be corrected, such as symbols, numbers and English words.
    Therefore, we need to restore these special tokens.
    :param src: e.g. 我今天非长高兴!
    :param output: e.g. 我今天非常高兴。
    :return: e.g. 我今天非常高兴！
    """
    if isinstance(output, list):
        output = ''.join(output)

    output = output.replace('[UNK]', '?').replace('[SEP]', '?').replace('[PAD]', '?') \
        .replace('[CLS]', '?').replace('[MASK]', '?')

    if len(src) != len(output):
        return output

    src = list(src)
    output = list(output)
    for i in range(len(src)):
        if src[i] in special_tokens or output[i] in special_tokens:
            output[i] = src[i]

    return ''.join(output)


def preprocess_text(sentence):
    sentence = sentence.replace(" ", "")

    char_list = list(sentence)
    for i, char in enumerate(char_list):
        char_list[i] = Q2B(char) # 全角转半角

    return ''.join(char_list)
