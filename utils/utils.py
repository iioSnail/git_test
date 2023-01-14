import os
import pickle
import random

import numpy as np
import torch
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

from sklearn.decomposition import PCA

from utils.confusions import confuse_char
from utils.str_utils import Q2B, is_chinese

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

    char_list = list(text.replace(" ", ""))
    for i in range(len(indices)):
        if indices[i]:
            char_list[i] = "\033[" + color_indices.get(color, '30') + "m" + char_list[i] + "\033[0m"

    return ''.join(char_list)


def compare_text(src: str, tgt: str):
    src_char_list = list(src.replace(" ", ""))
    tgt_char_list = list(tgt.replace(" ", ""))

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
        if not is_chinese(src[i]) or not is_chinese(output[i]):
            output[i] = src[i]

    return ''.join(output)


def preprocess_text(sentence):
    sentence = sentence.replace(" ", "")

    char_list = list(sentence)
    for i, char in enumerate(char_list):
        char_list[i] = Q2B(char)  # 全角转半角

    return ''.join(char_list)


def mask_tokens(ids, mask_id=130, hard_level=1):
    mask_id = int(mask_id)

    def mask_level_1(length):
        """难度1：对句子中的某一个token进行mask"""
        mask_index = [random.randint(1, length)]
        return mask_index

    for sequence_tokens in ids:
        length = torch.argwhere(sequence_tokens != 0)[-1].item()
        mask = []
        if hard_level == 1:
            mask = mask_level_1(length)

        sequence_tokens[mask] = int(mask_id)

    return ids


def mask_level_1(length):
    """难度1：对句子中的某一个token进行mask"""
    mask_index = [random.randint(0, length - 1)]
    return mask_index


def mask_sentence(sentence, hard_level=1):
    confuse_chars = list(sentence.replace(" ", ""))
    mask_chars = confuse_chars.copy()

    masks = []
    if hard_level == 1:
        masks = mask_level_1(len(confuse_chars))

    for mask in masks:
        confuse_chars[mask] = confuse_char(confuse_chars[mask])
        mask_chars[mask] = '[MASK]'

    return ' '.join(confuse_chars), masks


def mock_args(**kwargs):
    class MockArgs(dict):
        __setattr__ = dict.__setitem__
        __getattr__ = dict.__getitem__

    args = MockArgs()
    for key, value in kwargs.items():
        args[key] = value

    return args


def token_embeddings_visualise(embeddings, text):
    """
    将汉字文本embedding绘制成图像
    :param embeddings: 文本embedding后的向量，例如Shape为(55, 768)为55个token，每个token维度为768。
                       不要包含bos、eos和pad。embedding需要是numpy类型的
    :param text: 汉字文本，例如 “张 三 是 法 外 狂 徒”。text的长度需要和上面的token数一致
    """
    text = text.split(" ")
    assert embeddings.shape[0] == len(text), "embedding的token数与text不一致"

    if type(embeddings) == torch.Tensor:
        embeddings = embeddings.detach().numpy()

    assert type(embeddings) is np.ndarray, "embedding参数必须是numpy.ndarray类型或tensor类型"

    # 1. 将词向量降维到2维度
    pca = PCA(n_components=2)
    embeddings = pca.fit_transform(embeddings)
    # 2. 创建一个16x9大小的维度图像
    plt.figure(figsize=(16, 9))
    # 3. 循环绘制文字
    for i in range(len(embeddings)):
        plt.text(embeddings[i][0], embeddings[i][1], text[i])

    # 4. 设置坐标边界
    plt.xlim(embeddings[:, 0].min() - 0.5, embeddings[:, 0].max() + 0.5)
    plt.ylim(embeddings[:, 1].min() - 0.5, embeddings[:, 1].max() + 0.5)
    plt.show()
