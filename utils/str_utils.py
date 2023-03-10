import torch
from ltp import LTP
from torch.nn.utils.rnn import pad_sequence

ltp = None


def Q2B(uchar):
    """单个字符 全角转半角"""
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code)


def is_chinese(uchar):
    return '\u4e00' <= uchar <= '\u9fa5'


def word_segment(sentences):
    global ltp
    if ltp is None:
        ltp = LTP("LTP/small")

    outputs = ltp.pipeline(sentences, tasks=["cws"])
    return outputs.cws


def word_segment_labels(sentences):
    cws_list = word_segment(sentences)
    labels = []
    for si in range(len(sentences)):
        words = cws_list[si]
        chars = list(sentences[si])
        i = 0
        for word in words:
            if len(word) == 1:
                chars[i] = 'S'

            if len(word) == 2:
                chars[i] = 'B'
                chars[i + 1] = 'E'

            if len(word) > 2:
                chars[i] = 'B'
                chars[i + 1:i + len(word) - 1] = ['I'] * (len(word) - 2)
                chars[i + len(word) - 1] = 'E'
            i += len(word)
        labels.append(chars)
    return labels

def word_segment_targets(sentences):
    labels = word_segment_labels(sentences)
    word2idx = {
        'S': 1,
        'B': 2,
        'I': 3,
        'E': 4
    }
    targets = []
    for label in labels:
        targets.append(torch.LongTensor([word2idx[c] for c in label]))

    return pad_sequence(targets, batch_first=True)

if __name__ == '__main__':
    sentences = ["我 很 喜 欢 看 你 跳 无 ， 你 干 嘛 ！ ", "李 四 想 去 负 担 大 学 的 夜 市 摊 吃 甜 橘 子", "惊 弓 之 鸟"]
    print(word_segment(sentences))
    print(word_segment_labels(sentences))
    print(word_segment_targets(sentences))
