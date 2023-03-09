from ltp import LTP

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
                chars[i+1] = 'E'

            if len(word) > 2:
                chars[i] = 'B'
                chars[i+1:i + len(word) - 1] = ['I'] * (len(word) - 2)
                chars[i + len(word) - 1] = 'E'
            i += len(word)
        labels.append(chars)
    return labels

if __name__ == '__main__':
    sentences = ["我很喜欢看你跳无", "李四想去负担大学的夜市摊吃甜橘子", "惊弓之鸟"]
    print(word_segment(sentences))
    print(word_segment_labels(sentences))
