# 分词

import hanlp

tok = None


def _init_tok():
    global tok
    if tok is not None:
        return

    tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)


def word_segment(sentences):
    _init_tok()

    str_flag = False
    if type(sentences) == str:
        str_flag = True
        sentences = [sentences]

    result = tok(sentences)

    if str_flag:
        return result[0]

    return result