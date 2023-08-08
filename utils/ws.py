# 分词

import hanlp

tok = None


def _init_tok():
    global tok
    if tok is not None:
        return tok

    tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)


def word_segment(sentences):
    pass