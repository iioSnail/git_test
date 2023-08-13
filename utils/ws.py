# 分词
import time

import hanlp

tok = None

pos = None


def _init_tok():
    global tok
    if tok is not None:
        return

    tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)


def _init_pos():
    global pos
    if pos is not None:
        return

    pos = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)


def word_segment(sentence: str):
    _init_tok()

    words = tok([sentence])[0]
    pos_results = pos_words(words)

    final_words = []
    curr_word = ""
    for i, tag in enumerate(pos_results):
        if tag in ['FW', 'NN', 'NR', 'NT']:
            curr_word += words[i]
            continue

        if curr_word != "":
            final_words.append(curr_word)
            curr_word = ""

        final_words.append(words[i])

    if curr_word != "":
        final_words.append(curr_word)

    return final_words


def word_segment_bak(sentences):
    _init_tok()

    str_flag = False
    if type(sentences) == str:
        str_flag = True
        sentences = [sentences]

    result = tok(sentences)

    if str_flag:
        return result[0]

    return result


def pos_words(words, verbose=False):
    _init_pos()

    results = pos(words)

    if verbose:
        for i, tag in enumerate(results):
            print("%s/%s" % (words[i], tag), end=', ')

    return results


if __name__ == '__main__':
    while True:
        text = "柯雷白肝军肺炎是什么病"
        print(word_segment(text))
        time.sleep(1)
