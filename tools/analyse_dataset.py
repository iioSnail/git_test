import sys, os

sys.path.insert(1, os.path.abspath("."))
sys.path.insert(1, os.path.abspath(".."))


def _get_error_char_num(src, tgt):
    src = list(src)
    tgt = list(tgt)



def analyse_dataset(datasets):
    total_num = 0
    error_sentence_num = 0
    correct_sentence_num = 0

    invalid_sentence_num = 0

    error_char_num = 0


    for item in datasets:
        src, tgt = item

        src = src.replace(" ", "")
        tgt = tgt.replace(" ", "")

        if len(src) != len(tgt):
            invalid_sentence_num += 1
            continue

        if src == tgt:
            correct_sentence_num += 1
        else:
            error_sentence_num += 1

        total_num += 1

    print("句子数量：", total_num)
    print("异常句子的数量：", invalid_sentence_num)
    print("正确的句子数量：", correct_sentence_num)
    print("含错字的句子数量：", error_sentence_num)
