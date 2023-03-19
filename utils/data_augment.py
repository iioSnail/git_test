import random

from utils.utils import random_true

special_hanzi = list('的得地他她它哪那真这')


def special_hanzi_augment(src_sents, tgt_sents):
    """
    对原句子进行数据增强。
    增强方式：从特殊字符里随机挑一个字，随机从左边后右边找出来，然后进行替换。
             不同的特殊字符，替换规则不一样。
    """
    for i in range(len(src_sents)):
        sent = src_sents[i]

        hanzi = special_hanzi[random.randint(0, len(special_hanzi) - 1)]
        if random_true(0.5):
            index = sent.find(hanzi)
        else:
            index = sent.rfind(hanzi)

        if index < 0:
            continue

        replace_hanzi = None
        if sent[index] in ['的', '得', '地']:
            replace_hanzi = ['的', '得', '地'][random.randint(0, 2)]

        if sent[index] in ['哪', '那']:
            replace_hanzi = ['哪', '那'][random.randint(0, 1)]

        if sent[index] in ['真', '这']:
            replace_hanzi = ['真', '这'][random.randint(0, 1)]

        if sent[index] in ['他', '她', '它']:
            if sent.count(sent[index]) > 1:
                replace_hanzi = ['他', '她', '它'][random.randint(0, 2)]

        if replace_hanzi is None:
            continue

        sent = sent[:index] + replace_hanzi + sent[index+1:]
        src_sents[i] = sent

    return src_sents, tgt_sents