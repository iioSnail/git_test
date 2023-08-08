"""
将词编码成向量
"""
import os

os.chdir(os.path.pardir)

import pickle

import torch
from tqdm import tqdm

from utils.str_utils import is_chinese
from utils.utils import convert_char_to_pinyin, save_obj

max_length = 10
pinyin_size = 6


def load_entities(txt_file):
    """
    txt_file: A file that including many words. Every word is in one line.
    """
    with open(txt_file, encoding='utf-8') as f:
        lines = f.readlines()

    entities = []
    for word in lines:
        word = word.replace(" ", "").strip()

        if len(word) <= 1:
            continue

        if len(word) > 10:
            entities.append(word[:10])
            entities.append(word[-10:])
        else:
            entities.append(word)

    return entities


def embedding_entity(entity):
    pinyin_list = []
    for character in entity:
        pinyin = torch.LongTensor([0] * pinyin_size)
        if is_chinese(character):
            pinyin = convert_char_to_pinyin(character, size=pinyin_size)

        if character.isascii():
            pinyin = torch.LongTensor([ord(character)] + [0] * (pinyin_size - 1))

        pinyin_list.append(pinyin)

    pad_size = max_length - len(pinyin_list)
    pinyin_list += [torch.LongTensor([0] * pinyin_size)] * pad_size

    return torch.vstack(pinyin_list).view(-1)


def embedding_entities(entities):
    embeddings = []
    for entity in tqdm(entities):
        embedding = embedding_entity(entity)
        embeddings.append({
            "entity": entity,
            "embedding": embedding
        })

    return embeddings


def main(txt_file):
    entities = load_entities(txt_file)
    embeddings = embedding_entities(entities)
    save_obj(embeddings, txt_file[:-4] + ".pkl")


if __name__ == '__main__':
    main("./tools/temp/1_人文科学.txt")
    main("./tools/temp/2_农林渔畜.txt")
    main("./tools/temp/3_医学.txt")
    main("./tools/temp/4_城市信息大全.txt")
    main("./tools/temp/5_娱乐.txt")
    main("./tools/temp/6_工程与应用科学.txt")
    main("./tools/temp/7_生活.txt")
    main("./tools/temp/8_电子游戏.txt")
    main("./tools/temp/9_社会科学.txt")
    main("./tools/temp/10_自然科学.txt")
    main("./tools/temp/11_艺术.txt")
    main("./tools/temp/12_运动休闲.txt")
