import random
from pathlib import Path

from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory

same_pinyin_confusion_set = {}
simi_pinyin_confusion_set = {}
same_glyph_confusion_set = {}
all_confusion_set = {}
inited = False


def _load_confusion(in_file):
    confusion_datas = {}
    with open(in_file, encoding='utf-8') as f:
        lines = f.readlines()
    for line in tqdm(lines, desc="Load Confusion Set %s" % in_file):
        line = line.strip()
        tmps = line.split('\t')
        if len(tmps) != 2:
            continue
        key = tmps[0]
        values = tmps[1].split()
        confusion_datas[key] = values

    return confusion_datas


def _get_confusion_set(type='random'):
    confusion_set_list = [same_pinyin_confusion_set, simi_pinyin_confusion_set, same_glyph_confusion_set]

    if type == 'random':
        rand_i = random.randint(0, 2)
        return confusion_set_list[rand_i]

    if type == 'glyph':
        return confusion_set_list[2]

    if type == 'simi_pinyin':
        return confusion_set_list[1]

    if type == 'same_pinyin':
        return confusion_set_list[0]

    if type == 'pinyin':
        rand_i = random.randint(0, 1)
        return confusion_set_list[rand_i]


def confuse_char(char: str, type='random'):
    init_confusion_set()

    confusion_set = _get_confusion_set(type)
    if char not in confusion_set:
        return char

    confusion_list = confusion_set[char]

    rand_i = random.randint(0, len(confusion_list) - 1)
    return confusion_list[rand_i]


def init_confusion_set():
    global inited
    if inited:
        return

    global same_pinyin_confusion_set
    global simi_pinyin_confusion_set
    global same_glyph_confusion_set
    same_pinyin_confusion_set = _load_confusion(ROOT / "confusion/same_pinyin.txt")
    simi_pinyin_confusion_set = _load_confusion(ROOT / "confusion/simi_pinyin.txt")
    same_glyph_confusion_set = _load_confusion(ROOT / "confusion/same_stroke.txt")

    inited = True


def is_confusion_char(c1, c2):
    init_confusion_set()

    if c1 in same_pinyin_confusion_set and c2 in same_pinyin_confusion_set[c1]:
        return True

    if c1 in simi_pinyin_confusion_set and c2 in simi_pinyin_confusion_set[c1]:
        return True

    if c1 in same_glyph_confusion_set and c2 in same_glyph_confusion_set[c1]:
        return True

    return False

# init_confusion_set()
