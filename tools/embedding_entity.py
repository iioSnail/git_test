"""
将词编码成向量
"""
import os

from torch import nn

os.chdir(os.path.pardir)

import pickle

import torch
from tqdm import tqdm

from utils.str_utils import is_chinese
from utils.utils import convert_char_to_pinyin, save_obj, convert_char_to_image

max_length = 10
pinyin_size = 7


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
            pinyin = convert_char_to_pinyin(character, size=pinyin_size, tone=True)

        if character.isascii():
            pinyin = torch.LongTensor([ord(character)] + [0] * (pinyin_size - 1))

        pinyin_list.append(pinyin)

    pad_size = max_length - len(pinyin_list)
    pinyin_list += [torch.LongTensor([0] * pinyin_size)] * pad_size

    return torch.vstack(pinyin_list).view(-1).byte()


class GlyphDenseEmbedding(nn.Module):

    def __init__(self, font_size=32):
        super(GlyphDenseEmbedding, self).__init__()
        self.font_size = font_size
        self.embeddings = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 56),
            nn.Tanh()
        )

    def forward(self, images):
        batch_size = len(images)
        images = images.view(batch_size, -1) / 255.
        return self.embeddings(images)

    @staticmethod
    def from_pretrained(pretrained_model_path):
        glyph_embedding = torch.load(pretrained_model_path)
        return glyph_embedding

glyph_model = None

def embedding_glyph_entity(entity):
    global glyph_model
    if glyph_model is None:
        glyph_model = GlyphDenseEmbedding.from_pretrained("./ptm/glyph_dense_encoder.pt")
        glyph_model.eval()

    char_tensors = []
    for character in entity:
        char_tensor = convert_char_to_image(character, 32).view(-1)
        char_tensors.append(char_tensor)

    inputs = torch.vstack(char_tensors)
    outputs = glyph_model(inputs)

    outputs = outputs.view(-1)

    return outputs


def embedding_entities(entities):
    embeddings = []
    for entity in tqdm(entities):
        embedding = embedding_entity(entity)
        glyph_embedding = embedding_glyph_entity(entity)
        embeddings.append({
            "entity": entity,
            "embedding": embedding,
            "glyph_embedding": glyph_embedding,
        })

    return embeddings


def main(txt_file):
    entities = load_entities(txt_file)
    embeddings = embedding_entities(entities)
    save_obj(embeddings, txt_file[:-4] + ".pkl")


if __name__ == '__main__':
    main("./tools/temp/1_人物.txt")
    main("./tools/temp/2_化学.txt")
    main("./tools/temp/3_医学.txt")
    main("./tools/temp/4_城市信息.txt")
    main("./tools/temp/5_工业工程.txt")
    main("./tools/temp/6_法律.txt")
    main("./tools/temp/7_生物.txt")
    main("./tools/temp/8_电子游戏.txt")
    main("./tools/temp/9_计算机.txt")
    main("./tools/temp/10_诗词.txt")
