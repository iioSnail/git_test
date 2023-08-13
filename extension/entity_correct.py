import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizerFast, AutoModelForTokenClassification

from utils.str_utils import is_chinese
from utils.utils import load_obj, convert_char_to_pinyin, convert_char_to_image


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


class QueryEntity(object):

    def __init__(self, pkl_path):
        self.entities, self.embeddings, self.glyph_embeddings = self.load_entity_embeddings(pkl_path)

        self.glyph_model = GlyphDenseEmbedding.from_pretrained("./ptm/glyph_dense_encoder.pt")
        self.glyph_model = self.glyph_model.eval()

        self.max_length = 10
        self.pinyin_size = 7

    def load_entity_embeddings(self, pkl_path):
        entity_embeddings = load_obj(pkl_path)

        entities = []
        embeddings = []
        glyph_embeddings = []
        for item in entity_embeddings:
            entities.append(item['entity'])
            embeddings.append(item['embedding'])
            glyph_embedding = torch.concat([item['glyph_embedding'], torch.zeros(560 - len(item['glyph_embedding']))])
            glyph_embeddings.append(glyph_embedding)

        embeddings = torch.vstack(embeddings)
        glyph_embeddings = torch.vstack(glyph_embeddings)

        return entities, embeddings.byte(), glyph_embeddings

    def embedding_entity(self, entity):
        pinyin_list = []
        for character in entity:
            pinyin = torch.LongTensor([0] * self.pinyin_size)
            if is_chinese(character):
                pinyin = convert_char_to_pinyin(character, size=self.pinyin_size, tone=True)

            if character.isascii():
                pinyin = torch.LongTensor([ord(character)] + [0] * (self.pinyin_size - 1))

            pinyin_list.append(pinyin)

        pad_size = self.max_length - len(pinyin_list)
        pinyin_list += [torch.LongTensor([0] * self.pinyin_size)] * pad_size

        return torch.vstack(pinyin_list).view(-1)

    def glyph_embedding_entity(self, entity):
        char_tensors = []
        for character in entity:
            char_tensor = convert_char_to_image(character, 32).view(-1)
            char_tensors.append(char_tensor)

        inputs = torch.vstack(char_tensors)
        outputs = self.glyph_model(inputs)

        outputs = outputs.view(-1)

        return torch.concat([outputs, torch.zeros(560 - len(outputs))])

    def query_entity(self, entity, entity_class=None):
        if len(entity) > 10:
            return entity

        pinyin_tolerance = 3
        if entity_class == '人物':
            # 如果是人名的话，拼音容错低一点，也就是能跟某个名人真的对上了，才进行修改。
            pinyin_tolerance = 0

        embedding = self.embedding_entity(entity)
        sims = (embedding.byte() == self.embeddings).sum(-1)
        sims, indices = sims.sort(descending=True)
        max_value = sims[0]
        indices = indices[torch.nonzero(sims >= max_value - pinyin_tolerance).view(-1)].tolist()

        glyph_embedding = self.glyph_embedding_entity(entity)
        glyph_sims = F.cosine_similarity(glyph_embedding, self.glyph_embeddings)
        glyph_indices = torch.nonzero(glyph_sims >= 0.8).view(-1)
        glyph_indices = set(glyph_indices.tolist())

        for index in indices:
            # 拼音相似度优先级最高
            if index in glyph_indices:
                return self.entities[index]

        return entity


class MedicalNer(object):

    def __init__(self):
        self.tokenizer = BertTokenizerFast.from_pretrained('iioSnail/bert-base-chinese-medical-ner')
        self.model = AutoModelForTokenClassification.from_pretrained("iioSnail/bert-base-chinese-medical-ner")

    def extract_entities(self, sentences):
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, add_special_tokens=False)
        outputs = self.model(**inputs)
        outputs = outputs.logits.argmax(-1) * inputs['attention_mask']

        preds = []
        for i, pred_indices in enumerate(outputs):
            words = []
            start_idx = -1
            end_idx = -1
            flag = False
            for idx, pred_idx in enumerate(pred_indices):
                if pred_idx == 1:
                    start_idx = idx
                    flag = True
                    continue

                if flag and pred_idx != 2 and pred_idx != 3:
                    # 出现了不应该出现的index
                    print("Abnormal prediction results for sentence", sentences[i])
                    start_idx = -1
                    end_idx = -1
                    continue

                if pred_idx == 3:
                    end_idx = idx

                    words.append({
                        "start": start_idx,
                        "end": end_idx + 1,
                        "word": sentences[i][start_idx:end_idx + 1]
                    })
                    start_idx = -1
                    end_idx = -1
                    flag = False
                    continue

            preds.append(words)

        return preds


medical_query_entity = None
medical_ner = None


def medical_entity_correct(sentence):
    global medical_query_entity
    if medical_query_entity is None:
        medical_query_entity = QueryEntity("./ptm/entity_embeddings.pkl")

    global medical_ner
    if medical_ner is None:
        medical_ner = MedicalNer()

    entities = medical_ner.extract_entities([sentence])[0]

    if len(entities) <= 0:
        return sentence

    print(entities)

    for entity in entities:
        corrected_word = medical_query_entity.query_entity(entity['word'])

        start = entity['start']
        end = entity['end']
        sentence = sentence[:start] + corrected_word + sentence[end:]

    return sentence


if __name__ == '__main__':
    print(medical_entity_correct("暖官孕子有助于着床吗"))
    # nuan gong yun zi
    # ruan guan sui shi
    print()
