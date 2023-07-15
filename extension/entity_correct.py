import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast, AutoModelForTokenClassification

from utils.str_utils import is_chinese
from utils.utils import load_obj, convert_char_to_pinyin


class QueryEntity(object):

    def __init__(self, pkl_path):
        self.entities, self.embeddings = self.load_entity_embeddings(pkl_path)

        self.max_length = 10
        self.pinyin_size = 6

    def load_entity_embeddings(self, pkl_path):
        entity_embeddings = load_obj(pkl_path)

        entities = []
        embeddings = []
        for item in entity_embeddings:
            entities.append(item['entity'])
            embeddings.append(item['embedding'])

        embeddings = torch.vstack(embeddings)

        return entities, embeddings

    def embedding_entity(self, entity):
        pinyin_list = []
        for character in entity:
            pinyin = torch.LongTensor([0] * self.pinyin_size)
            if is_chinese(character):
                pinyin = convert_char_to_pinyin(character, size=self.pinyin_size)

            if character.isascii():
                pinyin = torch.LongTensor([ord(character)] + [0] * (self.pinyin_size - 1))

            pinyin_list.append(pinyin)

        pad_size = self.max_length - len(pinyin_list)
        pinyin_list += [torch.LongTensor([0] * self.pinyin_size)] * pad_size

        return torch.vstack(pinyin_list).view(-1)

    def query_entity(self, entity):
        embedding = self.embedding_entity(entity)
        sims = F.cosine_similarity(embedding, self.embeddings)
        index = sims.argmax(-1)

        if sims[index] < 0.95:
            return entity

        return self.entities[index]


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
