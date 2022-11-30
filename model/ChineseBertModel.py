import torch
from torch import nn

from ChineseBert.datasets.bert_dataset import BertDataset
from ChineseBert.models.modeling_glycebert import GlyceBertForMaskedLM
from utils.utils import mask_ids

torch.autograd.set_detect_anomaly(True)


class ChineseBertModelInput(object):

    def __init__(self, inputs: dict):
        self.inputs = inputs
        self.input_ids = inputs['input_ids']
        self.pinyin_ids = inputs['pinyin_ids']
        self.glyph_embeddings = inputs['glyph_embeddings']
        self.attention_mask = inputs['attention_mask']

    def to(self, device):
        for value in self.inputs.values():
            value.to(device)

        return self


class ChineseBertModel(nn.Module):
    tokenizer = None

    def __init__(self, args):
        super(ChineseBertModel, self).__init__()

        self.args = args
        self.tokenizer = ChineseBertModel.get_tokenizer()
        self.chinese_bert = GlyceBertForMaskedLM.from_pretrained("./ChineseBert/model/ChineseBERT-base")
        self.glyph_embeddings = self.chinese_bert.bert.embeddings.glyph_embeddings

        self.loss_function = nn.CrossEntropyLoss(ignore_index=0)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)

    def get_optimizer(self):
        return self.optimizer

    @staticmethod
    def get_tokenizer():
        if ChineseBertModel.tokenizer is None:
            ChineseBertModel.tokenizer = BertDataset("./ChineseBert/model/ChineseBERT-base")

        return ChineseBertModel.tokenizer

    def forward(self, inputs):
        input_ids = inputs.inputs['input_ids']
        pinyin_ids = inputs.inputs['pinyin_ids']
        glyph_embeddings = inputs.inputs['glyph_embeddings']
        output_hidden = self.chinese_bert.forward(input_ids, pinyin_ids, glyph_embeddings)
        return output_hidden.logits

    def compute_loss(self, outputs, targets):
        vocabulary_size = outputs.size(2)
        # TODO 这里如果改成只看[MASK]，不知道效果怎么样
        return self.loss_function(outputs.view(-1, vocabulary_size), targets.view(-1))

    def get_collate_fn(self):

        def collate_fn(batch):
            mask_id = "103"

            src, tgt = zip(*batch)
            src, tgt = list(src), list(tgt)

            batch_size = len(src)

            max_length = 0
            input_ids_list = []
            pinyin_ids_list = []
            with torch.no_grad():
                for sentence in tgt:
                    input_ids, pinyin_ids = self.tokenizer.tokenize_sentence(sentence)
                    length = input_ids.shape[0]
                    input_ids_list.append(input_ids)
                    pinyin_ids_list.append(pinyin_ids.view(length, 8))

                    if length > max_length:
                        max_length = length

                if max_length > 128:
                    max_length = 128

                for i in range(len(input_ids_list)):
                    length = input_ids_list[i].size(0)
                    if length >= max_length:
                        input_ids_list[i] = input_ids_list[i][:max_length]
                        pinyin_ids_list[i] = pinyin_ids_list[i][:max_length]
                    else:
                        input_ids_list[i] = torch.concat([input_ids_list[i], torch.zeros(max_length - length)])
                        pinyin_ids_list[i] = torch.concat([pinyin_ids_list[i], torch.zeros(max_length - length, 8)], dim=0)

                input_ids = torch.vstack(input_ids_list).long().to(self.args.device)
                pinyin_ids = torch.vstack(pinyin_ids_list).view(batch_size, max_length, 8).long().to(self.args.device)
                glyph_embeddings = self.glyph_embeddings(input_ids)

                attention_mask = (input_ids != 0).int()

                target_ids = input_ids.clone()
                input_ids = mask_ids(input_ids, mask_id)

            inputs = {
                "input_ids": input_ids,
                "pinyin_ids": pinyin_ids,
                "glyph_embeddings": glyph_embeddings,
                "attention_mask": attention_mask,
            }

            return ChineseBertModelInput(inputs), target_ids, (input_ids != target_ids).float()

        return collate_fn

    def predict(self, sentence):
        input_ids, pinyin_ids = self.tokenizer.tokenize_sentence(sentence)
        input_ids = input_ids.broadcast_to(9, 11)
        mask = torch.concat([torch.zeros(9, 1), (torch.eye(9) * 103).long(), torch.zeros(9, 1)], dim=1).long()

        pinyin_ids = pinyin_ids.view(-1, 8).unsqueeze(0)
        glyph_embeddings = self.glyph_embeddings(input_ids)
        attention_mask = (input_ids != 0).int()
        inputs = {
            "input_ids": input_ids,
            "pinyin_ids": pinyin_ids,
            "glyph_embeddings": glyph_embeddings,
            "attention_mask": attention_mask,
        }
        inputs = ChineseBertModelInput(inputs)

        output_hidden = self.forward(inputs)[0]
        return self.tokenizer.decode(output_hidden.argmax(1).squeeze()[1:-1], input_ids.squeeze()[1:-1])


if __name__ == '__main__':
    bert = ChineseBertModel(None)
    print(bert.predict("在圣文森我住在NORTH UNION,这里有很有明的花地方。"))