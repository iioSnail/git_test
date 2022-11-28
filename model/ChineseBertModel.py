import torch
from torch import nn

from ChineseBert.datasets.bert_dataset import BertDataset
from ChineseBert.models.modeling_glycebert import GlyceBertForMaskedLM
from model.common import BERT


class ChineseBertModel(nn.Module):

    tokenizer = None

    def __init__(self, args):
        super(ChineseBertModel, self).__init__()

        self.args = args
        self.tokenizer = ChineseBertModel.get_tokenizer()
        self.chinese_bert = GlyceBertForMaskedLM.from_pretrained("./ChineseBert/model/ChineseBERT-base")

    @staticmethod
    def get_tokenizer():
        if ChineseBertModel.tokenizer is None:
            ChineseBertModel.tokenizer = BertDataset("./ChineseBert/model/ChineseBERT-base")

        return ChineseBertModel.tokenizer

    def forward(self, inputs):
        pass

    def get_collate_fn(self):

        def collate_fn(batch):
            src, tgt = zip(*batch)
            src, tgt = list(src), list(tgt)

            tokenizer = ChineseBertModel.get_tokenizer()

            max_length = 0
            input_ids_list = []
            pinyin_ids_list = []
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
                input_ids_list[i] = torch.concat([input_ids_list[i], torch.zeros(max_length - length)])

            tokens = torch.vstack(input_ids_list)
            print(max_length)


            return src, tgt['input_ids'], (src['input_ids'] != tgt['input_ids']).float()

        return collate_fn

    def predict(self, sentence):
        input_ids, pinyin_ids = self.tokenizer.tokenize_sentence(sentence)
        length = input_ids.shape[0]
        input_ids = input_ids.view(1, length).to(self.args.device)
        pinyin_ids = pinyin_ids.view(1, length, 8).to(self.args.device)
        output_hidden = self.chinese_bert.forward(input_ids, pinyin_ids)[0]
        return self.tokenizer.decode(output_hidden.argmax(2).squeeze()[1:-1], input_ids.squeeze()[1:-1])

if __name__ == '__main__':
    bert = ChineseBertModel(None)
    print(bert.predict("在圣文森我住在NORTH UNION,这里有很有明的花地方。"))


