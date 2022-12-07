import torch
from torch import nn

from ChineseBert.datasets.bert_dataset import BertDataset
from ChineseBert.models.modeling_glycebert import GlyceBertForMaskedLM
from utils.utils import mask_tokens, mock_args, mask_sentence

# torch.autograd.set_detect_anomaly(True)
mask_id = 103

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

    def __init__(self, args, path="./ChineseBert/model/ChineseBERT-base"):
        super(ChineseBertModel, self).__init__()

        self.args = args
        self.tokenizer = ChineseBertModel.get_tokenizer(path)
        self.chinese_bert = GlyceBertForMaskedLM.from_pretrained(path)
        self.glyph_embeddings = self.chinese_bert.bert.embeddings.glyph_embeddings

        self.loss_function = nn.CrossEntropyLoss(ignore_index=0)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)

    def get_optimizer(self):
        return self.optimizer

    @staticmethod
    def get_tokenizer(path="./ChineseBert/model/ChineseBERT-base"):
        if ChineseBertModel.tokenizer is None:
            ChineseBertModel.tokenizer = BertDataset(path)

        return ChineseBertModel.tokenizer

    def forward(self, inputs):
        input_ids = inputs.inputs['input_ids']
        pinyin_ids = inputs.inputs['pinyin_ids']
        glyph_embeddings = inputs.inputs['glyph_embeddings']
        attention_mask = inputs.inputs['attention_mask']
        output_hidden = self.chinese_bert.forward(input_ids, pinyin_ids, glyph_embeddings, attention_mask)
        return output_hidden.logits

    def compute_loss(self, outputs, targets):
        vocabulary_size = outputs.size(2)
        # TODO 这里如果改成只看[MASK]，不知道效果怎么样
        return self.loss_function(outputs.view(-1, vocabulary_size), targets.view(-1))

    def get_collate_fn(self):

        def collate_fn(batch):
            src, tgt = zip(*batch)
            src, tgt = list(src), list(tgt)

            batch_size = len(src)

            max_length = 0
            input_ids_list = []
            target_ids_list = []
            pinyin_ids_list = []
            with torch.no_grad():
                for sentence in tgt:
                    target_ids, _ = self.tokenizer.tokenize_sentence(sentence)
                    confusion_sentence, mask_idx = mask_sentence(sentence)
                    input_ids, pinyin_ids = self.tokenizer.tokenize_sentence(confusion_sentence)
                    input_ids[1:-1][mask_idx] = 103
                    length = input_ids.shape[0]
                    input_ids_list.append(input_ids)
                    pinyin_ids_list.append(pinyin_ids.view(length, 8))
                    target_ids_list.append(target_ids)

                    if length > max_length:
                        max_length = length

                if max_length > 256:
                    max_length = 256

                for i in range(len(input_ids_list)):
                    length = input_ids_list[i].size(0)
                    if length >= max_length:
                        input_ids_list[i] = input_ids_list[i][:max_length]
                        target_ids_list[i] = target_ids_list[i][:max_length]
                        pinyin_ids_list[i] = pinyin_ids_list[i][:max_length]
                    else:
                        input_ids_list[i] = torch.concat([input_ids_list[i], torch.zeros(max_length - length)])
                        target_ids_list[i] = torch.concat([target_ids_list[i], torch.zeros(max_length - length)])
                        pinyin_ids_list[i] = torch.concat([pinyin_ids_list[i], torch.zeros(max_length - length, 8)], dim=0)

                input_ids = torch.vstack(input_ids_list).long().to(self.args.device)
                target_ids = torch.vstack(target_ids_list).long().to(self.args.device)
                pinyin_ids = torch.vstack(pinyin_ids_list).view(batch_size, max_length, 8).long().to(self.args.device)
                glyph_embeddings = self.glyph_embeddings(input_ids)

                attention_mask = (input_ids != 0).int()

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
        src_ids = input_ids.squeeze()[1:-1].clone()
        length = len(input_ids) - 2
        input_ids = input_ids.broadcast_to(length, length+2)

        pinyin_ids = pinyin_ids.view(-1, 8)
        pinyin_ids = pinyin_ids.broadcast_to(length, length + 2, 8)
        glyph_embeddings = self.glyph_embeddings(input_ids)
        attention_mask = (input_ids != 0).int()

        mask = torch.concat([torch.zeros(length, 1), (torch.eye(length)).long(), torch.zeros(length, 1)], dim=1).long()
        input_ids = input_ids.masked_fill(mask, mask_id)
        inputs = {
            "input_ids": input_ids,
            "pinyin_ids": pinyin_ids,
            "glyph_embeddings": glyph_embeddings,
            "attention_mask": attention_mask,
        }
        inputs = ChineseBertModelInput(inputs)

        outputs = self.forward(inputs).argmax(2)
        output = outputs.masked_select(mask.byte())
        return self.tokenizer.decode(output, src_ids)

    def test(self, sentence):
        confusion_sentence, mask_idx = mask_sentence(sentence)
        input_ids, pinyin_ids = self.tokenizer.tokenize_sentence(confusion_sentence)
        input_ids = input_ids.unsqueeze(0)
        pinyin_ids = pinyin_ids.view(-1, 8).unsqueeze(0)
        glyph_embeddings = self.glyph_embeddings(input_ids)
        attention_mask = (input_ids != 0).int()

        src_ids = input_ids.squeeze()[1:-1].clone()
        input_ids[0][1:-1][mask_idx] = 103

        inputs = {
            "input_ids": input_ids,
            "pinyin_ids": pinyin_ids,
            "glyph_embeddings": glyph_embeddings,
            "attention_mask": attention_mask,
        }
        inputs = ChineseBertModelInput(inputs)

        output_ids = self.forward(inputs).argmax(2).squeeze()[1:-1]
        output = self.tokenizer.decode(output_ids, src_ids)
        masked = self.tokenizer.decode(input_ids.squeeze()[1:-1], src_ids)
        print("masked:", masked)
        print("confuse:", confusion_sentence.replace(" ", ""))
        print("output:", output)
        print("target:", sentence)


def model_test():
    model = ChineseBertModel(mock_args(device='auto'), path="../ChineseBert/model/ChineseBERT-base").eval().to("cpu")
    model.load_state_dict(torch.load("../c_output/csc-best-model.pt", map_location='cpu'))
    model.test("中华人民共和国，简称中国，是一个位於东亚的社会主义国家")

if __name__ == '__main__':
    model_test()