from torch import nn

from ChineseBert.datasets.bert_dataset import BertDataset
from ChineseBert.models.modeling_glycebert import GlyceBertForMaskedLM


class ChineseBertModel(nn.Module):

    def __init__(self):
        super(ChineseBertModel, self).__init__()

        self.tokenizer = BertDataset("../ChineseBert/model/ChineseBERT-base")
        self.chinese_bert = GlyceBertForMaskedLM.from_pretrained("../ChineseBert/model/ChineseBERT-base")

    def predict(self, sentence):
        input_ids, pinyin_ids = self.tokenizer.tokenize_sentence(sentence)
        length = input_ids.shape[0]
        input_ids = input_ids.view(1, length)
        pinyin_ids = pinyin_ids.view(1, length, 8)
        output_hidden = self.chinese_bert.forward(input_ids, pinyin_ids)[0]
        return self.tokenizer.decode(output_hidden.argmax(2).squeeze()[1:-1])

if __name__ == '__main__':
    bert = ChineseBertModel()
    print(bert.predict("我喜欢吃火龙果"))
    print(bert.predict("我喜欢吃火聋果"))
    print(bert.predict("今天吃什么早菜"))
    print(bert.predict("我喜换你们的饺子"))
    print(bert.predict("今天我吃了一个打平果"))
    print(bert.predict("昨天你对卧做了什么事！！"))

