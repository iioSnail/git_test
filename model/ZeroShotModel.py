from torch import nn
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertForMaskedLM, BertTokenizerFast

from model.common import BERT
from utils.confusions import is_confusion_char
from utils.utils import convert_ids_to_tokens, get_top_n


class ZeroShotModel(nn.Module):

    def __init__(self, args):
        super(ZeroShotModel, self).__init__()
        self.args = args

        # self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        # self.model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.tokenizer = BertTokenizerFast.from_pretrained('hfl/chinese-macbert-base')
        self.model = BertForMaskedLM.from_pretrained('hfl/chinese-macbert-base')

    def predict(self, src, tgt=""):
        src_list = list(src.replace(" ", ""))
        tgt_list = list(tgt.replace(" ", ""))
        if len(src_list) == len(tgt_list):
            for i in range(len(src_list)):
                if src_list[i] != tgt_list[i]:
                    src_list[i] = '[MASK]'
        src = ' '.join(src_list)

        inputs = self.tokenizer(src, return_tensors='pt').to(self.args.device)
        outputs = self.model(**inputs).logits.squeeze()[1:-1, :]
        tokens_list = get_top_n(outputs, self.tokenizer, 10)

        tokens = []
        for i in range(len(tokens_list)):
            token_list = tokens_list[i]
            found = False
            for token in token_list:
                if token == src_list[i]:
                    tokens.append(token)
                    found = True
                    break

                if is_confusion_char(token, src_list[i]):
                    tokens.append(token)
                    found = True
                    break

            if not found:
                tokens.append(token_list[0])

        return ''.join(tokens)
