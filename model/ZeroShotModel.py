import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertForMaskedLM, BertTokenizerFast

from model.common import BERT
from utils.confusions import is_confusion_char
from utils.utils import convert_ids_to_tokens, get_top_n


class ZeroShotDetectModel(nn.Module):
    def __init__(self, args):
        super(ZeroShotDetectModel, self).__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-roberta-wwm-ext")

        self.n = 3

    def predict(self, src, tgt=None):
        src = " ".join(src.replace(" ", ""))
        src_inputs = self.tokenizer(src, return_tensors='pt').to(self.args.device)

        old_input_ids = src_inputs['input_ids'].squeeze()[1:-1]
        input_ids = src_inputs['input_ids'].broadcast_to(src_inputs['input_ids'].shape[-1],
                                                         src_inputs['input_ids'].shape[-1])
        input_ids = input_ids.clone().fill_diagonal_(103)[1:-1]

        src_inputs['input_ids'] = input_ids
        src_inputs['token_type_ids'] = src_inputs['token_type_ids'].broadcast_to(input_ids.shape)
        src_inputs['attention_mask'] = src_inputs['attention_mask'].broadcast_to(input_ids.shape)

        outputs = self.model(**src_inputs).logits[:, 1:-1, :]
        outputs_ids = outputs.argsort(descending=True)[:, :, :self.n]

        d_outputs = torch.zeros(old_input_ids.shape).int()
        for i in range(len(old_input_ids)):
            if old_input_ids[i] not in outputs_ids[i][i]:
                d_outputs[i] = 1
        if tgt:
            tgt = " ".join(tgt.replace(" ", ""))
            tgt_inputs = self.tokenizer(tgt, return_tensors='pt').to(self.args.device)
            d_targets = (old_input_ids != tgt_inputs['input_ids'].squeeze()[1:-1]).int()
            return d_outputs, d_targets
        else:
            return d_outputs


class ZeroShotModel(nn.Module):

    def __init__(self, args):
        super(ZeroShotModel, self).__init__()
        self.args = args
        self.max_length = 76

        # self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        # self.model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.tokenizer = BertTokenizerFast.from_pretrained('hfl/chinese-macbert-base')
        self.model = BertForMaskedLM.from_pretrained('hfl/chinese-macbert-base')
        self.d_model = ZeroShotDetectModel(args)
        self.d_cpu_model = ZeroShotDetectModel(args).to('cpu')

    def predict(self, src, tgt=""):
        src_list = list(src.replace(" ", ""))
        tgt_list = list(tgt.replace(" ", ""))
        if len(src_list) == len(tgt_list):
            for i in range(len(src_list)):
                if src_list[i] != tgt_list[i]:
                    src_list[i] = '[MASK]'
        src = ' '.join(src_list)

        inputs = self.tokenizer(src, return_tensors='pt').to(self.args.device)

        # FIXME，显存不足的临时方案
        if len(src_list) <= self.max_length:
            # mask src
            src_mask = self.d_model.predict(src).bool()
            inputs['input_ids'][0][1:-1][src_mask] = 103
        else:
            if str(next(self.d_cpu_model.parameters()).device) != 'cpu':
                self.d_cpu_model.to('cpu')

            src_mask = self.d_cpu_model.predict(src).bool()
            inputs['input_ids'][0][1:-1][src_mask] = 103


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
