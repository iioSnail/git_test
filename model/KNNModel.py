import torch

from torch import nn
from transformers import AutoTokenizer, AutoModelForMaskedLM

"""
使用mlm任务，对每一个字进行挖空，然后让其预测，如果前20个（softmax的argmax）都没有该字，那么该字就是错字
（效果极差）
"""
class DetectionKNNModel(nn.Module):

    def __init__(self, args, k=20):
        super(DetectionKNNModel, self).__init__()
        self.args = args
        self.k = k
        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-roberta-wwm-ext")

    def predict(self, src, tgt):
        inputs = self.tokenizer(src, return_tensors='pt').to(self.args.device)
        targets = self.tokenizer(tgt, return_tensors='pt').to(self.args.device)
        outputs = self.model(**inputs).logits.squeeze()[1:-1]
        candidates = outputs.argsort(1, descending=True)[:, :self.k]
        input_tokens = inputs['input_ids'].squeeze()[1:-1]
        target_tokens = targets['input_ids'].squeeze()[1:-1]
        d_outputs = []
        for i in range(len(input_tokens)):
            d_outputs.append(int(input_tokens[i] not in candidates[i]))
        return torch.tensor(d_outputs), input_tokens != target_tokens

if __name__ == '__main__':
    model = DetectionKNNModel()
    d_outputs, d_targets = model.predict("张三爱吃火聋果", "张三爱吃火龙果")
    print(d_outputs, d_targets)