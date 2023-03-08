import torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertForMaskedLM
import torch.nn.functional as F


class SDCLModel(nn.Module):

    def __init__(self, args):
        super(SDCLModel, self).__init__()
        self.args = args
        self.args.multi_forward_args = True

        self.tokenizer = BertTokenizerFast.from_pretrained('hfl/chinese-macbert-base')
        self.model = BertForMaskedLM.from_pretrained('hfl/chinese-macbert-base')

        self.alpha = 1
        self.beta = 0.5
        self.temperature = 0.9

    def forward(self, inputs, targets=None, detection_targets=None):
        if targets is not None:
            text_labels = targets['input_ids'].clone()
            text_labels[text_labels == 0] = -100  # -100计算损失时会忽略
        else:
            text_labels = None

        # TODO 需要W*H
        bert_outputs = self.model(**inputs, labels=text_labels, return_dict=True, output_hidden_states=True)

        return bert_outputs.logits, bert_outputs.hidden_states[-1], bert_outputs.loss

        # hidden_state = self.model.bert(**inputs).last_hidden_state
        # outputs = self.model.cls(hidden_state)

    def extract_outputs(self, outputs):
        logits, _, _ = outputs
        return logits.argmax(-1)

    def compute_loss(self, outputs, targets, inputs, detect_targets, *args, **kwargs):
        logits_x, hidden_states_x, loss_x = outputs
        logits_y, hidden_states_y, loss_y = self.forward(targets, targets)

        # FIXME
        anchor_samples = hidden_states_x[detect_targets.bool()]
        positive_samples = hidden_states_y[detect_targets.bool()]
        negative_samples = hidden_states_x[~detect_targets.bool() & inputs['attention_mask'].bool()]

        # 错字和对应正确的字计算余弦相似度
        positive_sim = F.cosine_similarity(anchor_samples, positive_samples)
        # 错字与所有batch内的所有其他字计算余弦相似度
        # （FIXME，这里与原论文不一致，原论文说的是与当前句子的其他字计算，但我除了for循环，不知道该怎么写）
        negative_sim = F.cosine_similarity(anchor_samples.unsqueeze(1), negative_samples.unsqueeze(0), dim=-1)

        sims = torch.concat([positive_sim.unsqueeze(1), negative_sim], dim=1) / self.temperature
        sim_labels = torch.zeros(sims.shape[0]).long().to(self.args.device)

        loss_c = F.cross_entropy(sims, sim_labels)

        return loss_x + self.alpha * loss_y + self.beta * loss_c

    def get_optimizer(self):
        return torch.optim.AdamW(self.parameters(), lr=7e-5)

    def predict(self, src):
        src = ' '.join(src.replace(" ", ""))
        inputs = self.tokenizer(src, return_tensors='pt').to(self.args.device)
        outputs = self.forward(inputs)
        outputs = self.extract_outputs(outputs)[0][1:-1]
        return self.tokenizer.decode(outputs).replace(' ', '')
