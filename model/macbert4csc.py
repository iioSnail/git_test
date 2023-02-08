import warnings
from abc import ABC

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from transformers import BertForMaskedLM, BertTokenizerFast, BertTokenizer

BASE_LR = 5e-5
WEIGHT_DECAY = 0.01
BIAS_LR_FACTOR = 2
WEIGHT_DECAY_BIAS = 0
OPTIMIZER_NAME = 'AdamW'


def make_optimizer(model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = BASE_LR
        weight_decay = WEIGHT_DECAY
        if "bias" in key:
            lr = BASE_LR * BIAS_LR_FACTOR
            weight_decay = WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = getattr(torch.optim, 'AdamW')(params)
    return optimizer


def build_lr_scheduler(optimizer):
    scheduler_args = {
        "optimizer": optimizer,
    }
    scheduler_args.update({'warmup_factor': 0.01,
                           'warmup_epochs': 1024,
                           'warmup_method': 'linear',
                           'milestones': (10,),
                           'gamma': 0.9999,
                           'max_iters': 10,
                           'delay_iters': 0,
                           'eta_min_lr': 3e-07})
    scheduler = WarmupExponentialLR(**scheduler_args)
    return scheduler


def _get_warmup_factor_at_iter(
        method: str, iter: int, warmup_iters: int, warmup_factor: float
) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See https://arxiv.org/abs/1706.02677 for more details.
    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).
    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))


class FocalLoss(nn.Module):
    """
    Softmax and sigmoid focal loss.
    copy from https://github.com/lonePatient/TorchBlocks
    """

    def __init__(self, num_labels, activation_type='softmax', gamma=2.0, alpha=0.25, epsilon=1.e-9):

        super(FocalLoss, self).__init__()
        self.num_labels = num_labels
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.activation_type = activation_type

    def forward(self, input, target):
        """
        Args:
            logits: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        if self.activation_type == 'softmax':
            idx = target.view(-1, 1).long()
            one_hot_key = torch.zeros(idx.size(0), self.num_labels, dtype=torch.float32, device=idx.device)
            one_hot_key = one_hot_key.scatter_(1, idx, 1)
            logits = torch.softmax(input, dim=-1)
            loss = -self.alpha * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss = loss.sum(1)
        elif self.activation_type == 'sigmoid':
            multi_hot_key = target
            logits = torch.sigmoid(input)
            zero_hot_key = 1 - multi_hot_key
            loss = -self.alpha * multi_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss += -(1 - self.alpha) * zero_hot_key * torch.pow(logits, self.gamma) * (1 - logits + self.epsilon).log()
        return loss.mean()


class WarmupExponentialLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, gamma, last_epoch=-1, warmup_epochs=2, warmup_factor=1.0 / 3, verbose=False,
                 **kwargs):
        self.gamma = gamma
        self.warmup_method = 'linear'
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_epochs, self.warmup_factor
        )

        if self.last_epoch <= self.warmup_epochs:
            return [base_lr * warmup_factor
                    for base_lr in self.base_lrs]
        return [group['lr'] * self.gamma
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** self.last_epoch
                for base_lr in self.base_lrs]


class MacBert4CscModel(nn.Module):
    def __init__(self, args):
        super(MacBert4CscModel, self).__init__()
        self.args = args
        self.args.multi_forward_args = True
        self.bert = BertForMaskedLM.from_pretrained('hfl/chinese-macbert-base')
        self.detection = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.tokenizer = BertTokenizerFast.from_pretrained('hfl/chinese-macbert-base')
        self.w = 0.3
        self.optimizer = make_optimizer(self)
        self.lr_scheduler = build_lr_scheduler(self.optimizer)

    def forward(self, inputs, targets=None, detection_targets=None):
        encoded_text = inputs

        if targets is not None:
            text_labels = targets['input_ids']
            text_labels[text_labels == 0] = -100  # -100计算损失时会忽略
        else:
            text_labels = None

        bert_outputs = self.bert(**encoded_text, labels=text_labels, return_dict=True, output_hidden_states=True)
        # 检错概率
        prob = self.detection(bert_outputs.hidden_states[-1])

        if text_labels is None:
            # 检错输出，纠错输出
            outputs = (prob, bert_outputs.logits)
        else:
            det_loss_fct = FocalLoss(num_labels=None, activation_type='sigmoid')
            det_labels = detection_targets
            active_loss = encoded_text['attention_mask'].view(-1, prob.shape[1]) == 1
            active_probs = prob.view(-1, prob.shape[1])[active_loss]
            active_labels = det_labels[active_loss]
            det_loss = det_loss_fct(active_probs, active_labels.float())
            # 检错loss，纠错loss，检错输出，纠错输出
            outputs = (det_loss,
                       bert_outputs.loss,
                       self.sigmoid(prob).squeeze(-1),
                       bert_outputs.logits)
        return outputs

    def compute_loss(self, outputs, *args, **kwargs):
        det_loss, bert_loss, _, _ = outputs
        loss = self.w * bert_loss + (1 - self.w) * det_loss
        return loss

    def extract_outputs(self, outputs):
        if len(outputs) == 2:
            outputs = outputs[1]
        elif len(outputs) == 4:
            outputs = outputs[3]

        return outputs.argmax(dim=2)

    def get_optimizer(self):
        return self.optimizer

    def get_lr_scheduler(self):
        return self.lr_scheduler


class HuggingFaceMacBert4CscModel(nn.Module):

    def __init__(self, args):
        super(HuggingFaceMacBert4CscModel, self).__init__()
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained("shibing624/macbert4csc-base-chinese")
        self.model = BertForMaskedLM.from_pretrained("shibing624/macbert4csc-base-chinese")

    def predict(self, src):
        src = ' '.join(src.replace(" ", ""))
        texts = [src]
        outputs = self.model(**self.tokenizer(texts, return_tensors='pt').to(self.args.device)).logits
        return self.tokenizer.decode(outputs.argmax(-1)[0], skip_special_tokens=True).replace(' ', '')
