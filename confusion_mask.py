"""
Confusion Mask定义：
若一句话中的某个字被[MASK]后，无法根据上下文确定唯一的正确字，则该位置就被称为Confusion Mask。

具体构建方法：
对于一句话，将每个字都依次进行[MASK]，然后让BERT进行预测，若BERT的预测结果与原字不一致，则认为该位置为confusion mask.
"""
import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

from utils.str_utils import is_chinese


class ConfusionMask(object):

    def __init__(self, model_path='hfl/chinese-roberta-wwm-ext'):
        self.args = self.parse_args()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_path).to(self.args.device)
        self.mask = self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]  # 103

    def get_confusion_mask(self, sentence):
        # 将句子中间增加空格
        sentence = ' '.join(sentence.replace(" ", ""))
        # 有几个字就构建几个相同的句子
        sentences = [sentence] * len(sentence.replace(" ", ""))
        inputs = self.tokenizer(sentences, return_tensors='pt')
        inputs = inputs.to(self.args.device)
        # 将对角线全部变成[MASK]
        inputs['input_ids'][:, 1:-1].fill_diagonal_(self.mask)
        outputs = self.model(**inputs).logits
        outputs = outputs[:, 1:-1, :]  # 不要<bos>和<eos>
        output = outputs.argmax(-1).diag()

        confusion_mask_result = []
        pred_sentence = self.tokenizer.convert_ids_to_tokens(output)
        sentence = sentence.split(" ")
        for i in range(len(pred_sentence)):
            if pred_sentence[i] != sentence[i] and is_chinese(sentence[i]):
                confusion_mask_result.append(i)

        return confusion_mask_result

    def generate_confusion_mask_file(self):
        """
        读取一个每一个行是一个句子的文件，然后生成confusion mask数据集
        """
        max_row = sum(1 for _ in open(self.args.file, encoding='utf-8'))

        row = 0
        try:
            with open(self.args.file, encoding='utf-8') as f:
                with open(self.args.output, mode='a+', encoding='utf-8') as fw:
                    for line in tqdm(f, total=max_row):
                        row += 1
                        if row < self.args.start_row:
                            continue

                        line = line.strip()
                        if line == "":
                            continue

                        confusion_mask_result = self.get_confusion_mask(line)
                        confusion_mask_result = '"%s"' % ','.join(map(str, confusion_mask_result))
                        fw.write("%s,%s\n" % (line, confusion_mask_result))
        except KeyboardInterrupt:
            print("Program exits. Number of rows processed: %d" % (row-1))


    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--file', type=str, default="sentences.txt")
        parser.add_argument('--output', type=str, default='sentences.csv')
        parser.add_argument('--start-row', type=int, default=-1)
        parser.add_argument('--device', type=str, default='auto',
                            help='The device for test. auto, cpu or cuda')

        args = parser.parse_known_args()[0]
        print(args)

        if args.device == 'auto':
            args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            args.device = torch.device(args.device)

        args.file = Path(args.file)
        args.output = Path(args.output)

        if args.start_row < 0:
            args.start_row = 1
            if os.path.exists(args.output):
                args.start_row = sum(1 for _ in open(args.output, encoding='utf-8')) + 1

        print("Start row:", args.start_row)
        print("Device:", args.device)

        return args


if __name__ == '__main__':
    confusion_mask = ConfusionMask()
    confusion_mask.generate_confusion_mask_file()
