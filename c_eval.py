import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from model.BertCorrectionModel import BertCorrectionModel
from model.MDCSpell import MDCSpellModel
from model.MDCSpellPlus import MDCSpellPlusModel
from model.MultiModalBert import MultiModalBertCorrectionModel
from model.macbert4csc import HuggingFaceMacBert4CscModel, MacBert4CscModel
from utils.dataset import CSCTestDataset
from utils.metrics import CSCMetrics
from utils.utils import restore_special_tokens


class Evaluation(object):

    def __init__(self):
        self.args = self.parse_args()
        self.test_set = CSCTestDataset(self.args)

        if self.args.model == 'MDCSpell':
            self.model = MDCSpellModel(self.args).eval()
        elif self.args.model == 'ChineseBertModel':
            from model.ChineseBertModel import ChineseBertModel
            self.model = ChineseBertModel(self.args).eval()
        elif self.args.model == 'Bert':
            self.model = BertCorrectionModel(self.args).eval()
        elif self.args.model == 'MultiModalBert':
            self.model = MultiModalBertCorrectionModel(self.args).eval()
        elif self.args.model == 'MDCSpellPlus':
            self.model = MDCSpellPlusModel(self.args).eval()
        elif self.args.model == 'HuggingFaceMacBert4Csc':
            self.model = HuggingFaceMacBert4CscModel(self.args).eval()
        elif self.args.model == 'MacBert4Csc':
            self.model = MacBert4CscModel(self.args).eval()
        else:
            raise Exception("Unknown model: " + str(self.args.model))

        try:
            self.model.load_state_dict(torch.load(self.args.model_path, map_location='cpu'))
        except Exception as e:
            print(e)
            print("Load model failed.")
        self.model.to(self.args.device)

        self.error_sentences = []

    def evaluate(self):
        self.compute_metrics()

    def compute_metrics(self):
        csc_metrics = CSCMetrics()

        progress = tqdm(range(len(self.test_set)), desc='Evaluation')
        for i in progress:
            src, tgt = self.test_set.__getitem__(i)
            src, tgt = src.replace(" ", ""), tgt.replace(" ", "")
            c_output = self.model.predict(src)
            c_output = restore_special_tokens(src, c_output)

            csc_metrics.add_sentence(src, tgt, c_output)

        csc_metrics.print_results()

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--test-data', type=str, default="./datasets/sighan_2015_test.csv",
                            help='The file path of test data.')
        parser.add_argument('--device', type=str, default='auto',
                            help='The device for test. auto, cpu or cuda')
        parser.add_argument('--model', type=str, default='Bert',
                            help='The model name you want to evaluate.')
        parser.add_argument('--model-path', type=str, default='./c_output/csc-best-model.pt',
                            help='The model file path. e.g. "./output/csc-best-model.pt"')
        parser.add_argument('--output-path', type=str, default='./output',
                            help='The model file path. e.g. "./output')

        args = parser.parse_known_args()[0]
        print(args)

        if args.device == 'auto':
            args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            args.device = torch.device(args.device)

        args.output_path = Path(args.output_path)
        args.test_data = Path(args.test_data)

        print("Device:", args.device)

        return args


if __name__ == '__main__':
    evaluation = Evaluation()
    evaluation.evaluate()

