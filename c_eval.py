import argparse
import pickle
import random
from pathlib import Path

import torch
from tqdm import tqdm

from model.BertCorrectionModel import BertCorrectionModel
from model.MDCSpell import MDCSpellModel
from model.MDCSpellPlus import MDCSpellPlusModel
from model.macbert4csc import HuggingFaceMacBert4CscModel
from utils.dataset import CSCTestDataset, CSCDataset
from utils.metrics import CSCMetrics
from utils.utils import restore_special_tokens


class Evaluation(object):

    def __init__(self):
        self.args = self.parse_args()
        if self.args.data_type == 'Wang271K':
            # FIXME , Please use args to set filepath.
            with open('./data/Wang271K_processed.pkl', mode='br') as f:
                train_data = pickle.load(f)

            # Because this dataset is large, you may be just eval the part of it.
            random.shuffle(train_data)
            self.test_set = CSCDataset(train_data, self.args)
        elif self.args.data_type == 'sighan':
            self.test_set = CSCTestDataset(self.args)

        if self.args.model == 'MDCSpell':
            self.model = MDCSpellModel(self.args).eval()
        elif self.args.model == 'ChineseBertModel':
            from model.ChineseBertModel import ChineseBertModel
            self.model = ChineseBertModel(self.args).eval()
        elif self.args.model == 'Bert':
            self.model = BertCorrectionModel(self.args).eval()
        elif self.args.model == 'MultiModalBert':
            from model.MultiModalBert import MultiModalBertCorrectionModel
            self.model = MultiModalBertCorrectionModel(self.args).eval()
        elif self.args.model == 'MultiModalBert_temp':
            from model.MultiModalBert_temp import MultiModalBertCorrectionModel
            self.model = MultiModalBertCorrectionModel(self.args).eval()
        elif self.args.model == 'MDCSpellPlus':
            self.model = MDCSpellPlusModel(self.args).eval()
        elif self.args.model == 'HuggingFaceMacBert4Csc':
            self.model = HuggingFaceMacBert4CscModel(self.args).eval()
        elif self.args.model == 'MacBert4Csc':
            from model.macbert4csc import MacBert4CscModel
            self.model = MacBert4CscModel(self.args).eval()
        elif self.args.model == 'MultiModalMacBert4Csc':
            from model.multimodal_macbert4csc import MacBert4CscModel
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
            c_output = self.model.predict(src, tgt)
            c_output = restore_special_tokens(src, c_output)

            csc_metrics.add_sentence(src, tgt, c_output)

        csc_metrics.print_results()
        # csc_metrics.print_errors()

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--test-data', type=str, default="./datasets/sighan_2015_test.csv",
                            help='The file path of test data.')
        parser.add_argument('--data-type', type=str, default="sighan",
                            help='The type of test data.')
        parser.add_argument('--limit-data-size', type=int, default=-1,
                            help='Limit the data size of the Wang271K for quickly testing if your model works.'
                                 '-1 means that there\'s no limit.')
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

