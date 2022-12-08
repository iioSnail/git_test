import argparse
import pickle
from pathlib import Path

import torch
from tqdm import tqdm

from model.BertDetectionModel import BertDetectionModel
from model.CLModel import DetectionCLModel
from model.CSCModelV1 import CSCModel
from model.KNNModel import DetectionKNNModel
from utils.dataset import CSCTestDataset

from train import Train
from utils.utils import save_obj, load_obj, render_color_for_text, compare_text


class Evaluation(object):

    def __init__(self):
        self.args = self.parse_args()
        self.test_set = CSCTestDataset(self.args)

        if self.args.model == "CL":
            self.model = DetectionCLModel(self.args).eval()
        elif self.args.model == "KNN":
            self.model = DetectionKNNModel(self.args, k=1).eval()
        elif self.args.model == "BertDetectionModel":
            self.model = BertDetectionModel(self.args).train().to(self.args.device)
        else:
            raise Exception("Unknown model: " + str(self.args.model))

        self.model.load_state_dict(torch.load(self.args.model_path, map_location='cpu'))
        self.model.to(self.args.device)

        self.error_sentences = []

    def evaluate(self):
        self.character_level_metrics()
        self.print_error_sentences()

    def character_level_metrics(self):
        d_tp, d_fp, d_tn, d_fn = 0, 0, 0, 0

        progress = tqdm(range(len(self.test_set)), desc='Evaluation')
        for i in progress:
            src, tgt = self.test_set.__getitem__(i)
            d_outputs, d_targets = self.model.predict(src, tgt)

            d_tp += (d_outputs[d_targets == 1] == 1).sum().item()
            d_fp_ = (d_outputs[d_targets == 0] == 1).sum().item()
            d_fp += d_fp_
            d_tn += (d_outputs[d_targets == 0] == 0).sum().item()
            d_fn_ = (d_outputs[d_targets == 1] == 0).sum().item()
            d_fn += d_fn_

            d_precision, d_recall, d_f1 = Train.compute_matrix(d_tp, d_fp, d_tn, d_fn)

            progress.set_postfix({
                'd_precision': d_precision,
                'd_recall': d_recall,
                'd_f1_score': d_f1
            })

            if (d_outputs != d_targets).sum() > 0:
                self.error_sentences.append({
                    "src": render_color_for_text(src, d_targets, "red"),
                    "pred": render_color_for_text(src, d_outputs, "red"),
                    "tgt": render_color_for_text(tgt, d_targets, "green")
                })

        print("Detection Character-level Precision {}, Recall {}, F1_Score {}".format(
            *Train.compute_matrix(d_tp, d_fp, d_tn, d_fn)))

    def print_error_sentences(self):
        for item in self.error_sentences:
            print("Source:", item['src'])
            print("Detect:", item['pred'])
            print("Target:", item['tgt'])
            print("-" * 30)

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--test-data', type=str, default="./datasets/sighan_2015_test.csv",
                            help='The file path of test data.')
        parser.add_argument('--device', type=str, default='auto',
                            help='The device for test. auto, cpu or cuda')
        parser.add_argument('--model', type=str, default='KNN',
                            help='The model name you want to evaluate.')
        parser.add_argument('--model-path', type=str, default='./output/csc-best-model.pt',
                            help='The model file path. e.g. "./output/csc-best-model.pt"')
        parser.add_argument('--output-path', type=str, default='./output',
                            help='The model file path. e.g. "./output')
        parser.add_argument('--error-threshold', type=float, default=0.5,
                            help='When detection logit greater than {error_threshold}, '
                                 'the token will be treated as error.')

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

