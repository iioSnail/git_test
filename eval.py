import argparse
import pickle
from pathlib import Path

import torch
from tqdm import tqdm

from model.CSCModelV1 import CSCModel
from utils.dataset import CSCTestDataset

from train import Train
from utils.utils import save_obj, load_obj, render_color_for_text, compare_text


class Evaluation(object):

    def __init__(self):
        self.args = self.parse_args()
        self.test_set = CSCTestDataset(self.args)

        self.model = CSCModel().eval()
        self.model.load_state_dict(torch.load(self.args.model_path, map_location='cpu'))
        self.model.to(self.args.device)

        self.error_sentences = []

    def evaluate(self):
        self.character_level_metrics()

        save_obj(self.error_sentences, self.args.output_path / self.args.test_data.name.replace(".pkl", ".result.pkl"))

        self.print_error_sentences()

    def character_level_metrics(self):
        d_tp, d_fp, d_tn, d_fn = 0, 0, 0, 0
        c_tp, c_fp, c_tn, c_fn = 0, 0, 0, 0

        progress = tqdm(range(len(self.test_set)), desc='Evaluation')
        for i in progress:
            src, tgt = self.test_set.__getitem__(i)
            outputs, output_rendered, detection_rendered = self.model.predict(src)

            detection_targets = torch.tensor(compare_text(src, tgt)).int().to(self.args.device)
            detection_outputs = torch.tensor(compare_text(src, outputs)).int().to(self.args.device)
            correction_outputs = torch.tensor(compare_text(outputs, tgt)).int().to(self.args.device)

            d_tp += (detection_outputs[detection_targets == 1] == 1).sum().item()
            d_fp_ = (detection_outputs[detection_targets == 0] == 1).sum().item()
            d_fp += d_fp_
            d_tn += (detection_outputs[detection_targets == 0] == 0).sum().item()
            d_fn_ = (detection_outputs[detection_targets == 1] == 0).sum().item()
            d_fn += d_fn_

            c_tp += (correction_outputs[detection_targets == 1] == 0).sum().item()
            c_fp_ = (detection_outputs[detection_targets == 0] == 1).sum().item()
            c_fp += c_fp_
            c_tn += (detection_outputs[detection_targets == 0] == 0).sum().item()
            c_fn_ = (detection_outputs[detection_targets == 1] == 0).sum().item()
            c_fn += c_fn_

            d_precision, d_recall, d_f1 = Train.compute_matrix(d_tp, d_fp, d_tn, d_fn)
            c_precision, c_recall, c_f1 = Train.compute_matrix(c_tp, c_fp, c_tn, c_fn)

            progress.set_postfix({
                'd_precision': d_precision,
                'd_recall': d_recall,
                'd_f1_score': d_f1,
                'c_precision': c_precision,
                'c_recall': c_recall,
                'c_f1_score': c_f1,
            })

            if d_fp_ or d_fn_ or c_fp_ or c_fn_:
                self.error_sentences.append({
                    "source": render_color_for_text(src, compare_text(src, tgt), 'yellow'),
                    "target": render_color_for_text(tgt, compare_text(src, tgt), 'green'),
                    "detect": detection_rendered,
                    "correct": output_rendered
                })

        print("Detection Character-level Precision {}, Recall {}, F1_Score {}".format(
            *Train.compute_matrix(d_tp, d_fp, d_tn, d_fn)))
        print("Correction Character-level Precision {}, Recall {}, F1_Score {}".format(
            *Train.compute_matrix(c_tp, c_fp, c_tn, c_fn)))

    def print_error_sentences(self):
        for item in self.error_sentences:
            print("Source:", item['source'])
            print("Detect:", item['detect'])
            print("Correct:", item['correct'])
            print("Target:", item['target'])
            print("-" * 30)

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--test-data', type=str, default="./data/sighan/Test/sighan15_test_set_simplified.pkl",
                            help='The file path of test data.')
        parser.add_argument('--device', type=str, default='auto',
                            help='The device for test. auto, cpu or cuda')
        parser.add_argument('--model-path', type=str, default='./output/csc-best-model.pt',
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
    # evaluation.error_sentences = load_obj('output/sighan15_test_set_simplified.result.pkl')
    evaluation.print_error_sentences()
