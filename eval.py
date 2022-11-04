import argparse
import pickle

import torch
from tqdm import tqdm

from model.CSCModelV1 import CSCModel
from utils.dataset import CSCTestDataset

from train import Train


class Evaluation(object):

    def __init__(self):
        self.args = self.parse_args()
        self.test_set = CSCTestDataset(self.args)

        self.model = CSCModel().eval()
        self.model.load_state_dict(torch.load(self.args.model_path, map_location='cpu'))
        self.model.to(self.args.device)

    def evaluate(self):
        self.character_level_metrics()

    def character_level_metrics(self):
        d_tp, d_fp, d_tn, d_fn = 0, 0, 0, 0
        c_tp, c_fp, c_tn, c_fn = 0, 0, 0, 0

        progress = tqdm(range(len(self.test_set)), desc='Evaluation')
        for i in progress:
            src, tgt, inputs, targets, detection_targets = self.test_set.__getitem__(i)
            inputs = inputs.to(self.args.device)
            targets = targets.to(self.args.device)
            detection_targets = detection_targets.to(self.args.device)

            outputs, detection_outputs = self.model(inputs)
            outputs = outputs.squeeze(0).argmax(dim=1)[1:-1]

            if False:  # detection_plus
                pass
            else:
                detection_outputs = (outputs != targets).int()

            d_tp += (detection_outputs[detection_targets == 1] == 1).sum().item()
            d_fp += (detection_targets[detection_outputs == 1] != 1).sum().item()
            d_tn += (detection_outputs[detection_targets == 0] == 0).sum().item()
            d_fn += (detection_targets[detection_outputs == 0] != 0).sum().item()

            c_tp += (outputs[detection_targets == 1] == targets[detection_targets == 1]).sum().item()
            c_fp += (outputs != targets)[detection_targets == 0].sum().item()
            c_tn += (outputs == targets)[detection_targets == 0].sum().item()
            c_fn += (outputs[detection_targets == 1] != targets[detection_targets == 1]).sum().item()

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

        print("Detection Character-level Precision {}, Recall {}, F1_Score {}".format(
            *Train.compute_matrix(d_tp, d_fp, d_tn, d_fn)))
        print("Correction Character-level Precision {}, Recall {}, F1_Score {}".format(
            *Train.compute_matrix(c_tp, c_fp, c_tn, c_fn)))



    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size', type=int, default=32, help='The batch size of training.')
        parser.add_argument('--test-data', type=str, default="./data/sighan/Test/sighan15_test_set_simplified.pkl",
                            help='The file path of test data.')
        parser.add_argument('--device', type=str, default='auto',
                            help='The device for test. auto, cpu or cuda')
        parser.add_argument('--error-threshold', type=float, default=0.5,
                            help='When detection logit greater than {error_threshold}, '
                                 'the token will be treated as error.')
        parser.add_argument('--model-path', type=str, default='./output/csc-best-model.pt',
                            help='The model file path. e.g. "./output/csc-best-model.pt"')

        args = parser.parse_known_args()[0]
        print(args)

        if args.device == 'auto':
            args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            args.device = torch.device(args.device)

        print("Device:", args.device)

        return args


if __name__ == '__main__':
    evaluation = Evaluation()
    evaluation.evaluate()
