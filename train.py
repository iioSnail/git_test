import argparse
import pickle

import numpy as np
import torch
from tqdm import tqdm

from model.CSCModelV1 import CSCModel
from utils.dataloader import create_dataloader
from utils.utils import setup_seed


class Train(object):

    def __init__(self):
        self.args = self.parse_args()
        self.train_loader, self.valid_loader = create_dataloader(self.args)
        self.model = CSCModel().train().to(self.args.device)

        self.detection_optimizer = torch.optim.Adam(self.model.detection_model.get_optimized_params(), lr=self.args.lr)
        self.correction_optimizer = torch.optim.Adam(self.model.correction_model.get_optimized_params(), lr=self.args.lr)

    def train_epoch(self):
        progress = tqdm(self.train_loader, desc="Training")
        for inputs, targets, detection_targets in progress:
            inputs, targets, detection_targets = inputs.to(self.args.device), \
                                                 targets.to(self.args.device), \
                                                 detection_targets.to(self.args.device)
            self.detection_optimizer.zero_grad()
            self.correction_optimizer.zero_grad()

            outputs, detection_outputs = self.model(inputs)
            d_loss, c_loss = self.model.compute_loss(outputs, targets, detection_outputs, detection_targets)
            d_loss.backward()
            c_loss.backward()
            self.detection_optimizer.step()
            self.correction_optimizer.step()

            outputs = outputs.argmax(dim=2)
            matrix = Train.character_level_confusion_matrix(outputs, targets,
                                                            detection_outputs, detection_targets,
                                                            inputs.attention_mask)

            detection_matrix = Train.compute_matrix(*matrix[0])
            correction_matrix = Train.compute_matrix(*matrix[1])

            progress.set_postfix({
                'd_loss': d_loss.item(),
                'c_loss': c_loss.item(),
                'd_precision': detection_matrix[0],
                'd_recall': detection_matrix[1],
                'd_f1_score': detection_matrix[2],
                'c_precision': correction_matrix[0],
                'c_recall': correction_matrix[1],
                'c_f1_score': correction_matrix[2],
            })

    def train(self):
        for epoch in range(self.args.epochs):
            self.train_epoch()

            self.validate()

            torch.save({
                'model': self.model.state_dict(),
                'd_optimizer': self.detection_optimizer.state_dict(),
                'c_optimizer': self.correction_optimizer.state_dict(),
                'epoch': epoch + 1,
            }, self.args.model_path)

    def validate(self):
        matrix = np.zeros([2, 4])

        progress = tqdm(self.valid_loader, desc="Validation")
        for inputs, targets, detection_targets in progress:
            inputs, targets, detection_targets = inputs.to(self.args.device), \
                                                 targets.to(self.args.device), \
                                                 detection_targets.to(self.args.device)

            outputs, detection_outputs = self.model(inputs)
            outputs = outputs.argmax(dim=2)

            matrix += Train.character_level_confusion_matrix(outputs, targets,
                                                             detection_outputs, detection_targets,
                                                             inputs.attention_mask)

            detection_matrix = Train.compute_matrix(*matrix[0])
            correction_matrix = Train.compute_matrix(*matrix[1])
            progress.set_postfix({
                'd_precision': detection_matrix[0],
                'd_recall': detection_matrix[1],
                'd_f1_score': detection_matrix[2],
                'c_precision': correction_matrix[0],
                'c_recall': correction_matrix[1],
                'c_f1_score': correction_matrix[2],
            })

        print("Detection Precision: {}, Recall: {}, F1-Score: {}".format(
            *Train.character_level_confusion_matrix(*matrix[0])))
        print("Correction Precision: {}, Recall: {}, F1-Score: {}".format(
            *Train.character_level_confusion_matrix(*matrix[1])))

    @staticmethod
    def character_level_confusion_matrix(outputs, targets, detection_outputs, detection_targets, mask):
        detection_targets[mask == 0] = -1
        # todo mask

        d_tp, d_fp, d_tn, d_fn = 0, 0, 0, 0
        c_tp, c_fp, c_tn, c_fn = 0, 0, 0, 0

        d_tp = (detection_outputs[detection_targets == 1] == 1).sum().item()
        d_fp = (detection_targets[detection_outputs == 1] != 1).sum().item()
        d_tn = (detection_outputs[detection_targets == 0] == 0).sum().item()
        d_fn = (detection_targets[detection_outputs == 0] != 0).sum().item()

        c_tp = (outputs[detection_targets == 1] == targets[detection_targets == 1]).sum().item()
        c_fp = (outputs != targets)[detection_targets == 0].sum().item()
        c_tn = (outputs == targets)[detection_targets == 0].sum().item()
        c_fn = (outputs[detection_targets == 1] != targets[detection_targets == 1]).sum().item()

        return np.array([[d_tp, d_fp, d_tn, d_fn],
                         [c_tp, c_fp, c_tn, c_fn]])

    @staticmethod
    def compute_matrix(tp, fp, tn, fn):
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)
        return precision, recall, f1_score

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size', type=int, default=32, help='The batch size of training.')
        parser.add_argument('--train-data', type=str, default="./data/Wang271K_processed.pkl",
                            help='The file path of training data.')
        parser.add_argument('--valid-ratio', type=float, default=0.2,
                            help='The ratio of splitting validation set.')
        parser.add_argument('--model-path', type=str, default='./output/models/csc-model.pt',
                            help='The save path of the model.')
        parser.add_argument('--device', type=str, default='auto',
                            help='The device for training. auto, cpu or cuda')
        parser.add_argument('--log-interval', type=int, default='20',
                            help='Print training info every {log_interval} steps.')
        parser.add_argument('--seed', type=int, default=0, help='The random seed.')
        parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate.')
        parser.add_argument('--epochs', type=int, default=100, help='The number of training epochs.')

        args = parser.parse_known_args()[0]
        print(args)

        if args.device == 'auto':
            args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            args.device = torch.device(args.device)

        setup_seed(args.seed)
        return args


if __name__ == '__main__':
    train = Train()
    train.train()
