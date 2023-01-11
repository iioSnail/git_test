import argparse
import collections
import os
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.BertCorrectionModel import BertCorrectionModel
from train_base import TrainBase
from utils.dataloader import create_dataloader
from utils.utils import setup_seed, mkdir


class C_Train(object):

    def __init__(self):
        super(C_Train, self).__init__()
        self.args = self.parse_args()
        if self.args.model == "ChineseBertModel":
            from model.ChineseBertModel import ChineseBertModel
            self.model = ChineseBertModel(self.args).train().to(self.args.device)
        elif self.args.model == "Bert":
            self.model = BertCorrectionModel(self.args).train().to(self.args.device)
        else:
            raise Exception("Unknown model: " + str(self.args.model))

        collate_fn = self.model.get_collate_fn() if 'get_collate_fn' in dir(self.model) else None
        self.train_loader, self.valid_loader = create_dataloader(self.args, collate_fn)

        if 'get_optimizer' in dir(self.model):
            self.optimizer = self.model.get_optimizer()
        else:
            # Default Optimizer.
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

        self.writer = SummaryWriter(log_dir=self.args.output_path / 'runs' / 'csc_model')
        self.total_step = 0
        self.current_epoch = 0

        self.recent_correction_f1_score = collections.deque(maxlen=5)
        self.correction_best_f1_score = 0

    def train_epoch(self):
        self.model = self.model.train()
        progress = tqdm(self.train_loader, desc="Epoch {} Training".format(self.current_epoch))
        for i, (inputs, targets, detection_targets) in enumerate(progress):

            if self.args.resume and self.total_step > self.current_epoch * len(self.train_loader) + i:
                # Resume the progress of training loader.
                continue
            else:
                self.args.resume = False

            inputs, targets, detection_targets = inputs.to(self.args.device), \
                                                 targets.to(self.args.device), \
                                                 detection_targets.to(self.args.device)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.model.compute_loss(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
            self.optimizer.step()

            self.total_step += 1

            outputs = outputs.argmax(dim=2)
            matrix = self.character_level_confusion_matrix(outputs, targets['input_ids'], detection_targets, inputs.attention_mask)

            correction_matrix = TrainBase.compute_matrix(*matrix)

            self.set_progress_postfix(progress, loss=loss, correction_matrix=correction_matrix)

            self.write_scalar(loss=loss, correction_matrix=correction_matrix)

    def set_progress_postfix(self, progress, **kwargs):
        progress.set_postfix({
            'loss': kwargs['loss'].item(),
            'c_precision': kwargs['correction_matrix'][0],
            'c_recall': kwargs['correction_matrix'][1],
            'c_f1_score': kwargs['correction_matrix'][2],
        })

    def write_scalar(self, **kwargs):
        self.writer.add_scalar(tag="correction/loss", scalar_value=kwargs['loss'].item(),
                               global_step=self.total_step)
        self.writer.add_scalar(tag="correction/c_precision", scalar_value=kwargs['correction_matrix'][0],
                               global_step=self.total_step)
        self.writer.add_scalar(tag="correction/c_recall", scalar_value=kwargs['correction_matrix'][1],
                               global_step=self.total_step)
        self.writer.add_scalar(tag="correction/c_f1_score", scalar_value=kwargs['correction_matrix'][2],
                               global_step=self.total_step)

    def train(self):
        if self.args.resume:
            self.resume()

        for epoch in range(self.current_epoch, self.args.epochs):
            try:
                self.train_epoch()
                self.validate()

                self.current_epoch += 1
            except KeyboardInterrupt:
                # This error can't be raised on the windows platform, but I don't know why.
                print("Received Ctrl-C command, training is about to terminal. Save model state to",
                      self.args.output_path)
                self.save_model_state(epoch)
                exit()
            except BaseException as e:
                traceback.print_exc()
                print("Unexpected exception happened. The program is about to exit. Save model state to",
                      self.args.output_path)
                exit()

            # Save model at the end of every epoch.
            self.save_model_state(epoch + 1)

            if self.recent_correction_f1_score[-1] > self.correction_best_f1_score:
                self.correction_best_f1_score = self.recent_correction_f1_score[-1]
                self.save_model()

            if len(self.recent_correction_f1_score) == self.recent_correction_f1_score.maxlen \
                    and self.correction_best_f1_score > max(self.recent_correction_f1_score):
                print("Early stop Training. The best model is saved to", self.args.model_path)
                break

        print("Finish Training. The best model is saved to", self.args.model_path)

    def save_model_state(self, epoch):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'total_step': self.total_step,
            'recent_correction_f1_score': self.recent_correction_f1_score,
            'correction_best_f1_score': self.correction_best_f1_score
        }, self.args.checkpoint_path)

    def save_model(self):
        torch.save(self.model.state_dict(), self.args.model_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.args.model_path))

    def resume(self):
        # Resume model training.
        if not os.path.exists(self.args.checkpoint_path):
            print("There is no model file in %s, so it can't resume training. "
                  "Training will start at the beginning." % self.args.output_path)
            return

        checkpoint = torch.load(self.args.checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.total_step = checkpoint['total_step']
        self.current_epoch = checkpoint['epoch']
        self.recent_correction_f1_score = checkpoint['recent_correction_f1_score']
        self.correction_best_f1_score = checkpoint['correction_best_f1_score']

        print("Resume Training. Epoch: {}. Total Step: {}.".format(self.current_epoch, self.total_step))

    def validate(self):
        self.model = self.model.eval()

        matrix = np.zeros([4])

        progress = tqdm(self.valid_loader, desc="Epoch {} Validation".format(self.current_epoch))
        for inputs, targets, detection_targets in progress:
            inputs, targets, detection_targets = inputs.to(self.args.device), \
                                                 targets.to(self.args.device), \
                                                 detection_targets.to(self.args.device)

            outputs = self.model(inputs)
            outputs = outputs.argmax(dim=2)

            matrix += self.character_level_confusion_matrix(outputs, targets['input_ids'], detection_targets, inputs.attention_mask)

            correction_matrix = TrainBase.compute_matrix(*matrix)
            progress.set_postfix({
                'c_precision': correction_matrix[0],
                'c_recall': correction_matrix[1],
                'c_f1_score': correction_matrix[2],
            })

        c_p, c_r, c_f1 = TrainBase.compute_matrix(*matrix)
        print("Correction Precision: {}, Recall: {}, F1-Score: {}".format(c_p, c_r, c_f1))

        self.recent_correction_f1_score.append(c_f1)

    def character_level_confusion_matrix(self, outputs, targets, detection_targets, mask):
        detection_targets[mask == 0] = -1

        c_tp = (outputs[detection_targets == 1] == targets[detection_targets == 1]).sum().item()
        c_fp = (outputs != targets)[detection_targets == 0].sum().item()  # FIXME
        c_tn = (outputs == targets)[detection_targets == 0].sum().item()  # FIXME
        c_fn = (outputs[detection_targets == 1] != targets[detection_targets == 1]).sum().item()  # FIXME

        return np.array([c_tp, c_fp, c_tn, c_fn])

    @staticmethod
    def compute_matrix(tp, fp, tn, fn):
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)
        return precision, recall, f1_score

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str, default='Bert',
                            help='The model name you want to evaluate.')
        parser.add_argument('--batch-size', type=int, default=32, help='The batch size of training.')
        parser.add_argument('--train-data', type=str, default="./data/Wang271K_processed.pkl",
                            help='The file path of training data.')
        parser.add_argument('--valid-ratio', type=float, default=0.2,
                            help='The ratio of splitting validation set.')
        parser.add_argument('--device', type=str, default='auto',
                            help='The device for training. auto, cpu or cuda')
        parser.add_argument('--seed', type=int, default=0, help='The random seed.')
        parser.add_argument('--epochs', type=int, default=100, help='The number of training epochs.')
        parser.add_argument('--output-path', type=str, default='./c_output',
                            help='The path of output files while running, '
                                 'including model state file, tensorboard files, etc.')
        parser.add_argument('--resume', action='store_true', help='Resume training.')
        parser.add_argument('--no-resume', dest='resume', action='store_false', help='Not Resume training.')
        parser.set_defaults(resume=True)
        parser.add_argument('--limit-data-size', type=int, default=-1,
                            help='Limit the data size of the Wang271K for quickly testing if your model works.'
                                 '-1 means that there\'s no limit.')
        parser.add_argument('--error-threshold', type=float, default=0.5,
                            help='When detection logit greater than {error_threshold}, '
                                 'the token will be treated as error.')

        args = parser.parse_known_args()[0]
        print(args)

        if args.device == 'auto':
            args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            args.device = torch.device(args.device)

        print("Device:", args.device)

        setup_seed(args.seed)
        mkdir(args.output_path)
        args.output_path = Path(args.output_path)
        args.checkpoint_path = str(args.output_path / 'csc-model.pt')
        args.model_path = str(args.output_path / 'csc-best-model.pt')

        return args


if __name__ == '__main__':
    train = C_Train()
    train.train()
