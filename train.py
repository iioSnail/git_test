import argparse
import collections
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.CSCModelV1 import CSCModel
from utils.dataloader import create_dataloader
from utils.utils import setup_seed, mkdir


class Train(object):

    def __init__(self):
        self.args = self.parse_args()
        self.train_loader, self.valid_loader = create_dataloader(self.args)
        self.model = CSCModel().train().to(self.args.device)

        self.detection_optimizer = torch.optim.Adam(self.model.detection_model.get_optimized_params(),
                                                    lr=self.args.d_lr)
        self.correction_optimizer = torch.optim.Adam(self.model.correction_model.get_optimized_params(),
                                                     lr=self.args.c_lr)

        self.writer = SummaryWriter(log_dir=self.args.output_path / 'runs' / 'csc_model')
        self.total_step = 0
        self.current_epoch = 0

        self.recent_detection_f1_score = collections.deque(maxlen=5)
        self.recent_correction_f1_score = collections.deque(maxlen=5)
        self.detection_best_f1_score = 0
        self.correction_best_f1_score = 0

        self.detection_stop_training = False

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
            self.detection_optimizer.zero_grad()
            self.correction_optimizer.zero_grad()

            outputs, detection_outputs = self.model(inputs)
            d_loss, c_loss = self.model.compute_loss(outputs, targets, detection_outputs, detection_targets)
            if not self.detection_stop_training:
                d_loss.backward()
                self.detection_optimizer.step()

            c_loss.backward()
            self.correction_optimizer.step()

            self.total_step += 1

            outputs = outputs.argmax(dim=2)
            matrix = self.character_level_confusion_matrix(outputs, targets,
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

            self.writer.add_scalar(tag="detection/d_loss", scalar_value=d_loss.item(),
                                   global_step=self.total_step)
            self.writer.add_scalar(tag="detection/d_precision", scalar_value=detection_matrix[0],
                                   global_step=self.total_step)
            self.writer.add_scalar(tag="detection/d_recall", scalar_value=detection_matrix[1],
                                   global_step=self.total_step)
            self.writer.add_scalar(tag="detection/d_f1_score", scalar_value=detection_matrix[2],
                                   global_step=self.total_step)

            self.writer.add_scalar(tag="correction/c_loss", scalar_value=c_loss.item(),
                                   global_step=self.total_step)
            self.writer.add_scalar(tag="correction/c_precision", scalar_value=correction_matrix[0],
                                   global_step=self.total_step)
            self.writer.add_scalar(tag="correction/c_recall", scalar_value=correction_matrix[1],
                                   global_step=self.total_step)
            self.writer.add_scalar(tag="correction/c_f1_score", scalar_value=correction_matrix[2],
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
                print(e)
                print("Unexpected exception happened. The program is about to exit. Save model state to",
                      self.args.output_path)
                exit()

            # Save model at the end of every epoch.
            self.save_model_state(epoch + 1)

            if not self.detection_stop_training and self.recent_detection_f1_score[-1] > self.detection_best_f1_score:
                self.detection_best_f1_score = self.recent_detection_f1_score[-1]
                self.save_model()

            if not self.detection_stop_training \
                    and len(self.recent_detection_f1_score) == self.recent_detection_f1_score.maxlen \
                    and self.detection_best_f1_score > max(self.recent_detection_f1_score):
                self.detection_stop_training = True
                self.load_model()
                self.recent_correction_f1_score.clear()
                continue

            if self.detection_stop_training and self.recent_correction_f1_score[-1] > self.correction_best_f1_score:
                self.correction_best_f1_score = self.recent_correction_f1_score[-1]
                self.save_model()

            if self.detection_stop_training \
                    and len(self.recent_correction_f1_score) == self.recent_correction_f1_score.maxlen \
                    and self.correction_best_f1_score > max(self.recent_correction_f1_score):
                print("Early stop Training. The best model is saved to", self.args.model_path)
                break

        print("Finish Training. The best model is saved to", self.args.model_path)

    def save_model_state(self, epoch):
        torch.save({
            'model': self.model.state_dict(),
            'd_optimizer': self.detection_optimizer.state_dict(),
            'c_optimizer': self.correction_optimizer.state_dict(),
            'epoch': epoch,
            'total_step': self.total_step,
            'recent_detection_f1_score': self.recent_detection_f1_score,
            'recent_correction_f1_score': self.recent_correction_f1_score,
            'detection_best_f1_score': self.detection_best_f1_score,
            'correction_best_f1_score': self.correction_best_f1_score,
            'detection_stop_training': self.detection_stop_training
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
        self.detection_optimizer.load_state_dict(checkpoint['d_optimizer'])
        self.correction_optimizer.load_state_dict(checkpoint['c_optimizer'])
        self.total_step = checkpoint['total_step']
        self.current_epoch = checkpoint['epoch']
        self.recent_detection_f1_score = checkpoint['recent_detection_f1_score']
        self.recent_correction_f1_score = checkpoint['recent_correction_f1_score']
        self.detection_best_f1_score = checkpoint['detection_best_f1_score']
        self.correction_best_f1_score = checkpoint['correction_best_f1_score']
        self.detection_stop_training = checkpoint['detection_stop_training']

        print("Resume Training. Epoch: {}. Total Step: {}.".format(self.current_epoch, self.total_step))

    def validate(self):
        self.model = self.model.eval()

        matrix = np.zeros([2, 4])

        progress = tqdm(self.valid_loader, desc="Epoch {} Validation".format(self.current_epoch))
        for inputs, targets, detection_targets in progress:
            inputs, targets, detection_targets = inputs.to(self.args.device), \
                                                 targets.to(self.args.device), \
                                                 detection_targets.to(self.args.device)

            outputs, detection_outputs = self.model(inputs)
            outputs = outputs.argmax(dim=2)

            matrix += self.character_level_confusion_matrix(outputs, targets,
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

        d_p, d_r, d_f1 = Train.compute_matrix(*matrix[0])
        print("Detection Precision: {}, Recall: {}, F1-Score: {}".format(d_p, d_r, d_f1))

        c_p, c_r, c_f1 = Train.compute_matrix(*matrix[1])
        print("Correction Precision: {}, Recall: {}, F1-Score: {}".format(c_p, c_r, c_f1))

        self.recent_detection_f1_score.append(d_f1)
        self.recent_correction_f1_score.append(c_f1)

    def character_level_confusion_matrix(self, outputs, targets,
                                         detection_outputs, detection_targets, mask):
        detection_targets[mask == 0] = -1
        detection_outputs[mask != 1] = -1
        detection_outputs[detection_outputs >= self.args.error_threshold] = 1
        detection_outputs[(detection_outputs < self.args.error_threshold) & (detection_outputs >= 0)] = 0

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
        parser.add_argument('--device', type=str, default='auto',
                            help='The device for training. auto, cpu or cuda')
        parser.add_argument('--seed', type=int, default=0, help='The random seed.')
        parser.add_argument('--d-lr', type=float, default=2e-5, help='The learning rate of Detection Network.')
        parser.add_argument('--c-lr', type=float, default=1e-4, help='The learning rate of Correction Network.')
        parser.add_argument('--epochs', type=int, default=100, help='The number of training epochs.')
        parser.add_argument('--error-threshold', type=float, default=0.5,
                            help='When detection logit greater than {error_threshold}, '
                                 'the token will be treated as error.')
        parser.add_argument('--output-path', type=str, default='./output',
                            help='The path of output files while running, '
                                 'including model state file, tensorboard files, etc.')
        parser.add_argument('--no-resume', type=bool, default=False, help='Resume training.')
        parser.add_argument('--limit-data-size', type=int, default=-1,
                            help='Limit the data size of the Wang271K for quickly testing if your model works.'
                                 '-1 means that there\'s no limit.')

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

        if not args.no_resume:
            args.resume = True

        return args


if __name__ == '__main__':
    train = Train()
    train.train()
