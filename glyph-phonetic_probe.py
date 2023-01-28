import argparse
import collections
import os
import traceback
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from model.GlyphPhoneticBert import GlyphPhoneticBertModel
from utils.dataloader import create_dataloader
from utils.utils import setup_seed, mkdir


class GlyphPhoneticProbeTrain(object):

    def __init__(self):
        super(GlyphPhoneticProbeTrain, self).__init__()
        self.args = self.parse_args()
        if self.args.model == "bert":
            self.model = GlyphPhoneticBertModel(self.args).train().to(self.args.device)
        else:
            raise Exception("Unknown model: " + str(self.args.model))

        def probe_collate_fn(batch):
            datas = [[], []]
            labels = []
            for data, label in batch:
                datas[0].append(data[0])
                datas[1].append(data[1])
                labels.append(label)
            return datas, torch.FloatTensor(labels)

        collate_fn = self.model.get_collate_fn() if 'get_collate_fn' in dir(self.model) else probe_collate_fn
        self.train_loader, self.valid_loader = create_dataloader(self.args, collate_fn)

        if 'get_optimizer' in dir(self.model):
            self.optimizer = self.model.get_optimizer()
        else:
            # Default Optimizer.
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

        self.total_step = 0
        self.current_epoch = 0

        self.recent_accuracy = collections.deque(maxlen=5)
        self.correction_best_accuracy = 0

        self.criteria = nn.BCELoss()

    def compute_loss(self, outputs, targets):
        return self.criteria(outputs, targets)

    def train_epoch(self):
        self.model = self.model.train()
        progress = tqdm(self.train_loader, desc="Epoch {} Training".format(self.current_epoch))
        for i, (inputs, targets) in enumerate(progress):

            if self.args.resume and self.total_step > self.current_epoch * len(self.train_loader) + i:
                # Resume the progress of training loader.
                continue
            else:
                self.args.resume = False

            inputs, targets = inputs.to(self.args.device) if 'to' in dir(inputs) else inputs, \
                              targets.to(self.args.device) if 'to' in dir(targets) else targets
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.model.compute_loss(outputs, targets) if 'compute_loss' in dir(self.model) \
                else self.compute_loss(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
            self.optimizer.step()

            self.total_step += 1

            accuracy = ((outputs >= 0.5) == targets.bool()).sum() / len(outputs)

            progress.set_postfix({
                'loss': loss.item(),
                'accuracy': accuracy.item()
            })

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
                self.save_model_state(epoch)
                print("Unexpected exception happened. The program is about to exit. Save model state to",
                      self.args.output_path)
                exit()

            # Save model at the end of every epoch.
            self.save_model_state(epoch + 1)

            if self.recent_accuracy[-1] > self.correction_best_accuracy:
                self.correction_best_accuracy = self.recent_accuracy[-1]
                self.save_model()

            if len(self.recent_accuracy) == self.recent_accuracy.maxlen \
                    and self.correction_best_accuracy > max(self.recent_accuracy):
                print("Early stop Training. The best model is saved to", self.args.model_path)
                break

        print("Finish Training. The best model is saved to", self.args.model_path)

    def save_model_state(self, epoch):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'total_step': self.total_step,
            'recent_accuracy': self.recent_accuracy,
            'correction_best_accuracy': self.correction_best_accuracy
        }, self.args.checkpoint_path)

    def save_model(self):
        torch.save(self.model.state_dict(), self.args.model_path)
        torch.save(self.model.bert, self.args.bert_path)

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
        self.recent_accuracy = checkpoint['recent_accuracy']
        self.correction_best_accuracy = checkpoint['correction_best_accuracy']

        print("Resume Training. Epoch: {}. Total Step: {}.".format(self.current_epoch, self.total_step))

    def validate(self):
        self.model = self.model.eval()

        total_correct_num = 0
        total_num = 0

        progress = tqdm(self.valid_loader, desc="Epoch {} Validation".format(self.current_epoch))
        for inputs, targets in progress:
            inputs, targets = inputs.to(self.args.device) if 'to' in dir(inputs) else inputs, \
                              targets.to(self.args.device) if 'to' in dir(targets) else targets

            outputs = self.model(inputs)
            total_correct_num += ((outputs >= 0.5) == targets.bool()).sum().item()
            total_num += len(outputs)

            progress.set_postfix({
                'accuracy': total_correct_num / total_num
            })

        accuracy = total_correct_num / total_num

        print("Accuracy: {}".format(accuracy))

        self.recent_accuracy.append(accuracy)

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str, default='bert',
                            help='The model name you want to evaluate.')
        parser.add_argument('--batch-size', type=int, default=32, help='The batch size of training.')
        parser.add_argument('--data-type', type=str, default="phonetic")
        parser.add_argument('--train-type', type=str, default="cls",
                            help='glyph, pinyin/phonetic or cls.')
        parser.add_argument('--valid-ratio', type=float, default=0.2,
                            help='The ratio of splitting validation set.')
        parser.add_argument('--device', type=str, default='auto',
                            help='The device for training. auto, cpu or cuda')
        parser.add_argument('--seed', type=int, default=-1, help='The random seed.')
        parser.add_argument('--epochs', type=int, default=100, help='The number of training epochs.')
        parser.add_argument('--output-path', type=str, default='./phonetic',
                            help='The path of output files while running, '
                                 'including model state file, tensorboard files, etc.')
        parser.add_argument('--glyph-model-path', type=str, default='./drive/MyDrive/Glyph/probe-best-model.pt')
        parser.add_argument('--phonetic-model-path', type=str, default='./drive/MyDrive/Phonetic/probe-best-model.pt')
        parser.add_argument('--resume', action='store_true', help='Resume training.')
        parser.add_argument('--no-resume', dest='resume', action='store_false', help='Not Resume training.')
        parser.set_defaults(resume=True)

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
        args.checkpoint_path = str(args.output_path / 'probe-model.pt')
        args.model_path = str(args.output_path / 'probe-best-model.pt')
        args.bert_path = str(args.output_path / 'multi-model-bert.pt')

        return args


if __name__ == '__main__':
    train = GlyphPhoneticProbeTrain()
    train.train()
