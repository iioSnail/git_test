import argparse
import pickle

import torch
from torch.utils.data import DataLoader

from model.CSCModelV1 import CSCModel
from utils.dataloader import collate_fn, create_dataloader
from utils.dataset import CSCDataset
from utils.utils import setup_seed


class Train(object):

    def __init__(self):
        self.args = self.parse_args()
        self.train_loader, self.valid_loader = create_dataloader(self.args)
        self.model = CSCModel()

    def train_epoch(self):

        for inputs, targets, detection_targets in self.train_loader:
            inputs, targets, detection_targets = inputs.to(self.args.device), \
                                                 targets.to(self.args.device), \
                                                 detection_targets.to(self.args.device)

            outputs, detection_outputs = self.model(inputs)
            print(outputs.size())
            print(detection_outputs.size())


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
    train.train_epoch()
