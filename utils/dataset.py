import pickle

import pandas as pd
from torch.utils.data import Dataset

from model.common import BERT
from utils.utils import preprocess_text
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]


class CSCDataset(Dataset):

    def __init__(self, dataset, args):
        super(CSCDataset, self).__init__()
        self.dataset = dataset
        self.args = args

    def __getitem__(self, index):
        src = self.dataset[index]['src']
        tgt = self.dataset[index]['tgt']
        return " ".join(src), " ".join(tgt)

    def __len__(self):
        if self.args.limit_data_size and self.args.limit_data_size > 0:
            return self.args.limit_data_size
        else:
            return len(self.dataset)


class SighanTrainDataset(Dataset):

    def __init__(self):
        super(SighanTrainDataset, self).__init__()

        sighan_2013 = pd.read_csv(ROOT / 'datasets/sighan_2013_train.csv')
        sighan_2014 = pd.read_csv(ROOT / 'datasets/sighan_2014_train.csv')
        sighan_2015 = pd.read_csv(ROOT / 'datasets/sighan_2015_train.csv')

        self.dataset = pd.concat([sighan_2013, sighan_2014, sighan_2015])

    def __getitem__(self, index):
        src = self.dataset.iloc[index]['src']
        tgt = self.dataset.iloc[index]['tgt']
        src = preprocess_text(src)
        tgt = preprocess_text(tgt)
        return " ".join(src), " ".join(tgt)

    def __len__(self):
        return len(self.dataset)


class CSCTestDataset(Dataset):

    def __init__(self, args):
        super(CSCTestDataset, self).__init__()
        self.dataset = pd.read_csv(ROOT / args.test_data)

    def __getitem__(self, index):
        src = self.dataset.iloc[index]['src']
        tgt = self.dataset.iloc[index]['tgt']
        src = preprocess_text(src)
        tgt = preprocess_text(tgt)
        return " ".join(src), " ".join(tgt)

    def __len__(self):
        return len(self.dataset)
