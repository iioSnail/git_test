import pickle

from torch.utils.data import Dataset

from model.common import BERT
from utils.utils import preprocess_text


class CSCDataset(Dataset):

    def __init__(self, dataset, args):
        super(CSCDataset, self).__init__()
        self.dataset = dataset
        self.args = args

    def __getitem__(self, index):
        src = self.dataset[index]['src']
        tgt = self.dataset[index]['tgt']
        return src, tgt

    def __len__(self):
        if self.args.limit_data_size and self.args.limit_data_size > 0:
            return self.args.limit_data_size
        else:
            return len(self.dataset)


class CSCTestDataset(Dataset):

    def __init__(self, args):
        super(CSCTestDataset, self).__init__()
        with open(args.test_data, mode='br') as f:
            self.dataset = pickle.load(f)

    def __getitem__(self, index):
        src = self.dataset[index]['src']
        tgt = self.dataset[index]['tgt']
        src = preprocess_text(src)
        tgt = preprocess_text(tgt)
        return src, tgt

    def __len__(self):
        return len(self.dataset)
