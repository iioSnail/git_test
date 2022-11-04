import pickle

from torch.utils.data import Dataset

from model.common import BERT


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

        self.tokenizer = BERT.get_tokenizer()

    def __getitem__(self, index):
        src = self.dataset[index]['src']
        tgt = self.dataset[index]['tgt']

        src_tokens = self.tokenizer(src, return_tensors='pt')
        tgt_tokens = self.tokenizer(tgt, return_tensors='pt')['input_ids'][0][1:-1]

        return src, tgt, src_tokens, tgt_tokens, (src_tokens['input_ids'][0][1:-1] != tgt_tokens).float()

    def __len__(self):
        return len(self.dataset)
