from torch.utils.data import Dataset


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
        if self.args.limit_data_size > 0:
            return self.args.limit_data_size
        else:
            return len(self.dataset)