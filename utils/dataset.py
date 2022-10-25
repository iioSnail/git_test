from torch.utils.data import Dataset


class CSCDataset(Dataset):

    def __init__(self, dataset):
        super(CSCDataset, self).__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        src = self.dataset[index]['src']
        tgt = self.dataset[index]['tgt']
        return src, tgt

    def __len__(self):
        return len(self.dataset)