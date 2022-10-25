from torch.utils.data import Dataset


class CSCDataset(Dataset):

    def __init__(self, training_data):
        super(CSCDataset, self).__init__()
        self.training_data = training_data

    def __getitem__(self, index):
        src = self.training_data[index]['src']
        tgt = self.training_data[index]['tgt']
        return src, tgt

    def __len__(self):
        return len(self.training_data)