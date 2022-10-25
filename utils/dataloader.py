import pickle

import torch
from torch.utils.data import DataLoader

from model.common import BERT
from utils.dataset import CSCDataset


def collate_fn(batch):
    src, tgt = zip(*batch)
    src, tgt = list(src), list(tgt)

    src = BERT.get_bert_inputs(src)
    tgt = BERT.get_bert_inputs(tgt)

    return src, tgt['input_ids'], (src['input_ids'] != tgt['input_ids']).float()


def create_dataloader(args):
    with open(args.train_data, mode='br') as f:
        train_data = pickle.load(f)

    dataset = CSCDataset(train_data)

    valid_size = int(len(dataset) * args.valid_ratio)
    train_size = len(dataset) - valid_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              collate_fn=collate_fn,
                              shuffle=True,
                              drop_last=True)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.batch_size,
                              collate_fn=collate_fn,
                              shuffle=True,
                              drop_last=True)

    return train_loader, valid_loader
