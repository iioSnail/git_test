import pickle

import torch
from torch.utils.data import DataLoader, ConcatDataset

from models.common import BERT
from utils.dataset import CSCDataset
from utils.log_utils import log
from utils.str_utils import word_segment_targets


def create_dataloader(args, collate_fn=None, tokenizer=None):
    dataset = None
    if args.data is not None:
        dataset = CSCDataset(args.data)
    elif args.datas is not None:
        dataset = ConcatDataset([CSCDataset(data) for data in args.datas.split(",")])
    else:
        log.exception("Please specify data or datas.")
        exit()

    valid_size = int(len(dataset) * args.valid_ratio)
    train_size = len(dataset) - valid_size

    if valid_size > 0:
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    else:
        log.warning("No any valid data.")
        train_dataset = dataset
        valid_dataset = None

    def default_collate_fn(batch):
        src, tgt = zip(*batch)
        src, tgt = list(src), list(tgt)

        src = BERT.get_bert_inputs(src, tokenizer=tokenizer, max_length=args.max_length)
        tgt = BERT.get_bert_inputs(tgt, tokenizer=tokenizer, max_length=args.max_length)

        return src.to(args.device), \
               tgt.to(args.device), \
               (src['input_ids'] != tgt['input_ids']).float().to(args.device), \
               {}  # 补充内容

    if collate_fn is None:
        collate_fn = default_collate_fn

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              collate_fn=collate_fn,
                              shuffle=True,
                              drop_last=True,
                              num_workers=args.workers)

    if valid_dataset is None:
        return train_loader, None

    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.batch_size,
                              collate_fn=collate_fn,
                              shuffle=False,
                              num_workers=args.workers)

    return train_loader, valid_loader


def create_test_dataloader(args):
    dataset = CSCDataset(args.data)

    return DataLoader(dataset,
                      batch_size=args.batch_size,
                      shuffle=False,
                      num_workers=args.workers)

