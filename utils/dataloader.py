import pickle
import random

import torch
from torch.utils.data import DataLoader

from model.common import BERT
from utils.dataset import CSCDataset, SighanTrainDataset, ConfusionMaskDataset, PhoneticProbeDataset, GlyphProbeDataset
from utils.str_utils import word_segment_targets


def create_dataloader(args, collate_fn=None, tokenizer=None):
    if 'data_type' not in dir(args) or args.data_type is None or args.data_type == 'none':
        with open(args.train_data, mode='br') as f:
            train_data = pickle.load(f)

        dataset = CSCDataset(train_data, args)
    elif args.data_type == 'sighan':
        dataset = SighanTrainDataset()
    elif args.data_type == 'confusion_mask':
        dataset = ConfusionMaskDataset(args)
    elif args.data_type == 'phonetic':
        dataset = PhoneticProbeDataset()
    elif args.data_type == 'glyph':
        dataset = GlyphProbeDataset()
    else:
        raise Exception("Unknown data type: %s!" % args.data_type)

    valid_size = int(len(dataset) * args.valid_ratio)
    train_size = len(dataset) - valid_size

    if valid_size > 0:
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    else:
        print("No any valid data.")
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
                              drop_last=True)

    if valid_dataset is None:
        return train_loader, None

    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.batch_size,
                              collate_fn=collate_fn,
                              shuffle=True,
                              drop_last=True)

    return train_loader, valid_loader


def get_word_segment_collate_fn(tokenizer, device):
    def word_segment_collate_fn(batch):
        src, tgt = zip(*batch)
        src, tgt = list(src), list(tgt)

        tgt_sents = [sent.replace(" ", "") for sent in tgt]

        src = BERT.get_bert_inputs(src, tokenizer=tokenizer)
        tgt = BERT.get_bert_inputs(tgt, tokenizer=tokenizer)

        tgt_ws_labels = word_segment_targets(tgt_sents)

        return src.to(device), tgt.to(device), (src['input_ids'] != tgt['input_ids']).float().to(
            device), tgt_ws_labels.to(device)

    return word_segment_collate_fn
