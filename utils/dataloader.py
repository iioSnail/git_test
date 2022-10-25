from model.common import BERT


def collate_fn(batch):
    src, tgt = zip(*batch)
    src, tgt = list(src), list(tgt)

    src = BERT.get_bert_inputs(src)
    tgt = BERT.get_bert_inputs(tgt)

    return src, tgt['input_ids'], (src['input_ids'] != tgt['input_ids']).float()

