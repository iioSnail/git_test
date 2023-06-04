"""
将SIGHAN测试集转换成其他模型（论文）所使用的格式
"""
import os.path
import pickle

from tqdm import tqdm
from transformers import BertTokenizer


def _read_dataset(filename):
    with open(filename, mode='r', encoding='utf-8') as f:
        lines = f.readlines()[1:]

    print("Total num:", len(lines))
    dataset = []
    for line in lines:
        items = line.split(",")
        if len(items) != 2:
            continue
        src, tgt = items
        src = src.strip()
        tgt = tgt.strip()

        if len(src) != len(tgt):
            continue

        dataset.append((src, tgt))

    print("Valid Num:", len(dataset))
    return dataset

def convert_to_realise(dataset):
    """
    论文代码地址：https://github.com/DaDaMrX/ReaLiSe
    """
    dir_path = "./realise"
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    pkl_list = []
    tsv_list = []
    for i, item in enumerate(dataset):
        src, tgt = item
        id = "A1-%d" % i
        tokens_size = [1] * len(src)
        src_idx = tokenizer(src)['input_ids']
        tgt_idx = tokenizer(tgt)['input_ids']
        lengths = len(src)

        error_tokens = []
        for j in range(1, len(src_idx)-1):
            if src_idx[j] != tgt_idx[j]:
                error_tokens.append("%d, %s" %(j, tgt[j-1]))

        if len(error_tokens) == 0:
            tsv_list.append("%s, 0\n" % id)
        else:
            tsv_list.append("%s, %s\n" % (id, ', '.join(error_tokens)))

        pkl_list.append({
            "id": id,
            "src": src,
            "tgt": tgt,
            "tokens_size": tokens_size,
            "src_idx": src_idx,
            "tgt_idx": tgt_idx,
            "lengths": lengths
        })

    pkl_file = dir_path + "/test.sighan.pkl"
    tsv_file = dir_path + "/test.sighan.lbl.tsv"

    with open(pkl_file, mode='wb') as f:
        pickle.dump(pkl_list, f)

    with open(tsv_file, mode='w', encoding='utf-8') as f:
        f.writelines(tsv_list)

    print("生成完成，文件：", pkl_file, ", ", tsv_file)


if __name__ == '__main__':
    filename = "../datasets/sighan_2015_test_revise.csv"
    dataset = _read_dataset(filename)
    convert_to_realise(dataset)