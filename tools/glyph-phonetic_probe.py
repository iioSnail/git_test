# coding:utf-8
import argparse
import os

from utils.str_utils import get_common_hanzi

os.chdir(os.path.pardir)

import pickle
import random
import traceback

import opencc
import pypinyin
import torch
from hanzi_chaizi import HanziChaizi
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.utils import mock_args

hanzi_list = get_common_hanzi()
hanzi_list = hanzi_list[:3000]

def load_obj(filepath):
    with open(filepath, "br") as f:
        return pickle.load(f)


def save_obj(obj, filepath):
    with open(filepath, "bw") as f:
        pickle.dump(obj, f)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def is_chinese(uchar):
    if uchar == u'\uf7ee':
        return False
    return '\u4e00' <= uchar <= '\u9fa5'


class PhoneticProbeDataset(Dataset):

    def __init__(self):
        super(PhoneticProbeDataset, self).__init__()
        # Get all chinese characters.
        tw2zh = opencc.OpenCC('t2s.json')
        chinese_chars = set()
        for token in hanzi_list:
            if is_chinese(token):
                chinese_chars.add(tw2zh.convert(token))
        chinese_chars = list(chinese_chars)

        # Create chinese pinyin list of chinese characters.
        chinese_chars_pinyin = []
        for char_ in chinese_chars:
            chinese_chars_pinyin.append(pypinyin.pinyin(char_, style=pypinyin.NORMAL)[0])

        # Create Positive samples.
        positive_samples = []
        for i in range(len(chinese_chars)):
            pinyin_i = chinese_chars_pinyin[i]
            for j in range(i + 1, len(chinese_chars)):
                pinyin_j = chinese_chars_pinyin[j]
                if pinyin_i == pinyin_j:
                    positive_samples.append(((chinese_chars[i], chinese_chars[j]), 1))

        # Create negative samples.
        negative_samples = []
        while True:
            i = random.randint(0, len(chinese_chars) - 1)
            j = random.randint(0, len(chinese_chars) - 1)
            if chinese_chars_pinyin[i] != chinese_chars_pinyin[j]:
                negative_samples.append(((chinese_chars[i], chinese_chars[j]), 0))

            if len(negative_samples) == len(positive_samples):
                break

        # Merge positive samples and negative samples and then shuffle them.
        dataset = positive_samples + negative_samples
        random.shuffle(dataset)
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

class GlyphProbeDataset(Dataset):
    chinese_chars_components = None

    def __init__(self, cache=True):
        super(GlyphProbeDataset, self).__init__()
        cache_path = '../cache/GlyphProbeDataset.pkl'
        if cache and os.path.exists(cache_path):
            self.dataset = load_obj(cache_path)
            print("Load GlyphProbeDataset from cache.")
            return

        # Get all chinese characters.
        tw2zh = opencc.OpenCC('t2s.json')
        chinese_chars = set()
        for token in hanzi_list:
            if is_chinese(token) and len(token) == 1:
                token = tw2zh.convert(token)
                if is_chinese(token):
                    chinese_chars.add(token)
        chinese_chars = list(chinese_chars)

        chaizi = HanziChaizi()
        chinese_chars_components = GlyphProbeDataset.get_chinese_chars_components()

        positive_samples = set()
        for u in tqdm(chinese_chars_components, desc="Init Glyph Dataset"):
            for w in chinese_chars:
                if w not in chaizi.data:
                    continue

                w_components = chaizi.query(w)
                if u in w_components:
                    if not is_chinese(u):
                        continue
                    positive_samples.add((u, w))

        negative_samples = set()
        while True:
            u = chinese_chars_components[random.randint(0, len(chinese_chars_components) - 1)]
            w = chinese_chars[random.randint(0, len(chinese_chars) - 1)]
            if not is_chinese(u):
                continue

            if not is_chinese(w):
                continue

            if (u, w) not in positive_samples:
                negative_samples.add((u, w))

            if len(negative_samples) == len(positive_samples):
                break

        positive_samples = [(item, 1) for item in positive_samples]
        negative_samples = [(item, 0) for item in negative_samples]

        # Merge positive samples and negative samples and then shuffle them.
        dataset = positive_samples + negative_samples
        random.shuffle(dataset)
        self.dataset = dataset
        if cache:
            mkdir('../cache')  # FIXME
            save_obj(self.dataset, cache_path)

    @staticmethod
    def get_chinese_chars_components():
        if GlyphProbeDataset.chinese_chars_components is not None:
            return GlyphProbeDataset.chinese_chars_components

        chaizi = HanziChaizi()
        chinese_chars_components = set()
        for values in chaizi.data.values():
            for value in values:
                for component in value:
                    chinese_chars_components.add(component)

        GlyphProbeDataset.chinese_chars_components = list(chinese_chars_components)
        return GlyphProbeDataset.chinese_chars_components

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


def create_dataloader(args, collate_fn=None):
    # dataset = PhoneticProbeDataset()
    dataset = GlyphProbeDataset()

    valid_size = int(len(dataset) * args.valid_ratio)
    train_size = len(dataset) - valid_size

    if valid_size > 0:
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    else:
        print("No any valid data.")
        train_dataset = dataset
        valid_dataset = None

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


#####################################################


def load_sota_model():
    from models.MultiModalMyModel_SOTA import MyModel
    model = MyModel(mock_args(hyper_params={}, device='cuda'))
    model.load_state_dict(torch.load("./temp/multimodal-sota.ckpt")['state_dict'])
    model = model.to("cuda")

    return model._tokenizer, model


def load_temp_model():
    from models.MultiModalMyModel_TEMP import MyModel
    model = MyModel(mock_args(hyper_params={}, device='cuda'))
    # model.load_state_dict(torch.load("./temp/multimodal-sota.ckpt")['state_dict'])
    model = model.to("cuda")

    return model._tokenizer, model


class GlyphPhoneticModel(nn.Module):

    def __init__(self):
        super(GlyphPhoneticModel, self).__init__()
        self.tokenizer, self.encoder = load_temp_model()

        self.cls = nn.Sequential(
            nn.Linear(830 * 2, 768),
            nn.ReLU(),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def build_inputs(self, chars):
        inputs = self.tokenizer(chars, return_tensors='pt', add_special_tokens=False)
        input_pinyins = self.encoder.input_helper.convert_tokens_to_pinyin_embeddings(inputs['input_ids'].view(-1))
        images = self.encoder.input_helper.convert_tokens_to_images(inputs['input_ids'].view(-1), chars)
        return inputs.to("cuda"), input_pinyins.to("cuda"), images.to("cuda")

    def forward(self, inputs):
        with torch.no_grad():
            _, a_features = self.encoder(*self.build_inputs(inputs[0]), output_hidden_states=True)
            _, b_features = self.encoder(*self.build_inputs(inputs[1]), output_hidden_states=True)

        # a_features = self.encoder(self.build_inputs(inputs[0]))
        # b_features = self.encoder(self.build_inputs(inputs[1]))

        return self.cls(torch.concat([a_features.squeeze(), b_features.squeeze()], dim=-1).squeeze(0)).view(-1)


class GlyphPhoneticProbeTrain(object):

    def __init__(self):
        super(GlyphPhoneticProbeTrain, self).__init__()
        self.args = self.parse_args()

        def probe_collate_fn(batch):
            datas = [[], []]
            labels = []
            for data, label in batch:
                datas[0].append(data[0])
                datas[1].append(data[1])
                labels.append(label)
            return datas, torch.FloatTensor(labels)

        self.train_loader, self.valid_loader = create_dataloader(self.args, probe_collate_fn)

        self.model = GlyphPhoneticModel().to("cuda")
        self.model.train()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

        self.total_step = 0
        self.current_epoch = 0

        self.criteria = nn.BCELoss()

    def compute_loss(self, outputs, targets):
        return self.criteria(outputs, targets)

    def train_epoch(self):
        self.model = self.model.train()
        progress = tqdm(self.train_loader, desc="Epoch {} Training".format(self.current_epoch))
        for i, (inputs, targets) in enumerate(progress):
            inputs, targets = inputs.to(self.args.device) if 'to' in dir(inputs) else inputs, \
                              targets.to(self.args.device) if 'to' in dir(targets) else targets

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.compute_loss(outputs, targets)
            loss.backward()
            self.optimizer.step()

            self.total_step += 1

            accuracy = ((outputs >= 0.5) == targets.bool()).sum() / len(outputs)

            progress.set_postfix({
                'loss': loss.item(),
                'accuracy': accuracy.item()
            })

    def train(self):
        for epoch in range(self.current_epoch, self.args.epochs):
            try:
                self.train_epoch()
                self.validate()

                self.current_epoch += 1
            except BaseException as e:
                traceback.print_exc()
                exit()

        print("Finish Training. The best model is saved to", self.args.model_path)

    def validate(self):
        self.model = self.model.eval()

        total_correct_num = 0
        total_num = 0

        progress = tqdm(self.valid_loader, desc="Epoch {} Validation".format(self.current_epoch))
        for inputs, targets in progress:
            inputs, targets = inputs.to(self.args.device) if 'to' in dir(inputs) else inputs, \
                              targets.to(self.args.device) if 'to' in dir(targets) else targets

            outputs = self.model(inputs)
            total_correct_num += ((outputs >= 0.5) == targets.bool()).sum().item()
            total_num += len(outputs)

            progress.set_postfix({
                'accuracy': total_correct_num / total_num
            })

        accuracy = total_correct_num / total_num

        print("Accuracy: {}".format(accuracy))

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size', type=int, default=32, help='The batch size of training.')
        parser.add_argument('--data-type', type=str, default="phonetic")
        parser.add_argument('--valid-ratio', type=float, default=0.2,
                            help='The ratio of splitting validation set.')
        parser.add_argument('--device', type=str, default='cuda',
                            help='The device for training. auto, cpu or cuda')
        parser.add_argument('--epochs', type=int, default=25, help='The number of training epochs.')

        args = parser.parse_known_args()[0]
        print(args)

        if args.device == 'auto':
            args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            args.device = torch.device(args.device)

        print("Device:", args.device)

        return args


if __name__ == '__main__':
    train = GlyphPhoneticProbeTrain()
    train.train()
