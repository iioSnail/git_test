from torch.utils.data import Dataset
from pathlib import Path

from utils.confusions import confuse_word
from utils.log_utils import log

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]


class CSCDataset(Dataset):

    def __init__(self, data_name: str, filepath=None, **kwargs):
        super(CSCDataset, self).__init__()

        if filepath is not None:
            filepath = filepath
        else:
            filepath = self.get_filepath_by_name(data_name)

        self.data, self.error_data = self.load_data_from_csv(filepath)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def load_data_from_csv(self, filepath):
        log.info("Load dataset from %s", filepath)
        with open(filepath, mode='r', encoding='utf-8') as f:
            lines = f.readlines()

        data = []
        error_data = []
        for line in lines[1:]:
            items = line.split(",")
            src = items[0].strip()
            tgt = items[1].strip()

            src = ' '.join(src.replace(" ", "").replace(u"\u3000", ""))
            tgt = ' '.join(tgt.replace(" ", "").replace(u"\u3000", ""))

            if len(src) == len(tgt):
                data.append((src, tgt))
            else:
                error_data.append((src, tgt))

        log.info("Load completed. Success num: %d, Failure num: %d.", len(data), len(error_data))

        return data, error_data

    def get_filepath_by_name(self, data_name):
        filepath = None

        data_name = data_name.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
        if data_name in ['sighan15test', 'sighan2015test']:
            filepath = ROOT / 'datasets' / 'sighan_2015_test.csv'
        elif data_name in ['sighan14test', 'sighan2014test']:
            filepath = ROOT / 'datasets' / 'sighan_2014_test.csv'
        elif data_name in ['sighan13test', 'sighan2013test']:
            filepath = ROOT / 'datasets' / 'sighan_2013_test.csv'
        elif data_name in ['sighan15testrevise', 'sighan2015testrevise']:
            filepath = ROOT / 'datasets' / 'sighan_2015_test_revise.csv'
        elif data_name in ['sighan14testrevise', 'sighan2014testrevise']:
            filepath = ROOT / 'datasets' / 'sighan_2014_test_revise.csv'
        elif data_name in ['sighan13testrevise', 'sighan2013testrevise']:
            filepath = ROOT / 'datasets' / 'sighan_2013_test_revise.csv'
        elif data_name in ['sighan15train', 'sighan2015train']:
            filepath = ROOT / 'datasets' / 'sighan_2015_train.csv'
        elif data_name in ['sighan14train', 'sighan2014train']:
            filepath = ROOT / 'datasets' / 'sighan_2014_train.csv'
        elif data_name in ['sighan13train', 'sighan2013train']:
            filepath = ROOT / 'datasets' / 'sighan_2013_train.csv'
        elif data_name in ['wang271k']:
            filepath = ROOT / 'datasets' / 'wang271k.csv'
        elif data_name in ['cscdimetrain']:
            filepath = ROOT / 'datasets' / 'cscd_ime_train.csv'
        elif data_name in ['cscdimetest']:
            filepath = ROOT / 'datasets' / 'cscd_ime_test.csv'
        elif data_name in ['cscdimetestsm']:
            filepath = ROOT / 'datasets' / 'cscd_ime_test_sm.csv'
        elif data_name in ['cscdimedev']:
            filepath = ROOT / 'datasets' / 'cscd_ime_dev.csv'
        elif data_name in ['cscdime2m']:
            filepath = ROOT / 'datasets' / 'cscd_ime_2m.csv'
        elif data_name in ['mcsctrain']:
            filepath = ROOT / 'datasets' / 'mcsc_train.csv'
        elif data_name in ['mcscdev']:
            filepath = ROOT / 'datasets' / 'mcsc_dev.csv'
        elif data_name in ['mcsctest']:
            filepath = ROOT / 'datasets' / 'mcsc_test.csv'
        elif data_name in ['mcscsm']:
            filepath = ROOT / 'datasets' / 'mcsc_sm.csv'
        elif data_name in ['customdata']:
            filepath = ROOT / 'datasets' / 'custom_data.csv'
        elif data_name in ['eclaw']:
            filepath = ROOT / 'datasets' / 'ec_law.csv'
        elif data_name in ['ecmed']:
            filepath = ROOT / 'datasets' / 'ec_med.csv'
        elif data_name in ['ecodw']:
            filepath = ROOT / 'datasets' / 'ec_odw.csv'

        if filepath is None:
            raise Exception("Can't find data file:%s" % data_name)

        return filepath


class WordsDataset(Dataset):

    def __init__(self, data_name: str, filepath=None, limit_size=0, **kwargs):
        super(WordsDataset, self).__init__()

        if filepath is not None:
            filepath = filepath
        else:
            filepath = self.get_filepath_by_name(data_name)

        self.words = self.load_words_from_txt(filepath)
        if limit_size > 0:
            self.words = self.words[:limit_size]

    def __getitem__(self, index):
        word = self.words[index]
        src = confuse_word(word)
        tgt = word
        src = ' '.join(src).replace("?", "[MASK]")
        tgt = ' '.join(tgt)
        return src, tgt

    def __len__(self):
        return len(self.words)

    def load_words_from_txt(self, filepath):
        log.info("Load dataset from %s", filepath)
        with open(filepath, mode='r', encoding='utf-8') as f:
            lines = f.readlines()

        words = []
        for item in lines:
            item = item.strip().replace(" ", "").replace(u"\u3000", "")

            if item != "":
                words.append(item)

        log.info("Load completed. Success num: %d.", len(words), )

        return words

    def get_filepath_by_name(self, data_name):
        filepath = ROOT / 'datasets' / ('%s.txt' % data_name)

        if filepath is None:
            raise Exception("Can't find data file:" % data_name)

        return filepath
