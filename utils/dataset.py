from torch.utils.data import Dataset
from pathlib import Path

from utils.log_utils import log

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]


class CSCDataset(Dataset):

    def __init__(self, data_name: str, filepath=None):
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
        elif data_name in ['customdata']:
            filepath = ROOT / 'datasets' / 'custom_data.csv'

        if filepath is None:
            raise Exception("Can't find data file:" % data_name)

        return filepath


