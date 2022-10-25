import argparse
import pickle
from torch.utils.data import DataLoader
from utils.dataset import CSCDataset


class Train(object):

    def __init__(self):
        self.args = self.parse_args()

        print(self.args)

        with open("data/trainall.times2.pkl", mode='br') as f:
            train_data = pickle.load(f)

        self.train_set = CSCDataset(train_data)
        self.train_loader = DataLoader(self.train_set,
                                       # batch_size=batch_size,
                                       # collate_fn=collate_fn,
                                       shuffle=True,
                                       drop_last=True)

    def train_epoch(self):
        pass

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size', '-b', type=int, default=32, help='The batch size of training.')


        # parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
        # parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
        # parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch.yaml', help='hyperparameters path')
        # parser.add_argument('--epochs', type=int, default=1)
        # parser.add_argument('--batch-size', type=int, default=-1,
        #                     help='total batch size for all GPUs, -1 for autobatch')
        # parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640,
        #                     help='train, val image size (pixels)')
        # parser.add_argument('--rect', action='store_true', help='rectangular training')
        # parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
        # parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
        # parser.add_argument('--noval', action='store_true', help='only validate final epoch')
        # parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
        # parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
        # parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
        # parser.add_argument('--cache', type=str, nargs='?', const='ram',
        #                     help='--cache images in "ram" (default) or "disk"')
        # parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
        # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        # parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
        # parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
        # parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
        # parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
        # parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
        # parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
        # parser.add_argument('--name', default='exp', help='save to project/name')
        # parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        # parser.add_argument('--quad', action='store_true', help='quad dataloader')
        # parser.add_argument('--linear-lr', action='store_true', help='linear LR')
        # parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
        # parser.add_argument('--patience', type=int, default=100,
        #                     help='EarlyStopping patience (epochs without improvement)')
        # parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze. backbone=10, all=24')
        # parser.add_argument('--save-period', type=int, default=-1,
        #                     help='Save checkpoint every x epochs (disabled if < 1)')
        # parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
        #
        # # Weights & Biases arguments
        # parser.add_argument('--entity', default=None, help='W&B: Entity')
        # parser.add_argument('--upload_dataset', action='store_true', help='W&B: Upload dataset as artifact table')
        # parser.add_argument('--bbox_interval', type=int, default=-1,
        #                     help='W&B: Set bounding-box image logging interval')
        # parser.add_argument('--artifact_alias', type=str, default='latest',
        #                     help='W&B: Version of dataset artifact to use')

        opt = parser.parse_known_args()[0]
        return opt

if __name__ == '__main__':
    train = Train()

