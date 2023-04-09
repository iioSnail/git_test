import argparse
import multiprocessing
import os.path
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, StochasticWeightAveraging

from common.callbacks import CheckpointCallback, SimpleProgressBar, TestMetricsCallback, TestCallback, \
    TrainMetricsCallback
from common.stochastic_weight_avg import CscStochasticWeightAveraging
from utils.dataloader import create_dataloader, create_test_dataloader
from utils.log_utils import log
from utils.utils import setup_seed, mkdir


class C_Train(object):

    def __init__(self):
        super(C_Train, self).__init__()
        self.args = self.parse_args()

        self.model = self.model_select()

    def model_select(self):
        model = self.args.model.lower()
        if model == 'bert':
            from models.BertCorrectionModel import BertCSCModel
            return BertCSCModel(self.args)

        if model == 'multimodalbert':
            from models.MultiModalBert import MultiModalBertCscModel
            return MultiModalBertCscModel(self.args)

        if model == 'mymodel':
            from models.MyModel import MyModel
            return MyModel(self.args)

        if model == 'pinyinmymodel':
            from models.PinyinMyModel import MyModel
            return MyModel(self.args)

    def train(self):
        collate_fn = self.model.collate_fn if 'collate_fn' in dir(self.model) else None
        tokenizer = self.model.tokenizer if hasattr(self.model, 'tokenizer') else None
        train_loader, valid_loader = create_dataloader(self.args, collate_fn, tokenizer)

        checkpoint_callback = CheckpointCallback(dir_path=self.args.ckpt_dir)

        ckpt_path = None
        if self.args.resume:
            if not os.path.exists(checkpoint_callback.ckpt_path):
                log.warning("Resume failed due to can't find checkpoint file at " + str(checkpoint_callback.ckpt_path))
                log.warning("Training without resuming!")
            else:
                ckpt_path = checkpoint_callback.ckpt_path
                log.info("Resume training from last checkpoint.")

        if not self.args.resume and self.args.finetune:
            self.model.load_state_dict(torch.load(self.args.ckpt_path)['state_dict'])

        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=3,
            mode='min',
        )

        limit_train_batches = None
        limit_val_batches = None
        if self.args.limit_batches > 0:
            limit_train_batches = self.args.limit_batches
            limit_val_batches = int(self.args.limit_batches * self.args.valid_ratio / (1 - self.args.valid_ratio))

        precision = '16'
        if str(self.args.device) == 'cpu':
            precision = '32-true'

        train_metrics_callback = TrainMetricsCallback()

        trainer = pl.Trainer(
            default_root_dir=self.args.work_dir,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            callbacks=[checkpoint_callback,
                       early_stop_callback,
                       train_metrics_callback,
                       SimpleProgressBar(train_metrics_callback),
                       CscStochasticWeightAveraging(train_metrics_callback),
                       ],
            max_epochs=self.args.epochs,
            num_sanity_val_steps=0,
            enable_progress_bar=False,  # Use custom progress bar
            precision=precision,
            gradient_clip_val=0.5,
            gradient_clip_algorithm="norm"
        )

        trainer.fit(self.model,
                    train_dataloaders=train_loader,
                    val_dataloaders=valid_loader,
                    ckpt_path=ckpt_path
                    )

    def test(self):
        trainer = pl.Trainer(
            default_root_dir=self.args.work_dir,
            callbacks=[TestMetricsCallback()]
        )

        assert self.args.ckpt_path and os.path.exists(self.args.ckpt_path), \
            "Checkpoint file is not found! ckpt_path:%s" % self.args.ckpt_path

        trainer.test(self.model, dataloaders=create_test_dataloader(self.args), ckpt_path=self.args.ckpt_path)

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str, default='bert',
                            help='The model name you want to evaluate.')
        parser.add_argument('--data', type=str, default=None, help='The data you want to load. e.g. wang271k.')
        parser.add_argument('--datas', type=str, default=None,
                            help='The data you want to load together. e.g. sighan15train,sighan14train')
        parser.add_argument('--valid-ratio', type=float, default=0.2,
                            help='The ratio of splitting validation set.')
        parser.add_argument('--batch-size', type=int, default=32,
                            help='The batch size of training, validation and test.')
        parser.add_argument('--workers', type=int, default=-1,
                            help="The num_workers of dataloader. -1 means auto select.")
        parser.add_argument('--work-dir', type=str, default='./outputs',
                            help='The path of output files while running, '
                                 'including model state file, tensorboard files, etc.')
        parser.add_argument('--ckpt-dir', type=str, default=None,
                            help='The filepath of last checkpoint and best checkpoint. '
                                 'The default value is ${work_dir}')
        parser.add_argument('--epochs', type=int, default=100, help='The number of training epochs.')
        parser.add_argument('--resume', action='store_true', help='Resume training.')
        parser.add_argument('--no-resume', dest='resume', action='store_false', help='Not Resume training.')
        parser.set_defaults(resume=True)
        parser.add_argument('--limit-batches', type=int, default=-1,
                            help='Limit the batches of datasets for quickly testing if your model works.'
                                 '-1 means that there\'s no limit.')
        parser.add_argument('--test', action='store_true', default=False, help='Test model.')
        parser.add_argument('--ckpt-path', type=str, default=None,
                            help='The filepath of checkpoint for test. '
                                 'Default: ${ckpt_dir}/best.ckpt')
        parser.add_argument('--finetune', action='store_true', default=False,
                            help="The finetune flag means that the training into the fine-tuning phase.")

        ###############################################################################################################

        parser.add_argument('--model-path', type=str, default=None,
                            help='The filepath of pretrain model.')
        parser.add_argument('--bert-path', type=str, default='../drive/MyDrive/MultiModalBertModel/multi-modal-bert.pt')

        parser.add_argument('--data-type', type=str, default="none",
                            help='The type of training data.')
        parser.add_argument('--train-data', type=str, default="./data/Wang271K_processed.pkl",
                            help='The file path of training data.')

        parser.add_argument('--device', type=str, default='auto',
                            help='The device for training. auto, cpu or cuda')
        parser.add_argument('--seed', type=int, default=0, help='The random seed.')

        parser.add_argument('--output-path', type=str, default='./c_output',
                            help='The path of output files while running, '
                                 'including model state file, tensorboard files, etc.')

        parser.add_argument('--error-threshold', type=float, default=0.5,
                            help='When detection logit greater than {error_threshold}, '
                                 'the token will be treated as error.')
        parser.add_argument('--eval', action='store_true', default=False, help='Eval model after every epoch.')
        parser.add_argument('--max-length', type=int, default=256,
                            help='The max length of sentence. Sentence will be truncated if its length long than it.')

        args = parser.parse_known_args()[0]
        print(args)

        if args.device == 'auto':
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            args.device = args.device

        print("Device:", args.device)

        setup_seed(args.seed)
        mkdir(args.work_dir)
        args.work_dir = Path(args.work_dir)

        if args.ckpt_dir is None:
            args.ckpt_dir = args.work_dir
        else:
            mkdir(args.ckpt_dir)
            args.ckpt_dir = Path(args.ckpt_dir)

        if args.workers < 0:
            if args.device == 'cpu':
                args.workers = 0
            else:
                args.workers = os.cpu_count()

        multiprocessing.set_start_method("spawn", force=True)

        return args


if __name__ == '__main__':
    train = C_Train()
    if train.args.test:
        train.test()
    else:
        train.train()
