import argparse
import json
import math
import multiprocessing
import os.path
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from common.callbacks import CheckpointCallback, SimpleProgressBar, TestMetricsCallback, \
    TrainMetricsCallback, EvalInTrainMetricsCallback
from common.stochastic_weight_avg import CscStochasticWeightAveraging
from utils.dataloader import create_dataloader, create_test_dataloader
from utils.log_utils import log, init_log, add_file_handler
from utils.str_utils import is_float
from utils.utils import setup_seed, mkdir

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class C_Train(object):

    def __init__(self):
        super(C_Train, self).__init__()
        self.args = self.parse_args()

        self.model = self.model_select(self.args)

    @staticmethod
    def model_select(args):
        model = args.model.lower()
        if model == 'bert':
            from models.BertCorrectionModel import BertCSCModel
            return BertCSCModel(args)

        if model == 'pinyinmymodel':
            from models.PinyinMyModel import MyModel
            return MyModel(args)

        if model == 'multimodalmodel':
            from models.MultiModalModel import MMModelForCSC
            return MMModelForCSC(args)

        if model == 'multimodalmymodel':
            from models.MultiModalMyModel import MyModel
            return MyModel(args)

        if model == 'multimodalmymodel_sota':
            from models.MultiModalMyModel_SOTA import MyModel
            return MyModel(args)

        if model == 'multimodalmymodel_temp':
            from models.MultiModalMyModel_TEMP import MyModel
            return MyModel(args)

        if model == 'multimodalmymodel_wo_multi':
            from models.MultiModalMyModel_wo_multi import MyModel
            return MyModel(args)

        if model == 'multimodalmymodel_wo_focal_loss':
            from models.MultiModalMyModel_wo_focal_loss import MyModel
            return MyModel(args)

        if model == 'multimodalmymodel_wo_multi_and_focal_loss':
            from models.MultiModalMyModel_wo_multi_and_focal_loss import MyModel
            return MyModel(args)

        if model == 'multimodalmymodel_wo_token_embeddings':
            from models.MultiModalMyModel_wo_token_embeddings import MyModel
            return MyModel(args)

        if model == 'multimodalmymodel_wo_fix_index':
            from models.MultiModalMyModel_wo_fix_index import MyModel
            return MyModel(args)

        if model == 'multimodalmymodel_wo_multi_and_token_embeddings':
            from models.MultiModalMyModel_wo_multi_and_token_embeddings import MyModel
            return MyModel(args)

        if model == 'multimodalmymodel_wo_glyph':
            from models.MultiModalMyModel_wo_glyph import MyModel
            return MyModel(args)

        if model == 'multimodalmymodel_wo_glyph_and_token_embeddings':
            from models.MultiModalMyModel_wo_glyph_and_token_embeddings import MyModel
            return MyModel(args)

        if model == 'multimodalmymodel_wo_pinyin':
            from models.MultiModalMyModel_wo_pinyin import MyModel
            return MyModel(args)

        if model == 'multimodalmymodel_wo_pinyin_and_token_embeddings':
            from models.MultiModalMyModel_wo_pinyin_and_token_embeddings import MyModel
            return MyModel(args)

        if model == 'multimodalmymodel_new_ft':
            from models.MultiModalMyModel_new_ft import MyModel
            return MyModel(args)

        if model == 'multimodalmymodel_wo_ft':
            from models.MultiModalMyModel_wo_ft import MyModel
            return MyModel(args)

        if model == 'multimodalmymodel_cpp':
            from models.MultiModalMyModel_CPP import MyModel
            return MyModel(args)

        if model == 'multimodalmymodel_smooth':
            from models.MultiModalMyModel_smooth import MyModel
            return MyModel(args)

        if model == 'zeroshot':
            from models.zero_shot import AdjustProbByPinyin
            return AdjustProbByPinyin(args)

        if model == 'pinyinbert':
            from models.PinyinBert import BertCSCModel
            return BertCSCModel(args)

        if model == 'scope':
            from models.SCOPE import SCOPE_CSC_Model
            return SCOPE_CSC_Model(args)

        if model == 'scope-p':
            from models.ScopeWithPlugin import ScopeWithPlugin
            return ScopeWithPlugin(args)

        if model == 'scope_fl':
            from models.SCOPE_FL import SCOPE_CSC_Model
            return SCOPE_CSC_Model(args)

        if model == 'macbert4csc':
            from models.MacBert4CSC import MacBert4CSC_Model
            return MacBert4CSC_Model(args)

        if model == 'soft-masked':
            from models.soft_masked_bert import SoftMaskedBERT
            return SoftMaskedBERT(args)

        if model == 'bert-mft':
            from models.bert_mft import BertMFT
            return BertMFT(args)

        if model == 'word-csc':
            from models.WordCSCModel import ScopeWithPlugin
            return ScopeWithPlugin(args)

        raise Exception("Can't find any model!")

    def train(self):
        collate_fn = self.model.collate_fn if 'collate_fn' in dir(self.model) else None
        tokenizer = self.model.tokenizer if hasattr(self.model, 'tokenizer') else None
        train_loader, val_loader = create_dataloader(self.args, collate_fn, tokenizer)

        self.args.train_loader = train_loader
        self.args.val_loader = val_loader

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
            if self.args.ckpt_path is None:
                log.error("If you finetune your model, you must specify the pre-trained model by ckpt-path parameters.")
                exit(0)
            log.info("Load pre-trained model from " + str(self.args.ckpt_path))
            self.model.load_state_dict(torch.load(self.args.ckpt_path)['state_dict'])

        limit_train_batches = None
        limit_val_batches = None
        if self.args.limit_batches > 0:
            limit_train_batches = self.args.limit_batches
            limit_val_batches = int(self.args.limit_batches * self.args.valid_ratio / (1 - self.args.valid_ratio))

        train_metrics_callback = TrainMetricsCallback()

        # Building callbacks
        callbacks = []
        callbacks.append(checkpoint_callback)
        if self.args.early_stop > 0:
            callbacks.append(EarlyStopping(
                monitor="val_loss",
                patience=self.args.early_stop,
                mode='min',
            ))
        callbacks.append(train_metrics_callback)
        callbacks.append(SimpleProgressBar(train_metrics_callback)),
        if self.args.swa:
            callbacks.append(CscStochasticWeightAveraging(train_metrics_callback))

        if self.args.eval:
            # FIXME Only for adjust hyper-parameters.
            callbacks.append(EvalInTrainMetricsCallback(self.args))

        trainer = pl.Trainer(
            default_root_dir=self.args.work_dir,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            callbacks=callbacks,
            max_epochs=self.args.epochs,
            min_epochs=self.args.min_epochs,
            num_sanity_val_steps=self.args.num_sanity_val_steps,
            enable_progress_bar=False,  # Use custom progress bar
            precision=self.args.precision,
            gradient_clip_val=self.args.gradient_clip_val,
            gradient_clip_algorithm=self.args.gradient_clip_algorithm,
            logger=TensorBoardLogger(self.args.work_dir),
            accumulate_grad_batches=self.args.accumulate_grad_batches
        )

        trainer.fit(self.model,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader,
                    ckpt_path=ckpt_path
                    )

    def test(self):
        trainer = pl.Trainer(
            default_root_dir=self.args.work_dir,
            callbacks=[TestMetricsCallback(print_errors=self.args.print_errors,
                                           ignore_de='13' in self.args.data
                                           )]
        )

        test_dataloader = create_test_dataloader(self.args)

        if self.args.ckpt_path == 'None':
            trainer.test(self.model, dataloaders=test_dataloader)
            return

        assert self.args.ckpt_path and os.path.exists(self.args.ckpt_path), \
            "Checkpoint file is not found! ckpt_path:%s" % self.args.ckpt_path

        ckpt_states = torch.load(self.args.ckpt_path, map_location='cpu')
        self.model.load_state_dict(ckpt_states['state_dict'])
        self.model = self.model.to(self.args.device)

        trainer.test(self.model, dataloaders=test_dataloader)

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str, default='bert',
                            help='The model name you want to train.')
        parser.add_argument('--device', type=str, default='auto',
                            help='The device for training. auto, cpu or cuda')
        parser.add_argument('--seed', type=int, default=0, help='The random seed.')
        parser.add_argument('--data', type=str, default=None, help='The data you want to load. e.g. wang271k.')
        parser.add_argument('--data-type', type=str, default='sentence', help='sentence or word')
        parser.add_argument('--val-data', type=str, default=None,
                            help='The data you want to load for validation. e.g. wang271k.')
        parser.add_argument('--test-data', type=str, default=None,  # Fixme
                            help='The data you want to load for te. e.g. wang271k.')
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
        parser.add_argument('--min-epochs', type=int, default=10, help='The minimum number of training epochs.')
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
        parser.add_argument('--print-errors', action='store_true', default=False,
                            help="Print sentences which is failure to predict.")
        parser.add_argument('--hyper-params', type=str, default="",
                            help='The hyper parameters of your model. The type must be json.')
        parser.add_argument('--bert-path', type=str, help="The pretrained model path.")
        parser.add_argument('--accumulate_grad_batches', type=int, default=1)
        parser.add_argument('--precision', type=str, default="32-true")
        parser.add_argument('--max-length', type=int, default=256,
                            help='The max length of sentence. Sentence will be truncated if its length long than it.')
        parser.add_argument('--num_sanity_val_steps', type=int, default=0,
                            help='The parameters of pytorch lightning Trainer.')
        parser.add_argument('--gradient_clip_val', type=int, default=None,
                            help='The parameters of pytorch lightning Trainer.')
        parser.add_argument('--gradient_clip_algorithm', type=int, default=None,
                            help='The parameters of pytorch lightning Trainer.')
        parser.add_argument('--early-stop', type=int, default=-1,
                            help="The epochs for early stop check. -1 means do not early stop.")
        parser.add_argument('--swa', action='store_true', default=False, help='Introduce Stochastic Weight Averaging.')

        ###############################################################################################################

        parser.add_argument('--model-path', type=str, default=None,
                            help='The filepath of pretrain model.')

        parser.add_argument('--train-data', type=str, default="./data/Wang271K_processed.pkl",
                            help='The file path of training data.')

        parser.add_argument('--output-path', type=str, default='./c_output',
                            help='The path of output files while running, '
                                 'including model state file, tensorboard files, etc.')

        parser.add_argument('--error-threshold', type=float, default=0.5,
                            help='When detection logit greater than {error_threshold}, '
                                 'the token will be treated as error.')
        parser.add_argument('--eval', action='store_true', default=False, help='Eval model after every epoch.')
        parser.add_argument('--finetune', action='store_true', default=False,
                            help="The finetune flag means that the training into the fine-tuning phase.")

        args = parser.parse_known_args()[0]

        if args.device == 'auto':
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            args.device = args.device

        print("Device:", args.device)

        setup_seed(args.seed)
        mkdir(args.work_dir)
        args.work_dir = Path(args.work_dir)

        add_file_handler(filename=args.work_dir / 'output.log')
        log.info(args)

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

        try:
            hyper_params = {}
            for param in args.hyper_params.split(","):
                if len(param.split("=")) != 2:
                    continue

                key, value = param.split("=")
                if is_float(value):
                    value = float(value)
                    if value == int(value):
                        value = int(value)

                hyper_params[key] = value
            args.hyper_params = hyper_params
        except:
            log.error(
                "Failed to resolve hyper-params. The pattern must look like 'key=value,key=value'. hyper_params: %s" % args.hyper_params)
            exit(0)

        if len(args.hyper_params) > 0:
            print("Hyper parameters:", args.hyper_params)

        # multiprocessing.set_start_method("spawn", force=True)

        return args


if __name__ == '__main__':
    from multiprocessing import freeze_support

    import torch.multiprocessing

    torch.multiprocessing.set_sharing_strategy('file_system')

    freeze_support()

    train = C_Train()
    if train.args.test:
        train.test()
    else:
        train.train()
