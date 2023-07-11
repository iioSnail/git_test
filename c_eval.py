import argparse
import os
from pathlib import Path

import torch

from c_train import C_Train
from common.callbacks import TestMetricsCallback
from utils.dataloader import create_test_dataloader
from utils.log_utils import add_file_handler, log
from utils.str_utils import is_float
import lightning.pytorch as pl

"""
python c_eval.py \
    --model SCOPE \
    --bert-path FPT \
    --data sighan15test \
    --test \
    --batch-size 1 \
    --ckpt-path my_model/scope-last.ckpt \
    --print-errors \
    --export-sighan-format
"""
class C_Eval(object):

    def __init__(self):
        self.args = self.parse_args()
        self.model = C_Train.model_select(self.args)

    def eval(self):
        trainer = pl.Trainer(
            default_root_dir=self.args.work_dir,
            callbacks=[TestMetricsCallback(print_errors=self.args.print_errors,
                                           ignore_de='13' in self.args.data,
                                           export_sighan_format=self.args.export_sighan_format
                                           )]
        )

        test_dataloader = create_test_dataloader(self.args)

        if self.args.ckpt_path is None or self.args.ckpt_path == 'None':
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
                            help='The model name you want to evaluate.')
        parser.add_argument('--device', type=str, default='auto',
                            help='The device for training. auto, cpu or cuda')
        parser.add_argument('--data', type=str, default=None, help='The data you want to test. e.g. sighan15test.')
        parser.add_argument('--batch-size', type=int, default=32,
                            help='The batch size of training, validation and test.')
        parser.add_argument('--ckpt-path', type=str, default=None, help='The filepath of checkpoint for test.')
        parser.add_argument('--print-errors', action='store_true', default=False,
                            help="Print sentences which is failure to predict.")
        parser.add_argument('--hyper-params', type=str, default="",
                            help='The hyper parameters of your model. The type must be json.')
        parser.add_argument('--bert-path', type=str, help="The pretrained model path.")
        parser.add_argument('--work-dir', type=str, default='./outputs',
                            help='The path of output files while running.')
        parser.add_argument('--export-sighan-format', action='store_true', default=False,
                            help="Export the results file in sighan format.")

        args = parser.parse_known_args()[0]

        if args.device == 'auto':
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            args.device = args.device

        print("Device:", args.device)

        args.work_dir = Path(args.work_dir)

        add_file_handler(filename=args.work_dir / 'output.log')
        log.info(args)

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

        return args


if __name__ == '__main__':
    eval_ = C_Eval()
    eval_.eval()
