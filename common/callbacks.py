import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from lightning.pytorch.callbacks import Callback, TQDMProgressBar, ProgressBar, StochasticWeightAveraging
import lightning.pytorch as pl
from lightning.pytorch.callbacks.progress.tqdm_progress import convert_inf, Tqdm
from lightning.pytorch.utilities.types import STEP_OUTPUT
from tqdm import tqdm

from utils.dataloader import create_test_dataloader
from utils.dataset import CSCDataset
from utils.log_utils import log
from utils.metrics import CSCMetrics
from utils.utils import mock_args


class TestCallback(Callback):

    def __init__(self, name="test"):
        super().__init__()
        self.name = name

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print("on_train_epoch_start, name:%s" % self.name)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print("on_train_epoch_end, name:%s" % self.name)

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print("on_validation_epoch_start, name:%s" % self.name)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print("on_validation_epoch_end, name:%s" % self.name)

    def on_train_batch_end(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any,
            batch_idx: int
    ) -> None:
        print("on_train_batch_end, batch_idx: %s, name:%s" % (batch_idx, self.name))

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: Optional[STEP_OUTPUT],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        print("on_validation_batch_end, name:%s" % self.name)


class CheckpointCallback(Callback):

    def __init__(self, dir_path: Path):
        super().__init__()
        self.dir_path = dir_path
        self.ckpt_path = dir_path / 'last.ckpt'
        self.best_ckpt_path = dir_path / 'best.ckpt'

        self.val_loss = 0.
        self.best_val_loss = 9999999.

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        trainer.save_checkpoint(self.ckpt_path)

    def on_exception(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: BaseException) -> None:
        log.error("Program occurred exception, save the last checkpoint at " + str(self.ckpt_path))
        trainer.save_checkpoint(self.ckpt_path)

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.val_loss = 0.

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: Optional[STEP_OUTPUT],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        loss = outputs['loss']
        self.val_loss += loss.item()

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.val_loss >= self.best_val_loss:
            return

        self.best_val_loss = self.val_loss
        trainer.save_checkpoint(self.best_ckpt_path)


class TrainMetricsCallback(Callback):

    def __init__(self):
        super(TrainMetricsCallback, self).__init__()

        self.train_matrix = np.zeros([4])
        self.valid_matrix = np.zeros([4])

        self.val_total_loss = 0.
        self.val_total_num = 0

        self.val_f1_list = []
        self.val_pr_list = []

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.train_matrix = np.zeros([4])

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        c_p, c_r, c_f1 = self.compute_matrix(*self.valid_matrix)
        self.valid_matrix = np.zeros([4])
        self.val_f1_list.append(c_f1)
        self.val_pr_list.append((c_p, c_r))

    def on_train_batch_end(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any,
            batch_idx: int
    ) -> None:
        targets = outputs['targets'].view(-1)
        d_targets = outputs['d_targets'].view(-1)
        attention_mask = outputs['attention_mask'].view(-1)
        outputs = outputs['outputs'].view(-1)

        self.train_matrix *= 0.96  # 每次让之前的值衰减0.96，差不多100次刚好衰减完，相当于matrix展示的是近100次的平均值
        self.train_matrix += self.character_level_confusion_matrix(outputs, targets, d_targets, attention_mask)

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: Optional[STEP_OUTPUT],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        loss = outputs['loss']

        targets = outputs['targets'].view(-1)
        d_targets = outputs['d_targets'].view(-1)
        attention_mask = outputs['attention_mask'].view(-1)
        outputs = outputs['outputs'].view(-1)

        self.valid_matrix += self.character_level_confusion_matrix(outputs, targets, d_targets,
                                                                   attention_mask)

        self.val_total_loss += loss.item()
        self.val_total_num += 1

        pl_module.log("val_loss", loss)

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.val_total_loss = 0.
        self.val_total_num = 0

    def get_train_matrix(self):
        c_p, c_r, c_f1 = self.compute_matrix(*self.train_matrix)
        return c_p, c_r, c_f1

    def get_val_matrix(self):
        c_p, c_r, c_f1 = self.compute_matrix(*self.valid_matrix)
        return c_p, c_r, c_f1

    def get_val_avg_loss(self):
        return self.val_total_loss / self.val_total_num

    @staticmethod
    def character_level_confusion_matrix(outputs, targets, detection_targets, mask):
        detection_targets[mask == 0] = -1

        c_tp = (outputs[detection_targets == 1] == targets[detection_targets == 1]).sum().item()
        c_fp = (outputs != targets)[detection_targets == 0].sum().item()
        c_tn = (outputs == targets)[detection_targets == 0].sum().item()
        c_fn = (outputs[detection_targets == 1] != targets[detection_targets == 1]).sum().item()

        return np.array([c_tp, c_fp, c_tn, c_fn])

    @staticmethod
    def compute_matrix(tp, fp, tn, fn):
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)
        return precision, recall, f1_score


class SimpleProgressBar(Callback):

    def __init__(self, train_metrics: TrainMetricsCallback):
        super(SimpleProgressBar, self).__init__()

        self.train_metrics = train_metrics

        self.train_progress_bar = None
        self.val_progress_bar = None

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.logger:
            trainer.logger.log_hyperparams(pl_module.args.hyper_params)

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        total = len(trainer.train_dataloader)
        if type(trainer.limit_train_batches) == int and trainer.limit_train_batches:
            total = trainer.limit_train_batches

        if type(trainer.limit_train_batches) == float and trainer.limit_train_batches < 1.0:
            total = math.ceil(total * trainer.limit_train_batches)

        self.train_progress_bar = tqdm(None, desc="Epoch {} Training".format(trainer.current_epoch), total=total)

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.train_progress_bar.close()

        total = len(trainer.val_dataloaders)
        if type(trainer.limit_val_batches) == int and trainer.limit_val_batches:
            total = trainer.limit_val_batches

        if type(trainer.limit_val_batches) == float and trainer.limit_val_batches < 1.0:
            total = math.ceil(total * trainer.limit_val_batches)

        self.val_progress_bar = tqdm(None, desc="Epoch {} Validation".format(trainer.current_epoch), total=total)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.val_progress_bar.close()

        c_p, c_r, c_f1 = self.train_metrics.get_train_matrix()
        log.info("Epoch {} train, Correction Precision: {}, Recall: {}, F1-Score: {}".format(trainer.current_epoch,
                                                                                             c_p, c_r, c_f1))

        val_avg_loss = self.train_metrics.get_val_avg_loss()
        c_p, c_r = self.train_metrics.val_pr_list[-1]
        c_f1 = self.train_metrics.val_f1_list[-1]
        log.info(
            "Epoch {} val_loss {:.5f}, Correction Precision: {}, Recall: {}, F1-Score: {}".format(trainer.current_epoch,
                                                                                                  val_avg_loss,
                                                                                                  c_p, c_r, c_f1))
        if trainer.logger:
            trainer.logger.log_metrics({
                "val_avg_loss": val_avg_loss,
                "val_pre": c_p,
                "val_rec": c_r,
                "val_f1": c_f1,
            }, step=trainer.current_epoch)

    def on_train_batch_end(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any,
            batch_idx: int
    ) -> None:
        loss = outputs['loss']

        c_p, c_r, c_f1 = self.train_metrics.get_train_matrix()

        postfix = {
            'loss': loss.item(),
            'c_precision': c_p,
            'c_recall': c_r,
            'c_f1_score': c_f1,
        }
        if 'bar_postfix' in outputs:
            postfix.update(outputs['bar_postfix'])
        self.train_progress_bar.set_postfix(postfix)

        if trainer.logger:
            trainer.logger.log_metrics({
                "train_loss": loss.item(),
                "train_pre": c_p,
                "train_rec": c_r,
                "train_f1": c_f1,
                "lr": trainer.optimizers[0].state_dict()['param_groups'][-1]['lr'],
            }, step=trainer.current_epoch * len(trainer.train_dataloader) + batch_idx)

        self.train_progress_bar.update(1)

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: Optional[STEP_OUTPUT],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        loss = outputs['loss']

        c_p, c_r, c_f1 = self.train_metrics.get_val_matrix()

        self.val_progress_bar.set_postfix({
            'loss': self.train_metrics.get_val_avg_loss(),
            'c_precision': c_p,
            'c_recall': c_r,
            'c_f1_score': c_f1,
        })

        self.val_progress_bar.update(1)

        pl_module.log("val_loss", loss)


class TestMetricsCallback(Callback):

    def __init__(self, print_errors=False, ignore_de=False, export_sighan_format=False):
        super().__init__()
        self.print_errors = print_errors
        self.ignore_de = ignore_de
        self.export_sighan_format = export_sighan_format

        self.csc_metrics = CSCMetrics()

    def on_test_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: Optional[STEP_OUTPUT],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        src, tgt = batch
        pred = outputs

        for i in range(len(src)):
            src_i, tgt_i, pred_i = src[i].replace(" ", ""), tgt[i].replace(" ", ""), pred[i].replace(" ", "")

            if self.ignore_de:
                tgt_tokens = list(tgt_i)
                pred_tokens = list(pred_i)
                for j in range(len(tgt_tokens)):
                    if tgt_tokens[j] in ['的', '地', '得']:
                        pred_tokens[j] = tgt_tokens[j]
                pred_i = ''.join(pred_tokens)

            self.csc_metrics.add_sentence(src_i, tgt_i, pred_i)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.csc_metrics.print_results()

        if self.print_errors:
            self.csc_metrics.print_errors()
            self.csc_metrics.print_abnormal_pairs()

        if self.export_sighan_format:
            self.csc_metrics.export_sigan_format()


class EvalInTrainMetricsCallback(Callback):

    def __init__(self, args):
        super().__init__()
        self.args = args
        if not args.eval:
            return

        self.csc_metrics = CSCMetrics()
        test_data = "sighan15test"
        if self.args.test_data:
            test_data = self.args.test_data

        self.dataset = CSCDataset(test_data)

    def on_test_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: Optional[STEP_OUTPUT],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        src, tgt = batch
        pred = outputs

        assert len(src) == len(tgt) == len(pred)

        for i in range(len(src)):
            self.csc_metrics.add_sentence(src[i].replace(" ", ""), tgt[i].replace(" ", ""), pred[i].replace(" ", ""))

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.csc_metrics.print_results()

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        FIXME Only for adjust hyper-parameters
        """
        if not self.args.eval:
            return

        pl_module.eval()
        for batch_idx, batch in tqdm(enumerate(self.dataset.data), total=len(self.dataset), desc="Test"):
            batch = ([batch[0]], [batch[1]])
            outputs = pl_module.test_step(batch, batch_idx)
            self.on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, 0)

        self.on_test_end(trainer, pl_module)
        self.csc_metrics = CSCMetrics()
        pl_module.train()
