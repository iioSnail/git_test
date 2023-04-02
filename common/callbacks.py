import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from lightning.pytorch.callbacks import Callback, TQDMProgressBar, ProgressBar
import lightning.pytorch as pl
from lightning.pytorch.callbacks.progress.tqdm_progress import convert_inf, Tqdm
from lightning.pytorch.utilities.types import STEP_OUTPUT
from tqdm import tqdm

from utils.log_utils import log


class CheckpointCallback(Callback):

    def __init__(self, dir_path: Path):
        super().__init__()
        self.dir_path = dir_path
        self.ckpt_path = dir_path / 'last.ckpt'

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        trainer.save_checkpoint(self.ckpt_path)

    def on_exception(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: BaseException) -> None:
        log.error("Program occurred exception, save the last checkpoint at " + str(self.ckpt_path))
        trainer.save_checkpoint(self.ckpt_path)


# class MetricsProgressBar(TQDMProgressBar):
#
#     def __init__(self, refresh_rate: int = 1, process_position: int = 0):
#         super().__init__(refresh_rate, process_position)
#         self.train_matrix = np.zeros([4])
#         self.valid_matrix = np.zeros([4])
#
#     def on_train_batch_end(
#             self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any,
#             batch_idx: int
#     ) -> None:
#         super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
#
#         loss = outputs['loss']
#
#         targets = outputs['targets'].view(-1)
#         d_targets = outputs['d_targets'].view(-1)
#         attention_mask = outputs['attention_mask'].view(-1)
#         outputs = outputs['outputs'].view(-1)
#
#         self.train_matrix *= 0.96  # 每次让之前的值衰减0.96，差不多100次刚好衰减完，相当于matrix展示的是近100次的平均值
#         self.train_matrix += self.character_level_confusion_matrix(outputs, targets, d_targets,
#                                                                    attention_mask)
#
#         c_p, c_r, c_f1 = self.compute_matrix(*self.train_matrix)
#
#         self._train_progress_bar.set_postfix({
#             'loss': loss.item(),
#             'c_precision': c_p,
#             'c_recall': c_r,
#             'c_f1_score': c_f1,
#         })
#
#     def on_train_epoch_start(self, trainer: "pl.Trainer", *_: Any) -> None:
#         super().on_train_epoch_start(trainer)
#         self.train_progress_bar.set_description(f"Epoch {trainer.current_epoch} Training")
#
#     def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
#         super().on_train_epoch_end(trainer, pl_module)
#         self.train_matrix = np.zeros([4])
#
#     def on_validation_batch_end(
#             self,
#             trainer: "pl.Trainer",
#             pl_module: "pl.LightningModule",
#             outputs: Optional[STEP_OUTPUT],
#             batch: Any,
#             batch_idx: int,
#             dataloader_idx: int = 0,
#     ) -> None:
#         super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
#
#         loss = outputs['loss']
#
#         targets = outputs['targets'].view(-1)
#         d_targets = outputs['d_targets'].view(-1)
#         attention_mask = outputs['attention_mask'].view(-1)
#         outputs = outputs['outputs'].view(-1)
#
#         self.valid_matrix += self.character_level_confusion_matrix(outputs, targets, d_targets,
#                                                                    attention_mask)
#
#         c_p, c_r, c_f1 = self.compute_matrix(*self.train_matrix)
#
#         self._val_progress_bar.set_postfix({
#             'loss': loss.item(),
#             'c_precision': c_p,
#             'c_recall': c_r,
#             'c_f1_score': c_f1,
#         })
#
#         pl_module.log("val_loss", loss)
#
#     def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
#         super().on_validation_epoch_start(trainer, pl_module)
#         self.val_progress_bar.set_description(f"Epoch {trainer.current_epoch} Validation")
#
#     @staticmethod
#     def character_level_confusion_matrix(outputs, targets, detection_targets, mask):
#         detection_targets[mask == 0] = -1
#
#         c_tp = (outputs[detection_targets == 1] == targets[detection_targets == 1]).sum().item()
#         c_fp = (outputs != targets)[detection_targets == 0].sum().item()
#         c_tn = (outputs == targets)[detection_targets == 0].sum().item()
#         c_fn = (outputs[detection_targets == 1] != targets[detection_targets == 1]).sum().item()
#
#         return np.array([c_tp, c_fp, c_tn, c_fn])
#
#     @staticmethod
#     def compute_matrix(tp, fp, tn, fn):
#         precision = tp / (tp + fp + 1e-9)
#         recall = tp / (tp + fn + 1e-9)
#         f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)
#         return precision, recall, f1_score
#
#     # overwrite the source code for some custom features.
#     def on_validation_batch_start(
#             self,
#             trainer: "pl.Trainer",
#             pl_module: "pl.LightningModule",
#             batch: Any,
#             batch_idx: int,
#             dataloader_idx: int = 0,
#     ) -> None:
#         pass
#         # if not self.has_dataloader_changed(dataloader_idx):
#         #     return
#         #
#         # self.val_progress_bar.reset(convert_inf(self.total_val_batches_current_dataloader))
#         # self.val_progress_bar.initial = 0
#         # desc = self.sanity_check_description if trainer.sanity_checking else self.validation_description
#         # self.val_progress_bar.set_description(f"{desc} DataLoader {dataloader_idx}")
#

class MetricsProgressBar(Callback):

    def __init__(self):
        super(MetricsProgressBar, self).__init__()
        self.train_progress_bar: tqdm
        self.val_progress_bar: tqdm

        self.train_matrix = np.zeros([4])
        self.valid_matrix = np.zeros([4])

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        total = len(trainer.train_dataloader)
        if trainer.limit_train_batches:
            total = trainer.limit_train_batches

        self.train_progress_bar = tqdm(None, desc="Epoch {} Training".format(trainer.current_epoch), total=total)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.train_matrix = np.zeros([4])
        self.train_progress_bar.close()

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        total = len(trainer.val_dataloaders)
        if trainer.limit_val_batches:
            total = trainer.limit_val_batches

        self.val_progress_bar = tqdm(None, desc="Epoch {} Validation".format(trainer.current_epoch), total=total)
        # If you run the program in Colab or Pycharm, you must set tqdm.pos to 0.
        # Otherwise, the progress bar will be printed many times.
        self.val_progress_bar.pos = 0

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        c_p, c_r, c_f1 = self.compute_matrix(*self.valid_matrix)
        log.info("Correction Precision: {}, Recall: {}, F1-Score: {}".format(c_p, c_r, c_f1))
        self.valid_matrix = np.zeros([4])
        self.val_progress_bar.close()

    def on_train_batch_end(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any,
            batch_idx: int
    ) -> None:
        loss = outputs['loss']

        targets = outputs['targets'].view(-1)
        d_targets = outputs['d_targets'].view(-1)
        attention_mask = outputs['attention_mask'].view(-1)
        outputs = outputs['outputs'].view(-1)

        self.train_matrix *= 0.96  # 每次让之前的值衰减0.96，差不多100次刚好衰减完，相当于matrix展示的是近100次的平均值
        self.train_matrix += self.character_level_confusion_matrix(outputs, targets, d_targets,
                                                                   attention_mask)

        c_p, c_r, c_f1 = self.compute_matrix(*self.train_matrix)

        self.train_progress_bar.set_postfix({
            'loss': loss.item(),
            'c_precision': c_p,
            'c_recall': c_r,
            'c_f1_score': c_f1,
        })

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

        targets = outputs['targets'].view(-1)
        d_targets = outputs['d_targets'].view(-1)
        attention_mask = outputs['attention_mask'].view(-1)
        outputs = outputs['outputs'].view(-1)

        self.valid_matrix += self.character_level_confusion_matrix(outputs, targets, d_targets,
                                                                   attention_mask)

        c_p, c_r, c_f1 = self.compute_matrix(*self.valid_matrix)

        self.val_progress_bar.set_postfix({
            'loss': loss.item(),
            'c_precision': c_p,
            'c_recall': c_r,
            'c_f1_score': c_f1,
        })

        self.val_progress_bar.update(1)

        pl_module.log("val_loss", loss)

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
