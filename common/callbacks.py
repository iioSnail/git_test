import os
from pathlib import Path
from typing import Any

from lightning.pytorch.callbacks import Callback
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT

from utils.log_utils import log


class CheckpointCallback(Callback):

    def __init__(self, dir_path: Path):
        super().__init__()
        self.dir_path = dir_path
        self.ckpt_path = dir_path / 'last.ckpt'

    def on_train_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
    ) -> None:
        trainer.save_checkpoint(self.ckpt_path)

    def on_exception(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: BaseException) -> None:
        log.error("Program occurred exception, save the last checkpoint at ", str(self.ckpt_path))
        trainer.save_checkpoint(self.ckpt_path)
