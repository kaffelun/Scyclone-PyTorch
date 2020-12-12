import itertools

import torch
from torch.nn import functional as F
from torch.tensor import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.core.datamodule import LightningDataModule

from models import ScycloneWaveRNN

class ScycloneWaveRNNTrainer(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters()

        self.model = ScycloneWaveRNN(args.model)

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        wave, spec = batch
        wave = wave.to(args.device)
        spec = spec.to(args.device)
        m, s, samples = self.model(spec)
        loss = torch.mean(.5 * (np.log(2 * np.pi) + 2 * s + ((wave - m) ** 2) * torch.exp(-2 * s)))
        log = {
            "Loss": loss
        }
        return {"loss": loss, "log_losses": log, "wave": samples}

    def validation_step(self, batch, batch_idx):
        o = self.training_step(batch, batch_idx)
        self.log("val_loss", o["loss"])
        return {
            "val_loss": o["loss"],
            "wave": o["wave"],
            "log_losses": o["log_losses"]
        }
    
    def validation_step_end(self, out):
        for i in range(0, 2):
            self.logger.experiment.add_audio(
                "Validation/Generated",
                o["wave"],
                global_step=self.current_epoch,
                sample_rate=24000
            )
        self.log("Validation/loss", o["log_losses"]["Loss"], on_step=False, on_epoch=True)
    
    def configure_optimizers(self):
        optimizer_args = self.args.optimizer
        decay_rate = optimizer_args.decay_rate
        decay_iter = optimizer_args.decay_iter
        optim = Adam(
            self.model.parameters(),
            lr=optimizer_args.lr,
            betas=(0.5, 0.999),
        )
        sched = {
            "scheduler": MultiStepLR(optim, milestones=decay_iter, gamma=decay_rate),
            "interval": "step"
        }
        return [optim], [sched]
