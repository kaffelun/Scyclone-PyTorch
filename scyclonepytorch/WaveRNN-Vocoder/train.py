import os
import os.path as osp
import re
import sys
import yaml
import shutil
import numpy as np
import torch
import click
from attrdict import AttrDict
import warnings

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.profiler import AdvancedProfiler

from trainer import ScycloneWaveRNNTrainer
from data import ScycloneWaveRNNDataModule

@click.command()
@click.option('-c', '--config_path', default='configs/base.yml', type=str)
@click.option('-r', '--resume_path', default=None, type=str)
def main(config_path, resume_path):
    config = yaml.safe_load(open('configs/base.yml', 'r'))
    config.update(yaml.safe_load(open(config_path, 'r')))
    config = AttrDict(config)

    pl.seed_everything(config.seed)

    datamodule = ScycloneWaveRNNDataModule(config.dataset)

    train(config, resume_path, datamodule)

def train(config, resume_path, datamodule):
    log_dir = config.log_dir

    if config.pretrained_model != "":
        model = ScycloneWaveRNNTrainer.load_from_checkpoint(
            checkpoint_path=config.pretrained_model
        )
    else:
        model = ScycloneWaveRNNTrainer(config)
    
    ckpt = ModelCheckpoint(
        period=config.checkpoint_period, save_last=True, save_top_k=2, monitor="val_loss"
    )

    n_gpus = config.n_gpus or 0
    n_tpus = config.n_tpus or 0

    trainer = pl.Trainer(
        tpu_cores=n_tpus if n_tpus > 0 else None,
        gpus=n_gpus,
        auto_select_gpus=n_gpus > 0,
        precision=16 if config.use_amp else 32,
        max_epochs=config.max_epochs or 1000000,
        check_val_every_n_epoch=config.validation_period,
        resume_from_checkpoint=resume_path,
        default_root_dir=log_dir,
        checkpoint_callback=ckpt,
        logger=pl_loggers.TensorBoardLogger(log_dir),
        profiler=AdvancedProfiler() if config.use_profiler else None
    )

    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()


