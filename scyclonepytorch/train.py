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

from trainer import ScycloneTrainer
from datamodule import DataLoaderPerformance, ScycloneDataModule

@click.command()
@click.option('-p', '--config_path', default='Configs/base.yml', type=str)
@click.option('-r', '--resume_path', default=None, type=str)
def cli_main(config_path, resume_path):

    # args
    config = yaml.safe_load(open('Configs/base.yml'))
    config.update(yaml.safe_load(open(config_path)))
    config = AttrDict(config)

    pl.seed_everything(config.seed)

    # datamodule
    loader_perf = DataLoaderPerformance(
        config.dataset.num_workers or 1,
        config.dataset.pin_memory or False
    )
    datamodule = ScycloneDataModule(
        config.dataset, config.batch_size, loader_perf)

    # train
    train(config, resume_path, datamodule)

def train(config, resume_path, datamodule: LightningDataModule) -> None:

    log_dir = config.log_dir

    # setup
    if config.pretrained_model != "":
        model = ScycloneTrainer.load_from_checkpoint(
            checkpoint_path=config.pretrained_model)
    else:
        model = ScycloneTrainer(config)
    
    # Save at `{default_root_dir}/default/version_{n}/checkpoints/last.ckpt`
    ckpt_cb = ModelCheckpoint(
        period=config.ckpt_period, save_last=True, save_top_k=2, monitor="val_loss"
    )

    n_gpus = config.n_gpus or 0
    n_tpus = config.n_tpus or 0

    trainer = pl.Trainer(
        tpu_cores=n_tpus if n_tpus > 0 else None,
        gpus=n_gpus,
        auto_select_gpus=n_gpus > 0,
        precision=16 if config.use_amp else 32,  # default AMP
        max_epochs=config.max_epochs or 100000,
        check_val_every_n_epoch=config.val_period,  # about 1 validation per 10 min
        # load/resume
        resume_from_checkpoint=resume_path,
        # save
        default_root_dir=log_dir,
        # weights_save_path=args_scpt.weights_save_path,
        checkpoint_callback=ckpt_cb,
        logger=pl_loggers.TensorBoardLogger(log_dir),
        # reload_dataloaders_every_epoch=True,
        profiler=AdvancedProfiler() if config.use_profiler else None,
    )

    # training
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":  # pragma: no cover
    cli_main()
