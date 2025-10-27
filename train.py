import os
import glob
import numpy as np
import torch
import torch.nn as nn
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
import pytorch_lightning as pl
from pytorch_lightning.profilers import AdvancedProfiler
from src.utils.lightning_callbacks import (
    DumpProfileCallback,
    SaveHighLossesCallback,
    SaveBeforeNaNCallback,
    UpdateEpochCallback,
    VisualizeLossVsTimeCallback,
    TimingCallback
)


from evodiff.utils import Tokenizer

import src

from nets import get_model_setup
from data import get_dataloaders
from src.utils.ema import EMA
from datetime import datetime

import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
import logging


@hydra.main(version_base=None, config_path="configs", config_name="basic")
def train(cfg: DictConfig) -> None:

    @rank_zero_only
    def init_wandb():
        wandb.login()
        wandb.init()
    init_wandb()
    pl.seed_everything(cfg.model.seed, workers=True)

    print("Getting dataloaders.")
    train_dataloader, test_dataloader = get_dataloaders(cfg)
    tokenizer = train_dataloader.tokenizer if hasattr(train_dataloader, "tokenizer") else None

    print("Setting up model.")
    x0_model_class, nn_params = get_model_setup(cfg, tokenizer) 
    
    print(cfg)
    
    if not cfg.model.restart:
        model = getattr(src, cfg.model.model)(
            x0_model_class,
            nn_params,
            num_classes=len(tokenizer) if tokenizer else cfg.data.N,
            gamma=cfg.model.gamma,
            forward_kwargs=OmegaConf.to_container(cfg.model.forward_kwargs, resolve=True),
            schedule_type=cfg.model.schedule_type,
            gen_trans_step=cfg.sampling.gen_trans_step,
            t_max=cfg.model.t_max,
            seed=cfg.model.seed,
            tokenizer=tokenizer if cfg.data.data != 'uniref50' else Tokenizer(),
            **OmegaConf.to_container(cfg.train, resolve=True),
        )
        ckpt_path = None
    else:
        ckpt_path = f'checkpoints/{cfg.model.restart}'        
        ckpt_path = max(glob.glob(os.path.join(ckpt_path, '*.ckpt')), key=os.path.getmtime)
        model = getattr(src, cfg.model.model).load_from_checkpoint(ckpt_path)
        train_dataloader.dataset.global_step = model.global_step

    print("PRE-CONFIGURING MODEL")
    ##### Load data
    model.pre_configure_model(train_dataloader, test_dataloader)

    ##### Train
    # wandb.init()
    wandb_logger = WandbLogger()
    lightning_model = model
    torch.set_float32_matmul_precision('high')
    @rank_zero_only
    def update_wandb_config():
        wandb.config.update(lightning_model.hparams)
    update_wandb_config()

    if 'uniref50' in cfg.data.data:
        val_check_interval = round(0.2 * (210000//cfg.train.batch_size))
    else:
        val_check_interval = 1.0
    profiler = AdvancedProfiler(dirpath=".", filename="profiler.txt")
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = f"{time}_{wandb_logger.experiment.name}"
    os.makedirs("checkpoints/" + save_dir, exist_ok=True)
    
    loss_threshold = cfg.train.get("logging_loss_threshold", 30.0)

    
    devices = getattr(cfg.train, "n_devices", torch.cuda.device_count())
    print(f"Devices: {devices}")
    
    if cfg.model.model == 'SimplicialDiffusion':
        callbacks =  [
            DumpProfileCallback(f"checkpoints/{save_dir}", profiler, every_n_steps=1000),
            SaveHighLossesCallback(save_path=f'checkpoints/{save_dir}', loss_threshold=loss_threshold),
            SaveBeforeNaNCallback(save_path=f'checkpoints/{save_dir}'),
            UpdateEpochCallback(),
            VisualizeLossVsTimeCallback(every_n_steps=100),
            TimingCallback(),
        ]
    else:
        callbacks = []

    trainer = Trainer(
        max_epochs=cfg.train.n_epoch, 
        profiler=profiler,
        accelerator='auto', 
        devices=devices,
        logger=wandb_logger, 
        strategy=getattr(cfg.train, "strategy",  DDPStrategy(broadcast_buffers=True)),
        callbacks=([EMA(0.9999)] * cfg.train.ema
                   + [ModelCheckpoint(
                       dirpath=f'checkpoints/{save_dir}',
                        save_on_train_epoch_end=False,
                        save_top_k=3,
                        monitor="val_loss"
                        )]
                   + callbacks
                   ),
        val_check_interval=val_check_interval,
        accumulate_grad_batches=cfg.train.accumulate,
    )
    trainer.fit(lightning_model, train_dataloader, test_dataloader, ckpt_path=ckpt_path)
    wandb.finish()

if __name__ == "__main__":
    train()
