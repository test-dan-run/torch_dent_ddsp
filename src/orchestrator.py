from typing import List

import torch
import torch.nn as nn
from sklearn import metrics
from torch.optim import Adam
from omegaconf import DictConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR

import pytorch_lightning as pl

from .src.model import DENT

class LightningWALNet(pl.LightningModule):
    def __init__(self, model_cfg: DictConfig, optim_cfg: DictConfig, run_cfg: DictConfig):

        super(LightningWALNet, self).__init__()

        self.model = WALNet(**model_cfg)
        self.model.xavier_init()

        self.loss_fn = nn.BCELoss()
        
        self.optim_cfg = optim_cfg
        self.run_cfg = run_cfg

    def forward(self, batch):
        
        x, _ = batch 

        return self.model(x)

    # TODO: for inference
    # def predict_step(self, batch, batch_idx):
    #     return self.model(batch)

    def training_step(self, batch, batch_idx):

        x, y = batch
        out = self.model(x)

        loss = self.loss_fn(out, y)
        self.log('train_loss', loss, on_step=True, on_epoch=False, sync_dist=self.run_cfg.distributed)

        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        out = self.model(x)
        
        loss = self.loss_fn(out, y)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, sync_dist=self.run_cfg.distributed)
        
        return {'predictions': out, 'labels': y, 'loss': loss}	

    def validation_epoch_end(self, validation_step_outputs):

        labels_list = [x['labels'] for x in validation_step_outputs]
        preds_list = [x['predictions'] for x in validation_step_outputs]

        y_true = torch.cat(labels_list).cpu().numpy()
        y_pred = torch.cat(preds_list).cpu().numpy()
        print(y_true.shape)
        print(y_pred.shape)

        val_ap = metrics.average_precision_score(y_true, y_pred)
        try:
            val_auc = metrics.roc_auc_score(y_true, y_pred)
        except:
            val_auc = 0.0

        self.log('valid_ap', val_ap, on_step=False, on_epoch=True, sync_dist=self.run_cfg.distributed)
        self.log('valid_auc', val_auc, on_step=False, on_epoch=True, sync_dist=self.run_cfg.distributed)

    def test_step(self, batch, batch_idx):

        x, y = batch
        out = self.model(x)
        
        loss = self.loss_fn(out, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, sync_dist=self.run_cfg.distributed)
        
        return {'predictions': out, 'labels': y, 'loss': loss}	

    def test_epoch_end(self, test_step_outputs):

        labels_list = [x['labels'] for x in test_step_outputs]
        preds_list = [x['predictions'] for x in test_step_outputs]

        y_true = torch.cat(labels_list).cpu().numpy()
        y_pred = torch.cat(preds_list).cpu().numpy()

        test_ap = metrics.average_precision_score(y_true, y_pred)
        try:    
            test_auc = metrics.roc_auc_score(y_true, y_pred)
        except:
            test_auc = 0.0

        self.log('test_ap', test_ap, on_step=False, on_epoch=True, sync_dist=self.run_cfg.distributed)
        self.log('test_auc', test_auc, on_step=False, on_epoch=True, sync_dist=self.run_cfg.distributed)

    def configure_optimizers(self):

        if self.optim_cfg.optim == 'adam':
            optim = Adam(self.parameters(), lr = self.optim_cfg.lr)

        if self.optim_cfg.scheduler == 'decay':
            scheduler = ReduceLROnPlateau(
                optimizer = optim,
                mode = self.optim_cfg.mode,
                factor = self.optim_cfg.factor,
                patience = self.optim_cfg.patience,
                cooldown = self.optim_cfg.cooldown,
                eps = self.optim_cfg.eps
            )
        elif self.optim_cfg.scheduler == 'cyclic':
            scheduler = CyclicLR(
                optimizer = optim,
                mode = self.optim_cfg.mode,
                lr = self.optim_cfg.lr,
                max_lr = self.optim_cfg.max_lr,
                epoch_size_up = self.optim_cfg.epoch_size_up,
                epoch_size_down = self.optim_cfg.epoch_size_down
            )

        return {
            'optimizer': optim,
            'monitor': 'valid_loss',
            'lr_scheduler': scheduler,
            'interval': self.optim_cfg.interval
        }
