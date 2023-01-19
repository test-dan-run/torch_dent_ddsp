from torch.optim import Adam
from omegaconf import DictConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
import pytorch_lightning as pl

from model import DENT
from loss import SpectralLoss

class LightningDENT(pl.LightningModule):
    def __init__(self, cfg: DictConfig):

        super(LightningDENT, self).__init__()

        self.cfg = cfg
        self.model = DENT(**cfg.model)
        self.loss_fn = SpectralLoss()

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

    def test_step(self, batch, batch_idx):

        x, y = batch
        out = self.model(x)
        
        loss = self.loss_fn(out, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, sync_dist=self.run_cfg.distributed)
        
        return {'predictions': out, 'labels': y, 'loss': loss}	

    def configure_optimizers(self):

        optimizer = Adam(self.parameters(), lr=self.cfg.optim.lr)
        scheduler = ReduceLROnPlateau(
            optimizer = optimizer,
            mode = self.cfg.optim.mode,
            factor = self.cfg.optim.factor,
            patience = self.cfg.optim.patience,
            cooldown = self.cfg.optim.cooldown,
            eps = self.cfg.optim.eps
        )

        return {
            'optimizer': optimizer,
            'monitor': 'valid_loss',
            'lr_scheduler': scheduler,
            'interval': self.cfg.optim.interval
        }

if __name__ == '__main__':

    from omegaconf import OmegaConf

    cfg = {
        'model': {
            'waveshaper_config': {
                'threshold': 15.0,
            },
            'compressor_config': {
                'sample_rate': 16000,
                'threshold': -10,
                'ratio': 30.0,
                'makeup': 0.0,
                'attack': 1.0e-7,
                'release': 1.0e-3,
                'downsample_factor': 2.0
            },
            'equalizer_config': {
                'n_frequency_bins': 1000,
            },
            'noise_config': {
                'n_frequency_bins': 1000
            }
        },
        'optim': {
            'lr': 5.0e-4,
            'mode': 'min',
            'factor': 0.1,
            'patience': 5,
            'cooldown': 0,
            'eps': 1.0e-7,
            'interval': 'epoch'
        }
    }

    cfg = OmegaConf.create(cfg)
    lightning_dent = LightningDENT(cfg)
    batch_tensor_1 = torch.rand(size=(4, 1, 16000))
    batch_tensor_2 = torch.rand(size=(4, 1, 16000))
    print(lightning_dent.training_step((batch_tensor_1, batch_tensor_2), 0))