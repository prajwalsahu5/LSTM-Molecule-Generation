from rdkit import Chem
import torch
from torch import nn
from pytorch_lightning import LightningModule
import torchmetrics
from fsr_fg_model import FsrFgModel
from data import FsrFgDataModule
from pytorch_lightning.cli import LightningCLI


class FsrFgLightning(LightningModule):
    def __init__(self, fg_input_dim=2786, mfg_input_dim=2586, num_input_dim=208,
                 enc_dec_dims=(500, 100), output_dims=(200, 100, 50), num_tasks=2, dropout=0.8,
                 method='FGR', lr=1e-4, **kwargs):
        super(FsrFgLightning, self).__init__()
        self.save_hyperparameters('fg_input_dim', 'mfg_input_dim', 'num_input_dim', 'enc_dec_dims',
                                  'output_dims', 'num_tasks', 'dropout', 'method', 'lr')
        self.net = FsrFgModel(fg_input_dim, mfg_input_dim, num_input_dim, enc_dec_dims, output_dims, num_tasks, dropout,
                              method)
        self.lr = lr
        self.method = method
        self.criterion = nn.CrossEntropyLoss()
        self.recon_loss = nn.BCEWithLogitsLoss()
        self.softmax = nn.Softmax(dim=1)
        self.train_auc = torchmetrics.AUROC(num_classes=num_tasks)
        self.valid_auc = torchmetrics.AUROC(num_classes=num_tasks)
        self.test_auc = torchmetrics.AUROC(num_classes=num_tasks)

    def forward(self, fg, mfg, num_features):
        if self.method == 'FG':
            y_pred = self.net(fg=fg)
        elif self.method == 'MFG':
            y_pred = self.net(mfg=mfg)
        elif self.method == 'FGR':
            y_pred = self.net(fg=fg, mfg=mfg)
        else:
            y_pred = self.net(fg=fg, mfg=mfg, num_features=num_features)
        return y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=0.3)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-2, total_steps=self.trainer.estimated_stepping_batches)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        fg, mfg, num_features, y = batch
        y_pred, recon = self(fg, mfg, num_features)
        if self.method == 'FG':
            loss_r_pre = 1e-4 * self.recon_loss(recon, fg)
        elif self.method == 'MFG':
            loss_r_pre = 1e-4 * self.recon_loss(recon, mfg)
        else:
            loss_r_pre = 1e-4 * self.recon_loss(recon, torch.cat([fg, mfg], dim=1))
        loss = self.criterion(y_pred, y) + loss_r_pre
        self.train_auc(self.softmax(y_pred), y)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log('train_auc', self.train_auc, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        fg, mfg, num_features, y = batch
        y_pred, recon = self(fg, mfg, num_features)
        loss = self.criterion(y_pred, y)
        self.valid_auc(self.softmax(y_pred), y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_auc',  self.valid_auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        fg, mfg, num_features, y = batch
        y_pred, recon = self(fg, mfg, num_features)
        loss = self.criterion(y_pred, y)
        self.test_auc(self.softmax(y_pred), y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_auc', self.test_auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)


if __name__ == '__main__':
    cli = LightningCLI(model_class=FsrFgLightning, datamodule_class=FsrFgDataModule,
                       save_config_callback=None, run=False)
    cli.trainer.fit(cli.model, cli.datamodule)
    cli.trainer.test(cli.model, cli.datamodule, ckpt_path='best')
