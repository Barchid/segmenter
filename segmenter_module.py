from argparse import ArgumentParser
from os import times

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
import torchmetrics
from get_segmenter import get_segmenter

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class SegmenterModule(pl.LightningModule):
    def __init__(self, config: dict, learning_rate: float, in_channels: int, num_classes: int, pretrain_path: str = None, **kwargs):
        super().__init__()
        self.save_hyperparameters(
            ignore=["config_path", "pretrain_path", "in_channels", "num_classes"])
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.model = get_segmenter(config, in_channels, num_classes, pretrain_path)

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = torchmetrics.functional.iou(
            y_hat.clone().detach(), y, num_classes=self.num_classes)

        self.log('train_loss', loss, on_epoch=True, prog_bar=False)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = torchmetrics.functional.iou(
            y_hat.clone().detach(), y, num_classes=self.num_classes)

        # logs
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = torchmetrics.functional.iou(
            y_hat.clone().detach(), y, num_classes=self.num_classes)

        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        self.log('test_acc', acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        # Here, you add every arguments needed for your module
        # NOTE: they must appear as arguments in the __init___() function
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--config_path', type=str)
        parser.add_argument('--pretrain_path', type=str, default=None)
        parser.add_argument('--num_classes', type=int, default=40)
        return parser
