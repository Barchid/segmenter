from argparse import ArgumentParser, Namespace
from os import times

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
import torchmetrics
from get_segmenter import get_segmenter
from segm.optim.factory import create_optimizer, create_scheduler

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class SegmenterModule(pl.LightningModule):
    def __init__(self, config: dict, learning_rate: float, in_channels: int, num_classes: int, num_samples_train: int, epochs: int, pretrain_path: str = None, **kwargs):
        super().__init__()
        self.save_hyperparameters(
            ignore=["config_path", "pretrain_path", "in_channels", "num_classes", "num_samples_train", "epochs"])
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.model = get_segmenter(
            config, in_channels, num_classes, pretrain_path)
        self.config = config
        self.num_samples_train = num_samples_train
        self.epochs = epochs
        self.learning_rate = learning_rate

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
        self.config['optimizer_kwargs']['iter_max'] = self.num_samples_train * self.epochs
        self.config['optimizer_kwargs']['iter_warmup'] = 0.0
        self.config['optimizer_kwargs']['lr'] = self.learning_rate
        optimizer = create_optimizer(Namespace(**self.config['optimizer_kwargs']), self)
        scheduler = create_scheduler(
            Namespace(**self.config['optimizer_kwargs']), optimizer)

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        # Here, you add every arguments needed for your module
        # NOTE: they must appear as arguments in the __init___() function
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--pretrain_path', type=str, default=None)
        return parser
