from argparse import ArgumentParser
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from get_dataloaders import get_nyuv2, get_sunrgbd
from segmenter_module import SegmenterModule
import yaml
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor


def main():
    # seeds the random from numpy, pytorch, etc for reproductibility
    pl.seed_everything(1234)

    args = get_args()

    config = yaml.load(
        open(args.config_path, "r"), Loader=yaml.FullLoader
    )

    if args.dataset == 'nyudv2':
        train_loader, val_loader = get_nyuv2(
            args.data_dir, config, args.batch_size, args.num_workers)

    else:
        train_loader, val_loader = get_sunrgbd(
            args.data_dir, config, args.batch_size, args.num_workers)

    module = create_module(args, config, train_loader)

    trainer = create_trainer(args)

    # Launch training/validation
    if args.mode == "train":
        trainer.fit(module, ckpt_path=args.ckpt_path,
                    train_dataloaders=train_loader, val_dataloaders=val_loader)

        # report results in a txt file
        report_path = os.path.join(args.default_root_dir, 'train_report.txt')
        report = open(report_path, 'a')

        # TODO: add any data you want to report here
        # here, we put the model's hyperparameters and the resulting val accuracy
        report.write(
            f"{config['net_kwargs']['backbone']} {args.dataset} {args.learning_rate}  {trainer.checkpoint_callback.best_model_score}\n")
    elif args.mode == "lr_find":
        lr_finder = trainer.tuner.lr_find(
            module, train_dataloaders=train_loader)
        fig = lr_finder.plot(suggest=True)
        fig.show()
        print(f'SUGGESTION IS :', lr_finder.suggestion())
    else:
        trainer.validate(module, ckpt_path=args.ckpt_path,
                         val_dataloaders=val_loader)


def create_module(args, config: dict, train_loader: DataLoader) -> pl.LightningModule:
    # vars() is required to pass the arguments as parameters for the LightningModule
    dict_args = vars(args)
    dict_args['config'] = config
    dict_args['num_classes'] = 40 if args.dataset == 'nyudv2' else 37
    dict_args['num_samples_train'] = len(train_loader)
    dict_args['epochs'] = args.max_epochs
    dict_args['in_channels'] = 3

    # TODO: you can change the module class here
    module = SegmenterModule(**dict_args)

    return module


def create_trainer(args) -> pl.Trainer:
    # saves the best model checkpoint based on the accuracy in the validation set
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",  # TODO: select the logged metric to monitor the checkpoint saving
        filename="model-{epoch:03d}-{val_acc:.4f}",
        save_top_k=1,
        mode="max",
    )

    logger = pl.loggers.TensorBoardLogger(
        os.path.join(args.default_root_dir, "logger"),
        name=f"{args.name} {args.dataset} {args.event_representation}"
    )

    # create trainer
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step')], logger=logger)
    return trainer


def get_args():
    # Program args
    # TODO: you can add program-specific arguments here
    parser = ArgumentParser()
    parser.add_argument(
        '--mode', type=str, choices=["train", "validate", "lr_find"], default="train")
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help="Path of a checkpoint file. Defaults to None, meaning the training/testing will start from scratch.")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--dataset', type=str, choices=["nyudv2", "sunrgbd"])
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data')

    # Args for model
    parser = SegmenterModule.add_model_specific_args(parser)

    # Args for Trainer
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
