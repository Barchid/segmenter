from cv2 import phase
from datasets.nyuv2.pytorch_dataset import NYUv2
from datasets.sunrgbd.pytorch_dataset import SUNRGBD
from preprocessing import get_preprocessor
from torch.utils.data import DataLoader


def get_nyuv2(data_dir: str, config: dict, batch_size: int, num_workers: int, debug: bool, **kwargs):
    train_set = NYUv2(
        data_dir=data_dir,
        n_classes=40,
        split='train',
        depth_mode='refined'
    )
    train_transforms = get_preprocessor(
        depth_mean=train_set.depth_mean,
        depth_std=train_set.depth_std,
        height=config['dataset_kwargs']['image_size'],
        width=config['dataset_kwargs']['image_size'],
        phase='train'
    )
    train_set.preprocessor = train_transforms
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              num_workers=num_workers, drop_last=True, shuffle= not debug)

    val_set = NYUv2(
        data_dir=data_dir,
        n_classes=40,
        split='test',
        depth_mode='refined'
    )
    val_transforms = get_preprocessor(
        depth_mean=train_set.depth_mean,
        depth_std=train_set.depth_std,
        height=config['dataset_kwargs']['image_size'],
        width=config['dataset_kwargs']['image_size'],
        phase='test'
    )
    val_set.preprocessor = val_transforms
    val_loader = DataLoader(train_set, batch_size=batch_size,
                            num_workers=num_workers, shuffle=False)

    return train_loader, val_loader


def get_sunrgbd(data_dir: str, config: dict, batch_size: int, num_workers: int, debug: bool, **kwargs):
    train_set = SUNRGBD(
        data_dir=data_dir,
        split='train',
        depth_mode='refined'
    )
    train_transforms = get_preprocessor(
        depth_mean=train_set.depth_mean,
        depth_std=train_set.depth_std,
        height=config['dataset_kwargs']['image_size'],
        width=config['dataset_kwargs']['image_size'],
        phase='train'
    )
    train_set.preprocessor = train_transforms
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              num_workers=num_workers, drop_last=True, shuffle=not debug)

    val_set = SUNRGBD(
        data_dir=data_dir,
        split='test',
        depth_mode='refined'
    )
    val_transforms = get_preprocessor(
        depth_mean=train_set.depth_mean,
        depth_std=train_set.depth_std,
        height=config['dataset_kwargs']['image_size'],
        width=config['dataset_kwargs']['image_size'],
        phase='test'
    )
    val_set.preprocessor = val_transforms
    val_loader = DataLoader(train_set, batch_size=batch_size,
                            num_workers=num_workers, shuffle=False)

    return train_loader, val_loader
