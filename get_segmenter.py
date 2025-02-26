import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from timm.models.layers import trunc_normal_
from segm.model.factory import create_segmenter
from segm.model.vit import PatchEmbedding
from segm.model.utils import init_weights


def get_segmenter(config: dict, in_channels: int, num_classes: int, pretrain_path: str = None):
    segmenter = create_segmenter(config['net_kwargs'])

    if pretrain_path is not None:
        checkpoint = torch.load(pretrain_path)
        segmenter.load_state_dict(checkpoint['model'])

    if config['net_kwargs']['decoder']['name'] == "linear":
        print('Linear Decoder')
        segmenter.decoder.n_cls = num_classes
        segmenter.decoder.head = nn.Linear(
            config['net_kwargs']['d_model'], num_classes)
        segmenter.decoder.apply(init_weights)
    else:
        print('Mask Transformer as Decoder')
        segmenter.decoder.n_cls = num_classes
        segmenter.decoder.cls_emb = nn.Parameter(torch.randn(
            1, num_classes, config['net_kwargs']['d_model']))
        trunc_normal_(segmenter.decoder.cls_emb, std=0.02)
        segmenter.decoder.mask_norm = nn.LayerNorm(num_classes)

    if in_channels != 3:
        segmenter.encoder.patch_embed = PatchEmbedding(
            config['net_kwargs']['image_size'], config['net_kwargs']['patch_size'], config['net_kwargs']['d_model'], in_channels)

    return segmenter
