# coding=utf-8
"""
Functions for model initialization, parameter counting, and saving checkpoints.
"""
import logging
import os

import torch
import numpy as np
from models.modeling import VisionTransformer, CONFIGS

logger = logging.getLogger(__name__)


def count_parameters(model):
    """
    Returns the number of trainable parameters in millions.
    """
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def setup(args):
    """
    Initializes the Vision Transformer model, loads pretrained weights,
    and moves the model to the specified device.
    """
    config = CONFIGS[args.model_type]
    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=2)
    model.load_from(np.load("/path/to/pretrained/imagenet21k_ViT-B_16.npz"))
    model.to(args.device)
    num_params = count_parameters(model)
    logger.info("Total Parameter: \t%2.1fM", num_params)
    return args, model


def save_model(args, model, fold):
    """
    Saves the model checkpoint to the output directory.
    """
    model_to_save = model.module if hasattr(model, 'module') else model
    checkpoint_name = f"{args.name}_fold{fold}_checkpoint_with_weight.bin"
    model_checkpoint = os.path.join(args.output_dir, checkpoint_name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", model_checkpoint)
