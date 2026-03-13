import torch
import numpy as np
import logging
from models.modeling import VisionTransformer, CONFIGS

logger = logging.getLogger(__name__)

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def setup(args):
    config = CONFIGS[args.model_type]
    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=2)
    model.load_from(np.load("/path/to/pretrained/imagenet21k_ViT-B_16.npz"))
    model.to(args.device)
    num_params = count_parameters(model)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model
