# coding=utf-8
"""
Vision Transformer (ViT) model implementation in PyTorch.
Includes modules for Attention, MLP, Embeddings, and the main Transformer Encoder.
"""
from __future__ import absolute_import, division, print_function

import copy
import logging
import math
from os.path import join as pjoin

import torch
from torch import nn
import numpy as np
from scipy import ndimage

from models import configs
from .modeling_resnet import ResNetV2

logger = logging.getLogger(__name__)

# Weight mapping constants
ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO weights to OIHW and return as torch tensor."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    """Swish activation function."""
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


# pylint: disable=too-few-public-methods
class Attention(nn.Module):
    """Multi-head self-attention mechanism."""
    def __init__(self, config, vis):
        super().__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        """Reshape and transpose input for multi-head attention scores."""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        """Forward pass for Multi-head attention."""
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        attention_output = self.proj_dropout(self.out(context_layer))
        return attention_output, weights


class Mlp(nn.Module):
    """Multi-layer Perceptron block used in Transformer layers."""
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = nn.Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform and small biases."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        """Forward pass for MLP."""
        x = self.dropout(self.act_fn(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch and position embeddings."""
    def __init__(self, config, img_size, in_channels=3):
        super().__init__()
        self.hybrid = config.patches.get("grid") is not None
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size

        if self.hybrid:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid_model = ResNetV2(
                block_units=config.resnet.num_layers,
                width_factor=config.resnet.width_factor
            )
            in_channels = self.hybrid_model.width * 16
        else:
            patch_size = config.patches["size"]
            patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.patch_embeddings = nn.Conv2d(
            in_channels=in_channels,
            out_channels=config.hidden_size,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches + 1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        """Apply hybrid ResNet (optional), Patch Conv, and Position Embeddings."""
        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x).flatten(2).transpose(-1, -2)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return self.dropout(x + self.position_embeddings)


class Block(nn.Module):
    """Transformer layer block consisting of Attention and MLP."""
    def __init__(self, config, vis):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        """Forward pass with residual connections."""
        h = x
        x, weights = self.attn(self.attention_norm(x))
        x = x + h

        h = x
        x = self.ffn(self.ffn_norm(x))
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        """Loads weights into the block from a pretrained numpy dictionary."""
        root = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            # Linear Weights Mapping
            for target, source in [
                (self.attn.query, ATTENTION_Q), (self.attn.key, ATTENTION_K),
                (self.attn.value, ATTENTION_V), (self.attn.out, ATTENTION_OUT)
            ]:
                target.weight.copy_(np2th(weights[pjoin(root, source, "kernel")]).view(self.hidden_size, -1).t())
                target.bias.copy_(np2th(weights[pjoin(root, source, "bias")]).view(-1))

            self.ffn.fc1.weight.copy_(np2th(weights[pjoin(root, FC_0, "kernel")]).t())
            self.ffn.fc2.weight.copy_(np2th(weights[pjoin(root, FC_1, "kernel")]).t())
            self.ffn.fc1.bias.copy_(np2th(weights[pjoin(root, FC_0, "bias")]).t())
            self.ffn.fc2.bias.copy_(np2th(weights[pjoin(root, FC_1, "bias")]).t())

            self.attention_norm.weight.copy_(np2th(weights[pjoin(root, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(root, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(root, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(root, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    """Transformer Encoder containing multiple Transformer Blocks."""
    def __init__(self, config, vis):
        super().__init__()
        self.vis = vis
        self.layer = nn.ModuleList([Block(config, vis) for _ in range(config.transformer["num_layers"])])
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, hidden_states):
        """Iterate through blocks and apply final normalization."""
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        return self.encoder_norm(hidden_states), attn_weights


class VisionTransformer(nn.Module):
    """Vision Transformer for image classification."""
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super().__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)
        self.head = nn.Linear(config.hidden_size, num_classes)

    def forward(self, x, labels=None):
        """Forward pass for logits. If labels provided, returns loss."""
        x, attn_weights = self.encoder(self.embeddings(x))
        logits = self.head(x[:, 0])

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, self.num_classes), labels.view(-1))
            return logits, attn_weights, loss
        return logits, attn_weights

    def _resize_pos_embed(self, weights, emb):
        """Handles interpolation of position embeddings for different resolutions."""
        posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
        if posemb.size() == emb.position_embeddings.size():
            emb.position_embeddings.copy_(posemb)
            return

        ntok_new = emb.position_embeddings.size(1)
        posemb_tok, posemb_grid = (posemb[:, :1], posemb[0, 1:]) if self.classifier == "token" \
                                   else (posemb[:, :0], posemb[0])

        gs_old, gs_new = int(np.sqrt(len(posemb_grid))), int(np.sqrt(ntok_new - (1 if self.classifier == "token" else 0)))
        posemb_grid = ndimage.zoom(posemb_grid.reshape(gs_old, gs_old, -1), (gs_new/gs_old, gs_new/gs_old, 1), order=1)
        posemb = np.concatenate([posemb_tok, posemb_grid.reshape(1, -1, posemb.size(-1))], axis=1)
        emb.position_embeddings.copy_(np2th(posemb))

    def load_from(self, weights):
        """Loads pretrained weights into the VisionTransformer."""
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], True))
            self.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            self._resize_pos_embed(weights, self.embeddings)

            for i, block in enumerate(self.encoder.layer):
                block.load_from(weights, n_block=str(i))


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'testing': configs.get_testing(),
}
