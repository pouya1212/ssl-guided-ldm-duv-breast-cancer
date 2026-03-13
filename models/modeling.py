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
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
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
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch and position embeddings."""
    def __init__(self, config, img_size, in_channels=3):
        super().__init__()
        self.hybrid = False
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = config.patches["size"]
            patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        if self.hybrid:
            self.hybrid_model = ResNetV2(
                block_units=config.resnet.num_layers,
                width_factor=config.resnet.width_factor
            )
            in_channels = self.hybrid_model.width * 16

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
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


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
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        """Loads weights into the block from a pretrained numpy dictionary."""
        root = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            # Attention Weights
            q_w = np2th(weights[pjoin(root, ATTENTION_Q, "kernel")]).view(self.hidden_size, -1).t()
            k_w = np2th(weights[pjoin(root, ATTENTION_K, "kernel")]).view(self.hidden_size, -1).t()
            v_w = np2th(weights[pjoin(root, ATTENTION_V, "kernel")]).view(self.hidden_size, -1).t()
            o_w = np2th(weights[pjoin(root, ATTENTION_OUT, "kernel")]).view(self.hidden_size, -1).t()

            self.attn.query.weight.copy_(q_w)
            self.attn.key.weight.copy_(k_w)
            self.attn.value.weight.copy_(v_w)
            self.attn.out.weight.copy_(o_w)

            # Attention Biases
            self.attn.query.bias.copy_(np2th(weights[pjoin(root, ATTENTION_Q, "bias")]).view(-1))
            self.attn.key.bias.copy_(np2th(weights[pjoin(root, ATTENTION_K, "bias")]).view(-1))
            self.attn.value.bias.copy_(np2th(weights[pjoin(root, ATTENTION_V, "bias")]).view(-1))
            self.attn.out.bias.copy_(np2th(weights[pjoin(root, ATTENTION_OUT, "bias")]).view(-1))

            # MLP Weights
            self.ffn.fc1.weight.copy_(np2th(weights[pjoin(root, FC_0, "kernel")]).t())
            self.ffn.fc2.weight.copy_(np2th(weights[pjoin(root, FC_1, "kernel")]).t())
            self.ffn.fc1.bias.copy_(np2th(weights[pjoin(root, FC_0, "bias")]).t())
            self.ffn.fc2.bias.copy_(np2th(weights[pjoin(root, FC_1, "bias")]).t())

            # Norms
            self.attention_norm.weight.copy_(np2th(weights[pjoin(root, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(root, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(root, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(root, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    """Transformer Encoder containing multiple Transformer Blocks."""
    def __init__(self, config, vis):
        super().__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    """Full Transformer including embeddings and encoder."""
    def __init__(self, config, img_size, vis):
        super().__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class VisionTransformer(nn.Module):
    """Vision Transformer for image classification."""
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super().__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, vis)
        self.head = nn.Linear(config.hidden_size, num_classes)

    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return logits, attn_weights, loss
        return logits, attn_weights

    def load_from(self, weights):
        """Loads pretrained weights into the VisionTransformer."""
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            emb = self.transformer.embeddings
            enc = self.transformer.encoder
            emb.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            emb.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            emb.cls_token.copy_(np2th(weights["cls"]))
            enc.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            enc.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = emb.position_embeddings
            if posemb.size() == posemb_new.size():
                emb.position_embeddings.copy_(posemb)
            else:
                logger.info("Resizing position embeddings: %s to %s", posemb.size(), posemb_new.size())
                ntok_new = posemb_new.size(1)
                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                logger.info("Grid-size changing from %s to %s", gs_old, gs_new)
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                emb.position_embeddings.copy_(np2th(posemb))

            for _, block in enc.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if emb.hybrid:
                h_model = emb.hybrid_model
                h_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                h_model.root.gn.weight.copy_(np2th(weights["gn_root/scale"]).view(-1))
                h_model.root.gn.bias.copy_(np2th(weights["gn_root/bias"]).view(-1))

                for bname, block in h_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'testing': configs.get_testing(),
}
