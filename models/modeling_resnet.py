# coding=utf-8
"""Bottleneck ResNet v2 with GroupNorm and Weight Standardization."""

from collections import OrderedDict
from os.path import join as pjoin

import torch
from torch import nn
import torch.nn.functional as F


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW and convert to torch tensor."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


# pylint: disable=too-few-public-methods
class StdConv2d(nn.Conv2d):
    """Conv2d layer with Weight Standardization."""

    def forward(self, x):
        """Forward pass with weight standardization."""
        weight = self.weight
        var, mean = torch.var_mean(weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
        weight = (weight - mean) / torch.sqrt(var + 1e-5)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    """3x3 convolution with Weight Standardization."""
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    """1x1 convolution with Weight Standardization."""
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block."""

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        """Initialize the Pre-activation Bottleneck block."""
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or cin != cout:
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):
        """Forward pass for the bottleneck block."""
        residual = x
        if self.downsample is not None:
            residual = self.gn_proj(self.downsample(x))

        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        return self.relu(residual + y)

    def load_from(self, weights, n_block, n_unit):
        """Load weights from a numpy dictionary."""
        # pylint: disable=too-many-locals
        unit_path = pjoin(n_block, n_unit)
        
        with torch.no_grad():
            self.conv1.weight.copy_(np2th(weights[pjoin(unit_path, "conv1/kernel")], True))
            self.conv2.weight.copy_(np2th(weights[pjoin(unit_path, "conv2/kernel")], True))
            self.conv3.weight.copy_(np2th(weights[pjoin(unit_path, "conv3/kernel")], True))

            self.gn1.weight.copy_(np2th(weights[pjoin(unit_path, "gn1/scale")]).view(-1))
            self.gn1.bias.copy_(np2th(weights[pjoin(unit_path, "gn1/bias")]).view(-1))
            self.gn2.weight.copy_(np2th(weights[pjoin(unit_path, "gn2/scale")]).view(-1))
            self.gn2.bias.copy_(np2th(weights[pjoin(unit_path, "gn2/bias")]).view(-1))
            self.gn3.weight.copy_(np2th(weights[pjoin(unit_path, "gn3/scale")]).view(-1))
            self.gn3.bias.copy_(np2th(weights[pjoin(unit_path, "gn3/bias")]).view(-1))

            if self.downsample is not None:
                w_proj = np2th(weights[pjoin(unit_path, "conv_proj/kernel")], True)
                self.downsample.weight.copy_(w_proj)
                self.gn_proj.weight.copy_(np2th(weights[pjoin(unit_path, "gn_proj/scale")]).view(-1))
                self.gn_proj.bias.copy_(np2th(weights[pjoin(unit_path, "gn_proj/bias")]).view(-1))


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        """Initialize the ResNetV2 model."""
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', self._make_block(width, width * 4, width, block_units[0])),
            ('block2', self._make_block(width * 4, width * 8, width * 2, block_units[1], 2)),
            ('block3', self._make_block(width * 8, width * 16, width * 4, block_units[2], 2)),
        ]))

    @staticmethod
    def _make_block(cin, cout, cmid, num_units, stride=1):
        """Helper to create a sequential block of units."""
        units = [('unit1', PreActBottleneck(cin, cout, cmid, stride))]
        for i in range(2, num_units + 1):
            units.append((f'unit{i}', PreActBottleneck(cout, cout, cmid)))
        return nn.Sequential(OrderedDict(units))

    def forward(self, x):
        """Forward pass for the ResNetV2 model."""
        return self.body(self.root(x))
