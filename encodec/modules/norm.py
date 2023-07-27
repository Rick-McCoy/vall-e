# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Normalization modules."""


import torch
from torch import nn


class ConvLayerNorm(nn.LayerNorm):
    """
    Convolution-friendly LayerNorm that moves channels to last dimensions
    before running the normalization and moves them back to original position right
    after.
    """

    def __init__(self, normalized_shape: int | list[int] | torch.Size, **kwargs):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        x = torch.einsum("b ... t -> b t ...", x)
        x = super().forward(x)
        x = torch.einsum("b t ... -> b ... t", x)
        return
