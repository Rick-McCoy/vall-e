import torch
from torch import Tensor, nn
from torch.nn import functional as F


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, d_model: int, n_channels: int, eps: float):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_embedding = nn.Parameter(torch.zeros(n_channels, d_model))
        self.bias_embedding = nn.Parameter(torch.zeros(n_channels, d_model))

    def forward(self, data: Tensor, layer: int):
        weight = self.weight_embedding[layer].exp()
        bias = self.bias_embedding[layer]
        return F.layer_norm(data, self.normalized_shape, weight, bias, self.eps)
