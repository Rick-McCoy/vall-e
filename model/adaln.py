from torch import Tensor, nn
from torch.nn import functional as F


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, d_model: int, n_channels: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = 1e-5
        self.weight_embedding = nn.Embedding(n_channels, embedding_dim=d_model)
        self.bias_embedding = nn.Embedding(n_channels, embedding_dim=d_model)

    def forward(self, data: Tensor, layer: int):
        weight = self.weight_embedding.weight[layer].exp()
        bias = self.bias_embedding.weight[layer]
        return F.layer_norm(data, self.normalized_shape, weight, bias, self.eps)
