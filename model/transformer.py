from typing import Callable, Optional

import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import functional as F

from model.adaln import AdaptiveLayerNorm


class TransformerEncoder(nn.TransformerEncoder):
    def forward(
        self,
        src: Tensor,
        layer: int,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        output = src

        for mod in self.layers:
            output = torch.utils.checkpoint.checkpoint(
                mod, output, layer, mask, src_key_padding_mask, is_causal
            )
            assert output is not None

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    __constants__ = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        n_channels: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = F.relu,
        layer_norm_eps: float = 0.00001,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            device=device,
            dtype=dtype,
        )
        self.norm1 = AdaptiveLayerNorm(
            d_model=d_model, n_channels=n_channels, eps=layer_norm_eps
        )
        self.norm2 = AdaptiveLayerNorm(
            d_model=d_model, n_channels=n_channels, eps=layer_norm_eps
        )

    def forward(
        self,
        src: Tensor,
        layer: int,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x, layer),
                src_mask,
                src_key_padding_mask,
                is_causal=is_causal,
            )
            x = x + self._ff_block(self.norm2(x, layer))
        else:
            x = self.norm1(
                x
                + self._sa_block(
                    x, src_mask, src_key_padding_mask, is_causal=is_causal
                ),
                layer,
            )
            x = self.norm2(x + self._ff_block(x), layer)

        return x
