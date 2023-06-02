from typing import Callable, Optional

import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.modules.activation import MultiheadAttention

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


class TransformerEncoderLayer(nn.Module):
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
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = AdaptiveLayerNorm(
            d_model=d_model, n_channels=n_channels, eps=layer_norm_eps
        )
        self.norm2 = AdaptiveLayerNorm(
            d_model=d_model, n_channels=n_channels, eps=layer_norm_eps
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

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

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
