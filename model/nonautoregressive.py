from typing import Callable, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from config.config import Config
from data.text import VOCAB_SIZE
from model.adaln import AdaptiveLayerNorm
from model.positional_encoding import PositionalEncoding


class TransformerDecoder(nn.TransformerDecoder):
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        layer: int,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        output = tgt

        for mod in self.layers:
            output = mod(
                output,
                memory,
                layer,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        n_channels: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable[[Tensor], Tensor] = F.relu,
        layer_norm_eps: float = 0.00001,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            device,
            dtype,
        )
        self.norm1 = AdaptiveLayerNorm(d_model=d_model, n_channels=n_channels)
        self.norm2 = AdaptiveLayerNorm(d_model=d_model, n_channels=n_channels)
        self.norm3 = AdaptiveLayerNorm(d_model=d_model, n_channels=n_channels)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        layer: int,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x, layer), tgt_mask, tgt_key_padding_mask, tgt_is_causal
            )
            x = x + self._mha_block(
                self.norm2(x, layer),
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            x = x + self._ff_block(self.norm3(x, layer))
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal),
                layer,
            )
            x = self.norm2(
                x
                + self._mha_block(
                    x, memory, memory_mask, memory_key_padding_mask, memory_is_causal
                ),
                layer,
            )
            x = self.norm3(x + self._ff_block(x), layer)

        return x


class NonAutoRegressive(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.text_embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=cfg.model.hidden_dim,
        )
        self.shared_audio_weight = nn.Parameter(
            torch.randn(
                cfg.data.codec_channels, 2**cfg.data.codec_bits, cfg.model.hidden_dim
            )
        )
        self.register_buffer(
            "offset",
            torch.arange(0, cfg.data.codec_channels).reshape(1, -1, 1)
            * 2**cfg.data.codec_bits,
        )
        self.offset: Tensor
        self.index_embedding_weight = nn.Parameter(
            torch.randn(cfg.data.codec_channels, cfg.model.hidden_dim)
        )
        self.positional_encoding = PositionalEncoding(
            d_model=cfg.model.hidden_dim, dropout=cfg.model.dropout
        )
        self.transformer_decoder = TransformerDecoder(
            decoder_layer=TransformerDecoderLayer(
                d_model=cfg.model.hidden_dim,
                nhead=cfg.model.nhead,
                n_channels=cfg.data.codec_channels,
                dim_feedforward=cfg.model.dim_feedforward,
                dropout=cfg.model.dropout,
                activation=cfg.model.activation,
                norm_first=True,
            ),
            num_layers=cfg.model.num_layers,
        )

    def forward(
        self,
        text: Tensor,
        audio: Tensor,
        enrolled_audio: Tensor,
        text_len_batch: Tensor,
        audio_len_batch: Tensor,
        enrolled_audio_len_batch: Tensor,
        index: int,
    ):
        text_embedding = self.positional_encoding(self.text_embedding(text))
        audio = audio + self.offset[:, :index]
        audio_embedding = self.positional_encoding(
            F.embedding(audio, self.shared_audio_weight.flatten(0, 1)).sum(dim=1)
        )
        enrolled_audio = enrolled_audio + self.offset
        enrolled_audio_embedding = self.positional_encoding(
            F.embedding(enrolled_audio, self.shared_audio_weight.flatten(0, 1)).sum(
                dim=1
            )
        )
        index_embedding = self.index_embedding_weight[index].unsqueeze(0)

        embed_list = []
        for (
            text_embed,
            audio_embed,
            enrolled_audio_embed,
            text_len,
            audio_len,
            enrolled_audio_len,
        ) in zip(
            text_embedding,
            audio_embedding,
            enrolled_audio_embedding,
            text_len_batch,
            audio_len_batch,
            enrolled_audio_len_batch,
        ):
            embed_list.append(
                torch.cat(
                    [
                        text_embed[:text_len],
                        enrolled_audio_embed[:enrolled_audio_len],
                        audio_embed[:audio_len],
                        index_embedding,
                    ],
                    dim=0,
                ),
            )

        embed = torch.nn.utils.rnn.pad_sequence(embed_list)
        transformer_output = self.transformer_decoder(embed, embed, layer=index)
        return torch.einsum(
            "lbc,dc->bld", transformer_output, self.shared_audio_weight[index]
        )
