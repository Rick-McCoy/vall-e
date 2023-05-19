import itertools
from typing import Callable

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
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
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
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
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
        self.enrolled_codec_len = (
            self.cfg.data.enrolled_codec_sec * self.cfg.data.codec_rate + 1
        )
        self.text_embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=cfg.model.hidden_dim,
        )
        self.audio_embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    num_embeddings=2**cfg.data.codec_bits,
                    embedding_dim=cfg.model.hidden_dim,
                )
                for _ in range(cfg.data.codec_channels)
            ]
        )
        self.positional_encoding = PositionalEncoding(
            d_model=cfg.model.hidden_dim,
            dropout=cfg.model.dropout,
        )
        self.index_embedding = nn.Embedding(
            num_embeddings=cfg.data.codec_channels,
            embedding_dim=cfg.model.hidden_dim,
        )
        self.transformer_decoder = TransformerDecoder(
            decoder_layer=TransformerDecoderLayer(
                d_model=cfg.model.hidden_dim,
                nhead=cfg.model.nhead,
                dim_feedforward=cfg.model.dim_feedforward,
                dropout=cfg.model.dropout,
                activation=cfg.model.activation,
                n_channels=cfg.data.codec_channels,
            ),
            num_layers=cfg.model.num_layers,
        )
        self.linears = nn.ModuleList(
            [
                nn.Linear(
                    in_features=cfg.model.hidden_dim,
                    out_features=2**cfg.data.codec_bits,
                    bias=False,
                )
                for _ in range(cfg.data.codec_channels)
            ]
        )
        for linear, audio_embedding in zip(
            itertools.islice(self.linears, 1, None),
            itertools.islice(self.audio_embeddings, None, cfg.data.codec_channels - 1),
        ):
            linear.weight = audio_embedding.weight

    def forward(
        self,
        text: Tensor,
        audio: Tensor,
        enrolled_audio: Tensor,
        text_len_batch: Tensor,
        audio_len_batch: Tensor,
        index: int,
    ):
        text_embedding = self.positional_encoding(self.text_embedding(text))
        audio_embed_list = []
        for i, embedding in enumerate(self.audio_embeddings):
            audio_embed_list.append(embedding(audio[:, i]))
            if i == index - 1:
                break
        audio_embedding = self.positional_encoding(
            torch.stack(audio_embed_list, dim=1).sum(dim=1)
        )
        enrolled_audio_embed_list = [
            embedding(enrolled_audio[:, i])
            for i, embedding in enumerate(self.audio_embeddings)
        ]
        enrolled_audio_embedding = self.positional_encoding(
            torch.stack(enrolled_audio_embed_list, dim=1).sum(dim=1)
        )
        index_embedding = self.positional_encoding(
            self.index_embedding.weight[index].reshape(1, 1, -1)
        ).squeeze(0)

        embed_list = []
        max_len = (
            int((text_len_batch + audio_len_batch).max().item())
            + self.enrolled_codec_len
            + 1
        )
        for text_embed, audio_embed, enrolled_audio_embed, text_len, audio_len in zip(
            text_embedding,
            audio_embedding,
            enrolled_audio_embedding,
            text_len_batch,
            audio_len_batch,
        ):
            item_len = int((text_len + audio_len).item()) + self.enrolled_codec_len + 1
            embed_list.append(
                nn.functional.pad(
                    torch.cat(
                        [
                            text_embed[:text_len],
                            enrolled_audio_embed,
                            audio_embed[:audio_len],
                            index_embedding,
                        ],
                        dim=0,
                    ),
                    (0, 0, 0, max_len - item_len),
                )
            )

        embed = torch.einsum("blc->lbc", torch.stack(embed_list, dim=0))
        transformer_output = self.transformer_decoder(embed, embed, layer=index)
        for i, linear in enumerate(self.linears):
            if i == index - 1:
                output = linear(torch.einsum("lbc->blc", transformer_output))
                break
        else:
            raise ValueError(f"index {index} is out of range")
        return output
