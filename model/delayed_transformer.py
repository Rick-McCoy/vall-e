import torch
from torch import nn
from torch.nn import functional as F

from config.config import Config
from model.positional_encoding import PositionalEncoding
from utils.text import CHAR_TO_CODE, VOCAB_SIZE


class DelayedTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_channels = cfg.data.codec_channels
        self.text_embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=cfg.model.hidden_dim,
            padding_idx=CHAR_TO_CODE["<PAD>"],
        )
        self.shared_audio_weight = nn.Parameter(
            torch.randn(
                cfg.data.codec_channels,
                2**cfg.data.codec_bits + 3,
                cfg.model.hidden_dim,
            )
        )
        self.padding_idx = 2**cfg.data.codec_bits + 2
        with torch.no_grad():
            self.shared_audio_weight[:, -1].fill_(0)
        self.positional_encoding = PositionalEncoding(
            d_model=cfg.model.hidden_dim, dropout=cfg.model.dropout
        )
        self.transformer = nn.Transformer(
            d_model=cfg.model.hidden_dim,
            nhead=cfg.model.nhead,
            num_encoder_layers=cfg.model.num_layers,
            num_decoder_layers=cfg.model.num_layers,
            dim_feedforward=cfg.model.dim_feedforward,
            dropout=cfg.model.dropout,
            batch_first=True,
        )

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        text_len: torch.Tensor,
        audio_len: torch.Tensor,
    ):
        text_embedding = self.positional_encoding(self.text_embedding(text))
        audio_embedding = torch.stack(
            [
                self.positional_encoding(
                    F.embedding(
                        audio[:, i],
                        self.shared_audio_weight[i],
                        padding_idx=self.padding_idx,
                    )
                )
                for i in range(self.n_channels)
            ],
            dim=1,
        ).sum(dim=1)
        max_text_len = text_embedding.shape[1]
        max_audio_len = audio_embedding.shape[1]
        tgt_mask = torch.ones(
            (max_audio_len, max_audio_len), dtype=torch.bool, device=audio.device
        ).triu(diagonal=1)
        src_key_padding_mask = torch.arange(max_text_len).to(
            text_embedding.device
        ).unsqueeze(0) >= text_len.unsqueeze(1)
        tgt_key_padding_mask = torch.arange(max_audio_len).to(
            audio_embedding.device
        ).unsqueeze(0) >= audio_len.unsqueeze(1)
        transformer_output = self.transformer(
            text_embedding,
            audio_embedding,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        output = torch.stack(
            [
                F.linear(
                    transformer_output,
                    self.shared_audio_weight[i],
                    bias=None,
                )
                for i in range(self.n_channels)
            ],
            dim=1,
        )
        return output
