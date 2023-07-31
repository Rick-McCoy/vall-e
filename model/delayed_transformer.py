import torch
from torch import Tensor, nn

from config.config import Config
from model.positional_encoding import PositionalEncoding
from utils.text import CHAR_TO_CODE, VOCAB_SIZE


class DelayedTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_channels = cfg.data.codec_channels
        self.max_audio_len = cfg.data.max_codec_len
        self.max_text_len = cfg.data.max_text_len
        self.text_embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=cfg.model.hidden_dim,
            padding_idx=CHAR_TO_CODE["<PAD>"],
        )
        self.audio_embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    num_embeddings=cfg.data.codec_num,
                    embedding_dim=cfg.model.hidden_dim,
                    padding_idx=cfg.data.codec_pad,
                )
                for _ in range(self.n_channels)
            ]
        )
        self.padding_idx = cfg.data.codec_pad
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
            norm_first=True,
        )
        self.norm = nn.LayerNorm(cfg.model.hidden_dim)
        self.linear = nn.Linear(
            cfg.model.hidden_dim, cfg.data.codec_num * self.n_channels
        )

    def forward(self, text: Tensor, audio: Tensor, text_len: Tensor, audio_len: Tensor):
        text_embedding = self.positional_encoding(self.text_embedding(text))
        batch_size, _, length = audio.shape
        audio_embedding = self.positional_encoding(
            torch.stack(
                [
                    audio_embedding(audio[:, i])
                    for i, audio_embedding in enumerate(self.audio_embeddings)
                ],
                dim=1,
            ).sum(dim=1)
        )
        tgt_mask = torch.ones(
            (self.max_audio_len, self.max_audio_len),
            dtype=torch.bool,
            device=audio_len.device,
        ).triu(diagonal=1)
        src_key_padding_mask = torch.arange(self.max_text_len).to(
            text_len.device
        ).unsqueeze(0) >= text_len.unsqueeze(1)
        tgt_key_padding_mask = torch.arange(self.max_audio_len).to(
            audio_len.device
        ).unsqueeze(0) >= audio_len.unsqueeze(1)
        transformer_output = self.transformer(
            text_embedding,
            audio_embedding,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        transformer_output = self.norm(transformer_output)
        output = (
            self.linear(transformer_output)
            .reshape(batch_size, length, self.n_channels, -1)
            .permute(0, 2, 1, 3)
        )
        return output
