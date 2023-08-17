import torch
from torch import Tensor, nn
from torch.nn import functional as F

from config.config import Config
from model.positional_encoding import PositionalEncoding
from model.transformer import Transformer
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
        self.positional_encoding = PositionalEncoding(
            d_model=cfg.model.hidden_dim, dropout=cfg.model.dropout
        )
        self.transformer = Transformer(
            d_model=cfg.model.hidden_dim,
            nhead=cfg.model.nhead,
            num_encoder_layers=cfg.model.num_layers,
            num_decoder_layers=cfg.model.num_layers,
            dim_feedforward=cfg.model.dim_feedforward,
            dropout=cfg.model.dropout,
            activation=F.gelu,
            batch_first=True,
            norm_first=True,
            bias_ff=False,
            bias_attn=False,
        )
        self.linear = nn.Linear(
            cfg.model.hidden_dim, cfg.data.codec_num * self.n_channels, bias=False
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
        max_audio_len = audio_embedding.shape[1]
        max_text_len = text_embedding.shape[1]
        tgt_mask = torch.full(
            (max_audio_len, max_audio_len),
            float("-inf"),
            dtype=audio_embedding.dtype,
            device=audio_embedding.device,
        ).triu(diagonal=1)
        src_key_padding_mask = torch.arange(max_text_len).to(text_len.device).unsqueeze(
            0
        ) >= text_len.unsqueeze(1)
        src_key_padding_mask = torch.zeros_like(
            src_key_padding_mask, dtype=text_embedding.dtype
        ).masked_fill(src_key_padding_mask, float("-inf"))
        tgt_key_padding_mask = torch.arange(max_audio_len).to(
            audio_len.device
        ).unsqueeze(0) >= audio_len.unsqueeze(1)
        tgt_key_padding_mask = torch.zeros_like(
            tgt_key_padding_mask, dtype=audio_embedding.dtype
        ).masked_fill(tgt_key_padding_mask, float("-inf"))
        transformer_output = self.transformer(
            text_embedding,
            audio_embedding,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        output = (
            self.linear(transformer_output)
            .reshape(batch_size, length, self.n_channels, -1)
            .permute(0, 2, 1, 3)
        )
        return output
