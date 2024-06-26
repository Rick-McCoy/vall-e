import torch
from torch import Tensor, nn

from config.config import Config
from model.positional_encoding import PositionalEncoding
from model.transformer import TransformerEncoder, TransformerEncoderLayer
from utils.text import CHAR_TO_CODE, VOCAB_SIZE


class AutoRegressive(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.nhead = cfg.model.nhead
        self.text_embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=cfg.model.hidden_dim,
            padding_idx=CHAR_TO_CODE["<PAD>"],
        )
        self.audio_embedding = nn.Embedding(
            num_embeddings=2**cfg.data.codec_bits + 2,
            embedding_dim=cfg.model.hidden_dim,
            padding_idx=2**cfg.data.codec_bits + 1,
        )
        self.positional_encoding = PositionalEncoding(
            d_model=cfg.model.hidden_dim, dropout=cfg.model.dropout
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=cfg.model.hidden_dim,
                nhead=cfg.model.nhead,
                n_channels=1,
                dim_feedforward=cfg.model.dim_feedforward,
                dropout=cfg.model.dropout,
                batch_first=True,
            ),
            num_layers=cfg.model.num_layers,
        )
        self.linear = nn.Linear(
            in_features=cfg.model.hidden_dim,
            out_features=2**cfg.data.codec_bits + 2,
            bias=False,
        )
        self.linear.weight = self.audio_embedding.weight

    def forward(
        self,
        text: Tensor,
        audio: Tensor,
        text_len_batch: Tensor,
        audio_len_batch: Tensor,
    ):
        text_embedding = self.positional_encoding(self.text_embedding(text))
        audio_embedding = self.positional_encoding(self.audio_embedding(audio))

        max_len = int((text_len_batch + audio_len_batch).max().item())
        embed_list = []
        for text_embed, audio_embed, text_len, audio_len in zip(
            text_embedding, audio_embedding, text_len_batch, audio_len_batch
        ):
            embed_list.append(
                torch.cat(
                    [
                        text_embed[:text_len],
                        audio_embed[:audio_len],
                    ],
                    dim=0,
                ),
            )

        embed = torch.nn.utils.rnn.pad_sequence(embed_list, batch_first=True)
        total_len = text_len_batch + audio_len_batch
        mask = torch.ones(
            (max_len, max_len), dtype=torch.bool, device=text.device
        ).triu(diagonal=1)
        padding_mask = torch.arange(max_len, device=text.device).unsqueeze(
            0
        ) >= total_len.unsqueeze(1)
        transformer_output = self.transformer_encoder(
            embed, layer=0, mask=mask, src_key_padding_mask=padding_mask, is_causal=True
        )
        output = self.linear(transformer_output)
        return output
