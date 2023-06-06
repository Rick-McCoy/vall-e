import torch
from torch import Tensor, nn
from torch.nn import functional as F

from config.config import Config
from model.positional_encoding import PositionalEncoding
from model.transformer import TransformerEncoder, TransformerEncoderLayer
from utils.text import CHAR_TO_CODE, VOCAB_SIZE


class NonAutoRegressive(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.codec_channels = cfg.data.codec_channels
        self.padding_idx = 2**cfg.data.codec_bits + 1
        self.text_embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=cfg.model.hidden_dim,
            padding_idx=CHAR_TO_CODE["<PAD>"],
        )
        self.shared_audio_weight = nn.Parameter(
            torch.randn(
                cfg.data.codec_channels,
                2**cfg.data.codec_bits + 2,
                cfg.model.hidden_dim,
            )
        )
        with torch.no_grad():
            self.shared_audio_weight[:, -1].fill_(0)
        self.positional_encoding = PositionalEncoding(
            d_model=cfg.model.hidden_dim, dropout=cfg.model.dropout
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=cfg.model.hidden_dim,
                nhead=cfg.model.nhead,
                n_channels=cfg.data.codec_channels,
                dim_feedforward=cfg.model.dim_feedforward,
                dropout=cfg.model.dropout,
                batch_first=True,
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
    ):
        text_embedding = self.positional_encoding(self.text_embedding(text))
        index = audio.shape[1]
        audio_embedding = self.positional_encoding(
            torch.stack(
                [
                    F.embedding(
                        audio[:, i],
                        self.shared_audio_weight[i],
                        padding_idx=self.padding_idx,
                    )
                    for i in range(index)
                ],
                dim=1,
            ).sum(dim=1)
        )
        enrolled_audio_embedding = self.positional_encoding(
            torch.stack(
                [
                    F.embedding(
                        enrolled_audio[:, i],
                        self.shared_audio_weight[i],
                        padding_idx=self.padding_idx,
                    )
                    for i in range(self.codec_channels)
                ],
                dim=1,
            ).sum(dim=1)
        )

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
                    ],
                    dim=0,
                ),
            )

        embed = torch.nn.utils.rnn.pad_sequence(embed_list, batch_first=True)
        total_len = text_len_batch + audio_len_batch + enrolled_audio_len_batch + 1
        padding_mask = torch.arange(embed.shape[1], device=embed.device).unsqueeze(
            0
        ) >= total_len.unsqueeze(1)

        transformer_output = self.transformer_encoder(
            embed, layer=index, src_key_padding_mask=padding_mask
        )
        output = F.linear(transformer_output, self.shared_audio_weight[index])
        return output
