import torch
from torch import Tensor, nn
from torch.nn import functional as F

from config.config import Config
from data.text import VOCAB_SIZE
from model.positional_encoding import PositionalEncoding
from model.transformer import TransformerDecoder, TransformerDecoderLayer


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
                cfg.data.codec_channels,
                2**cfg.data.codec_bits + 2,
                cfg.model.hidden_dim,
            )
        )
        self.register_buffer(
            "offset",
            torch.arange(0, cfg.data.codec_channels).reshape(1, -1, 1)
            * (2**cfg.data.codec_bits + 2),
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
                batch_first=True,
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

        embed = torch.nn.utils.rnn.pad_sequence(embed_list, batch_first=True)
        transformer_output = self.transformer_decoder(embed, embed, layer=index)
        output = torch.einsum(
            "blc,dc->bld", transformer_output, self.shared_audio_weight[index]
        )
        return output
