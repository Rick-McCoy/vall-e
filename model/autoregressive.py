import torch
from torch import Tensor, nn

from config.config import Config
from data.text import VOCAB_SIZE
from model.positional_encoding import PositionalEncoding


class AutoRegressive(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.cfg = config
        self.enrolled_codec_len = (
            self.cfg.data.enrolled_codec_sec * self.cfg.data.codec_rate + 1
        )
        self.text_embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=config.model.hidden_dim,
        )
        self.audio_embedding = nn.Embedding(
            num_embeddings=2**config.data.codec_bits,
            embedding_dim=config.model.hidden_dim,
        )
        self.positional_encoding = PositionalEncoding(
            d_model=config.model.hidden_dim,
            dropout=config.model.dropout,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=config.model.hidden_dim,
                nhead=config.model.nhead,
                dim_feedforward=config.model.dim_feedforward,
                dropout=config.model.dropout,
                activation=config.model.activation,
            ),
            num_layers=config.model.num_layers,
        )
        self.linear = nn.Linear(
            in_features=config.model.hidden_dim,
            out_features=2**config.data.codec_bits,
        )

    def forward(
        self,
        text: Tensor,
        audio: Tensor,
        enrolled_audio: Tensor,
        text_len_batch: Tensor,
        audio_len_batch: Tensor,
    ):
        text_embedding = self.positional_encoding(self.text_embedding(text))
        audio_embedding = self.positional_encoding(self.audio_embedding(audio))
        enrolled_audio_embedding = self.positional_encoding(
            self.audio_embedding(enrolled_audio)
        )

        max_len = (
            int((text_len_batch + audio_len_batch).max().item())
            + self.enrolled_codec_len
        )
        embed_list = []
        mask_list = []
        for text_embed, audio_embed, enrolled_audio_embed, text_len, audio_len in zip(
            text_embedding,
            audio_embedding,
            enrolled_audio_embedding,
            text_len_batch,
            audio_len_batch,
        ):
            item_len = int((text_len + audio_len).item()) + self.enrolled_codec_len
            embed_list.append(
                nn.functional.pad(
                    torch.cat(
                        [
                            text_embed[:text_len],
                            enrolled_audio_embed,
                            audio_embed[:audio_len],
                        ],
                        dim=0,
                    ),
                    (0, 0, 0, max_len - item_len),
                )
            )
            mask_item = (
                torch.full((item_len, item_len), float("-inf")).triu(1).to(text.device)
            )
            mask_item[:, : text_len + self.enrolled_codec_len] = 0
            mask_list.append(
                nn.functional.pad(
                    mask_item,
                    (0, max_len - item_len, 0, max_len - item_len),
                )
            )

        mask = torch.stack(mask_list, dim=0).repeat_interleave(
            repeats=self.cfg.model.nhead, dim=0
        )
        embed = torch.einsum("blc->lbc", torch.stack(embed_list, dim=0))
        transformer_output = self.transformer_decoder(embed, embed, tgt_mask=mask)
        output = self.linear(torch.einsum("lbc->blc", transformer_output))
        return output
