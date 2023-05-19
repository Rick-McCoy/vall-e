import torch
from torch import Tensor, nn

from config.config import Config
from data.text import VOCAB_SIZE
from model.positional_encoding import PositionalEncoding


class AutoRegressive(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.text_embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=cfg.model.hidden_dim,
        )
        self.audio_embedding = nn.Embedding(
            num_embeddings=2**cfg.data.codec_bits,
            embedding_dim=cfg.model.hidden_dim,
        )
        self.positional_encoding = PositionalEncoding(
            d_model=cfg.model.hidden_dim,
            dropout=cfg.model.dropout,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=cfg.model.hidden_dim,
                nhead=cfg.model.nhead,
                dim_feedforward=cfg.model.dim_feedforward,
                dropout=cfg.model.dropout,
                activation=cfg.model.activation,
            ),
            num_layers=cfg.model.num_layers,
        )
        self.linear = nn.Linear(
            in_features=cfg.model.hidden_dim,
            out_features=2**cfg.data.codec_bits,
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
        mask_list = []
        for text_embed, audio_embed, text_len, audio_len in zip(
            text_embedding,
            audio_embedding,
            text_len_batch,
            audio_len_batch,
        ):
            item_len = int((text_len + audio_len).item())
            embed_list.append(
                nn.functional.pad(
                    torch.cat(
                        [
                            text_embed[:text_len],
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
            mask_item[:, :text_len] = 0
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
