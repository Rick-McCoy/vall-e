import torch
from torch import Tensor, nn

from config.config import Config
from data.text import VOCAB_SIZE


class NonAutoRegressive(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.cfg = config
        self.text_embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=config.model.hidden_dim,
        )
        self.audio_embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    num_embeddings=2**config.data.codec_bits,
                    embedding_dim=config.model.hidden_dim,
                )
                for _ in range(config.data.codec_channels)
            ]
        )
        self.index_embedding = nn.Embedding(
            num_embeddings=config.data.codec_channels,
            embedding_dim=config.model.hidden_dim,
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
        self.linears = nn.ModuleList(
            [
                nn.Linear(
                    in_features=config.model.hidden_dim,
                    out_features=2**config.data.codec_bits,
                )
            ]
        )

    def forward(
        self,
        text: Tensor,
        audio: Tensor,
        enrolled_audio: Tensor,
        text_len_batch: Tensor,
        audio_len_batch: Tensor,
        index: int,
    ):
        text_embedding = self.text_embedding(text)
        audio_embed_list = []
        for i, embedding in enumerate(self.audio_embeddings):
            audio_embed_list.append(embedding(audio[:, i]))
            if i == index:
                break
        audio_embedding = torch.stack(audio_embed_list, dim=1).sum(dim=1)
        enrolled_audio_embedding = torch.stack(
            [
                embedding(enrolled_audio[:, i])
                for i, embedding in enumerate(self.audio_embeddings)
            ],
            dim=1,
        ).sum(
            dim=1,
        )
        index_embedding = (
            self.index_embedding(torch.tensor(index))
            .reshape(1, 1, -1)
            .repeat(text.shape[0], 1, 1)
        )

        embed_list = []
        max_len = (
            int((text_len_batch + audio_len_batch).max().item())
            + self.cfg.data.enrolled_codec_len
        )
        for text_embed, audio_embed, enrolled_audio_embed, text_len, audio_len in zip(
            text_embedding,
            audio_embedding,
            enrolled_audio_embedding,
            text_len_batch,
            audio_len_batch,
        ):
            text_len_item = int(text_len.item())
            audio_len_item = int(audio_len.item())
            item_len = (
                text_len_item + audio_len_item + self.cfg.data.enrolled_codec_len + 1
            )
            embed_list.append(
                nn.functional.pad(
                    torch.cat(
                        [
                            text_embed[:, :text_len_item],
                            audio_embed[:, :audio_len_item],
                            enrolled_audio_embed,
                            index_embedding,
                        ],
                        dim=1,
                    ),
                    (0, 0, 0, max_len - item_len),
                )
            )

        embed = torch.stack(embed_list, dim=0).transpose(0, 1)
        transformer_output = self.transformer_decoder(embed, embed)
        for i, linear in enumerate(self.linears):
            if i == index:
                output = linear(transformer_output[:, -1, :])
                break
        else:
            raise ValueError(f"index {index} is out of range")
        return output
