import json
from math import ceil
from pathlib import Path
from typing import Literal, Optional

import hydra
import numpy as np
import pandas as pd
import torch
from encodec.model import EncodecModel
from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config.config import Config
from config.data.config import DataConfig
from config.model.config import ModelConfig
from config.train.config import TrainConfig
from data.audio import audio_to_codec, load_audio

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="train", name="base_train", node=TrainConfig)
cs.store(group="data", name="base_data", node=DataConfig)
cs.store(group="model", name="base_model", node=ModelConfig)


class PreprocessDataset(Dataset):
    def __init__(self, cfg: Config, mode: Literal["train", "val"]):
        self.cfg = cfg
        train_path = self.cfg.data.path / "train"
        val_path = self.cfg.data.path / "val"
        if mode == "train":
            self.json_list = list(train_path.glob("**/*.json"))
        else:
            self.json_list = list(val_path.glob("**/*.json"))

    def __len__(self):
        return len(self.json_list)

    def __getitem__(self, idx):
        json_path = self.json_list[idx]
        wav_path = Path(str(json_path).replace("label", "source")).with_suffix(".wav")
        with open(json_path, "r") as f:
            contents = json.load(f)
            text = contents["전사정보"]["TransLabelText"]
            speaker = contents["화자정보"]["SpeakerName"]
        if not wav_path.exists():
            return None
        audio = torch.from_numpy(
            load_audio(wav_path, self.cfg.data.sample_rate, self.cfg.data.channels)
        )
        return audio, wav_path, text, speaker


def collate_fn(batch: list[Optional[tuple[torch.Tensor, Path, str, str]]]):
    filtered_batch = [x for x in batch if x is not None]
    return (
        torch.nn.utils.rnn.pad_sequence(
            [x[0].T for x in filtered_batch], batch_first=True
        ).transpose(1, 2),
        [x[0].shape[1] for x in filtered_batch],
        [x[1] for x in filtered_batch],
        [x[2] for x in filtered_batch],
        [x[3] for x in filtered_batch],
    )


def preprocess(
    mode: Literal["train"] | Literal["val"], cfg: Config, encodec_model: EncodecModel
):
    dataset = PreprocessDataset(cfg, mode)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    compression_factor = cfg.data.sample_rate // cfg.data.codec_rate

    metadata_list = []
    tqdm_dataloader = tqdm(dataloader)
    for batch in tqdm_dataloader:
        audio_batch, audio_len_batch, wav_path_list, text_list, speaker_list = batch
        codec_batch = (
            audio_to_codec(audio_batch.to("cuda"), encodec_model)
            .detach()
            .cpu()
            .numpy()
            .astype(np.int16)
        )
        # Show max length of audio
        tqdm_dataloader.set_description(
            f"Max length of audio: {max(audio_len_batch)} samples"
        )
        for codec, audio_len, wav_path, text, speaker in zip(
            codec_batch, audio_len_batch, wav_path_list, text_list, speaker_list
        ):
            codec_len = ceil(audio_len / compression_factor)
            codec = codec[:, :codec_len]
            codec_path = Path(str(wav_path).replace("source", "codec")).with_suffix(
                ".npy"
            )
            codec_path.parent.mkdir(exist_ok=True, parents=True)
            np.save(codec_path, codec)
            relative_path = codec_path.relative_to(cfg.data.path / mode / "codec")
            metadata_list.append((speaker, text, relative_path))

    df = pd.DataFrame(metadata_list, columns=["speaker", "text", "codec_path"])
    if mode == "train":
        df.to_csv(cfg.data.path / "train_val.csv", index=False)
    else:
        df.to_csv(cfg.data.path / "test.csv", index=False)


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: Config):
    if cfg.data.sample_rate == 24000:
        encodec_model = EncodecModel.encodec_model_24khz()
    elif cfg.data.sample_rate == 48000:
        encodec_model = EncodecModel.encodec_model_48khz()
    else:
        raise NotImplementedError(f"Sample rate {cfg.data.sample_rate} not supported")
    encodec_model.set_target_bandwidth(
        cfg.data.codec_rate * cfg.data.codec_channels * cfg.data.codec_bits / 1000
    )

    encodec_model.eval()
    encodec_model.to("cuda")

    preprocess("train", cfg, encodec_model)
    preprocess("val", cfg, encodec_model)


if __name__ == "__main__":
    main()
