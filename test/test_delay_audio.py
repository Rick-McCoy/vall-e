import unittest
from typing import cast

import torch
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore

from config.config import Config
from config.data.config import DataConfig
from config.model.config import ModelConfig
from config.train.config import TrainConfig
from model.delay_audio import DelayAudio

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="train", name="base_train", node=TrainConfig)
cs.store(group="data", name="base_data", node=DataConfig)
cs.store(group="model", name="base_model", node=ModelConfig)


class TestDelayAudio(unittest.TestCase):
    def setUp(self) -> None:
        with initialize(config_path="../config", version_base=None):
            cfg = compose(config_name="config")
            self.cfg = cast(Config, cfg)
            self.delay_audio = DelayAudio(self.cfg)
            self.codec_channels = self.cfg.data.codec_channels
            self.sos = 2**self.cfg.data.codec_bits
            self.eos = 2**self.cfg.data.codec_bits + 1
            self.pad = 2**self.cfg.data.codec_bits + 2

        self.batch_len = 4
        self.max_len = 100

    def test_delay_audio(self):
        data = torch.randint(
            0,
            2**self.cfg.data.codec_bits,
            (self.batch_len, self.codec_channels, self.max_len),
        )
        data_len = torch.randint(1, self.max_len, (self.batch_len,))
        data_len[-1] = self.max_len
        delay_target_audio, delay_target_audio_length = self.delay_audio(data, data_len)
        self.assertEqual(
            delay_target_audio.shape,
            (self.batch_len, self.codec_channels, self.max_len + self.codec_channels),
        )
        for i in range(self.batch_len):
            self.assertEqual(
                delay_target_audio_length[i].item(),
                data_len[i].item() + self.codec_channels,
            )
            data_len_item = int(data_len[i].item())
            for j in range(self.codec_channels):
                for k in range(j):
                    self.assertEqual(delay_target_audio[i, j, k].item(), self.sos)
                for k in range(j, data_len_item + j):
                    self.assertEqual(
                        delay_target_audio[i, j, k].item(), data[i, j, k - j].item()
                    )
                for k in range(
                    data_len_item + j,
                    data_len_item + self.codec_channels,
                ):
                    self.assertEqual(delay_target_audio[i, j, k].item(), self.eos)
                for k in range(
                    data_len_item + self.codec_channels,
                    self.max_len + self.codec_channels,
                ):
                    self.assertEqual(delay_target_audio[i, j, k].item(), self.pad)

    def test_remove_delay(self):
        data = torch.randint(
            0,
            2**self.cfg.data.codec_bits,
            (self.batch_len, self.codec_channels, self.max_len),
        )
        data_len = torch.randint(1, self.max_len, (self.batch_len,))
        data_len[-1] = self.max_len
        delay_target_audio, delay_target_audio_length = self.delay_audio(data, data_len)
        remove_delay_audio, remove_delay_audio_length = self.delay_audio.remove_delay(
            delay_target_audio, delay_target_audio_length
        )
        self.assertEqual(
            remove_delay_audio.shape,
            (self.batch_len, self.codec_channels, self.max_len),
        )
        for i in range(self.batch_len):
            self.assertEqual(remove_delay_audio_length[i].item(), data_len[i].item())
            data_len_item = int(data_len[i].item())
            for j in range(self.codec_channels):
                for k in range(data_len_item):
                    self.assertEqual(
                        remove_delay_audio[i, j, k].item(),
                        delay_target_audio[i, j, k + j].item(),
                    )
