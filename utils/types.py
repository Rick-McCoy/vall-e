from dataclasses import dataclass
from enum import Enum

import numpy as np
from torch import Tensor


@dataclass
class Batch:
    text: np.ndarray
    audio: np.ndarray
    enrolled_audio: np.ndarray


@dataclass
class CollatedBatch:
    text: Tensor
    text_len: Tensor
    audio: Tensor
    audio_len: Tensor
    enrolled_audio: Tensor
    enrolled_audio_len: Tensor


class ChannelEnum(Enum):
    SINGLE = 1
    DOUBLE = 2
