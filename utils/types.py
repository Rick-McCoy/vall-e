from dataclasses import dataclass

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
