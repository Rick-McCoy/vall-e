from typing import TypeVar

import numpy as np
import torch
from torch import Tensor

T = TypeVar("T", Tensor, np.ndarray)


def unpad_sequence(
    padded_sequences: T, lengths: T, batch_first: bool = False
) -> list[T]:
    """Unpad padded Tensor into a list of variable length Tensors
    The padded dimension must be the second dimension if batch_first is False
    The padded dimension must be the first dimension if batch_first is True

    Example:
        >>> tensor_1 = torch.zeros((1, 8, 20))
        >>> tensor_2 = torch.zeros((1, 4, 20))
        >>> tensor_3 = torch.zeros((1, 2, 20))
        >>> padded_tensor = torch.nn.utils.rnn.pad_sequence(
        ...     [tensor_1, tensor_2, tensor_3], batch_first=True
        ... )
        >>> lengths = torch.tensor([8, 4, 2])
        >>> unpadded_tensors = unpad_sequence(padded_tensor, lengths, batch_first=True)
        >>> unpadded_tensors[0].shape
        torch.Size([1, 8, 20])
        >>> unpadded_tensors[1].shape
        torch.Size([1, 4, 20])
        >>> unpadded_tensors[2].shape
        torch.Size([1, 2, 20])

    Args:
        padded_sequences (Tensor): Padded Tensor
        lengths (Tensor): Lengths of each sequence
        batch_first (bool): Whether the batch dimension is the first

    Returns:
        list[Tensor]: List of unpadded Tensors
    """

    assert padded_sequences.shape[0] == lengths.shape[0]
    if not batch_first:
        padded_sequences = padded_sequences.transpose(0, 1)

    unpadded_sequences = []
    for seq, length in zip(padded_sequences, lengths):
        unpacked_seq = seq[:length]
        unpadded_sequences.append(unpacked_seq)

    if not batch_first:
        unpadded_sequences = [seq.transpose(0, 1) for seq in unpadded_sequences]

    return unpadded_sequences


def remove_delay(audio: Tensor, audio_len: Tensor, n_channels: int, audio_pad: float):
    with torch.no_grad():
        split_audio = unpad_sequence(audio.transpose(1, 2), audio_len, batch_first=True)
        unpad_audio = [
            torch.stack(
                [audio_item[i : i - n_channels, i] for i in range(n_channels)], dim=0
            )
            for audio_item in split_audio
        ]

        padded_audio = torch.nn.utils.rnn.pad_sequence(
            [unpad_audio_item.T for unpad_audio_item in unpad_audio],
            batch_first=True,
            padding_value=audio_pad,
        ).transpose(1, 2)

        return padded_audio
