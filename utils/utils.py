import torch


def unpad_sequence(
    padded_sequences: torch.Tensor, lengths: torch.Tensor, batch_first: bool = False
) -> list[torch.Tensor]:
    unpadded_sequences = []

    if not batch_first:
        padded_sequences.transpose_(0, 1)

    for seq, length in zip(padded_sequences, lengths):
        unpacked_seq = seq[:length]
        unpadded_sequences.append(unpacked_seq)

    return unpadded_sequences
