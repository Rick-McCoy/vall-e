from torch import Tensor


def unpad_sequence(
    padded_sequences: Tensor, lengths: Tensor, batch_first: bool = False
) -> list[Tensor]:
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
    assert padded_sequences.shape[1] == lengths.max()
    unpadded_sequences: list[Tensor] = []

    if not batch_first:
        padded_sequences.transpose_(0, 1)

    for seq, length in zip(padded_sequences, lengths):
        unpacked_seq = seq[:length]
        unpadded_sequences.append(unpacked_seq)

    if not batch_first:
        unpadded_sequences = [seq.transpose(0, 1) for seq in unpadded_sequences]

    return unpadded_sequences
