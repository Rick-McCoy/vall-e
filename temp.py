import torch

A = torch.Tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
A_lengths = torch.LongTensor([3, 5])
B = torch.Tensor([[9, 10, 11, 12, 13, 14], [15, 0, 0, 0, 0, 0]])
B_lengths = torch.LongTensor([6, 1])

total_elements = A_lengths.sum() + B_lengths.sum()
C = torch.zeros((A.size(0), total_elements), dtype=A.dtype, device=A.device)

A_indices = A_lengths.cumsum(dim=0) - A_lengths
B_indices = B_lengths.cumsum(dim=0) - B_lengths

C[:, A_indices] = A[:, : A_lengths.max()]
C[:, B_indices] = B[:, : B_lengths.max()]

C_lengths = torch.cat([A_lengths, B_lengths], dim=0)
max_length = C_lengths.max()
C = C[:, :max_length]

C_lengths = C_lengths.unsqueeze(0)  # Add a batch dimension to C_lengths

print("C:", C)
print("C_lengths:", C_lengths)
