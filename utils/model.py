import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.weight_norm import remove_weight_norm


def remove_norm(module: nn.Module):
    try:
        remove_weight_norm(module)
    except ValueError:
        pass
    for child in module.children():
        remove_norm(child)


def nucleus_sample(logits: Tensor, top_p: float = 0.9) -> Tensor:
    """
    Truncate the distribution to the top_p percentile using the
    cumulative probability distribution of the sorted logit values.

    Args:
        logits (Tensor): Logits of the next token.
        top_p (float): The probability mass of the top_p tokens.

    Returns:
        Tensor: The sampled token.
    """
    *prefix_shape, vocab_size = logits.shape
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    sorted_probs.masked_fill_(sorted_indices_to_remove, 0.0)
    sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-6)
    sample = torch.multinomial(sorted_probs.reshape(-1, vocab_size), num_samples=1)
    sample = sample.reshape(*prefix_shape, 1)
    return sorted_indices.gather(dim=-1, index=sample).squeeze(-1)
