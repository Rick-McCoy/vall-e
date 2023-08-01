import torch
import torch.nn.functional as F
from torch import Tensor, nn
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
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    sorted_logits[sorted_indices_to_remove] = float("-inf")
    sample = torch.multinomial(
        F.softmax(sorted_logits.reshape(-1, vocab_size), dim=-1), num_samples=1
    )
    sample = sample.reshape(*prefix_shape, 1)
    return sorted_indices.gather(dim=-1, index=sample).squeeze(-1)
