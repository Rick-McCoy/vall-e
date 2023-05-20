from torch import nn


def remove_weight_norm(module: nn.Module):
    try:
        nn.utils.remove_weight_norm(module)
    except ValueError:
        pass
    for child in module.children():
        remove_weight_norm(child)
