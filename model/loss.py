from torch import Tensor, nn

from config.config import Config


class SimpleLoss(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logit: Tensor, target: Tensor) -> Tensor:
        return self.cross_entropy(logit, target)
