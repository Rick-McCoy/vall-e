from torch import Tensor, nn

from config.config import Config


class VallELoss(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=2**cfg.data.codec_bits + 1
        )

    def forward(self, logit: Tensor, target: Tensor) -> Tensor:
        return self.cross_entropy(logit, target)
