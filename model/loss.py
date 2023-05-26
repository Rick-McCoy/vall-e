from torch import Tensor, nn

from config.config import Config


class VallELoss(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=2**cfg.data.codec_bits + 1
        )

    def forward(self, logit: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        ar_loss = self.cross_entropy(logit[:, :, 0], target[:, 0])
        nar_loss = self.cross_entropy(logit[:, :, 1:], target[:, 1:])
        total_loss = self.cross_entropy(logit, target)
        return ar_loss, nar_loss, total_loss
