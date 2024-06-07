# From DaaT merge. Fix here T145981161
# pyre-ignore-all-errors

from typing import Any, Mapping, Optional

import torch as th
import torch.nn as nn

from ca_code.loss import register_loss
from ca_code.utils.image import erode

from .effnet import EfficientNetLoss
from .vgg import VGGLossMasked


class BasePerceptualLoss(nn.Module):
    def __init__(
        self,
        assets,
        net,
        src_key: str = "rendered_rgb",
        tgt_key: str = "image",
        dst_key: Optional[str] = None,
        mask_key: str = "image_mask",
        mask_erode: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.src_key = src_key
        self.tgt_key = tgt_key
        self.dst_key = dst_key
        self.mask_key = mask_key
        self.mask_erode = mask_erode
        self.net = net

    def forward(
        self, preds: Mapping[str, Any], targets: Mapping[str, Any]
    ) -> th.Tensor:
        # NOTE: for relighting training, mask comes from rendered alpha
        fg_mask = (
            targets[self.mask_key] if self.mask_key in targets else preds[self.mask_key]
        )
        if self.mask_erode is not None:
            fg_mask = erode(fg_mask, self.mask_erode)
        src = preds[self.src_key]
        tgt = targets[self.tgt_key] if self.dst_key is None else preds[self.dst_key]
        return self.net(src, tgt, fg_mask)


@register_loss("vgg")
class VGGLoss(BasePerceptualLoss):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(net=VGGLossMasked(), *args, **kwargs)


@register_loss("effnet")
class EfficientNetLossImpl(BasePerceptualLoss):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(net=EfficientNetLoss(), *args, **kwargs)
