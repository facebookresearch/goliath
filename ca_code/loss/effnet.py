# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.efficientnet import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetLoss(nn.Module):
    def __init__(
        self,
        activation_idxs: List[int] = None,
        weights: List[float] = None,
    ):
        super().__init__()
        self.ftrs_net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        self.register_buffer(
            "mean",
            th.as_tensor([0.485, 0.456, 0.406], dtype=th.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            th.as_tensor([0.229, 0.224, 0.225], dtype=th.float32).view(1, 3, 1, 1),
        )

        self.activation_idxs = [1, 2, 3]
        self.weights = [0.8, 0.1, 0.1]

        self.ftrs_net.eval()

        for param in self.parameters():
            param.requires_grad = False

    def normalize(self, batch):
        return ((batch / 255.0).clamp(0.0, 1.0) - self.mean) / self.std

    def _compute_features(self, x: th.Tensor):
        ftrs = []
        for i in range(0, max(self.activation_idxs) + 1):
            x = self.ftrs_net.features[i](x)
            if i in self.activation_idxs:
                ftrs.append(x)
        return ftrs

    def forward(self, x: th.Tensor, y: th.Tensor, mask: Optional[th.Tensor] = None):

        x = self.normalize(x)
        y = self.normalize(y)

        ftrs_x = self._compute_features(x)
        ftrs_y = self._compute_features(y)
        loss = 0.0
        for i, (fx, fy) in enumerate(zip(ftrs_x, ftrs_y)):
            if isinstance(mask, th.Tensor):
                m = F.interpolate(
                    mask, size=(fx.shape[-2], fx.shape[-1]), mode="bilinear"
                ).detach()
            else:
                m = 1.0
            loss += self.weights[i] * ((fx - fy) * m).abs().mean()
        return loss
