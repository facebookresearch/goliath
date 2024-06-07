# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

""" Raymarcher for a mixture of volumetric primitives """
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from extensions.mvpraymarch.mvpraymarch import mvpraymarch


class Raymarcher(nn.Module):
    def __init__(self, volradius, dt: float = 1.0):
        super(Raymarcher, self).__init__()

        self.volume_radius = volradius

        # step size
        self.dt = dt / self.volume_radius

    def forward(
        self,
        raypos: torch.Tensor,
        raydir: torch.Tensor,
        tminmax: torch.Tensor,
        decout: Dict[str, torch.Tensor],
        renderoptions={},
        rayterm=None,
        with_shadow=False
    ):        
        primpos = decout["primpos"] / self.volume_radius
        primrot = decout["primrot"]
        primscale = decout["primscale"]
        template = decout["primrgba"]
        
        if decout.get("valid_prims", None) is not None:
            valid_prims = decout["valid_prims"]
            assert decout["valid_prims"].shape[0] == template.shape[1]
            template = template[:, valid_prims].contiguous()
            primpos = primpos[:, valid_prims].contiguous()
            primrot = primrot[:, valid_prims].contiguous()
            primscale = primscale[:, valid_prims].contiguous()

        out = mvpraymarch(
            raypos,
            raydir,
            self.dt,
            tminmax,
            (primpos, primrot, primscale),
            template=template,
            warp=decout["warp"] if "warp" in decout else None,
            rayterm=rayterm,
            with_shadow=with_shadow,
            **{k: v for k, v in renderoptions.items() if k in mvpraymarch.__code__.co_varnames}
        )
        if with_shadow:
            rayrgba, shadow = out
        else:
            rayrgba = out
            shadow = None

        assert rayrgba is not None

        rayrgba = rayrgba.permute(0, 3, 1, 2)
        rayrgb, rayalpha = rayrgba[:, :3].contiguous(), rayrgba[:, 3:4].contiguous()

        return rayrgb, rayalpha, rayrgba, shadow