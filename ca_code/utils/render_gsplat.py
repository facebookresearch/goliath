# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch as th
from gsplat import rasterization

def render(
    cam_img_w: int,
    cam_img_h: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    Rt: th.Tensor,
    primpos: th.Tensor,
    primqvec: th.Tensor,
    primscale: th.Tensor,
    opacity: th.Tensor,
    colors: th.Tensor,
    return_depth: bool = True,
    bg_color: Optional[th.Tensor] = None,
    block_width: int = 16,
    global_scale: float = 1.0,
    z_near: float = 0.1,
):

    # NOTE(julieta) Rt comes in shape [3, 4] for some reason, make it [4, 4]
    if Rt.size(dim=0) == 3:
        Rt = th.cat((
            Rt,
            th.tensor([0,0,0,1])[None, :].to(Rt.device)),
            dim=0,
        )

    means3D = primpos.view(-1, 3).contiguous()
    scales = primscale.view(-1, 3).contiguous()
    rotations = primqvec.view(-1, 4).contiguous()
    opacity = opacity.view(-1, 1).contiguous()
    colors = colors.view(-1, 3).contiguous()

    if bg_color is None:
        bg_color = th.zeros(3, device=Rt.device)

    K = th.tensor([[fx, 0, cx], [0, fy, cy], [0., 0., 1.]], device=Rt.device)
    out_img, alpha, meta = rasterization(
        means=means3D, # [N, 3]
        quats=rotations, # [N, 4]
        scales=scales, # [N, 3]
        opacities=opacity.squeeze(-1), # [N]
        colors=colors, # [N, 3]
        viewmats=Rt[None, ...], # [1, 4, 4]
        Ks=K[None, ...], # [1, 3, 3]
        width=cam_img_w,
        height=cam_img_h,
        render_mode="RGB" + "+D" if return_depth else ""
    )

    out_img = out_img[0]
    alpha = alpha[0]

    radii = meta['radii'][0]

    assert alpha is not None
    out_color = out_img[..., :3]
    final_T = 1.0 - alpha

    out = {
        "render": out_color.permute(2, 0, 1),
        "final_T": final_T[None],
        "alpha": alpha[None],
        "radii": radii,
    }

    if return_depth:
        depth = out_img[..., -1]
        depth = depth[..., 0]
        out["depth"] = depth[None]

    return out
