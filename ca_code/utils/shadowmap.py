# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch as th
import torch.nn.functional as thf

from ca_code.utils.geom import project_points_multi

import cv2

def get_shadow_map(rl, Rt, K, verts, postex, nml = None):
    batch_size = postex.shape[0]
    height = postex.shape[2]
    width = postex.shape[3]

    K = th.eye(3)[None].expand(Rt.shape[0]).to(Rt.device)
    K[:, 0, 0] = 1000
    K[:, 1, 1] = 1000
    K[:, 0, 2] = rl.w / 2
    K[:, 1, 2] = rl.h / 2

    points = postex.permute(0, 2, 3, 1).contiguous().view((batch_size, -1, 3))
    v_pix, v_cam = project_points_multi(verts, Rt, K)

    center = th.tensor([rl.w, rl.h], dtype=th.float32, device=Rt.device) / 2
    pix_ratio = 1.02 * ((v_pix[..., :2] - center[None, None]) / center[None, None])
    focal = focal / abs(pix_ratio).max(1)[0]
    v_pix, v_cam = project_points_multi(points, Rt, K)

    # TODO: just use the rasterizer directly?
    tex = th.empty(batch_size, 1, 1024, 1024, device=Rt.device)

    # TODO TMP
    if isinstance(verts, (list, tuple)):
        z = th.empty(batch_size, 1, 256, 256, device=Rt.device)
        tex = [z, z, tex]

    rlout = rl(
        verts,
        tex,
        K,
        Rt,
        output_filters=["depth_img", "index_img", "mask"],
    )

    depth = rlout["depth_img"][:, None, :, :]
    v_depth_1 = v_cam[:, :, [2]].view(batch_size, height, width, 1).permute(0, 3, 1, 2).contiguous()

    v_pix = v_pix[:, :, 0:2].view(batch_size, height, width, 2)

    v_pix[..., 0] = (v_pix[..., 0] - depth.shape[3] / 2.0 - 0.5) / (depth.shape[3] / 2.0)
    v_pix[..., 1] = (v_pix[..., 1] - depth.shape[2] / 2.0 - 0.5) / (depth.shape[2] / 2.0)
    
    # compute backface
    if nml is not None:
        v_dir = thf.normalize(campos - postex, dim=1)
        nv_dot = (nml * v_dir).sum(1, keepdim=True)
        bcull_mask = th.sigmoid(10 * nv_dot) # softer boundary

    kernel = 3
    sigma = 0.3 * ((kernel - 1) * 0.5 - 1) + 0.8
    valid = []
    in_shadow = []
    dx = 2.0 / depth.shape[-1]
    dy = 2.0 / depth.shape[-2]
    for x in range(kernel):
        for y in range(kernel):
            weight = math.exp(
                -((x - kernel // 2) ** 2 + (y - kernel // 2) ** 2) / (2.0 * sigma**2)
            )
            v_pix_i = v_pix.clone()
            v_pix_i[..., 0] += dx * (x - kernel // 2)
            v_pix_i[..., 1] += dy * (y - kernel // 2)
            d = thf.grid_sample(depth, v_pix_i, mode="nearest", align_corners=False)
            w = thf.grid_sample((depth > 0.0).float(), v_pix_i, mode="nearest", align_corners=False)

            v_depth_2 = d / (w + 1e-8)
            valid.append(weight * (w > 1e-4).float())

            diff = valid[-1] * (v_depth_1 - v_depth_2).clamp(min=0)
            in_shadow.append(diff)

    valid_all = th.stack(valid, 0).sum(0)
    in_shadow = th.stack(in_shadow, 0).sum(0) / (valid_all + 1e-6)
    
    if nml is not None:
        in_shadow = bcull_mask * in_shadow + (1.0 - bcull_mask) * 1e3
    
    return in_shadow