# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Dict
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# NOTE: using new drtk API
from drtk import rasterize, render, transform, interpolate, edge_grad_estimator

class RenderLayer(nn.Module):

    def __init__(self, h, w, vi, vt, vti, flip_uvs=False):
        super().__init__()
        self.h = h
        self.w = w
        self.register_buffer("vi", vi, persistent=False)
        self.register_buffer("vt", vt, persistent=False)
        self.register_buffer("vti", vti, persistent=False)
        self.flip_uvs = flip_uvs
        if flip_uvs:
            self.vt[:, 1] = 1 - self.vt[:, 1]
            
        image_size = th.as_tensor([h, w], dtype=th.int32)
        self.register_buffer("image_size", image_size)

    def forward(
        self,
        verts: th.Tensor,
        tex: th.Tensor,
        K: th.Tensor,
        Rt: th.Tensor,
        background: th.Tensor = None,
        output_filters: List[str] = None,
        edge_grad: bool = True,
    ):

        assert output_filters is None
        assert background is None

        v_pix = transform(verts, K=K, Rt=Rt)

        index_img = rasterize(v_pix, self.vi, self.h, self.w)
        depth_img, bary_img = render(v_pix, self.vi, index_img)

        vt_img = interpolate(
            (self.vt * 2.0 - 1.0)[None].expand(verts.shape[0], -1, -1),
            self.vti,
            index_img,
            bary_img,
        )

        mask = (index_img != -1)[:, None].float()

        img = (
            F.grid_sample(
                tex, vt_img.permute(0, 2, 3, 1), mode="bilinear", align_corners=False
            )
            * mask
        )

        if edge_grad:
            img = edge_grad_estimator(
                v_pix=v_pix,
                vi=self.vi,
                bary_img=bary_img,
                img=img,
                index_img=index_img,
            )
        
        return {
            "render": img,
            "depth_img": depth_img,
            "v_pix": v_pix,
            "vt_img": vt_img,
            "index_img": index_img,
            "bary_img": bary_img,
            "mask": mask,
        }
