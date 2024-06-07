# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging

from typing import Any, Callable, Dict, List, Optional, Tuple

import ca_code.nn.layers as la
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ca_code.models.hand_mvp import PoseEncoder
from ca_code.models.hand_mvp import AutoEncoder as BaseAE

from ca_code.utils.image import linear2srgb, scale_diff_image
from ca_code.utils.envmap import compose_envmap

from drtk import transform

from extensions.utils.utils import compute_raydirs

from torchvision.utils import make_grid


def build_cam_rot_mat(campos, objcenter=None):
    campos[(campos[:, 0].abs() + campos[:, 2].abs()) < 1e-8, 2] += 1e-2

    if objcenter is None:
        z = F.normalize(-campos, dim=1)
    else:
        z = F.normalize(objcenter - campos, dim=1)
    up = th.zeros_like(campos)
    up[:, 1] = 1

    x = F.normalize(th.cross(z, up, dim=1), dim=1)
    y = F.normalize(th.cross(z, x, dim=1), dim=1)

    mat = th.zeros(campos.shape[0], 3, 3).to(campos.device)
    mat[:, 0, 0:3] = x
    mat[:, 1, 0:3] = y
    mat[:, 2, 0:3] = z

    return mat


class AutoEncoder(BaseAE):
    def __init__(
        self,
        assets,
        image_height,
        image_width,
        cal=None,
        n_pose_dims: int = 54,
        n_embs: int = 64,
        volradius: int = 2000.0,
        primsize: Tuple[int] = (16, 16, 8),
        learn_blur: bool = True,
    ):
        super().__init__(
            assets,
            image_height,
            image_width,
            cal,
            n_pose_dims,
            n_embs,
            volradius,
            primsize,
            learn_blur,
        )
        
        self.poseencoder2 = PoseEncoder(n_pose_dims, n_embs, self.n_prim_x)

        self.relightdecoder = OLATRGBDecoder(
            self.uv_size,
            self.primsize,
            self.n_prim_x,
            self.n_prim_y,
            self.raymarcher,
            self.volradius,
        )
        
        # this avoids reinitializing primitives
        self.geomdecoder.eval()

    def forward(
        self,
        pose: th.Tensor,
        campos: th.Tensor,
        K: th.Tensor,
        Rt: th.Tensor,
        light_intensity: th.Tensor,
        light_pos: th.Tensor,
        camera_id: Optional[List[str]] = None,
        frame_id: Optional[th.Tensor] = None,
        iteration: Optional[int] = None,
        background: Optional[th.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:

        joint = self.poseencoder(pose)

        geo_preds = self.geomdecoder(pose, joint, iteration)

        # compute view_cos
        joint2 = self.poseencoder2(pose)
        dec_preds = self.relightdecoder(
            campos,
            K,
            Rt,
            geo_preds["primpos"],
            geo_preds["primrot"],
            geo_preds["primscale"],
            geo_preds["primalpha"],
            self.valid_prims.float(),
            joint2,
            light_pos,
            light_intensity,
            iteration,
        )
        primrgb = dec_preds["primrgb"]

        preds = {
            "primrgb": primrgb,
            "valid_prims": self.valid_prims,
            **geo_preds,
            **dec_preds,
        }

        # # rendering
        rgb, alpha, _ = self.render(K, Rt, preds, with_shadow=False)

        if not th.jit.is_scripting() and self.cal_enabled:
            rgb = self.cal(rgb, self.cal.name_to_idx(camera_id))

        if self.training:
            if background is not None:
                bg = background[:, :3].clone()
                rgb = rgb + (1.0 - alpha) * bg

        if "envbg" in kwargs:
            rgb = compose_envmap(rgb / 255.0, alpha, kwargs["envbg"], K, Rt)

        preds.update(
            rgb=rgb,
            alpha=alpha,
            ae=self,
        )

        if not th.jit.is_scripting() and self.learn_blur_enabled:
            preds["rgb"] = self.learn_blur(preds["rgb"], camera_id)
            preds["learn_blur_weights"] = self.learn_blur.reg(camera_id)

        return preds


class OLATRGBDecoder(nn.Module):
    def __init__(
        self,
        uv_size,
        primsize,
        n_prim_x,
        n_prim_y,
        raymarcher,
        volradius,
        n_init_channels=64,
        n_enc_dims=[64, 64, 64, 64, 64],
        shadow_img_size=1024,
    ):
        super().__init__()

        # initializing primitives
        self.chunksize = 5
        self.uv_size = uv_size
        self.primsize = primsize
        self.n_prim_x = n_prim_x
        self.n_prim_y = n_prim_y

        self.uv_multiple = 1

        in_feats = 2 * 3 + 1  # light_dir, view_dir, shadow

        self.n_enc_dims = [
            (in_feats * self.primsize[2], n_enc_dims[0]),
            (n_enc_dims[0], n_enc_dims[1]),
            (n_enc_dims[1], n_enc_dims[2]),
            (n_enc_dims[2], n_enc_dims[3]),
            (n_enc_dims[3], n_enc_dims[4]),
        ]

        self.n_dec_dims = [
            (n_enc_dims[4] + n_init_channels, n_enc_dims[3]),
            (n_enc_dims[3] * 2, n_enc_dims[2]),
            (n_enc_dims[2] * 2, n_enc_dims[1]),
            (n_enc_dims[1] * 2, n_enc_dims[0]),
            (n_enc_dims[0] * 2, self.primsize[2] * 4),
        ]

        self.sizes = [
            (self.primsize[0] * self.n_prim_x) // (2**i)
            for i in range(len(self.n_enc_dims))
        ]

        self.enc_layers = nn.ModuleList()
        for i, size in enumerate(self.sizes):
            n_in, n_out = self.n_enc_dims[i]
            self.enc_layers.append(
                nn.Sequential(
                    la.Conv2dWNUB(
                        n_in,
                        n_out,
                        kernel_size=3,
                        height=size,
                        width=size,
                        stride=1,
                        padding=1,
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )

        self.dec_layers = nn.ModuleList()
        for i in range(len(self.sizes)):
            size = self.sizes[-i - 1]
            n_in, n_out = self.n_dec_dims[i]
            self.dec_layers.append(
                nn.Sequential(
                    la.Conv2dWNUB(
                        n_in,
                        n_out,
                        kernel_size=3,
                        height=size,
                        width=size,
                        stride=1,
                        padding=1,
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )

        self.volradius = volradius
        self.raymarcher = raymarcher

        # for shadow map
        y, x = th.meshgrid(th.arange(shadow_img_size), th.arange(shadow_img_size))
        pyx = th.stack([y, x], dim=-1).float()
        self.register_buffer("pixel_coords", pyx, persistent=False)

        self.apply(lambda x: la.glorot(x, 0.2))

    def forward_rgb(
        self,
        campos: th.Tensor,
        K: th.Tensor,
        Rt: th.Tensor,
        primpos: th.Tensor,
        primrot: th.Tensor,
        primscale: th.Tensor,
        primalpha: th.Tensor,
        valid_prims: th.Tensor,
        joint_feat: th.Tensor,
        light_pos: th.Tensor,
        light_intensity: th.Tensor,
        iteration: Optional[int] = None,
    ):

        B = light_pos.shape[0]
        L = light_pos.shape[1]
        with th.no_grad():
            # Compute deep shadow maps
            postex = primpos[:, valid_prims.bool()]
            postex = postex[:, None].expand(-1, L, -1, -1).reshape(B * L, -1, 3)
            template = (
                th.ones(
                    (
                        B * L,
                        self.n_prim_x * self.n_prim_y,
                        self.primsize[2],
                        self.primsize[1],
                        self.primsize[0],
                        3,
                    ),
                    device=light_pos.device,
                )
                * 255.0
            )
            primalpha = primalpha.reshape(
                B,
                self.primsize[2],
                1,
                self.n_prim_y,
                self.primsize[1],
                self.n_prim_x,
                self.primsize[0],
            )
            primalpha = (
                primalpha.permute(0, 3, 5, 1, 4, 6, 2)[:, None]
                .expand(-1, L, -1, -1, -1, -1, -1, -1)
                .reshape(
                    B * L,
                    self.n_prim_x * self.n_prim_y,
                    self.primsize[2],
                    self.primsize[1],
                    self.primsize[0],
                    1,
                )
            )

            primalpha = valid_prims[None, :, None, None, None, None] * primalpha

            template = th.cat([template, primalpha], dim=-1)

            posc = (postex.max(1)[0] + postex.min(1)[0]) / 2
            lpos = light_pos.view(-1, 3)
            lrot = build_cam_rot_mat(lpos, posc)

            pixelcoords = (B * L) * [self.pixel_coords[None]]
            pixelcoords = th.cat(pixelcoords, dim=0)

            focal = th.diag(th.ones(2, device=lpos.device) * 1000.0)[None].expand(B * L, -1, -1)
            princpt = th.ones((B * L), 2, device=lpos.device)
            princpt[:, 0] *= self.pixel_coords.shape[1] / 2
            princpt[:, 1] *= self.pixel_coords.shape[0] / 2

            v_pix = transform(postex, campos=lpos, camrot=lrot, focal=focal, princpt=princpt)

            pixelcoords_shape = self.pixel_coords.shape[:2][::-1]
            pixelcoords_shape = th.tensor(pixelcoords_shape, device=pixelcoords.device)
            pix_ratio = (v_pix[..., :2] - princpt[:, None]) / (
                0.45 * pixelcoords_shape[None, None]
            )
            focal = th.diagonal(focal, 0, 1, 2) / abs(pix_ratio).max(1)[0]
            viewrot = lrot
            viewpos = lpos
            raypos, raydir, tminmax = compute_raydirs(
                viewpos, viewrot, focal, princpt, pixelcoords, self.volradius
            )

            primpos_ = primpos[:, None].expand(-1, L, -1, -1).reshape(B * L, -1, 3)

            primrot_ = (
                primrot[:, None].expand(-1, L, -1, -1, -1).reshape(B * L, -1, 3, 3)
            )

            primscale_ = primscale[:, None].expand(-1, L, -1, -1).reshape(B * L, -1, 3)

            inputs = {
                "primrgba": template,
                "primpos": primpos_.contiguous(),
                "primrot": primrot_.contiguous(),
                "primscale": primscale_.contiguous(),
            }

            _, _, shadow_img, shadow = self.raymarcher(
                raypos, raydir, tminmax, inputs, with_shadow=True
            )
            shadow = shadow.reshape(
                B,
                L,
                self.n_prim_y,
                self.n_prim_x,
                1,
                self.primsize[2],
                self.primsize[1],
                self.primsize[0],
            )  # B x L x H x W x C x Z x Y x X

            # B x L x Z x C x H x Y x W x X
            shadow_feat = shadow.permute(0, 1, 5, 4, 2, 6, 3, 7)
            shadow_feat = shadow_feat.reshape(
                B * L,
                -1,
                self.uv_size,
                self.uv_size,
            )  # B * L x Z * C x H * Y x W * X

            intervx = th.linspace(-1.0, 1.0, self.primsize[0]).cuda()
            intervy = th.linspace(-1.0, 1.0, self.primsize[1]).cuda()
            intervz = th.linspace(-1.0, 1.0, self.primsize[2]).cuda()
            prims = th.stack(th.meshgrid([intervz, intervy, intervx])).transpose(
                1, 3
            )  # 3 x Z x Y x X

            prims = prims.reshape(3, -1)
            prims = prims[None, None] / primscale[..., None]
            prims = primrot @ prims
            prims = self.volradius * (
                primpos[..., None] + prims
            )  # (B, N, 3, 16, 16, 16)
            prims = prims.view(
                B,
                self.n_prim_y,
                self.n_prim_x,
                3,
                self.primsize[2],
                self.primsize[1],
                self.primsize[0],
            )  # B x H x W x C x Z x Y x X
            prims = prims.permute(0, 4, 3, 1, 5, 2, 6)
            # B x Z x C x H x Y x W x X
            lightvechq = (
                light_pos[:, :, None, :, None, None, None, None] - prims[:, None, :]
            )  # B x L x Z x C x H x Y x W x X

            light_intensity = light_intensity[:, :, None, :, None, None]

            lightvechq = F.normalize(lightvechq, dim=3)
            viewdirhq = (
                campos[:, None, :, None, None, None, None] - prims
            )  # B x Z x C x H x Y x W x X
            viewdirhq = F.normalize(viewdirhq, dim=2)

            primrot = primrot.reshape(B, self.n_prim_y, self.n_prim_x, 3, 3)
            lightvechq = th.einsum("bhwef,blzehywx->blzfhywx", primrot, lightvechq)
            viewdirhq = th.einsum("bhwef,bzehywx->bzfhywx", primrot, viewdirhq)

            valid_prims_ = valid_prims.reshape(self.n_prim_y, self.n_prim_x)
            lightvechq = (
                valid_prims_[None, None, None, None, :, None, :, None] * lightvechq
            )
            viewdirhq = valid_prims_[None, None, None, :, None, :, None] * viewdirhq

            lightvechq = lightvechq.reshape(
                B * L, -1, self.uv_size, self.uv_size
            )  # B * L x Z * C x H * Y x W * X
            viewdirhq = viewdirhq.reshape(
                B, -1, self.uv_size, self.uv_size
            )  # B x Z * C x H * Y x W * X
            viewdirhq = viewdirhq[:, None].expand(-1, L, -1, -1, -1)
            viewdirhq = viewdirhq.reshape(-1, *viewdirhq.shape[2:])

        if self.training:
            lightvechq.requires_grad = True
            viewdirhq.requires_grad = True
            shadow_feat.requires_grad = True

        x = th.cat([lightvechq, viewdirhq, 1.0 - shadow_feat], dim=1)

        joint_feat = joint_feat[:, None].expand(-1, L, -1, -1, -1)
        joint_feat = joint_feat.reshape(B * L, *joint_feat.shape[-3:])

        enc_acts = []
        # unet enc
        for i, layer in enumerate(self.enc_layers):
            x = layer(x)
            enc_acts.append(x)
            if i < len(self.sizes) - 1:
                x = F.interpolate(
                    x,
                    scale_factor=0.5,
                    mode="bilinear",
                    recompute_scale_factor=True,
                    align_corners=True,
                )

        for i, layer in enumerate(self.dec_layers):
            if i == 0:
                x = th.cat([x, joint_feat], dim=1)
            elif i > 0:
                x_prev = enc_acts[-i - 1]
                x = F.interpolate(
                    x, size=x_prev.shape[2:4], mode="bilinear", align_corners=True
                )
                x = th.cat([x, x_prev], dim=1)
            x = layer(x)
        tex = x.view(B, L, self.primsize[2], 4, *x.shape[2:])

        if self.training and iteration is not None and iteration < 1000:
            shadowolat = shadow_feat.reshape(
                B, L, self.primsize[2], 1, self.uv_size, self.uv_size
            )
        else:
            shadowolat = th.sigmoid(tex[:, :, :, :1])
        texolat = 25.0 * tex[:, :, :, 1:] + 100.0

        rgb = (shadowolat * F.relu(texolat) * light_intensity).sum(1)
        rgb = rgb.view(B, self.primsize[2], 3, self.uv_size, self.uv_size)
        
        # for debug
        primshadow = (
            shadow_feat[:, :, None]
            .expand(-1, -1, 3, -1, -1)
            .reshape(B, L, self.primsize[2], 3, self.uv_size, self.uv_size)
            .sum(1)
            / L
        )

        output = {"primrgb": rgb, "primshadow": primshadow}
        if self.training:
            output["texolat"] = texolat
        return output


    def forward(
        self,
        campos: th.Tensor,
        K: th.Tensor,
        Rt: th.Tensor,
        primpos: th.Tensor,
        primrot: th.Tensor,
        primscale: th.Tensor,
        primalpha: th.Tensor,
        valid_prims: th.Tensor,
        joint_feat: th.Tensor,
        light_pos: th.Tensor,
        light_intensity: th.Tensor,
        iteration: Optional[int] = None,
    ):
        L = light_pos.shape[1]
        chunknum = (L - 1) // self.chunksize + 1
        rgb, shadow, texolat = None, None, None
        for i in range(chunknum):
            start = i * self.chunksize
            end = (i + 1) * self.chunksize
            tmp_light_pos = light_pos[:, start:end].contiguous()
            tmp_light_intensity = light_intensity[:, start:end].contiguous()

            output = self.forward_rgb(
                campos,
                K,
                Rt,
                primpos,
                primrot,
                primscale,
                primalpha,
                valid_prims,
                joint_feat,
                tmp_light_pos,
                tmp_light_intensity,
                iteration
            )

            tmp_rgb = output["primrgb"]
            tmp_shadow = output["primshadow"]
            if "texolat" in output:
                texolat = output["texolat"]

            if rgb is None:
                rgb = tmp_rgb.clone()
            else:
                rgb = rgb + tmp_rgb

            if shadow is None:
                shadow = tmp_shadow.clone()
            else:
                shadow = shadow + tmp_shadow

        output = {"primrgb": rgb, "primshadow": shadow}
        if self.training:
            output["texolat"] = texolat
 
        return output


class HandMVPSummary(Callable):

    def __call__(
        self, preds: Dict[str, Any], batch: Dict[str, Any]
    ) -> Dict[str, th.Tensor]:

        diag = {}

        bs = preds["primrgba"].shape[0]
        ps_x = preds["primrgba"].shape[-2]
        ps_y = preds["primrgba"].shape[-3]
        ps_z = preds["primrgba"].shape[-4]
        primrgba = preds["primrgba"].view(
            bs, 1024 // ps_y, 1024 // ps_x, ps_z, ps_y, ps_x, -1
        )
        primrgba = primrgba.permute(0, 3, 6, 1, 4, 2, 5).mean(1)
        primrgba = primrgba.view(bs, -1, 1024, 1024)

        diag["rgb_slab"] = linear2srgb(primrgba[:, :3] / 255.0).clamp(0, 1)
        diag["alpha_slab"] = primrgba[:, 3:].clamp(0, 1)

        render = preds["rgb"] / 255.0
        render = linear2srgb(render).clamp(0, 1)

        if "image" in batch:
            gt = batch["image"] / 255.0
            diff = ((preds["rgb"] - batch["image"]) / 255).clamp(-1.0, 1.0)
            if "segmentation_fgbg" in batch:
                diff *= batch["segmentation_fgbg"]
            diff = scale_diff_image(diff)

            diag["gt"] = linear2srgb(gt).clamp(0, 1)
            diag["diff"] = diff.clamp(0, 1)
            
        if "primshadow" in preds:
            preds["primrgb"] = 255.0 * preds["primshadow"]
            shadow = preds["ae"].render(batch["K"], batch["Rt"], preds, with_shadow=False)[0]
            diag["shadow"] = (shadow / 255.0).clamp(0, 1)

        if "segmentation_fgbg" in batch:
            diag["gt_alpha"] = batch["segmentation_fgbg"]

        diag["render"] = render
        diag["alpha"] = preds["alpha"].clamp(0, 1).expand(-1, 3, -1, -1)

        if "image_weight" in preds:
            diag["weight"] = preds["image_weight"]

        for k, v in diag.items():
            diag[k] = make_grid(255.0 * v, nrow=16).clip(0, 255).to(th.uint8)

        return diag
