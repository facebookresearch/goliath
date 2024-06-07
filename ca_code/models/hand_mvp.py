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
from ca_code.nn.blocks import (
    ConvBlock,
    tile2d,
)

from ca_code.nn.color_cal import CalV5

from ca_code.nn.dof_cal import LearnableBlur

from ca_code.utils.geom import (
    compute_tbn,
    compute_view_cos,
    GeometryModule,
    make_postex,
    values_to_uv,
)

from ca_code.utils.image import (
    linear2srgb,
    scale_diff_image,
)

from ca_code.utils.lbs import LBSModule

# from ca_code.utils.render_drtk import RenderLayer
from ca_code.utils.render_pytorch3d import RenderLayer
from ca_code.utils.render_raymarcher import Raymarcher

from extensions.utils.utils import compute_raydirs

from torchvision.utils import make_grid


logger = logging.getLogger(__name__)


def init_primitives(slab_size, n_prims, geo_fn):
    stride = slab_size // int(n_prims**0.5)
    device = geo_fn.vt.device
    # TODO: check if this handles seams properly
    _, face_index_imp, bary_index_imp = geo_fn.render_index_images(
        slab_size, impaint=True
    )

    bary_index_imp = th.as_tensor(bary_index_imp, device=device)

    prim_bary_img = bary_index_imp[stride // 2 :: stride, stride // 2 :: stride]
    prim_vidx_img = geo_fn.vi[
        face_index_imp[stride // 2 :: stride, stride // 2 :: stride]
    ]
    prim_vtidx_img = geo_fn.vti[
        face_index_imp[stride // 2 :: stride, stride // 2 :: stride]
    ]

    return prim_vidx_img, prim_vtidx_img, prim_bary_img


class AutoEncoder(nn.Module):
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
        super().__init__()

        self.uv_size = 1024
        self.primsize = primsize
        self.n_prim_x = self.uv_size // self.primsize[0]
        self.n_prim_y = self.uv_size // self.primsize[1]
        self.n_prims = self.n_prim_x * self.n_prim_y
        self.height = image_height
        self.width = image_width
        self.volradius = volradius
        self.lbs_fn = LBSModule(
            assets.lbs_model_json,
            assets.lbs_config_dict,
            assets.template_mesh_unscaled[None],
            assets.skeleton_scales,
            global_scaling=[10.0, 10.0, 10.0],  # meter to cm
        )

        self.geo_fn = GeometryModule(
            assets.topology.vi,
            assets.topology.vt,
            assets.topology.vti,
            assets.topology.v2uv,
            uv_size=1024,
            flip_uv=False,
            impaint=False,
        )

        self.poseencoder = PoseEncoder(n_pose_dims, n_embs, self.n_prim_x)

        self.geomdecoder = GeomDecoder(
            n_embs,
            primsize[2],
            self.uv_size,
            self.n_prims,
            self.lbs_fn,
            self.geo_fn,
            primposstart=1000,
        )

        self.rgbdecoder = RGBSlabDecoder(
            n_embs + 2, primsize[2], self.uv_size, self.geo_fn
        )

        self.raymarcher = Raymarcher(volradius=self.volradius, dt=1.0)

        self.renderer = RenderLayer(
            h=image_height,
            w=image_width,
            vt=self.geo_fn.vt,
            vi=self.geo_fn.vi,
            vti=self.geo_fn.vti,
            flip_uvs=False,
        )
        self.learn_blur_enabled = False
        if learn_blur:
            self.learn_blur_enabled = True
            self.learn_blur = LearnableBlur(assets.camera_ids)

        # training-only stuff
        self.cal_enabled = False
        if cal is not None:
            self.cal_enabled = True
            self.cal = CalV5(**cal, cameras=assets.camera_ids)

        y, x = th.meshgrid(th.arange(self.height), th.arange(self.width), indexing="ij")
        pxy = th.stack([x, y], dim=-1).float()
        self.register_buffer("pixel_coords", pxy, persistent=False)

        valid_mask = F.interpolate(
            self.geo_fn.valid_mask.float()[None].permute(0, 3, 1, 2),
            (self.n_prim_x, self.n_prim_y),
            mode="area",
        )
        self.register_buffer(
            "valid_prims", (valid_mask != 0).reshape(-1), persistent=False
        )

    def render(
        self,
        K: th.Tensor,
        Rt: th.Tensor,
        preds: Dict[str, Any],
        with_shadow: bool = False,
    ):
        B = K.shape[0]

        # finally reshape texture slab to primitive set
        primrgba = th.cat([preds["primrgb"], preds["primalpha"]], 2).view(
            B,
            self.primsize[2],
            4,
            self.n_prim_y,
            self.primsize[1],
            self.n_prim_x,
            self.primsize[0],
        )
        primrgba = primrgba.permute(0, 3, 5, 1, 4, 6, 2)
        primrgba = primrgba.reshape(
            B, self.n_prims, self.primsize[2], self.primsize[1], self.primsize[0], 4
        )
        preds["primrgba"] = primrgba

        focal = K[:, :2, :2].contiguous()
        focal = th.diagonal(focal, dim1=1, dim2=2).contiguous()
        princpt = K[:, :2, 2].contiguous()

        camrot = Rt[:, :3, :3].contiguous()
        campos = -(camrot.transpose(-2, -1) @ Rt[:, :3, 3:4])[..., 0]

        pixelcoords = self.pixel_coords[None].expand(B, -1, -1, -1).contiguous()

        # Compute ray directions for ray marching
        raypos, raydir, tminmax = compute_raydirs(
            campos, camrot, focal, princpt, pixelcoords, self.raymarcher.volume_radius
        )

        rayrgb, rayalpha, _, shadow = self.raymarcher(
            raypos, raydir, tminmax, preds, with_shadow=with_shadow
        )

        return rayrgb, rayalpha, shadow

    def forward(
        self,
        pose: th.Tensor,
        campos: th.Tensor,
        ambient_occlusion: Optional[th.Tensor] = None,
        K: Optional[th.Tensor] = None,
        Rt: Optional[th.Tensor] = None,
        camera_id: Optional[List[str]] = None,
        frame_id: Optional[th.Tensor] = None,
        embs: Optional[th.Tensor] = None,
        encode: bool = True,
        iteration: Optional[int] = None,
        background: Optional[th.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        B = pose.shape[0]

        joint = self.poseencoder(pose)

        geo_preds = self.geomdecoder(pose, joint, iteration)
        geom_lbs = geo_preds["geom_lbs"]
        primalpha = geo_preds["primalpha"]

        # compute view_cos
        view_cos = compute_view_cos(geom_lbs, self.geo_fn.vi.long(), campos)
        view_cos_uv = values_to_uv(
            view_cos[..., None],
            self.geomdecoder.prim_vidx_img,
            self.geomdecoder.prim_bary_img,
        )

        primrgb = self.rgbdecoder(view_cos_uv, joint, ambient_occlusion)

        preds = {
            "primrgb": primrgb,
            "valid_prims": self.valid_prims,
            **geo_preds,
        }

        # # rendering
        rgb, alpha, _ = self.render(K, Rt, preds, with_shadow=False)

        if not th.jit.is_scripting() and self.cal_enabled:
            rgb = self.cal(rgb, self.cal.name_to_idx(camera_id))

        if self.training:
            if background is not None:
                bg = background[:, :3].clone()
                rgb = rgb + (1.0 - alpha) * bg

        preds.update(
            rgb=rgb,
            alpha=alpha,
        )

        if not th.jit.is_scripting() and self.learn_blur_enabled:
            preds["rgb"] = self.learn_blur(preds["rgb"], camera_id)
            preds["learn_blur_weights"] = self.learn_blur.reg(camera_id)

        return preds


class PoseEncoder(nn.Module):
    def __init__(self, n_pose_dims, n_embs, in_size):
        super().__init__()

        self.in_size = in_size

        self.local_pose_conv_block = ConvBlock(
            n_pose_dims,
            16,
            in_size,
            kernel_size=1,
            padding=0,
        )

        self.joint_conv_block = ConvBlock(
            16,
            n_embs,
            in_size,
        )

    def forward(self, pose):
        local_pose = pose[:, 6:]

        pose_tile = tile2d(local_pose, self.in_size)
        pose_conv = self.local_pose_conv_block(pose_tile)
        return self.joint_conv_block(pose_conv)


class TransDecoder(nn.Module):
    def __init__(self, inch):
        super().__init__()

        self.dec0 = nn.Sequential(
            la.Conv2dWNUB(inch, 64, 64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            la.Conv2dWNUB(64, 128, 64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            la.Conv2dWNUB(128, 64, 64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            la.Conv2dWNUB(64, 64, 64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            la.Conv2dWNUB(64, 9, 64, 64, 3, 1, 1),
        )
        self.apply(lambda x: la.glorot(x, 0.2))
        la.glorot(self.dec0[-1], 1.0)

    def forward(self, local_encoding):
        out = self.dec0(local_encoding)
        out = out.view(local_encoding.size(0), 9, -1).permute(0, 2, 1).contiguous()
        primposdelta = out[:, :, 0:3] * 1.0e-4
        primrvecdelta = out[:, :, 3:6] * 0.01
        primscaledelta = th.exp(0.01 * out[:, :, 6:9])
        return primposdelta, primrvecdelta, primscaledelta


class DeconvContentDecoder(nn.Module):
    def __init__(self, primsize_z: Tuple[int], inch, outch):
        super().__init__()

        self.primsize_z = primsize_z
        self.outch = outch

        self.texbranch = nn.Sequential(
            la.ConvTranspose2dWNUB(inch, 32, 128, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            la.ConvTranspose2dWNUB(32, 32, 256, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),
            la.ConvTranspose2dWNUB(32, 16, 512, 512, 4, 2, 1),
            nn.LeakyReLU(0.2),
            la.ConvTranspose2dWNUB(
                16, self.primsize_z * self.outch, 1024, 1024, 4, 2, 1
            ),
        )

        self.apply(lambda x: la.glorot(x, 0.2))
        la.glorot(self.texbranch[-1], 1.0)

    def forward(self, local_enc: th.Tensor):
        tex = self.texbranch(local_enc)
        return tex


class GeomDecoder(nn.Module):
    def __init__(
        self,
        inch,
        primsize_z,
        uv_size,
        n_prims,
        lbs_fn,
        geo_fn,
        primposstart,
        prim_scale=512,
    ):
        super().__init__()

        self.lbs_fn = lbs_fn
        self.geo_fn = geo_fn
        self.primposstart = primposstart
        self.uv_size = uv_size
        self.n_prims = n_prims
        self.primsize_z = primsize_z
        self.prim_scale = prim_scale

        prim_vidx_img, prim_vtidx_img, prim_bary_img = init_primitives(
            uv_size,
            n_prims,
            self.geo_fn,
        )

        self.register_buffer("prim_vidx_img", prim_vidx_img, persistent=False)
        self.register_buffer("prim_vtidx_img", prim_vtidx_img, persistent=False)
        self.register_buffer("prim_bary_img", prim_bary_img, persistent=False)

        self.transdecoder = TransDecoder(inch)
        self.alphadecoder = DeconvContentDecoder(primsize_z, inch, 1)

    def forward(self, pose, joint, iteration=-1):
        B = pose.shape[0]

        with th.no_grad():
            geom_lbs = self.lbs_fn.pose(
                th.zeros_like(self.lbs_fn.lbs_template_verts), pose
            )
            primposbase = (
                make_postex(geom_lbs, self.prim_vidx_img, self.prim_bary_img)
                .permute(0, 2, 3, 1)
                .reshape(B, -1, 3)
            )
            # TODO: check the TBN computation?
            tbn = compute_tbn(
                geom_lbs, self.geo_fn.vt, self.prim_vidx_img, self.prim_vtidx_img
            )

            primrotbase = (
                th.stack(tbn, dim=-2)
                .reshape(B, self.n_prims, 3, 3)
                .permute(0, 1, 3, 2)
                .contiguous()
            )

        delta_pos, delta_rvec, delta_scale = self.transdecoder(joint)

        if self.training and iteration < self.primposstart:
            delta_pos = delta_pos * 0.0
            delta_rvec = delta_rvec * 0.0
            delta_scale = delta_scale * 0.0 + 1.0

        primpos = primposbase + th.bmm(
            primrotbase.view(-1, 3, 3), delta_pos.view(-1, 3, 1)
        ).reshape(B, -1, 3)

        primscale = self.prim_scale * delta_scale
        primrotdelta = axisangle_to_matrix(delta_rvec)
        primrot = th.bmm(primrotbase.view(-1, 3, 3), primrotdelta.view(-1, 3, 3)).view(
            B, -1, 3, 3
        )

        alpha = self.alphadecoder(joint).view(
            B,
            self.primsize_z,
            1,
            self.uv_size,
            self.uv_size,
        )
        alpha = F.relu(alpha)

        preds = {
            "primalpha": alpha,
            "primpos": primpos,
            "primscale": primscale,
            "primrot": primrot,
            "geom_lbs": geom_lbs,
        }

        return preds


class RGBSlabDecoder(nn.Module):
    def __init__(self, inch, primsize_z, uv_size, geo_fn):
        super().__init__()

        self.geo_fn = geo_fn
        self.primsize_z = primsize_z
        self.uv_size = uv_size

        self.texdecoder = DeconvContentDecoder(primsize_z, inch, 3)

    def forward(self, view_cos_uv, joint, ambient_occlusion):
        B = joint.shape[0]

        # This is fixed
        ao_ds = F.interpolate(ambient_occlusion, (64, 64), mode="bilinear")

        # decoding RGB texture
        view_cond = th.cat((joint, view_cos_uv, ao_ds), 1)
        rgb = (self.texdecoder(view_cond)).view(
            B,
            self.primsize_z,
            3,
            self.uv_size,
            self.uv_size,
        )
        rgb = F.relu(25.0 * rgb + 100.0)

        return rgb


def axisangle_to_matrix(rvec):
    theta = th.sqrt(1e-5 + th.sum(rvec**2, dim=-1))
    rvec = rvec / theta[..., None]
    costh = th.cos(theta)
    sinth = th.sin(theta)
    return th.stack(
        (
            th.stack(
                (
                    rvec[..., 0] ** 2 + (1.0 - rvec[..., 0] ** 2) * costh,
                    rvec[..., 0] * rvec[..., 1] * (1.0 - costh) - rvec[..., 2] * sinth,
                    rvec[..., 0] * rvec[..., 2] * (1.0 - costh) + rvec[..., 1] * sinth,
                ),
                dim=-1,
            ),
            th.stack(
                (
                    rvec[..., 0] * rvec[..., 1] * (1.0 - costh) + rvec[..., 2] * sinth,
                    rvec[..., 1] ** 2 + (1.0 - rvec[..., 1] ** 2) * costh,
                    rvec[..., 1] * rvec[..., 2] * (1.0 - costh) - rvec[..., 0] * sinth,
                ),
                dim=-1,
            ),
            th.stack(
                (
                    rvec[..., 0] * rvec[..., 2] * (1.0 - costh) - rvec[..., 1] * sinth,
                    rvec[..., 1] * rvec[..., 2] * (1.0 - costh) + rvec[..., 0] * sinth,
                    rvec[..., 2] ** 2 + (1.0 - rvec[..., 2] ** 2) * costh,
                ),
                dim=-1,
            ),
        ),
        dim=-2,
    )


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

        if "segmentation_fgbg" in batch:
            diag["gt_alpha"] = batch["segmentation_fgbg"]

        diag["render"] = render
        diag["alpha"] = preds["alpha"].clamp(0, 1).expand(-1, 3, -1, -1)

        if "image_weight" in preds:
            diag["weight"] = preds["image_weight"]

        for k, v in diag.items():
            diag[k] = make_grid(255.0 * v, nrow=16).clip(0, 255).to(th.uint8)

        return diag
