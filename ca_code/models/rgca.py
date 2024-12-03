# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import ca_code.nn.layers as la

import ca_code.utils.sh as sh
import cv2
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ca_code.nn.color_cal import CalV5

from ca_code.nn.dof_cal import LearnableBlur
from ca_code.nn.layers import make_conv_trans, make_linear
from ca_code.utils.envmap import compose_envmap, dir2uv

from ca_code.utils.geom import (
    depth2normals,
    GeometryModule,
)

from ca_code.utils.image import (
    linear2srgb,
    make_image_grid_batched,
    scale_diff_image,
)

from ca_code.utils.mipmap_sampler import mipmap_grid_sample

from ca_code.utils.render_gsplat import render as render_gs

from extensions.sgutils.sgutils import evaluate_gaussian

from torchvision.utils import make_grid

logger = logging.getLogger(__name__)

primscale_range: Tuple[float, float] = (0.1, 20.0)


class AutoEncoder(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        assets,
        image_height,
        image_width,
        cal=None,
        n_embs: int = 256,
        n_diff_sh: int = 8,
        learn_blur: bool = True,
        bg_weight: float = 1.0,
    ):
        super().__init__()

        self.height = image_height
        self.width = image_width
        self.n_diff_sh = n_diff_sh
        self.bg_weight = bg_weight

        self.geo_fn = GeometryModule(
            assets.topology.vi,
            assets.topology.vt,
            assets.topology.vti,
            None,
            uv_size=1024,
            flip_uv=True,
            impaint=False,
        )

        self.encoder = Encoder(
            n_embs=n_embs,
            n_verts_in=assets.topology.v.shape[0],
            **encoder,
        )

        self.geomdecoder = GeomDecoder(
            n_embs=n_embs,
            verts_std=math.sqrt(assets.verts_var),
            verts_mean=th.from_numpy(assets.verts_mean),
        )

        self.decoder = PrimDecoder(
            n_embs=n_embs,
            geo_fn=self.geo_fn,
            color_mean=assets.color_mean,
            n_diff_sh=self.n_diff_sh,
            **decoder,
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

    def render(self, K: th.Tensor, Rt: th.Tensor, preds: Dict[str, Any]):
        B = K.shape[0]

        rgbs: List[th.Tensor] = []
        Ts: List[th.Tensor] = []
        depths: List[th.Tensor] = []

        for b in range(B):
            render_output = render_gs(
                cam_img_w=self.width,
                cam_img_h=self.height,
                fx=K[b, 0, 0].item(),
                fy=K[b, 1, 1].item(),
                cx=K[b, 0, 2].item(),
                cy=K[b, 1, 2].item(),
                Rt=Rt[b],
                primpos=preds["primpos"][b],
                primqvec=preds["primqvec"][b],
                primscale=preds["primscale"][b],
                opacity=preds["opacity"][b],
                colors=preds["color"][b],
                return_depth=True,
            )

            rgbs.append(render_output["render"])
            Ts.append(render_output["final_T"].detach())
            depths.append(render_output["depth"])

        rgb = th.stack(rgbs)
        depth = th.stack(depths)

        rgb = rgb
        alpha = 1.0 - th.stack(Ts)
        
        depth = depth / alpha.clamp(0.05, 1.0)

        del rgbs
        del Ts
        del depths

        return rgb, alpha, depth

    def forward(
        self,
        head_pose: th.Tensor,
        campos: th.Tensor,
        registration_vertices: th.Tensor,
        color: th.Tensor,
        light_intensity: th.Tensor,
        light_pos: th.Tensor,
        n_lights: th.Tensor,
        K: th.Tensor,
        Rt: th.Tensor,
        background: Optional[th.Tensor] = None,
        is_fully_lit_frame: Optional[th.Tensor] = None,
        camera_id: Optional[List[str]] = None,
        frame_id: Optional[th.Tensor] = None,
        iteration: Optional[int] = None,
        preconv_envmap: Optional[th.Tensor] = None,
        lightrot: Optional[th.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        B = head_pose.shape[0]

        light_intensity = light_intensity.expand(-1, -1, 3)

        # convert everything into head relative coordinates
        head_pose_4x4 = th.cat([head_pose, th.zeros_like(head_pose[:, :1, :])], dim=1)
        head_pose_4x4[:, 3, 3] = 1.0
        headrel_Rt = Rt @ head_pose_4x4
        headrel_campos = (
            (campos - head_pose[:, :3, 3])[:, None] @ head_pose[:, :3, :3]
        )[:, 0]
        headrel_light_pos = (light_pos - head_pose[:, None, :3, 3]) @ head_pose[
            :, :3, :3
        ]
        headrel_light_dir = F.normalize(headrel_light_pos, p=2, dim=-1)
        sh_coeffs = sh.dir2sh_torch(self.n_diff_sh, headrel_light_dir)
        headrel_light_sh = (sh_coeffs[:, :, None] * light_intensity[..., None]).sum(
            dim=1
        )
        if lightrot is not None:
            lightrot = lightrot @ head_pose[:, :3, :3]
        # encoding
        enc_preds = self.encoder(registration_vertices, color)
        embs = enc_preds["embs"]

        # decoding
        geom_preds = self.geomdecoder(embs)
        geom = geom_preds["face_geom"]
        dec_preds = self.decoder(
            embs,
            geom,
            headrel_campos,
            light_intensity,
            headrel_light_pos,
            headrel_light_sh,
            n_lights,
            preconv_envmap,
            lightrot,
        )

        preds = {
            "geom": geom,
            "headrel_light_sh": headrel_light_sh,
            **enc_preds,
            **dec_preds,
        }

        # rendering
        rgb, alpha, depth = self.render(K, headrel_Rt, preds)

        if not th.jit.is_scripting() and self.cal_enabled:
            rgb = self.cal(rgb, self.cal.name_to_idx(camera_id))

        if self.training:
            if background is not None:
                bg = background[:, :3].clone()
                bg[th.logical_not(is_fully_lit_frame)] *= 0.0
                rgb = rgb + (1.0 - alpha) * bg

        if preconv_envmap is not None and "envbg" in kwargs:
            rgb = compose_envmap(rgb, alpha, kwargs["envbg"], K, Rt)

            rgbs = [rgb]

            preds["color"] = preds["diff_color"].clamp(min=0.0)
            rgb, alpha, depth = self.render(K, headrel_Rt, preds)
            rgbs.append(rgb)

            preds["color"] = preds["spec_color"].clamp(min=0.0)
            rgb, alpha, depth = self.render(K, headrel_Rt, preds)
            rgbs.append(rgb)

            rgb = th.cat(rgbs, -1)

        preds.update(rgb=rgb, alpha=alpha, depth=depth)

        if not th.jit.is_scripting() and self.learn_blur_enabled:
            preds["rgb"] = self.learn_blur(preds["rgb"], camera_id)
            preds["learn_blur_weights"] = self.learn_blur.reg(camera_id)

        return preds


class Encoder(nn.Module):
    """A joint encoder for tex and geometry."""

    def __init__(
        self,
        n_embs: int,
        n_verts_in: int,
        noise_std: float = 1.0,
        mean_scale: float = 0.1,
        logvar_scale: float = 0.01,
    ):
        """Fixed-width conv encoder."""
        super().__init__()

        self.noise_std = noise_std
        self.n_embs = n_embs
        self.mean_scale = mean_scale
        self.logvar_scale = logvar_scale

        self.n_verts_in = n_verts_in

        self.geommod = th.nn.Sequential(
            la.LinearWN(self.n_verts_in * 3, 256), th.nn.LeakyReLU(0.2, inplace=True)
        )

        self.texmod = th.nn.Sequential(
            la.Conv2dWNUB(3, 32, 512, 512, 4, 2, 1),
            th.nn.LeakyReLU(0.2, inplace=True),
            la.Conv2dWNUB(32, 32, 256, 256, 4, 2, 1),
            th.nn.LeakyReLU(0.2, inplace=True),
            la.Conv2dWNUB(32, 64, 128, 128, 4, 2, 1),
            th.nn.LeakyReLU(0.2, inplace=True),
            la.Conv2dWNUB(64, 64, 64, 64, 4, 2, 1),
            th.nn.LeakyReLU(0.2, inplace=True),
            la.Conv2dWNUB(64, 128, 32, 32, 4, 2, 1),
            th.nn.LeakyReLU(0.2, inplace=True),
            la.Conv2dWNUB(128, 128, 16, 16, 4, 2, 1),
            th.nn.LeakyReLU(0.2, inplace=True),
            la.Conv2dWNUB(128, 256, 8, 8, 4, 2, 1),
            th.nn.LeakyReLU(0.2, inplace=True),
            la.Conv2dWNUB(256, 256, 4, 4, 4, 2, 1),
            th.nn.LeakyReLU(0.2, inplace=True),
        )
        self.jointmod = th.nn.Sequential(
            la.LinearWN(256 + 256 * 4 * 4, 512), th.nn.LeakyReLU(0.2, inplace=True)
        )

        self.mean: th.nn.Module = la.LinearWN(512, self.n_embs)
        self.logvar: th.nn.Module = la.LinearWN(512, self.n_embs)

        self.apply(lambda m: la.glorot(m, 0.2))
        la.glorot(self.mean, 1.0)
        la.glorot(self.logvar, 1.0)

    def forward(self, geom: th.Tensor, color: th.Tensor) -> Dict[str, th.Tensor]:
        preds = {}

        geomout = self.geommod(geom.view(geom.shape[0], -1))
        texout = self.texmod(color / 255.0 - 0.5).view(-1, 256 * 4 * 4)
        encout = self.jointmod(th.cat([geomout, texout], dim=1))
        embs_mu = self.mean(encout) * self.mean_scale
        embs_logvar = self.logvar(encout) * self.logvar_scale

        # NOTE: the noise is only applied to the input-conditioned values
        if self.training:
            noise = th.randn_like(embs_mu)
            embs = embs_mu + th.exp(embs_logvar) * noise * self.noise_std
        else:
            embs = embs_mu.clone()

        preds.update(
            embs=embs,
            embs_mu=embs_mu,
            embs_logvar=embs_logvar,
        )

        return preds


class GeomDecoder(nn.Module):
    """A decoder for coarse geometry."""

    def __init__(
        self,
        n_embs: int,
        verts_mean: th.Tensor,
        verts_std: float,
    ):
        super().__init__()

        self.verts_std: float = verts_std
        self.register_buffer("verts_mean", verts_mean[None].float())

        self.n_embs = n_embs
        self.n_verts_out = verts_mean.shape[-2]

        self.geommod = th.nn.Sequential(
            la.LinearWN(self.n_embs, 256),
            th.nn.LeakyReLU(0.2, inplace=True),
            la.LinearWN(256, 3 * self.n_verts_out),
        )

        self.apply(lambda m: la.glorot(m, 0.2))
        la.glorot(self.geommod[-1], 1.0)

    def forward(self, embs: th.Tensor) -> Dict[str, th.Tensor]:
        preds = {}

        geom = self.geommod(embs).view(embs.shape[0], -1, 3)
        geom = geom * self.verts_std + self.verts_mean

        preds.update(face_geom=geom)

        return preds


class PrimDecoder(nn.Module):
    """A decoder for relightable Gaussians."""

    def __init__(
        self,
        n_embs,
        geo_fn,
        color_mean: th.Tensor,
        n_diff_sh: int = 8,
        n_color_sh: int = 3,
    ):
        super().__init__()

        self.slabsize = 1024
        self.n_splats = 1024**2
        self.n_embs = n_embs

        self.geo_fn = geo_fn

        self.viewmod = nn.Sequential(
            *make_linear(3, 8, "wn", nn.LeakyReLU(0.2, inplace=True))
        )
        self.encmod = nn.Sequential(
            *make_linear(n_embs, 256 * 8 * 8, "wn", nn.LeakyReLU(0.2, inplace=True))
        )

        self.diff_sh_degree = n_diff_sh
        self.color_sh_degree = n_color_sh
        self.n_color_sh_coeffs = (n_color_sh + 1) ** 2
        self.n_mono_sh_coeffs = (n_diff_sh + 1) ** 2 - self.n_color_sh_coeffs
        self.n_diff_coeffs = 3 * self.n_color_sh_coeffs + self.n_mono_sh_coeffs

        vind_ch = (
            self.n_diff_coeffs + 11 + 1
        )  # diffuse_sh + Gaussian params + roughness
        vd_ch = 4  # normal + visibility
        self.vnocond_mod = nn.Sequential(
            *make_conv_trans(
                256, 256, 4, 2, 1, "wn", nn.LeakyReLU(0.2, inplace=True), ub=(16, 16)
            ),
            *make_conv_trans(
                256, 128, 4, 2, 1, "wn", nn.LeakyReLU(0.2, inplace=True), ub=(32, 32)
            ),
            *make_conv_trans(
                128, 128, 4, 2, 1, "wn", nn.LeakyReLU(0.2, inplace=True), ub=(64, 64)
            ),
            *make_conv_trans(
                128, 64, 4, 2, 1, "wn", nn.LeakyReLU(0.2, inplace=True), ub=(128, 128)
            ),
            *make_conv_trans(
                64, 32, 4, 2, 1, "wn", nn.LeakyReLU(0.2, inplace=True), ub=(256, 256)
            ),
            *make_conv_trans(
                32, 16, 4, 2, 1, "wn", nn.LeakyReLU(0.2, inplace=True), ub=(512, 512)
            ),
            *make_conv_trans(16, vind_ch, 4, 2, 1, "wn", ub=(1024, 1024)),
        )
        self.vcond_mod = nn.Sequential(
            *make_conv_trans(
                256 + 8,
                256,
                4,
                2,
                1,
                "wn",
                nn.LeakyReLU(0.2, inplace=True),
                ub=(16, 16),
            ),
            *make_conv_trans(
                256, 128, 4, 2, 1, "wn", nn.LeakyReLU(0.2, inplace=True), ub=(32, 32)
            ),
            *make_conv_trans(
                128, 128, 4, 2, 1, "wn", nn.LeakyReLU(0.2, inplace=True), ub=(64, 64)
            ),
            *make_conv_trans(
                128, 64, 4, 2, 1, "wn", nn.LeakyReLU(0.2, inplace=True), ub=(128, 128)
            ),
            *make_conv_trans(
                64, 32, 4, 2, 1, "wn", nn.LeakyReLU(0.2, inplace=True), ub=(256, 256)
            ),
            *make_conv_trans(
                32, 16, 4, 2, 1, "wn", nn.LeakyReLU(0.2, inplace=True), ub=(512, 512)
            ),
            *make_conv_trans(16, vd_ch, 4, 2, 1, "wn", ub=(1024, 1024)),
        )

        self.apply(lambda m: la.glorot(m, 0.2))
        la.glorot(self.vnocond_mod[-1], 1.0)
        la.glorot(self.vcond_mod[-1], 1.0)

        rgb = color_mean / 255.0  # [3, tex_res, tex_res]
        albedo = (2.0 * rgb / 2.2974).permute(1, 2, 0).reshape(1, -1, 3)
        self.albedo = th.nn.Parameter(albedo)

    def forward(
        self,
        embs: th.Tensor,
        geom: th.Tensor,
        headrel_campos: th.Tensor,
        light_intensity: th.Tensor,
        headrel_light_pos: th.Tensor,
        headrel_light_sh: th.Tensor,
        n_lights: th.Tensor,
        preconv_envmap: Optional[th.Tensor] = None,
        lightrot: Optional[th.Tensor] = None,
    ):
        preds = {}
        B = embs.shape[0]

        # compute positional map on uv
        # TODO: check if we need this
        postex = self.geo_fn.to_uv(geom)
        primposbase = postex.permute(0, 2, 3, 1).view(B, -1, 3)

        # compute normal map on uv
        # TODO: check if we need this
        vn = self.geo_fn.vn(geom)
        tn = self.geo_fn.to_uv(vn)
        tn = F.normalize(tn, dim=1)
        primnmlbase = tn.permute(0, 2, 3, 1).reshape(B, -1, 3)

        # run view-independent decoder
        embs = self.encmod(embs).view(-1, 256, 8, 8)
        f_vnocond = self.vnocond_mod(embs)

        # run view-dependent decoder
        view = self.viewmod(F.normalize(headrel_campos, dim=1))[
            :, :, None, None
        ].expand(-1, -1, 8, 8)
        embs_v = th.cat([embs, view], dim=1)
        f_vcond = self.vcond_mod(embs_v)
        f_vcond = f_vcond.permute(0, 2, 3, 1).view(B, -1, 4)

        # diffuse sh
        diff_shs = f_vnocond[:, : self.n_diff_coeffs]
        diff_shs = diff_shs.permute(0, 2, 3, 1).view(B, -1, self.n_diff_coeffs)
        diff_shs_color = diff_shs[..., : self.n_color_sh_coeffs * 3].reshape(
            B, -1, 3, self.n_color_sh_coeffs
        )
        diff_shs_mono = diff_shs[..., self.n_color_sh_coeffs * 3 :].reshape(
            B, -1, 1, self.n_mono_sh_coeffs
        )
        diff_shs = th.cat([diff_shs_color, diff_shs_mono.expand(-1, -1, 3, -1)], -1)

        # Gaussian parameters
        f_geom = f_vnocond[:, self.n_diff_coeffs: self.n_diff_coeffs + 11]
        f_geom = f_geom.permute(0, 2, 3, 1).view(B, -1, 11)
        primpos = f_geom[..., 0:3] + primposbase
        primqvec = F.normalize(f_geom[..., 3:7], dim=-1)
        primscale = F.softplus(f_geom[..., 7:10])
        opacity = th.sigmoid(f_geom[..., 10:11])

        # roughness
        sigma = f_vnocond[:, self.n_diff_coeffs + 11:]
        sigma = sigma.permute(0, 2, 3, 1).view(B, -1)
        sigma = (th.exp(sigma) * 0.1).clamp(min=0.01)

        # view-dependent specular visibility
        spec_vis = th.sigmoid(f_vcond[..., :1])

        # view-dependent specular normal
        spec_dnml = f_vcond[..., 1:]
        spec_nml = F.normalize(spec_dnml + primnmlbase, dim=-1)

        # albedo
        albedo = self.albedo.expand(B, -1, -1)

        # compute diffuse color
        diff_color = albedo * (diff_shs * headrel_light_sh[:, None]).sum(dim=-1)

        # compute specular color
        view_local = F.normalize(primpos - headrel_campos[:, None], dim=-1, p=2.0)
        ref_dirs = (
            view_local - 2.0 * (view_local * spec_nml).sum(-1, keepdim=True) * spec_nml
        )

        if preconv_envmap is not None:
            # rotate ref vector not envmap itself
            ref_dirs = th.einsum("bxy,bny->bnx", lightrot, ref_dirs)
            ref_uv = dir2uv(ref_dirs, 2)
            miplevel = sigma * 5
            spec_color = mipmap_grid_sample(
                preconv_envmap, ref_uv[..., None, :], miplevel[..., None]
            )[..., 0]
            spec_color = spec_color.permute(0, 2, 1).clamp(max=1.0) * spec_vis
        else:
            # NOTE: it assumes the input lights are Dirac delta function
            spec_color = (
                evaluate_gaussian(
                    ref_dirs.contiguous(),
                    sigma.contiguous(),
                    light_intensity.contiguous(),
                    headrel_light_pos.contiguous(),
                    primpos.contiguous(),
                    n_lights.int(),
                    w_type=0,
                )
                * spec_vis
            )

        color = diff_color.clamp(min=0.0) + spec_color

        preds.update(
            color=color.clamp(min=0.0),
            opacity=opacity,
            primpos=primpos,
            primqvec=primqvec,
            primscale=primscale.clamp(*primscale_range),
            primscale_preclip=primscale,
            sigma=sigma,
            spec_vis=spec_vis,
            spec_nml=spec_nml,
            spec_dnml=spec_dnml,
            diff_color=diff_color,
            spec_color=spec_color,
            primnmlbase=primnmlbase,
        )

        if self.training:
            with th.no_grad():
                # Choose a random light direction
                light_dir = (
                    th.rand(
                        B,
                        1,
                        3,
                        device=headrel_light_pos.device,
                        dtype=headrel_light_pos.dtype,
                    )
                    - 0.5
                )
                light_dir = F.normalize(light_dir, p=2, dim=-1)  # [1, 1, 3]
                cos_weight = (light_dir * spec_nml).sum(
                    dim=-1, keepdims=True
                )  # [B, n_prims, 1]
                light_intensity = th.ones_like(light_intensity[:, :1])
                n_lights = th.ones_like(n_lights)
                headrel_light_pos = 10000.0 * light_dir
                sh_coeffs = sh.dir2sh_torch(self.diff_sh_degree, light_dir)
                light_sh = (sh_coeffs[:, :, None] * light_intensity[..., None]).sum(
                    dim=1
                )

            diff_color_rand = (diff_shs * light_sh[:, None]).sum(dim=-1)

            preds["cos_weight"] = cos_weight
            preds["color_rand"] = diff_color_rand.clamp(min=0.0)

        return preds


class RGCASummary(Callable):

    def __call__(
        self, preds: Dict[str, Any], batch: Dict[str, Any]
    ) -> Dict[str, th.Tensor]:

        diag = {}

        dev = preds["diff_color"].device
        bs = preds["diff_color"].shape[0]
        diff_color = preds["diff_color"].clamp(0, 1)
        spec_color = preds["spec_color"].clamp(0, 1)
        opacity = (preds["opacity"]).clamp(0, 1)
        spec_normal = preds["spec_nml"] * 0.5 + 0.5
        spec_rough = preds["sigma"].clamp(0, 1)
        spec_vis = (preds["spec_vis"]).clamp(0, 1)
        color = diff_color + spec_color

        fh, fw = 1024, 1024
        nf = fh * fw
        h = fh

        out = th.zeros(bs, 3, h, fw, device=dev)
        out[:, :, :fh] = color[:, 0:nf].view(bs, fh, fw, -1).permute(0, 3, 1, 2)
        diag["sh_slab"] = linear2srgb(out).clamp(0, 1)

        out = th.zeros(bs, 3, h, fw, device=dev)
        out[:, :, :fh] = diff_color[:, 0:nf].view(bs, fh, fw, -1).permute(0, 3, 1, 2)
        diag["diff_sh_slab"] = linear2srgb(out).clamp(0, 1)

        out = th.zeros(bs, 3, h, fw, device=dev)
        out[:, :, :fh] = spec_color[:, 0:nf].view(bs, fh, fw, -1).permute(0, 3, 1, 2)
        diag["spec_slab"] = linear2srgb(out).clamp(0, 1)

        out = th.zeros(bs, 3, h, fw, device=dev)
        out[:, :, :fh] = spec_normal[:, 0:nf].view(bs, fh, fw, -1).permute(0, 3, 1, 2)
        diag["spec_normal_slab"] = out.clamp(0, 1)

        out = th.zeros(bs, 1, h, fw, device=dev)
        out[:, :, :fh] = spec_vis[:, 0:nf].view(bs, fh, fw, -1).permute(0, 3, 1, 2)
        diag["spec_vis_slab"] = out.clamp(0, 1)

        out = th.zeros(bs, 1, h, fw, device=dev)
        out[:, :, :fh] = spec_rough[:, 0:nf].view(bs, fh, fw, -1).permute(0, 3, 1, 2)
        diag["spec_rough_slab"] = out.clamp(0, 1)

        out = th.zeros(bs, 1, h, fw, device=dev)
        out[:, :, :fh] = opacity[:, 0:nf].view(bs, fh, fw, -1).permute(0, 3, 1, 2)
        diag["opacity_slab"] = out.clamp(0, 1)

        light_sh = preds["headrel_light_sh"]
        h, w = 128, 128
        py, px = th.meshgrid(
            th.linspace(1.0, -1.0, h, device=dev), th.linspace(-1.0, 1.0, w, device=dev)
        )
        pixelcoords = th.stack([px, py], -1)
        zsq = pixelcoords.pow(2).sum(-1, keepdim=True)
        mask = (zsq < 1.0).float()[:, :, :1]
        nz = -(1.0 - zsq).clamp(min=0.0).sqrt()
        nml_n = th.cat([pixelcoords, nz], -1)
        nml_p = th.cat([pixelcoords, -nz], -1)
        nml = th.cat([nml_p, nml_n], 0)
        mask = th.cat([mask, mask], 0)
        color = sh.eval_sh(8, light_sh[:, None, None], nml)
        color = (mask * color).permute(0, 3, 1, 2)
        diag["light_sh"] = color / color.max()

        render = preds["rgb"]
        render = linear2srgb(render).clamp(0, 1)

        if "image" in batch:
            gt = batch["image"]
            diff = (preds["rgb"] - batch["image"]).clamp(-1, 1)
            diff = scale_diff_image(diff)

            diag["gt"] = linear2srgb(gt).clamp(0, 1)
            diag["diff"] = diff.clamp(0, 1)

        diag["render"] = render
        diag["alpha"] = preds["alpha"].clamp(0, 1).expand(-1, 3, -1, -1)

        if "image_weight" in preds:
            diag["weight"] = preds["image_weight"]

        if "mesh_nml" in preds:
            diag["mesh_nml"] = 0.5 * (-preds["mesh_nml"]) + 0.5

        diag["depth_nml"] = (
            diag["alpha"]
            * (
                0.5 * -depth2normals(preds["depth"], batch["focal"], batch["princpt"])
                + 0.5
            )
            + (1.0 - diag["alpha"]) * 0.5
        )

        bdi = make_image_grid_batched(diag, input_is_in_0_1=True)
        bdi = cv2.resize(
            bdi, fx=0.5, fy=0.5, dsize=None, interpolation=cv2.INTER_LINEAR
        )

        for k, v in diag.items():
            diag[k] = make_grid(255.0 * v, nrow=16).clip(0, 255).to(th.uint8)

        return diag
