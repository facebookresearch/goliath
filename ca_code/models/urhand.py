# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import functools
import numpy as np
import scipy.stats as st
import torch as th
import torch.nn as nn
import cv2
import torch.nn.functional as F
from torchvision.utils import make_grid
import copy
from ca_code.nn.color_cal import CalV5
from ca_code.utils.lbs import LBSModule
from ca_code.utils.seams import SeamSampler
from ca_code.nn.blocks import (
    ConvBlock,
    ConvDownBlock,
    UpConvBlockDeep,
    tile2d,
    weights_initializer,
)
from ca_code.utils.geom import (
    GeometryModule,
    compute_view_cos,
    depth_discontuity_mask,
    index_image_impaint,
    values_to_uv,
    compute_tbn_uv,
    compute_tbn_uv_given_normal,
    vert_normals, 
    xyz2normals, 
    make_uv_vert_index,
)
from ca_code.utils.image import linear2displayBatch
import ca_code.nn.layers as la
from ca_code.utils.torchutils import index
# from ca_code.utils.render_drtk import RenderLayer
from ca_code.utils.render_pytorch3d import RenderLayer
from pytorch3d.transforms import axis_angle_to_matrix, euler_angles_to_matrix, matrix_to_axis_angle
from scipy.ndimage import gaussian_filter
# from care.utils.lighting import get_shadow_map

logger = logging.getLogger(__name__)

def sample_uv(d, img):
    '''
    args:
        d: (B, 3, H, W)
        img: (B, 3, H', W')
    '''
    u = (1 / np.pi) * th.atan2(d[:, 0], d[:, 2])  # range: [-1, 1]
    v = (1 / np.pi) * th.acos(d[:, 1])  # range: [0, 1]
    v = 2 * v - 1.0
    uv = th.stack([u, v], -1)
    return F.grid_sample(img, uv, padding_mode="border", align_corners=False)

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

class FeatEncoderUNet(nn.Module):
    def __init__(self, n_diff_feat, n_spec_feat, out_ch, m=1):
        super(FeatEncoderUNet, self).__init__()
        c = 3
        nfc = [64, 64 * c, 128 * c, 128 * c, 256 * c]
        nbc = [64, 64 * m, 128 * m, 128 * m, 256 * m]

        self.gb_mod = nn.ModuleList()
        self.feat_mod = nn.ModuleList()
        self.proj = la.Conv2dWN(n_diff_feat + n_spec_feat, 64, 7, 1, 3, bias=False)
        for i in range(len(nfc) - 1):
            self.feat_mod.append(la.Conv2dWN(nfc[i], nfc[i + 1], 4, 2, 1, bias=False))
            self.gb_mod.append(la.Conv2dWN(nfc[i + 1], nbc[i + 1], 1, 1, 0, bias=False))
        self.n_layers = len(nfc) - 1
        self.enc = la.Conv2dWN(256 * c, out_ch, 4, 2, 1)  # Collapsed to one layer

    def forward(self, x):
        gb = []
        x = self.proj(x)
        for i in range(self.n_layers):
            x = self.feat_mod[i](x)
            b = self.gb_mod[i](x)
            gb.insert(0, b)
        z = self.enc(x)

        return z, gb

class DisplacementUNet(nn.Module):
    def __init__(self, uv_size, init_uv_size, output_scale, n_enc_dims=[64, 64, 64, 64, 64, 64]):
        super().__init__()
        self.uv_size = uv_size
        self.init_uv_size = init_uv_size
        self.n_blocks = int(np.log2(self.uv_size // self.init_uv_size))
        self.sizes = [init_uv_size * 2 ** s for s in range(self.n_blocks + 1)]

        in_feats = 2 * 3 # id mesh & normal
        init_channels = 69
        
        self.n_enc_dims = [
            (in_feats, n_enc_dims[0]),
            (n_enc_dims[0], n_enc_dims[1]),
            (n_enc_dims[1], n_enc_dims[2]),
            (n_enc_dims[2], n_enc_dims[3]),
            (n_enc_dims[3], n_enc_dims[4]),
            (n_enc_dims[4], n_enc_dims[5]),
        ]

        self.n_dec_dims = [
            (n_enc_dims[5] + init_channels, n_enc_dims[4]),
            (n_enc_dims[4] * 2, n_enc_dims[3]),
            (n_enc_dims[3] * 2, n_enc_dims[2]),
            (n_enc_dims[2] * 2, n_enc_dims[1]),
            (n_enc_dims[1] * 2, n_enc_dims[0]),
            (n_enc_dims[0] * 2, 1),
        ]

        self.n_dec_roughness_dims = [
            (n_enc_dims[5], n_enc_dims[4]),
            (n_enc_dims[4] * 2, n_enc_dims[3]),
            (n_enc_dims[3] * 2, n_enc_dims[2]),
            (n_enc_dims[2] * 2, n_enc_dims[1]),
            (n_enc_dims[1] * 2, n_enc_dims[0]),
            (n_enc_dims[0] * 2, 1),
        ]
        
        self.enc_layers = nn.ModuleList()
        for i in range(len(self.sizes)):
            size = self.sizes[-i - 1]
            n_in, n_out = self.n_enc_dims[i]
            logger.debug(f"EncoderLayers({i}): {n_in}, {n_out}, {size}")
            self.enc_layers.append(
                la.Conv2dWNUB(
                    n_in,
                    n_out,
                    kernel_size=3,
                    height=size,
                    width=size,
                    stride=1,
                    padding=1,
                )
            )

        self.dec_layers = nn.ModuleList()
        for i, size in enumerate(self.sizes):
            n_in, n_out = self.n_dec_dims[i]
            logger.debug(f"DecoderLayer({i}): {n_in}, {n_out}, {size}")
            self.dec_layers.append(
                la.Conv2dWNUB(
                    n_in,
                    n_out,
                    kernel_size=3,
                    height=size,
                    width=size,
                    stride=1,
                    padding=1,
                )
            )
        self.dec_layers_roughness = nn.ModuleList()
        for i, size in enumerate(self.sizes):
            n_in, n_out = self.n_dec_roughness_dims[i]
            logger.debug(f"DecoderLayer({i}): {n_in}, {n_out}, {size}")
            self.dec_layers_roughness.append(
                la.Conv2dWNUB(
                    n_in,
                    n_out,
                    kernel_size=3,
                    height=size,
                    width=size,
                    stride=1,
                    padding=1,
                )
            )
        self.output_scale = output_scale

        self.apply(weights_initializer(0.2))
        self.dec_layers[-1].apply(weights_initializer(1.0))
        self.dec_layers_roughness[-1].apply(weights_initializer(1.0))

    def forward(self, feat_uv, pose_cond):
        enc_acts = []
        x = feat_uv
        # unet enc
        for i, layer in enumerate(self.enc_layers):
            x = F.leaky_relu(layer(x), 0.2, inplace=True)
            enc_acts.append(x)
            if i < len(self.sizes) - 1:
                x = F.interpolate(
                    x,
                    scale_factor=0.5,
                    mode="bilinear",
                    recompute_scale_factor=True,
                    align_corners=True,
                )

        enc_x = x
        for i, layer in enumerate(self.dec_layers):
            if i == 0:
                x = th.cat([x, pose_cond], dim=1)
                interm_feat = x
            elif i > 0:
                x = F.leaky_relu(x, 0.2, inplace=True)
                x_prev = enc_acts[-i - 1]
                x = F.interpolate(x, size=x_prev.shape[2:4], mode="bilinear", align_corners=True)
                x = th.cat([x, x_prev], dim=1)
            x = layer(x)
        x = th.nn.functional.tanh(x)
        disp = x * self.output_scale

        x = enc_x
        for i, layer in enumerate(self.dec_layers_roughness):
            if i == 0:
                pass
            else:
                x = F.leaky_relu(x, 0.2, inplace=True)
                x_prev = enc_acts[-i - 1]
                x = F.interpolate(x, size=x_prev.shape[2:4], mode="bilinear", align_corners=True)
                x = th.cat([x, x_prev], dim=1)
            x = layer(x)
        x = th.nn.functional.tanh(x)
        roughness = (x + 1) / 4. + 0.3 #[-1, 1] -> [0.3, 0.8]
        return disp, roughness, interm_feat

class ConvTeacherDecoder(nn.Module):
    def __init__(
        self,
        assets,
        uv_size,
        init_uv_size,
        n_joint_enc_dims,
        n_view_enc_dims,
        disp_scale,
        init_channels=128,
        min_channels=16,
        n_enc_dims=[64, 64, 64, 64, 64, 64],
        refine_geo=True,
        feat_uv='texmean',
        view_cond=True,
        fresnel=0.04,
        scaled_albedo=True,
        masked_refiner_input=True,
        impaint_uv=True,
        geo_fn=None,
    ):
        super().__init__()
        self.geo_fn = geo_fn
        self.shadow = True
        self.view_cond = view_cond
        self.refine_geo = refine_geo
        self.feat_uv = feat_uv
        self.fresnel = fresnel
        self.scaled_albedo = scaled_albedo
        if masked_refiner_input:
            assert impaint_uv
        self.masked_refiner_input = masked_refiner_input
        self.impaint_uv = impaint_uv
        self.spec_powers = [1, 16, 32]
        self.env_scale = 12.0
        H, W = 16, 32
        theta, phi = np.meshgrid(
            (np.arange(H, dtype=np.float32) + 0.5) * 3.1415926 / H,
            (np.arange(W, dtype=np.float32) + 0.5) * 3.1415926 * 2.0 / W,
            indexing="ij",
        )
        sph = np.stack(
            [np.sin(theta) * np.sin(phi), np.cos(theta), np.sin(theta) * np.cos(phi)], axis=-1
        )
        sph = th.from_numpy(sph)

        sph = sph.view(-1, 3)
        dkernel = (sph[:, None] * sph[None, :]).sum(-1).clamp(min=0.0).reshape(H * W, H * W)
        self.register_buffer("diff_kernel", dkernel)
        raw_index_image = make_uv_vert_index(self.geo_fn.vt, self.geo_fn.vi, self.geo_fn.vti, uv_shape=1024, flip_uv=False)
        self.register_buffer("raw_index_image", raw_index_image)
        self.register_buffer("raw_index_mask", (raw_index_image != -1).any(dim=-1))

        self.uv_size = uv_size
        self.init_uv_size = init_uv_size
        self.n_view_enc_dims = n_view_enc_dims
        self.n_joint_enc_dims = n_joint_enc_dims
        if self.view_cond:
            self.n_joint_enc_dims += 3
        self.featenc = FeatEncoderUNet(1, 1 * len(self.spec_powers), 128, m = 1)
        nc = [128, 256, 128, 128, 64, 32, 16, 4]
        self.texmod0 = nn.ModuleList()
        self.texmod1 = nn.ModuleList()
        h = 64
        for i in range(len(nc) - 1):
            self.texmod0.append(la.Conv2dWNUB(nc[i],
                    nc[i+1],
                    kernel_size=3,
                    height=h,
                    width=h,
                    stride=1,
                    padding=1,))
            self.texmod1.append(la.Conv2dWN(
                nc[i],
                nc[i+1],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False))
            h *= 2
        self.n_layers_tex = len(self.texmod0)

        self.apply(lambda x: la.glorot(x, 0.2))
        la.glorot(self.texmod0[-1], 1.0)
        la.glorot(self.texmod1[-1], 1.0)        

        self.joint_conv_block_tex = ConvBlock(
            self.n_joint_enc_dims,
            init_channels,
            self.init_uv_size,
        )

        self.geo_refiner = DisplacementUNet(uv_size, init_uv_size, disp_scale)
        self.rl = RenderLayer(
            h=1024,
            w=1024,                 
            vt=self.geo_fn.vt,
            vi=self.geo_fn.vi,
            vti=self.geo_fn.vti,
            flip_uvs=False
        )

        self.global_scale = nn.Parameter(th.ones(1))
        if self.scaled_albedo:
            self.global_albedo_scale = nn.Parameter(th.ones(1)*0)

    def forward(
        self,
        lbs_motion,
        id_mesh,
        tex_mean,
        verts_rec,
        cam_pos,
        light_pos,
        light_intensity,
        seam_sampler = None,
        ccm = None,
        falloff_dist = None,
        nearfield = False,
        iteration: Optional[int] = None,
    ):
        B = verts_rec.shape[0]
        L = light_pos.shape[1]
        chunk_size = 128

        mask = (self.geo_fn.index_image != -1).any(dim=-1)
        idxs = self.geo_fn.index_image[mask]

        tri_uv = self.geo_fn.vt[self.geo_fn.v2uv[idxs, 0].to(th.long)]
        tri_xyz = verts_rec[:, idxs]

        # interpolate normal for smoothness
        vert_nml = vert_normals(verts_rec, self.geo_fn.vi.expand(B, -1, -1))
        face_idxs = self.geo_fn.face_index_image[mask]
        vi_img = self.geo_fn.vi[face_idxs]
        bary_img = self.geo_fn.bary_image[mask]
        vn_img0 = th.stack([index(vert_nml[i], vi_img[..., 0], 0) for i in range(B)])
        vn_img1 = th.stack([index(vert_nml[i], vi_img[..., 1], 0) for i in range(B)])
        vn_img2 = th.stack([index(vert_nml[i], vi_img[..., 2], 0) for i in range(B)])
        vn_img = (th.stack([vn_img0, vn_img1, vn_img2], dim=3) * bary_img[None, :, None, :]).sum(-1)
        n = vn_img / th.norm(vn_img, dim=-1, keepdim=True).clamp(min=1e-5)

        t, b, n = compute_tbn_uv_given_normal(tri_xyz, tri_uv, n)
        tbn_rot = th.stack((t, -b, n), dim=-2)
        tbn_rot_uv = th.zeros(
            (B, self.geo_fn.uv_size, self.geo_fn.uv_size, 3, 3),
            dtype=th.float32,
            device=verts_rec.device,
        )
        tbn_rot_uv[:, mask] = tbn_rot

        p_uv = self.geo_fn.to_uv(verts_rec) # [1k, 1k]
        if not self.impaint_uv:
            # no impaint means we use Stan's seamless uv
            vert_mask = (verts_rec.detach() - self.geo_fn.from_uv(p_uv).detach()).abs() > 1
        v_uv = F.normalize(cam_pos[..., None, None] - p_uv, dim=1)
        light_intensity = light_intensity[..., None, None]

        # compute shadow map
        if self.shadow:
            with th.no_grad():
                posc = (verts_rec.max(1)[0] + verts_rec.min(1)[0]) / 2
                posc = posc[:, None].expand(-1, L, -1).reshape(-1, 3)
                lightpos = light_pos.view(-1, 3)
                lightrot = build_cam_rot_mat(lightpos, posc)
                p_uv_shadow = p_uv[:, None].expand(-1, L, -1, -1, -1)
                p_uv_shadow = p_uv_shadow.reshape(B * L, p_uv_shadow.shape[2], p_uv_shadow.shape[3], p_uv_shadow.shape[4])
                verts = verts_rec[:, None].expand(-1, L, -1, -1)
                verts = verts.reshape(B * L, verts.shape[2], verts.shape[3])
                nml_shadow = tbn_rot_uv[:, :, :, 2:].permute(0, 3, 4, 1, 2).expand(-1, L, -1, -1, -1)
                nml_shadow = nml_shadow.reshape(B * L, 3, nml_shadow.shape[3], nml_shadow.shape[4])
                shadow_map = get_shadow_map(self.rl, lightpos, lightrot, verts, p_uv_shadow, nml_shadow) # range [0, 1000]
                shadow_map = th.exp(-shadow_map / 8.0) # [B*L, 1, H, W]
                shadow_map = shadow_map.reshape(B, L, 1, shadow_map.shape[-2], shadow_map.shape[-1])

        # calculate light energy for point light
        l_uv = F.normalize(light_pos[..., None, None] - p_uv[:, None], dim=2)
        view = - v_uv
        nml = tbn_rot_uv[:, :, :, 2:].permute(0, 3, 4, 1, 2)[:, 0, ...]
        ref = view - 2.0 * (view * nml).sum(1, keepdim=True) * nml
        # apply rotation
        diff = (
            (nml[:, None] * l_uv).sum(2, keepdim=True).clamp(min=0.0, max=1.0)
        )  # (B, L, 1, H, W)
        spec = (ref[:, None] * l_uv).sum(2, keepdim=True).clamp(min=0.0) # (B, L, 1, H, W)
        specs = []
        for v in self.spec_powers:
            specs.append((spec.pow(v)).clamp(max=1.0))
        spec = th.stack(specs, 2)
        if self.shadow:
            diff_p = (diff * light_intensity * shadow_map).sum(1)
            spec_p = (spec * light_intensity[:, :, None, :, :, :] * shadow_map[:, :, None, :, :, :]).sum(1)
        else:
            diff_p = (diff * light_intensity).sum(1)
            spec_p = (spec * light_intensity[:, :, None, :, :, :]).sum(1)
        lint_scale = light_intensity.sum(1)
        inv_lint_scale = 1.0 / (lint_scale + 1e-6)
        outputs = {"diff_feature_raw": inv_lint_scale * diff_p if lightmap is None else None, 
                "spec_feature_raw": inv_lint_scale[:, None] * spec_p if lightmap is None else None,
                "shadow_raw": shadow_map if self.shadow else None,
                "feature_normal_raw": nml,}

        # calculate displacement first
        uv_id_mesh = self.geo_fn.to_uv(id_mesh)
        pose_cond = tile2d(lbs_motion, self.init_uv_size)
        normalized_tex = (tex_mean / 255.0) * 2.0 - 1.0
        if self.masked_refiner_input:
            uv_id_mesh[:, :, ~self.raw_index_mask] *= 0
            normalized_tex[:, :, ~self.raw_index_mask] *= 0
        if self.feat_uv == 'texmean':
            uv_refiner_feat = th.concat([normalized_tex, normalized_tex], dim=1)
        elif self.feat_uv == 'texmean_geo':
            uv_refiner_feat = th.concat([normalized_tex, uv_id_mesh], dim=1)
        elif self.feat_uv == 'geo':
            uv_refiner_feat = th.concat([uv_id_mesh, nml], dim=1) # [bs, 6, uv_size, uv_size]
        else:
            raise NotImplementedError("{} not supported".format(self.feat_uv))
        displacement, roughness, id_pose_feat = self.geo_refiner(uv_refiner_feat, pose_cond)
        nml4disp = tbn_rot_uv[:, :, :, 2:].permute(0, 3, 4, 1, 2)[:, 0, ...]
        if not self.refine_geo:
            displacement = displacement * 0
        
        p_uv_displaced = p_uv + nml4disp.detach() * displacement
        n = xyz2normals(p_uv_displaced)
        n = n[:, :, mask].permute(0, 2, 1)

        # update the vertices by displacement
        verts_rec_displaced = self.geo_fn.from_uv(p_uv_displaced)
        if not self.impaint_uv:
            # no impaint means we use Stan's seamless uv
            verts_rec_displaced[vert_mask] = verts_rec[vert_mask]
        tri_xyz_displaced = verts_rec_displaced[:, idxs]

        t, b, n = compute_tbn_uv_given_normal(tri_xyz_displaced, tri_uv, n)
        tbn_rot = th.stack((t, -b, n), dim=-2)
        tbn_rot_uv = th.zeros(
            (B, self.geo_fn.uv_size, self.geo_fn.uv_size, 3, 3),
            dtype=th.float32,
            device=verts_rec.device,
        )
        tbn_rot_uv[:, mask] = tbn_rot

        p_uv = p_uv_displaced
        v_uv = F.normalize(cam_pos[..., None, None] - p_uv, dim=1)

        # compute shadow map
        if self.shadow:
            with th.no_grad():
                posc = (verts_rec_displaced.max(1)[0] + verts_rec_displaced.min(1)[0]) / 2
                posc = posc[:, None].expand(-1, L, -1).reshape(-1, 3)
                lightpos = light_pos.view(-1, 3)
                lightrot = build_cam_rot_mat(lightpos, posc)
                p_uv_shadow = p_uv[:, None].expand(-1, L, -1, -1, -1)
                p_uv_shadow = p_uv_shadow.reshape(B * L, p_uv_shadow.shape[2], p_uv_shadow.shape[3], p_uv_shadow.shape[4])
                verts = verts_rec_displaced[:, None].expand(-1, L, -1, -1)
                verts = verts.reshape(B * L, verts.shape[2], verts.shape[3])
                nml_shadow = tbn_rot_uv[:, :, :, 2:].permute(0, 3, 4, 1, 2).expand(-1, L, -1, -1, -1)
                nml_shadow = nml_shadow.reshape(B * L, 3, nml_shadow.shape[3], nml_shadow.shape[4])
                shadow_map = get_shadow_map(self.rl, lightpos, lightrot, verts, p_uv_shadow, nml_shadow) # range [0, 1000]
                shadow_map = th.exp(-shadow_map / 8.0) # [B*L, 1, H, W]
                shadow_map = shadow_map.reshape(B, L, 1, shadow_map.shape[-2], shadow_map.shape[-1])

        # calculate light energy for point light
        if True:
            ggx_L = F.normalize(light_pos[..., None, None] - p_uv[:, None], dim=2) #[bs, nlights, 3, uv_size, uv_size]
            ggx_V = v_uv #[bs, 3, uv_size, uv_size]
            ggx_H = F.normalize((ggx_L + ggx_V[:, None, ...]) / 2.0, dim=2) #[bs, nlights, 3, uv_size, uv_size]
            ggx_N = tbn_rot_uv[:, :, :, 2:].permute(0, 3, 4, 1, 2)[:, 0, ...] #[bs, 3, uv_size, uv_size]

            ggx_nov = th.sum(ggx_V * ggx_N, dim=1, keepdim=True) #[bs, 1, uv_size, uv_size]
            ggx_N = ggx_N * ggx_nov.sign()

            ggx_nol = th.sum(ggx_N[:, None, ...] * ggx_L, dim=2, keepdim=True).clamp_(1e-6, 1) #[bs, nlights, 1, uv_size, uv_size]
            ggx_nov = th.sum(ggx_N * ggx_V, dim=1, keepdim=True) #[bs, 1, uv_size, uv_size]
            ggx_noh = th.sum(ggx_N[:, None, ...] * ggx_H, dim=2, keepdim=True).clamp_(1e-6, 1)
            ggx_voh = th.sum(ggx_V[:, None, ...] * ggx_H, dim=2, keepdim=True).clamp_(1e-6, 1)

            alpha = roughness * roughness
            alpha2 = alpha * alpha
            ggx_k = (alpha + 2 * roughness + 1) / 8.0
            FMi = ((-5.55473) * ggx_voh - 6.98316) * ggx_voh
            frac0 = self.fresnel + (1 - self.fresnel) * th.pow(2.0, FMi)
            frac = frac0 * alpha2[:, None, ...]
            nom0 = ggx_noh * ggx_noh * (alpha2[:, None, ...] - 1) + 1

            nom1 = ggx_nov * (1 - ggx_k) + ggx_k
            nom2 = ggx_nol * (1 - ggx_k[:, None, ...]) + ggx_k[:, None, ...]
            nom = (4 * np.pi * nom0 * nom0 * nom1[:, None, ...] * nom2).clamp_(1e-6, 4 * np.pi)
            spec = frac / nom #[bs, nlights, 1, uv_size, uv_size]
            specular = spec

            l_uv = F.normalize(light_pos[..., None, None] - p_uv[:, None], dim=2)
            nml = tbn_rot_uv[:, :, :, 2:].permute(0, 3, 4, 1, 2)[:, 0, ...]
            # apply rotation
            diff_cos = (
                (nml[:, None] * l_uv).sum(2, keepdim=True).clamp(min=0.0, max=1.0)
            )  # (B, L, 1, H, W)
            specs = []
            for v in self.spec_powers:
                specs.append((spec.pow(v)).clamp(max=1.0))
            spec = th.stack(specs, 2)
            if self.shadow:
                diff_p = (diff_cos * light_intensity * shadow_map).sum(1)
                spec_p = (spec * light_intensity[:, :, None, :, :, :] * shadow_map[:, :, None, :, :, :] * ((diff_cos[:, :, None, ...] > 0) * 1.)).sum(1)
            else:
                diff_p = (diff_cos * light_intensity).sum(1)
                spec_p = (spec * light_intensity[:, :, None, :, :, :] * ((diff_cos[:, :, None, ...] > 0) * 1.)).sum(1)
            # hack to scale up spec_p
            spec_p = spec_p * 10

            lint_scale = light_intensity.sum(1)
            inv_lint_scale = 1.0 / (lint_scale + 1e-6)

            feat_p = inv_lint_scale[:, None] * th.cat([diff_p[:, None], spec_p], 1)
            
            if self.scaled_albedo:
                tex_mean = tex_mean.clone() * (th.sigmoid(self.global_albedo_scale) / 2. + 0.7)

            surface_brdf = (tex_mean[:, None, ...] / 255.) / (np.pi) + specular
            cosine = th.einsum("ijknm,iknm->ijnm", l_uv, nml) #[bs, nlights, uv_size, uv_size]
            cosine = th.clamp(cosine, min=0.0)
            # assume the light area is uniformly distributed on the hemi-sphere
            rgb = th.mean(4 * th.pi * surface_brdf * light_intensity * cosine[:, :, None], dim=1)
            rgb = rgb * (th.sigmoid(self.global_scale) / 2. + 0.3)
            outputs.update(
                phys_tex=rgb,
                roughness=roughness,
            )

        if self.view_cond:
            # v_uv [bs, 3, 1k, 1k]
            # tbn_rot_uv [bs, 1k, 1k, 3, 3]
            viewout = v_uv.permute(0, 2, 3, 1)[:, :, :, None, :] @ tbn_rot_uv.transpose(-2, -1)
            viewout = viewout[:, :, :, 0, :].permute(0, 3, 1, 2)
            viewout = F.interpolate(viewout, (id_pose_feat.shape[2:]), mode='bilinear')
            id_pose_feat = th.concat([id_pose_feat, viewout], dim=1)
        outputs.update(id_pose_conv=id_pose_feat)
        joint_feat = self.joint_conv_block_tex(id_pose_feat)
        def dec_feat(feat):
            feat = feat.reshape(feat.shape[0], -1, feat.shape[-2], feat.shape[-1])
            z, gainbias = self.featenc(feat)
            interm_features = gainbias
            # non-linear branch
            scale = 0.707107
            activations = []
            x = joint_feat
            hh = 64
            for i in range(self.n_layers_tex):
                x = F.interpolate(x, (hh, hh), mode='bilinear', align_corners=True)
                x = self.texmod0[i](x)
                x = th.nn.functional.leaky_relu(x, 0.2)
                activations.append(x)
                hh *= 2
            x = z
            hh = 64
            for i in range(self.n_layers_tex):
                x = F.interpolate(x, (hh, hh), mode='bilinear', align_corners=True)
                x = self.texmod1[i](x) * activations[i]
                hh *= 2
                if i < len(gainbias):
                    gb = gainbias[i]
                    x = (x + gb) * scale
            x = F.interpolate(x, (self.geo_fn.uv_size, self.geo_fn.uv_size), mode='bilinear', align_corners=True)
            return x, interm_features
        
        # energy in & energy out linear branch
        if True:
            rgb, interm_features = dec_feat(feat_p.detach())

            # for better shadow generalization
            if self.shadow and not self.training:
                rgb = rgb * ((light_intensity / lint_scale[:, None]) * shadow_map).sum(1)
            rgb = lint_scale * rgb

        outputs.update(
            tex=rgb.clamp(min=0),
            shadow=shadow_map if self.shadow else None,
            verts_displaced=verts_rec_displaced,
            diff_feature=inv_lint_scale * diff_p if lightmap is None else None,
            spec_feature=inv_lint_scale[:, None] * spec_p if lightmap is None else None,
            displacement=displacement,
            feature_normal=nml,
            interm_features2reg=interm_features,
        )
        return outputs

class AutoEncoder(nn.Module):
    def __init__(
        self,
        assets,
        cal=None,
        renderer=None,
        relight=None,
        blur_enable=False,
        blur_sig=1.0,
        blur_size=3,
        vis_feature=False,
        impaint_uv=True,
    ):
        super().__init__()
        self.geo_fn = GeometryModule(
            th.LongTensor(assets.topology.vi),
            assets.topology.vt,
            assets.topology.vti,
            assets.topology.v2uv,
            uv_size=1024,
            impaint=impaint_uv,
        )
        self.lbs_fn = LBSModule(
            assets.lbs_model_json,
            assets.lbs_config_dict,
            assets.template_mesh_unscaled[None],
            assets.skeleton_scales,
            global_scaling=[1.0, 1.0, 1.0],  # meter
        )

        tex_mean = th.as_tensor(assets.tex_mean)[np.newaxis]
        self.register_buffer("tex_mean", F.interpolate(tex_mean, (relight.uv_size, relight.uv_size), mode="bilinear"))

        if cal is not None:
            self.cal = CalV5(**cal, cameras=assets.camera_ids)
        self.tex_std = 64.0

        if relight is not None:
            self.relighting_enabled = True
            self.decoder_relight = ConvTeacherDecoder(
                **relight,
                geo_fn=self.geo_fn,
                assets=assets
            )
        else:
            self.relighting_enabled = False

        self.vis_feature = vis_feature
        self.impaint_uv = impaint_uv
        self.force_black_bkgd = False

        if renderer is not None:
            self.rendering_enabled = True
            self.renderer = RenderLayer(
                h=renderer.image_height,
                w=renderer.image_width,
                vt=self.geo_fn.vt,
                vi=self.geo_fn.vi,
                vti=self.geo_fn.vti,
                flip_uvs=False,
            )
        else:
            self.rendering_enabled = False

        self.blur_enable = blur_enable
        self.blur_size = blur_size
        blur_kernel_x = np.diff(
            st.norm.cdf(np.linspace(-blur_sig, blur_sig, blur_size + 1))
        ).astype(np.float32)
        blur_kernel = blur_kernel_x[:, np.newaxis] * blur_kernel_x[np.newaxis, :]
        blur_kernel /= np.sum(blur_kernel)

        blur_kernel = (
            th.tensor(blur_kernel, dtype=th.float32)
            .view(1, 1, blur_size, blur_size)
            .repeat(3, 1, 1, 1).contiguous()
        )
        self.register_buffer('blur_kernel', blur_kernel)
        self.seam_sampler = SeamSampler(assets.seam_data_1024)

    def forward_tex(
        self,
        relight_preds,
        tex_mean_rec: th.Tensor,
        tex_view_rec: th.Tensor,
        tex_mean: th.Tensor,
        shadow_map: th.Tensor,
        index: Optional[Dict[str, Any]] = None,
    ):
        interim = {}
        if relight_preds["tex"].shape[1] == 2:
            gain = relight_preds["tex"][:, 0:1]
            bias = relight_preds["tex"][:, 1:2]
        elif relight_preds["tex"].shape[1] == 4:
            gain = relight_preds["tex"][:, 0:3]
            bias = relight_preds["tex"][:, 3:4]
        elif relight_preds["tex"].shape[1] == 6:
            gain = relight_preds["tex"][:, 0:3]
            bias = relight_preds["tex"][:, 3:6]
        if self.decoder_relight.refine_geo:
            roughness = relight_preds["roughness"]
            interim["roughness"] = (roughness.detach() * 255).clamp_(min=0, max=255)
        interim["tex_mean_vis"] = tex_mean.detach()
        interim["gain"] = (gain.detach() * 255).clamp_(min=0, max=255)
        interim["bias"] = (bias.detach() * self.tex_std).clamp_(min=0, max=255)
        if relight_preds.get("diff_feature") is not None:
            interim["diffuse_rgb"] = (relight_preds["diff_feature"].detach() * tex_mean.detach()).clamp_(min=0, max=255)

        tex_rec = tex_mean
        interim["tex_ua"] = tex_rec.detach().clamp_(min=0, max=255)
        tex_rec = tex_rec * gain + bias * self.tex_std

        if not th.jit.is_scripting() and index is not None and hasattr(self, "cal"):
            ident_camera_id = self.cal.name_to_idx(index['camera'])
            tex_rec = self.cal(tex_rec, ident_camera_id)

        tex_rec = tex_rec.clamp_(min=0, max=255)
        return tex_rec, interim

    def forward(
        self,
        pose: th.Tensor,
        campos: th.Tensor,
        K: th.Tensor,
        Rt: th.Tensor,
        light_pos: Optional[th.Tensor] = None,
        light_intensity: Optional[th.Tensor] = None,
        camera_id: Optional[List[str]] = None,
        frame_id: Optional[th.Tensor] = None,
        iteration: Optional[int] = None,
        **kwargs,
    ):
        index = {'camera': camera_id, 'frame': frame_id}
        tex_mean = self.tex_mean
        preds = {}
        mesh_world = self.lbs_fn.pose(
            th.zeros_like(self.lbs_fn.lbs_template_verts), pose
        )
        mesh_id_only = self.lbs_fn.lbs_template_verts
        verts_rec = mesh_world * 1000  # meter -> mm

        hand_pose_aa = matrix_to_axis_angle(euler_angles_to_matrix(th.flip(pose, [2]), 'ZYX')).reshape(bs, -1)

        relight_preds = self.decoder_relight(
            lbs_motion=hand_pose_aa.detach(),
            id_mesh=mesh_id_only.detach(),
            tex_mean=tex_mean.detach(),
            verts_rec=verts_rec.detach(),
            cam_pos=campos,
            light_pos=light_pos,
            light_intensity=light_intensity,
            seam_sampler=self.seam_sampler,
            iteration=iteration
        )
        preds.update(interm_features2reg=relight_preds["interm_features2reg"])

        if relight_preds.get("phys_tex") is not None:
            phys_tex_rec = relight_preds["phys_tex"] * 255.
            phys_tex_rec = phys_tex_rec.clamp_(min=0, max=255)
        else:
            phys_tex_rec = th.zeros_like(tex_mean)

        tex_rec, interim = self.forward_tex(
            relight_preds,
            None,
            None,
            tex_mean,
            None,
            index,
        )
        preds.update(texrec_before_warp=tex_rec)

        # still do seam sampler for linear model texture
        if self.impaint_uv:
            # not impaint uv means we are using seamless uv
            tex_rec = self.seam_sampler.resample(tex_rec)

        old_verts = verts_rec
        verts_rec = relight_preds["verts_displaced"]
        preds.update(
            {
                'geom': verts_rec,
                'tex_rec': tex_rec,
            }
        )

        if self.decoder_relight.refine_geo:
            preds.update(displacement = relight_preds["displacement"])


        edge_grad = self.training
        if not th.jit.is_scripting() and self.rendering_enabled:
            tex_seg = th.ones_like(tex_rec[:, :1])
            tex_rgb_seg = th.cat([tex_rec, tex_seg], dim=1)
            phys_tex_rgb_seg = th.cat([phys_tex_rec, tex_seg], dim=1)
            green_background = th.zeros(verts_rec.shape[0], 4, self.renderer.h, self.renderer.w).to(verts_rec)
            if not self.training and not self.force_black_bkgd:
                green_background[:, 1, :, :] += 255.

            vn = vert_normals(old_verts, self.geo_fn.vi[np.newaxis].to(th.int64))
            vn = th.bmm(vn, Rt[:, :3, :3].permute(0, 2, 1)).contiguous()
            tmp_normal = relight_preds['feature_normal_raw'].detach()
            tmp_normal = th.bmm(tmp_normal.permute(0,  2, 3, 1).reshape(bs, -1, 3), Rt[:, :3, :3].permute(0, 2, 1))
            tmp_normal = tmp_normal.reshape(bs, 1024, 1024, 3).permute(0, 3, 1, 2)
            feat_normal_raw = (1 - tmp_normal) * 127.5
            feat_normal_raw = th.cat([feat_normal_raw, th.ones_like(feat_normal_raw[:, :1])], dim=1)
            feat_render_preds = self.renderer(
                old_verts,
                feat_normal_raw,
                K=K,
                Rt=Rt,
                background=green_background,
                vn=vn,
                output_filters=["render"],
            )
            rendered_feat_normal_raw = feat_render_preds['render'][:, :3]
            uhm_geo_mask = feat_render_preds['render'][:, 3:4]
            preds.update(old_normals=rendered_feat_normal_raw, uhm_geo_mask=uhm_geo_mask, old_normals_uv = feat_normal_raw)

            # computing normals in camera space
            vn = vert_normals(verts_rec, self.geo_fn.vi[np.newaxis].to(th.int64))
            vn = th.bmm(vn, Rt[:, :3, :3].permute(0, 2, 1)).contiguous()
            tmp_normal = relight_preds['feature_normal'].detach()
            tmp_normal = th.bmm(tmp_normal.permute(0,  2, 3, 1).reshape(bs, -1, 3), Rt[:, :3, :3].permute(0, 2, 1))
            tmp_normal = tmp_normal.reshape(bs, 1024, 1024, 3).permute(0, 3, 1, 2)
            feat_normal = (1 - tmp_normal) * 127.5
            feat_normal = th.cat([feat_normal, th.ones_like(feat_normal[:, :1])], dim=1)
            feat_render_preds = self.renderer(
                verts_rec,
                feat_normal,
                K=K,
                Rt=Rt,
                background=green_background,
                vn=vn,
                output_filters=["render"],
            )
            rendered_feat_normal = feat_render_preds['render'][:, :3]
            preds.update(normals=rendered_feat_normal, normals_uv=feat_normal)

            render_preds = self.renderer(
                verts_rec,
                tex_rgb_seg,
                K=K,
                Rt=Rt,
                background=green_background,
                vn=vn,
                output_filters=[
                    "render",
                    "depth_img",
                    "mask",
                    "alpha",
                    "index_img",
                    "bary_img",
                    "v_pix",
                    "vn_img",
                ],
                edge_grad = edge_grad,
            )
            rgb_seg = render_preds["render"][:, :4].contiguous()

            phys_render_preds = self.renderer(
                verts_rec,
                phys_tex_rgb_seg,
                K=K,
                Rt=Rt,
                background=green_background,
                vn=vn,
                output_filters=[
                    "render",
                    "depth_img",
                    "mask",
                    "alpha",
                    "index_img",
                    "bary_img",
                    "v_pix",
                    "vn_img",
                ],
                edge_grad = edge_grad,
            )
            phys_rgb_seg = phys_render_preds["render"][:, :4].contiguous()

            if self.vis_feature:
                diff_feat = relight_preds['diff_feature'].detach() * 255.
                feat_render_preds = self.renderer(
                    verts_rec,
                    diff_feat,
                    K=K,
                    Rt=Rt,
                    background=green_background,
                    vn=vn,
                    output_filters=["render"],
                )
                rendered_diff_feat = feat_render_preds['render'][:, :3]

                if self.decoder_relight.shadow:
                    shadow_map = relight_preds['shadow'].detach().sum(1) * 255.
                    feat_render_preds = self.renderer(
                        verts_rec,
                        shadow_map,
                        K=K,
                        Rt=Rt,
                        background=green_background,
                        vn=vn,
                        output_filters=["render"],
                    )
                    rendered_shadow_map = feat_render_preds['render'][:, :3]
                else:
                    rendered_shadow_map = None

                rendered_spec_feat = []
                for spec_pow in [0, 2, len(self.decoder_relight.spec_powers) - 1]:
                    uv_spec_feature = relight_preds['spec_feature'][:, spec_pow, ...].detach() * 255.
                    feat_render_preds = self.renderer(
                        verts_rec,
                        uv_spec_feature,
                        K=K,
                        Rt=Rt,
                        background=green_background,
                        vn=vn,
                        output_filters=["render"],
                    )
                    rendered_spec_feat.append(feat_render_preds['render'][:, :3])
                preds.update(
                    diff_feat=rendered_diff_feat,
                    spec_feat=th.stack(rendered_spec_feat, dim=1),
                    shadow_map=rendered_shadow_map,
                    diff_feat_uv=diff_feat,
                    spec_feat_uv=relight_preds['spec_feature'].detach() * 255.,
                )

                # raw vertices
                if self.decoder_relight.shadow:
                    shadow_map = relight_preds['shadow_raw'].detach().sum(1) * 255.
                    feat_render_preds = self.renderer(
                        old_verts,
                        shadow_map,
                        K=K,
                        Rt=Rt,
                        background=green_background,
                        vn=vn,
                        output_filters=["render"],
                    )
                    rendered_shadow_map_raw = feat_render_preds['render'][:, :3]
                else:
                    rendered_shadow_map_raw = None

                diff_feat = relight_preds['diff_feature_raw'].detach() * 255.
                feat_render_preds = self.renderer(
                    old_verts,
                    diff_feat,
                    K=K,
                    Rt=Rt,
                    background=green_background,
                    vn=vn,
                    output_filters=["render"],
                )
                rendered_diff_feat_raw = feat_render_preds['render'][:, :3]
                rendered_spec_feat_raw = []
                for spec_pow in [0, 2, len(self.decoder_relight.spec_powers) - 1]:
                    uv_spec_feature = relight_preds['spec_feature_raw'][:, spec_pow, ...].detach() * 255.
                    feat_render_preds = self.renderer(
                        old_verts,
                        uv_spec_feature,
                        K=K,
                        Rt=Rt,
                        background=green_background,
                        vn=vn,
                        output_filters=["render"],
                    )
                    rendered_spec_feat_raw.append(feat_render_preds['render'][:, :3])
                preds.update(
                    diff_feat_raw=rendered_diff_feat_raw,
                    spec_feat_raw=th.stack(rendered_spec_feat_raw, dim=1),
                    shadow_map_raw=rendered_shadow_map_raw,
                )

                for k, v in interim.items():
                    if v is not None:
                        if v.shape[1] == 1:
                            v_seg = v
                        elif v.shape[1] == 3:
                            seg_temp = th.ones_like(v[:, :1])
                            v_seg = th.cat([v, seg_temp], dim=1)
                        interm_render_preds = self.renderer(
                            verts_rec,
                            v_seg,
                            K=K,
                            Rt=Rt,
                            background=green_background,
                            vn=vn,
                            output_filters=["render"],
                        )
                        rendered_interm = interm_render_preds['render'][:, :3]
                        preds[k] = rendered_interm
                        preds[k+'_uv'] = v

            rgb = rgb_seg[:, :3]
            seg = rgb_seg[:, 3:4]
            phys_rgb = phys_rgb_seg[:, :3]

            if self.blur_enable:
                blur_padding = int((self.blur_size - 1) // 2)
                rgb_blur = th.nn.functional.conv2d(
                    rgb, self.blur_kernel, padding=blur_padding, groups=3
                )
                preds.update(rendered_rgb_blur=rgb_blur)

            preds.update(
                depth=render_preds['depth_img'],
                rendered_rgb=rgb,
                rendered_mask=seg,
                rendered_phys_rgb=phys_rgb,
            )

            depth = render_preds["depth_img"].detach()[:, np.newaxis]
            depth_disc_mask = depth_discontuity_mask(depth)
            preds.update(depth_disc_mask=depth_disc_mask)
        return preds

class URHandSummary(Callable):
    def __call__(
        self, preds: Dict[str, Any], batch: Dict[str, Any], alpha = 0.75, nrow = 16, ccm=False
    ) -> Dict[str, th.Tensor]:
        filters = [
            "rgb",
            "phys_rgb",
            "rgb_gt",
            "diff_rgb",
            "old_normals",
            "normals",
            "diff_feat",
            "spec_feat1",
            "spec_feat2",
            "tex_mean_vis",
            "gain",
            "bias",
            "roughness",
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        font_thickness = 3
        text_color = (255, 0, 0)  # red color

        if ccm:
            rgb = linear2displayBatch(preds["rendered_rgb"][:, :3], gamma=2.4, wbscale=np.array([1.0, 1.0, 1.0], dtype=np.float32), black=0.0)
            phys_rgb = linear2displayBatch(preds["rendered_phys_rgb"][:, :3], gamma=2.4, wbscale=np.array([1.0, 1.0, 1.0], dtype=np.float32), black=0.0)
            rgb_gt = linear2displayBatch(batch["image"], gamma=2.4, wbscale=np.array([1.0, 1.0, 1.0], dtype=np.float32), black=0.0)
        else:
            rgb = linear2displayBatch(preds["rendered_rgb"][:, :3])
            phys_rgb = linear2displayBatch(preds["rendered_phys_rgb"][:, :3])
            rgb_gt = linear2displayBatch(batch["image"])
        diff_feat = preds["diff_feat"]
        spec_feat1 = preds["spec_feat"][:, 0] * 10
        spec_feat2 = preds["spec_feat"][:, 1] * 10
        tex_mean_vis = preds["tex_mean_vis"]
        gain = preds["gain"]
        bias = preds["bias"]
        roughness = preds["roughness"]
        mask_preds = preds["rendered_mask"].detach()
        green_background = th.zeros_like(rgb_gt)
        green_background[:, 1, :, :] += 255.
        rgb = rgb * mask_preds + green_background * (1 - mask_preds)
        phys_rgb = phys_rgb * mask_preds + green_background * (1 - mask_preds)

        diff_rgb = ((preds["rendered_rgb"] - batch["image"])).abs().mean(dim=1, keepdims=True)
        diff_rgb_error = (diff_rgb * preds["rendered_mask"]).mean(dim=[2,3])
        grid_diff_rgb = (
            make_grid(diff_rgb, nrow=nrow).permute(1, 2, 0).to(th.uint8).cpu().numpy()[..., 0]
        )
        tmp_diff_rgb = np.ascontiguousarray(cv2.applyColorMap(grid_diff_rgb, cv2.COLORMAP_JET)[..., ::-1]).copy()
        cv2.putText(tmp_diff_rgb, "{}".format(diff_rgb_error.detach().cpu().numpy()), (200, 200), font, font_scale, text_color, font_thickness)
        cv2.putText(tmp_diff_rgb, "{}".format(batch['_index']['frame'].detach().cpu().numpy()), (200, 400), font, font_scale, text_color, font_thickness)
        cv2.putText(tmp_diff_rgb, " ".join(batch['_index']['camera']), (200, 600), font, font_scale, text_color, font_thickness)
        capid = []
        for cid in batch['_index']['capture_id']:
            capid.append(cid[19:25])
        cv2.putText(tmp_diff_rgb, " ".join(capid), (200, 800), font, font_scale, text_color, font_thickness)
        grid_diff_rgb = th.as_tensor(
            tmp_diff_rgb,
            device=rgb.device,
            dtype=th.uint8,
        )

        # is mask useful?
        diff_mask = 255.0 * alpha * preds["rendered_mask"] + (1.0 - alpha) * rgb_gt
        grid_diff_mask = make_grid(diff_mask, nrow=nrow).permute(1, 2, 0).to(th.uint8)

        grid_rgb = make_grid(rgb, nrow=nrow).permute(1, 2, 0).clip(0, 255).to(th.uint8)
        grid_phys_rgb = make_grid(phys_rgb, nrow=nrow).permute(1, 2, 0).clip(0, 255).to(th.uint8)
        grid_rgb_gt = make_grid(rgb_gt, nrow=nrow).permute(1, 2, 0).clip(0, 255).to(th.uint8)
        normals = preds["normals"].detach().clip(0, 255) * mask_preds + (1.0 - mask_preds) * rgb_gt * (
            1.0 - alpha
        )
        old_normals = preds["old_normals"].detach().clip(0, 255) * mask_preds + (1.0 - mask_preds) * rgb_gt * (
            1.0 - alpha
        )
        grid_diff_feat = make_grid(diff_feat, nrow=nrow).permute(1, 2, 0).clip(0, 255).to(th.uint8)
        grid_spec_feat1 = make_grid(spec_feat1, nrow=nrow).permute(1, 2, 0).clip(0, 255).to(th.uint8)
        grid_spec_feat2 = make_grid(spec_feat2, nrow=nrow).permute(1, 2, 0).clip(0, 255).to(th.uint8)
        grid_tex_mean = make_grid(tex_mean_vis, nrow=nrow).permute(1, 2, 0).clip(0, 255).to(th.uint8)
        grid_gain = make_grid(gain, nrow=nrow).permute(1, 2, 0).clip(0, 255).to(th.uint8)
        grid_bias = make_grid(bias, nrow=nrow).permute(1, 2, 0).clip(0, 255).to(th.uint8)
        grid_roughness = make_grid(roughness, nrow=nrow).permute(1, 2, 0).clip(0, 255).to(th.uint8)

        grid_normals = make_grid(normals, nrow=nrow).permute(1, 2, 0).clip(0, 255).to(th.uint8)
        grid_old_normals = make_grid(old_normals, nrow=nrow).permute(1, 2, 0).clip(0, 255).to(th.uint8)

        outputs = [
            ("rgb", grid_rgb),
            ("phys_rgb", grid_phys_rgb),
            ("rgb_gt", grid_rgb_gt),
            ("diff_rgb", grid_diff_rgb),
            ("old_normals", grid_old_normals),
            ("normals", grid_normals),
            ("diff_mask", grid_diff_mask),
            ("diff_feat", grid_diff_feat),
            ("spec_feat1", grid_spec_feat1),
            ("spec_feat2", grid_spec_feat2),
            ("tex_mean_vis", grid_tex_mean),
            ("gain", grid_gain),
            ("bias", grid_bias),
            ("roughness", grid_roughness),
        ]

        progress_image = []
        for name, tensor in outputs:
            if name in filters:
                progress_image.append(tensor)

        progress_image = th.cat(
            progress_image,
            dim=0,
        )

        texture = preds["tex_rec"][0].permute(1, 2, 0)
        texture = texture.clip(0, 255).to(th.uint8)

        summaries = {
            "progress_image": (progress_image, "png"),
            "texture": (texture, "png"),
        }
        return summaries