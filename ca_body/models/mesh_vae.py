# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
from typing import Callable, Dict, Optional, Tuple, Any, List

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import make_grid
from torchvision.transforms.functional import gaussian_blur

import ca_body.nn.layers as la

from ca_body.nn.blocks import (
    ConvBlock,
    ConvDownBlock,
    UpConvBlockDeep,
    tile2d,
    weights_initializer,
)
from ca_body.nn.dof_cal import LearnableBlur

from ca_body.utils.geom import (
    GeometryModule,
    compute_view_cos,
    depth_discontuity_mask,
    depth2normals,
)

from ca_body.nn.shadow import ShadowUNet, PoseToShadow
from ca_body.nn.unet import UNetWB
from ca_body.nn.color_cal import CalV5

from ca_body.utils.image import linear2displayBatch

from ca_body.utils.lbs import LBSModule
# from care.models.body.lbs import LBSModule
from ca_body.utils.seams import SeamSampler

# from ca_body.utils.render_pytorch3d import RenderLayer
from ca_body.utils.render_drtk import RenderLayer

logger = logging.getLogger(__name__)


class CameraPixelBias(nn.Module):
    def __init__(self, image_height, image_width, cameras, ds_rate) -> None:
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.cameras = cameras
        self.n_cameras = len(cameras)

        bias = th.zeros(
            (self.n_cameras, 1, image_width // ds_rate, image_height // ds_rate),
            dtype=th.float32,
        )
        self.register_parameter("bias", nn.Parameter(bias))

    def forward(self, idxs: th.Tensor):
        bias_up = F.interpolate(
            self.bias[idxs], size=(self.image_height, self.image_width), mode="bilinear"
        )
        return bias_up


class AutoEncoder(nn.Module):
    def __init__(
        self,
        encoder,
        encoder_face,
        decoder,
        decoder_view,
        shadow_net,
        upscale_net,
        assets,
        pose_to_shadow=None,
        renderer=None,
        cal=None,
        pixel_cal=None,
        learn_blur: bool = True,
    ):
        super().__init__()
        # TODO: should we have a shared LBS here?

        self.geo_fn = GeometryModule(
            assets.topology.vi,
            assets.topology.vt,
            assets.topology.vti,
            assets.topology.v2uv,
            uv_size=1024,
            impaint=True,
        )

        self.lbs_fn = LBSModule(
            assets.lbs_model_json,
            assets.lbs_config_dict,
            assets.template_mesh[0],
            assets.skeleton_scales,
            assets.global_scaling,
        )

        self.seam_sampler = SeamSampler(assets.seam_data_1024)
        self.seam_sampler_2k = SeamSampler(assets.seam_data_2048)

        # joint tex -> body and clothes
        # TODO: why do we have a joint one in the first place?
        tex_mean = gaussian_blur(
            th.as_tensor(assets.color_mean, dtype=th.float32)[np.newaxis],
            kernel_size=11,
        )
        self.register_buffer(
            "tex_mean", F.interpolate(tex_mean, (2048, 2048), mode="bilinear")
        )

        # this is shared
        self.tex_std = assets.tex_var if "tex_var" in assets else 64.0

        face_cond_mask = th.as_tensor(assets.face_cond_mask, dtype=th.float32)[
            np.newaxis, np.newaxis
        ]
        self.register_buffer("face_cond_mask", face_cond_mask)

        meye_mask = self.geo_fn.to_uv(
            th.as_tensor(assets.mouth_eyes_mask_geom[np.newaxis, :, np.newaxis])
        )
        meye_mask = F.interpolate(meye_mask, (2048, 2048), mode="bilinear")
        self.register_buffer("meye_mask", meye_mask)

        self.decoder = ConvDecoder(
            geo_fn=self.geo_fn,
            seam_sampler=self.seam_sampler,
            **decoder,
            assets=assets,
        )

        # embs for everything but face
        non_head_mask = 1.0 - assets.face_mask
        self.encoder = Encoder(
            mask=non_head_mask,
            **encoder,
        )

        self.encoder_face = FaceEncoder(
            mask=assets.face_mask,
            **encoder_face,
        )

        self.decoder_view = UNetViewDecoder(
            self.geo_fn,
            seam_sampler=self.seam_sampler,
            **decoder_view,
        )

        self.shadow_net = ShadowUNet(
            ao_mean=assets.ambient_occlusion_mean,
            interp_mode="bilinear",
            biases=False,
            **shadow_net,
        )

        self.pose_to_shadow_enabled = False
        if pose_to_shadow is not None:
            self.pose_to_shadow_enabled = True
            self.pose_to_shadow = PoseToShadow(**pose_to_shadow)

        self.upscale_net = UpscaleNet(
            in_channels=6, size=1024, upscale_factor=2, out_channels=3, **upscale_net
        )

        self.pixel_cal_enabled = False
        if pixel_cal is not None:
            self.pixel_cal_enabled = True
            self.pixel_cal = CameraPixelBias(**pixel_cal, cameras=assets.camera_ids)

        self.learn_blur_enabled = False
        if learn_blur:
            self.learn_blur_enabled = True
            self.learn_blur = LearnableBlur(assets.camera_ids)

        # training-only stuff
        self.cal_enabled = False
        if cal is not None:
            self.cal_enabled = True
            self.cal = CalV5(**cal, cameras=assets.camera_ids)

        self.rendering_enabled = False
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

    def forward_tex(self, tex_mean_rec, tex_view_rec, shadow_map):
        x = th.cat([tex_mean_rec, tex_view_rec], dim=1)
        tex_rec = tex_mean_rec + tex_view_rec

        tex_rec = self.seam_sampler.impaint(tex_rec)
        tex_rec = self.seam_sampler.resample(tex_rec)

        tex_rec = F.interpolate(
            tex_rec, size=(2048, 2048), mode="bilinear", align_corners=False
        )
        tex_rec = tex_rec + self.upscale_net(x)

        tex_rec = tex_rec * self.tex_std + self.tex_mean

        shadow_map = self.seam_sampler_2k.impaint(shadow_map)
        shadow_map = self.seam_sampler_2k.resample(shadow_map)
        shadow_map = self.seam_sampler_2k.resample(shadow_map)

        tex_rec = tex_rec * shadow_map

        tex_rec = self.seam_sampler_2k.impaint(tex_rec)
        tex_rec = self.seam_sampler_2k.resample(tex_rec)
        tex_rec = self.seam_sampler_2k.resample(tex_rec)

        return tex_rec

    def encode(
        self, registration_vertices: th.Tensor, pose: th.Tensor
    ) -> Dict[str, th.Tensor]:
        with th.no_grad():
            verts_unposed = self.lbs_fn.unpose(registration_vertices, pose)
            verts_unposed_uv = self.geo_fn.to_uv(verts_unposed)
        # TODO: add separate face embeddings ?
        enc_preds = self.encoder(verts_unposed_uv)
        face_enc_preds = self.encoder_face(verts_unposed_uv)
        return {**enc_preds, **face_enc_preds}

    def forward(
        self,
        # TODO: should we try using this as well for cond?
        pose: th.Tensor,
        campos: th.Tensor,
        registration_vertices: Optional[th.Tensor] = None,
        ambient_occlusion: Optional[th.Tensor] = None,
        K: Optional[th.Tensor] = None,
        Rt: Optional[th.Tensor] = None,
        camera_id: Optional[List[str]] = None,
        frame_id: Optional[th.Tensor] = None,
        embs: Optional[th.Tensor] = None,
        encode: bool = True,
        iteration: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        B = pose.shape[0]

        if not th.jit.is_scripting() and encode:
            # NOTE: these are `face_embs_hqlp`
            enc_preds = self.encode(registration_vertices, pose)
            embs = enc_preds["embs"]
            face_embs = enc_preds["face_embs"]

        dec_preds = self.decoder(
            pose=pose,
            embs=embs,
            face_embs=face_embs,
        )

        geom_rec = self.lbs_fn.pose(dec_preds["geom_delta_rec"], pose)

        dec_view_preds = self.decoder_view(
            geom_rec=geom_rec,
            tex_mean_rec=dec_preds["tex_mean_rec"],
            camera_pos=campos,
        )

        if self.training and self.pose_to_shadow_enabled:
            shadow_preds = self.shadow_net(ao_map=ambient_occlusion)
            pose_shadow_preds = self.pose_to_shadow(pose)
            shadow_preds["pose_shadow_map"] = pose_shadow_preds["shadow_map"]
        elif self.pose_to_shadow_enabled:
            shadow_preds = self.pose_to_shadow(pose)
        else:
            shadow_preds = self.shadow_net(ao_map=ambient_occlusion)

        tex_rec = self.forward_tex(
            dec_preds["tex_mean_rec"],
            dec_view_preds["tex_view_rec"],
            shadow_preds["shadow_map"],
        )

        if not th.jit.is_scripting() and self.cal_enabled:
            tex_rec = self.cal(tex_rec, self.cal.name_to_idx(camera_id))

        preds = {
            "geom": geom_rec,
            "tex_rec": tex_rec,
            **dec_preds,
            **shadow_preds,
            **dec_view_preds,
        }

        if not th.jit.is_scripting() and encode:
            preds.update(**enc_preds)

        if not th.jit.is_scripting() and self.rendering_enabled:
            tex_seg = th.ones_like(tex_rec[:, :1])
            # NOTE: this is a reduced version tested for forward only
            renders = self.renderer(
                preds["geom"],
                tex=th.cat([tex_rec, tex_seg], dim=1),
                K=K,
                Rt=Rt,
            )

            render_rgb = renders["render"][:, :3]
            render_alpha = renders["render"][:, 3:]
            render_depth = renders["depth_img"][:, None].detach()

            depth_disc_mask = depth_discontuity_mask(render_depth)

            preds.update(
                rgb=render_rgb,
                alpha=render_alpha,
                depth_disc_mask=depth_disc_mask,
                depth=render_depth,
            )

        if not th.jit.is_scripting() and self.learn_blur_enabled:
            preds["rgb"] = self.learn_blur(preds["rgb"], camera_id)
            preds["learn_blur_weights"] = self.learn_blur.reg(camera_id)

        if not th.jit.is_scripting() and self.pixel_cal_enabled:
            assert self.cal_enabled
            cam_idxs = self.cal.name_to_idx(camera_id)
            pixel_bias = self.pixel_cal(cam_idxs)
            preds["rgb"] = preds["rgb"] + pixel_bias

        return preds


class Encoder(nn.Module):
    """A joint encoder for tex and geometry."""

    def __init__(
        self,
        n_embs: int,
        mask: th.Tensor,
        noise_std: float = 1.0,
        mean_scale: float = 0.1,
        logvar_scale: float = 0.1,
        verts_scale: float = 1.0,
    ):
        """Fixed-width conv encoder."""
        super().__init__()

        self.noise_std = noise_std
        self.n_embs = n_embs
        self.mean_scale = mean_scale
        self.logvar_scale = logvar_scale
        self.verts_scale = verts_scale

        self.verts_conv = ConvDownBlock(3, 8, 512)

        mask = th.as_tensor(mask[np.newaxis, np.newaxis], dtype=th.float32)
        mask = F.interpolate(mask, size=(512, 512), mode="bilinear").to(th.bool)
        self.register_buffer("mask", mask)

        self.joint_conv_blocks = nn.Sequential(
            ConvDownBlock(8, 16, 256),
            ConvDownBlock(16, 32, 128),
            ConvDownBlock(32, 32, 64),
            ConvDownBlock(32, 64, 32),
            ConvDownBlock(64, 128, 16),
            ConvDownBlock(128, 128, 8),
            # ConvDownBlock(128, 128, 4),
        )

        # TODO: should we put initializer
        self.mu = la.LinearWN(4 * 4 * 128, self.n_embs)
        self.logvar = la.LinearWN(4 * 4 * 128, self.n_embs)

        self.apply(lambda m: la.glorot(m, 0.2))
        la.glorot(self.mu, 1.0)
        la.glorot(self.logvar, 1.0)

    def forward(self, verts_unposed_uv: th.Tensor) -> Dict[str, th.Tensor]:
        preds = {}

        B = verts_unposed_uv.shape[0]

        verts_cond = (
            F.interpolate(
                verts_unposed_uv * self.verts_scale, size=(512, 512), mode="bilinear"
            )
            * self.mask
        )
        verts_cond = self.verts_conv(verts_cond)

        joint_cond = verts_cond
        x = self.joint_conv_blocks(joint_cond)
        x = x.reshape(B, -1)
        embs_mu = self.mean_scale * self.mu(x)
        embs_logvar = self.logvar_scale * self.logvar(x)

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


class FaceEncoder(nn.Module):
    def __init__(
        self,
        mask: th.Tensor,
        **kwargs,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(**kwargs, mask=mask[512:, :512])

    def forward(self, verts_unposed_uv: th.Tensor) -> Dict[str, th.Tensor]:
        face_verts_uv = verts_unposed_uv[:, :, 512:, :512]
        preds = self.encoder(face_verts_uv)
        return {f"face_{k}": v for k, v in preds.items()}


class ConvDecoder(nn.Module):

    def __init__(
        self,
        geo_fn,
        uv_size,
        seam_sampler,
        init_uv_size,
        n_pose_dims,
        n_pose_enc_channels,
        n_embs,
        n_embs_enc_channels,
        n_face_embs,
        n_init_channels,
        n_min_channels,
        assets,
        tex_scale: float = 0.001,
        verts_scale: float = 0.01,
    ):
        super().__init__()

        self.geo_fn = geo_fn

        self.tex_scale = tex_scale
        self.verts_scale = verts_scale

        self.uv_size = uv_size
        self.init_uv_size = init_uv_size
        self.n_pose_dims = n_pose_dims
        self.n_pose_enc_channels = n_pose_enc_channels
        self.n_embs = n_embs
        self.n_embs_enc_channels = n_embs_enc_channels
        self.n_face_embs = n_face_embs

        self.n_blocks = int(np.log2(self.uv_size // init_uv_size))
        self.sizes = [init_uv_size * 2**s for s in range(self.n_blocks + 1)]

        # TODO: just specify a sequence?
        self.n_channels = [
            max(n_init_channels // 2**b, n_min_channels)
            for b in range(self.n_blocks + 1)
        ]

        logger.info(f"ConvDecoder: n_channels = {self.n_channels}")

        self.local_pose_conv_block = ConvBlock(
            n_pose_dims,
            n_pose_enc_channels,
            init_uv_size,
            kernel_size=1,
            padding=0,
        )

        self.embs_fc = nn.Sequential(
            la.LinearWN(n_embs, 4 * 4 * 128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # TODO: should we switch to the basic version?
        self.embs_conv_block = nn.Sequential(
            UpConvBlockDeep(128, 128, 8),
            UpConvBlockDeep(128, 128, 16),
            UpConvBlockDeep(128, 64, 32),
            UpConvBlockDeep(64, n_embs_enc_channels, 64),
        )

        self.face_embs_fc = nn.Sequential(
            la.LinearWN(n_face_embs, 4 * 4 * 32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.face_embs_conv_block = nn.Sequential(
            UpConvBlockDeep(32, 64, 8),
            UpConvBlockDeep(64, 64, 16),
            UpConvBlockDeep(64, n_embs_enc_channels, 32),
        )

        n_groups = 2

        self.joint_conv_block = ConvBlock(
            n_pose_enc_channels + n_embs_enc_channels,
            n_init_channels,
            self.init_uv_size,
        )

        self.conv_blocks = nn.ModuleList([])
        for b in range(self.n_blocks):
            self.conv_blocks.append(
                UpConvBlockDeep(
                    self.n_channels[b] * n_groups,
                    self.n_channels[b + 1] * n_groups,
                    self.sizes[b + 1],
                    groups=n_groups,
                ),
            )

        self.verts_conv = la.Conv2dWNUB(
            in_channels=self.n_channels[-1],
            out_channels=3,
            kernel_size=3,
            height=self.uv_size,
            width=self.uv_size,
            padding=1,
        )
        self.tex_conv = la.Conv2dWNUB(
            in_channels=self.n_channels[-1],
            out_channels=3,
            kernel_size=3,
            height=self.uv_size,
            width=self.uv_size,
            padding=1,
        )

        self.apply(lambda x: la.glorot(x, 0.2))

        la.glorot(self.verts_conv, 1.0)
        la.glorot(self.tex_conv, 1.0)

        self.verts_scale = verts_scale
        self.tex_scale = tex_scale

        self.seam_sampler = seam_sampler

        # NOTE: removing head region from pose completely
        pose_cond_mask = th.as_tensor(
            assets.pose_cond_mask[np.newaxis]
            * (1 - assets.head_cond_mask[np.newaxis, np.newaxis]),
            dtype=th.int32,
        )
        self.register_buffer("pose_cond_mask", pose_cond_mask)
        face_cond_mask = th.as_tensor(assets.face_cond_mask, dtype=th.float32)[
            np.newaxis, np.newaxis
        ]
        self.register_buffer("face_cond_mask", face_cond_mask)

        body_cond_mask = th.as_tensor(assets.body_cond_mask, dtype=th.float32)[
            np.newaxis, np.newaxis
        ]
        self.register_buffer("body_cond_mask", body_cond_mask)

    def forward(
        self, pose: th.Tensor, embs: th.Tensor, face_embs: th.Tensor
    ) -> Dict[str, th.Tensor]:
        B = pose.shape[0]

        # processing pose
        local_pose = pose[:, 6:]

        non_head_mask = (self.body_cond_mask * (1.0 - self.face_cond_mask)).clip(
            0.0, 1.0
        )

        pose_masked = tile2d(local_pose, self.init_uv_size) * self.pose_cond_mask
        pose_conv = self.local_pose_conv_block(pose_masked) * non_head_mask

        embs_conv = self.embs_conv_block(self.embs_fc(embs).reshape(B, 128, 4, 4))

        face_conv = self.face_embs_conv_block(
            self.face_embs_fc(face_embs).reshape(B, 32, 4, 4)
        )
        # merging embeddings with spatial masks
        embs_conv[:, :, 32:, :32] = (
            face_conv * self.face_cond_mask[:, :, 32:, :32]
            + embs_conv[:, :, 32:, :32] * non_head_mask[:, :, 32:, :32]
        )

        joint = th.cat([pose_conv, embs_conv], axis=1)
        joint = self.joint_conv_block(joint)

        x = th.cat([joint, joint], axis=1)
        for b in range(self.n_blocks):
            x = self.conv_blocks[b](x)

        # NOTE: here we do resampling at feature level
        x = self.seam_sampler.impaint(x)
        x = self.seam_sampler.resample(x)
        x = self.seam_sampler.resample(x)

        verts_features, tex_features = th.split(x, self.n_channels[-1], 1)

        verts_uv_delta_rec = self.verts_conv(verts_features) * self.verts_scale
        # TODO: need to get values
        verts_delta_rec = self.geo_fn.from_uv(verts_uv_delta_rec)
        tex_mean_rec = self.tex_conv(tex_features) * self.tex_scale

        preds = {
            "geom_delta_rec": verts_delta_rec,
            "geom_uv_delta_rec": verts_uv_delta_rec,
            "tex_mean_rec": tex_mean_rec,
            "embs_conv": embs_conv,
            "pose_conv": pose_conv,
        }

        return preds


class UNetViewDecoder(nn.Module):
    def __init__(self, geo_fn, net_uv_size, seam_sampler, n_init_ftrs=8):
        super().__init__()
        self.geo_fn = geo_fn
        self.net_uv_size = net_uv_size
        self.unet = UNetWB(4, 3, n_init_ftrs=n_init_ftrs, size=net_uv_size)
        self.register_buffer("faces", self.geo_fn.vi.to(th.int64), persistent=False)

    def forward(self, geom_rec, tex_mean_rec, camera_pos):

        with th.no_grad():
            view_cos = compute_view_cos(geom_rec, self.faces, camera_pos)
            view_cos_uv = self.geo_fn.to_uv(view_cos[..., np.newaxis])
        cond_view = th.cat([view_cos_uv, tex_mean_rec], dim=1)
        tex_view = self.unet(cond_view)
        # TODO: should we try warping here?
        return {"tex_view_rec": tex_view, "cond_view": cond_view}


class UpscaleNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_ftrs, size=1024, upscale_factor=2):
        super().__init__()

        self.conv_block = nn.Sequential(
            la.Conv2dWNUB(in_channels, n_ftrs, size, size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.out_block = la.Conv2dWNUB(
            n_ftrs,
            out_channels * upscale_factor**2,
            size,
            size,
            kernel_size=1,
            padding=0,
        )

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=upscale_factor)

        self.apply(lambda m: la.glorot(m, 0.2))
        la.glorot(self.out_block, 1.0)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.out_block(x)
        return self.pixel_shuffle(x)


class MeshVAESummary(Callable):

    def __call__(
        self, preds: Dict[str, Any], batch: Dict[str, Any]
    ) -> Dict[str, th.Tensor]:

        rgb = linear2displayBatch(preds["rgb"][:, :3])
        rgb_gt = linear2displayBatch(batch["image"])
        depth = preds["depth"]

        mask = depth > 0.0
        normals = (
            255 * (1.0 - depth2normals(depth, batch["focal"], batch["princpt"])) / 2.0
        ) * mask
        grid_rgb = make_grid(rgb, nrow=16).permute(1, 2, 0).clip(0, 255).to(th.uint8)
        grid_rgb_gt = (
            make_grid(rgb_gt, nrow=16).permute(1, 2, 0).clip(0, 255).to(th.uint8)
        )
        grid_normals = (
            make_grid(normals, nrow=16).permute(1, 2, 0).clip(0, 255).to(th.uint8)
        )
        progress_image = th.cat([grid_rgb, grid_rgb_gt, grid_normals], dim=0)
        return {
            "progress_image": progress_image.permute(2, 0, 1),
        }
