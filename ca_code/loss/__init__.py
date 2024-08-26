# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-ignore-all-errors

import copy
from glob import glob

from typing import Any, Mapping, Optional

import numpy as np

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ca_code.loss.registry import (  # noqa
    get_loss,
    logger,
    loss_registry,  # noqa
    register_loss,  # noqa
    register_loss_by_fn,  # noqa
)

from ca_code.loss import perceptual

from ca_code.utils.image import erode
from ca_code.utils.module_loader import load_from_config, load_module
from ca_code.utils.ssim import ssim
from omegaconf import DictConfig


class StepWeightSchedule:
    def __init__(self, start: int, end: int, value: float):
        self.start = start
        self.end = end
        self.value = value

    def __call__(self, iteration: int):
        if iteration < self.start:
            return 0.0
        elif iteration > self.end:
            return 0.0
        return self.value


class MonotonicWeightSchedule:
    def __init__(self, start: int, end: int, init_value: float, target_value: float):
        self.start = start
        self.end = end
        self.init_value = init_value
        self.target_value = target_value
        self.delta = (target_value - init_value) / (end - start)

    def __call__(self, iteration: int):
        if iteration < self.start:
            return self.init_value
        elif iteration > self.end:
            return self.target_value
        t = min(iteration, self.end) - self.start
        return self.init_value + t * self.delta


class CyclicWeightSchedule:
    def __init__(self, cycle: int, min_value: float, max_value: float):
        self.cycle = cycle
        self.min_value = min_value
        self.max_value = max_value
        self.delta = (max_value - min_value) / cycle

    def __call__(self, iteration: int) -> float:
        # iteration within each cycle
        it = iteration % (self.cycle * 2)
        return min(self.min_value + self.delta * it, self.max_value)


class ModularLoss(nn.Module):
    def __init__(self, losses, assets, extra_modules_loaded=None):
        super().__init__()

        if extra_modules_loaded is None:
            extra_modules_loaded = []

        # constructing
        self.weights = {}
        self.args = {}
        self.start_at = {}
        self.end_at = {}
        self.schedule = {}
        # TODO: should this be a module dict?
        self.fns = nn.ModuleDict()
        # load potential loss modules
        for extra_module in extra_modules_loaded:
            load_module(extra_module)
        # load losses from config `loss.losses`
        for loss_name, loss_def in losses.items():
            loss_def = copy.deepcopy(loss_def)

            # TODO: probably can get rid of this, keeping for compat-ty
            # get kwargs for arguments of loss class constructor
            if isinstance(loss_def, DictConfig):
                loss_init_kwargs = loss_def.pop("init_kwargs", {})
            else:
                loss_init_kwargs = {}

            # load the loss class
            loss_class_name = loss_name
            if isinstance(loss_def, DictConfig) and "loss_type" in loss_def:
                loss_class_name = loss_def.pop("loss_type")

            # decide the weights, etc.
            if isinstance(loss_def, DictConfig):
                assert "weight" in loss_def or "schedule" in loss_def
                if "weight" in loss_def:
                    self.weights[loss_name] = float(loss_def.pop("weight"))
                elif "schedule" in loss_def:
                    self.schedule[loss_name] = load_from_config(
                        loss_def.pop("schedule")
                    )
                if "start_at" in loss_def:
                    self.start_at[loss_name] = loss_def.pop("start_at")
                if "end_at" in loss_def:
                    self.end_at[loss_name] = loss_def.pop("end_at")
                if loss_def:
                    loss_init_kwargs.update(**loss_def)
            elif isinstance(loss_def, (float, int)):
                self.weights[loss_name] = float(loss_def)
            else:
                raise ValueError("unsupported loss definition")

            self.fns[loss_name] = get_loss(loss_class_name, assets, loss_init_kwargs)

    def forward(self, preds, targets, iteration=None):
        loss_total = 0.0
        losses_dict = {"loss_total": loss_total}
        for loss_name, loss_fn in self.fns.items():
            args = self.args.get(loss_name, {})
            loss_value = loss_fn(preds, targets, **args)
            losses_dict[f"loss_{loss_name}"] = loss_value
            if (
                iteration is not None
                and loss_name in self.start_at
                and iteration < self.start_at[loss_name]
            ):
                # TODO: we should make this more generic and specify this in register_loss?
                if loss_name == "rgb_l1" or loss_name == "rgb_face":
                    loss_total += loss_value * 0.0
                continue
            if (
                iteration is not None
                and loss_name in self.end_at
                and iteration > self.end_at[loss_name]
            ):
                if loss_name == "rgb_l1" or loss_name == "rgb_face":
                    loss_total += loss_value * 0.0
                continue

            if loss_name in self.weights:
                loss_total += self.weights[loss_name] * loss_value
            elif loss_name in self.schedule:
                assert (
                    iteration is not None
                ), "Provide `iteration` when using schedules!"
                weight = self.schedule[loss_name](iteration)
                loss_total += weight * loss_value
            else:
                logger.warning(f"no weight or schedule specified for {loss_name}")

        losses_dict.update(loss_total=loss_total)

        return loss_total, losses_dict


def kl_loss(mu, logvar):
    return -0.5 * th.mean(1.0 + logvar - mu**2 - th.exp(logvar))


def compute_laplacian(
    x: th.Tensor, nbs_idxs: th.Tensor, nbs_weights: th.Tensor
) -> th.Tensor:
    # TODO: check if this is sufficiently fast?
    return x + (x[:, nbs_idxs] * nbs_weights[np.newaxis, :, :, np.newaxis]).sum(2)


@register_loss("geom_lap")
class LaplacianLoss(nn.Module):
    def __init__(
        self, assets, src_key: str = "geom", tgt_key: str = "registration_vertices"
    ) -> None:
        super().__init__()
        # NOTE: using a slightly confusing notation where:
        # `src` - prediction
        # `tgt` - target (inside of `batch` or `targets`)
        # `dst` - target (inside of `preds`)
        self.src_key = src_key
        self.tgt_key = tgt_key
        self.register_buffer("nbs_idxs", th.as_tensor(assets.topology.nbs_idxs))
        self.register_buffer("nbs_weights", th.as_tensor(assets.topology.nbs_weights))

    def forward(self, preds, targets):
        l_preds = compute_laplacian(
            preds[self.src_key], self.nbs_idxs, self.nbs_weights
        )
        l_targets = compute_laplacian(
            targets[self.tgt_key], self.nbs_idxs, self.nbs_weights
        )
        return (l_preds - l_targets).pow(2).mean()


@register_loss("geom_lap_penalty")
class LaplacianPenaltyLoss(nn.Module):
    def __init__(self, assets, src_key: str = "geom") -> None:
        super().__init__()
        self.src_key = src_key
        self.register_buffer("nbs_idxs", th.as_tensor(assets.topology.nbs_idxs))
        self.register_buffer("nbs_weights", th.as_tensor(assets.topology.nbs_weights))

    def forward(self, preds, targets):
        x = preds[self.src_key]
        return (
            (
                (
                    x
                    + (
                        x[:, self.nbs_idxs]
                        * self.nbs_weights[np.newaxis, ..., np.newaxis]
                    ).sum(2)
                )
                ** 2
            )
            .mean(dim=-1)
            .mean()
        )


@register_loss("geom_lap_template")
class LaplacianTemplateLoss(nn.Module):
    def __init__(
        self, assets, src_key: str = "geom", dst_key: str = "geom_template_posed"
    ) -> None:
        super().__init__()
        self.src_key = src_key
        self.dst_key = dst_key
        self.register_buffer("nbs_idxs", th.as_tensor(assets.topology.nbs_idxs))
        self.register_buffer("nbs_weights", th.as_tensor(assets.topology.nbs_weights))

    def forward(self, preds, targets):
        l_preds = compute_laplacian(
            preds[self.src_key], self.nbs_idxs, self.nbs_weights
        )
        l_targets = compute_laplacian(
            preds[self.dst_key].detach(), self.nbs_idxs, self.nbs_weights
        )
        return (l_preds - l_targets).pow(2).mean()


@register_loss_by_fn("geom_l2")
def loss_geom_l2(
    preds: Mapping[str, Any],
    targets: Mapping[str, Any],
    src_key: str = "geom",
    tgt_key: str = "registration_vertices",
    dst_key: Optional[str] = None,
) -> th.Tensor:
    src = preds[src_key]
    tgt = preds[dst_key].detach() if dst_key is not None else targets[tgt_key]
    return (src - tgt).pow(2).mean()


# TODO: switch to standalone `loss_fn` lib and make sure we can do same losses multiple times
@register_loss("region_geom_l2")
class RegionGeomL2Loss(nn.Module):
    def __init__(
        self,
        assets,
        region_mask_name: str,
        src_key: str = "geom",
        tgt_key: str = "registration_vertices",
    ):
        super().__init__()
        self.src_key = src_key
        self.tgt_key = tgt_key
        self.register_buffer("weight_mask", th.as_tensor(assets[region_mask_name]))

    def forward(self, preds, batch):
        return (
            (
                (preds[self.src_key] - batch[self.tgt_key])
                * self.weight_mask[np.newaxis, :, np.newaxis]
            )
            .pow(2)
            .mean()
        )


@register_loss("region_lap")
class RegionLaplacianLoss(nn.Module):
    def __init__(
        self,
        assets,
        region_mask_name: str,
        src_key: str = "geom",
        tgt_key: str = "registration_vertices",
    ):
        super().__init__()
        self.src_key = src_key
        self.tgt_key = tgt_key
        self.register_buffer("weight_mask", th.as_tensor(assets[region_mask_name]))
        self.register_buffer("nbs_idxs", th.as_tensor(assets.topology.nbs_idxs))
        self.register_buffer("nbs_weights", th.as_tensor(assets.topology.nbs_weights))

    def forward(self, preds, batch):
        l_preds = compute_laplacian(
            preds[self.src_key], self.nbs_idxs, self.nbs_weights
        )
        l_targets = compute_laplacian(
            batch[self.tgt_key], self.nbs_idxs, self.nbs_weights
        )
        return (
            ((l_preds - l_targets) * self.weight_mask[np.newaxis, :, np.newaxis])
            .pow(2)
            .mean()
        )


@register_loss("region_lap_penalty")
class RegionLaplacianPenaltyLoss(nn.Module):
    def __init__(self, assets, region_mask_name: str, src_key: str = "geom"):
        super().__init__()
        self.src_key = src_key
        self.register_buffer("weight_mask", th.as_tensor(assets[region_mask_name]))
        self.register_buffer("nbs_idxs", th.as_tensor(assets.topology.nbs_idxs))
        self.register_buffer("nbs_weights", th.as_tensor(assets.topology.nbs_weights))

    def forward(self, preds, batch):
        l_preds = compute_laplacian(
            preds[self.src_key], self.nbs_idxs, self.nbs_weights
        )
        return (l_preds * self.weight_mask[np.newaxis, :, np.newaxis]).pow(2).mean()


@register_loss("head_geom_l2")
class HeadGeomL2Loss(RegionGeomL2Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, region_mask_name="full_head_mask_geom")


@register_loss("head_lap")
class HeadLaplacianLoss(RegionLaplacianLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, region_mask_name="full_head_mask_geom")


@register_loss("mouth_eyes_lap_penalty")
class MouthEyesLaplacianLoss(RegionLaplacianPenaltyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, region_mask_name="mouth_eyes_mask_geom")


@register_loss_by_fn()
# TODO: we need to normalize this properly
def rgb_l2(
    preds,
    targets,
    src_key: str = "rendered_rgb",
    tgt_key: str = "image",
    mask_key: str = "image_mask",
    ddisc_key: str = "depth_disc_mask",
    mask_erode: Optional[int] = None,
):
    # TODO: should this be all defined with unique names?
    mask = targets.get(mask_key, preds.get(mask_key, None))
    if mask is None:
        mask = th.ones_like(preds[src_key])
    if mask_erode is not None:
        mask = erode(mask.to(th.float32), mask_erode).to(th.bool)
    if ddisc_key in preds:
        try:
            mask = mask * (1 - preds[ddisc_key])
        except Exception:
            mask = mask * ~preds[ddisc_key]
    return ((preds[src_key] - targets[tgt_key]) * mask).pow(2).mean()


@register_loss_by_fn()
# TODO: we need to normalize this properly
def rgb_l1(
    preds,
    targets,
    src_key: str = "rendered_rgb",
    tgt_key: str = "image",
    mask_key: str = "image_mask",
    ddisc_key: str = "depth_disc_mask",
    mask_erode: Optional[int] = None,
):
    # TODO: should this be all defined with unique names?
    mask = targets.get(mask_key, preds.get(mask_key, None))
    if mask is None:
        mask = th.ones_like(preds[src_key])
    if mask_erode is not None:
        mask = erode(mask.to(th.float32), mask_erode).to(th.bool)
    if ddisc_key in preds:
        try:
            mask = mask * (1 - preds[ddisc_key])
        except Exception:
            mask = mask * ~preds[ddisc_key]
    return ((preds[src_key] - targets[tgt_key]) * mask).abs().mean()


@register_loss_by_fn()
def psnr(
    preds,
    targets,
    src_key: str = "rendered_rgb",
    tgt_key: str = "image",
    mask_key: str = "image_mask",
    data_range: float = 1.,
    ddisc_key: str = "depth_disc_mask",
    mask_erode: Optional[int] = None,
):
    mask = targets.get(mask_key, preds.get(mask_key, None))
    if mask is None:
        mask = th.ones_like(preds[src_key])
    if mask_erode is not None:
        mask = erode(mask.to(th.float32), mask_erode).to(th.bool)
    if ddisc_key in preds:
        try:
            mask = mask * (1 - preds[ddisc_key])
        except Exception:
            mask = mask * ~preds[ddisc_key]

    msqerr = ((preds[src_key] - targets[tgt_key]) * mask).pow(2).mean()

    dev = preds[src_key].device
    
    base = th.tensor(10.).to(dev)
    data_range = th.tensor(data_range).to(dev)

    psnr_base_e = 2 * th.log(data_range) - th.log(msqerr)
    psnr_vals = psnr_base_e * (10 / th.log(base))
    return psnr_vals



@register_loss_by_fn()
def mask_l1(
    preds, targets, src_key: str = "rendered_mask", tgt_key: str = "image_mask"
):
    return ((preds[src_key] - targets[tgt_key])).abs().mean()


@register_loss("region_rgb_l1")
class RegionRGBL1Loss(nn.Module):
    def __init__(
        self,
        assets,
        src_key: str = "rgb",
        tgt_key: str = "image",
        mask_key: str = "seg_fg",
        region_mask_key: str = "rendered_region_mask",
    ) -> None:
        super().__init__()
        self.src_key = src_key
        self.tgt_key = tgt_key
        self.mask_key = mask_key
        self.region_mask_key = region_mask_key

    def forward(self, preds, targets):
        mask = targets[self.mask_key] * (preds[self.region_mask_key].detach())
        rgb_delta = preds[self.src_key] - targets[self.tgt_key]
        return (rgb_delta * mask).abs().sum() / (1.0 + mask.sum())


@register_loss_by_fn()
def rgb_ssim(
    preds,
    targets,
    src_key: str = "rendered_rgb",
    tgt_key: str = "image",
    mask_key: str = "image_mask",
    normalize_mask: bool = True,
):
    mask = targets.get(mask_key, preds.get(mask_key, None))
    if mask is None:
        mask = th.ones_like(preds[src_key])

    if normalize_mask:
        return 1.0 - ssim(targets[tgt_key], preds[src_key], mask=mask)
    else:
        return 1.0 - ssim(mask * targets[tgt_key], mask * preds[src_key])


@register_loss_by_fn("learn_blur")
def learn_blur_reg_loss(preds, batch=None):
    return (preds["learn_blur_weights"] - 1.0).abs().mean()


@register_loss_by_fn("kl")
def loss_kl(preds, batch=None, prefix: str = "embs_"):
    return kl_loss(preds[f"{prefix}mu"], preds[f"{prefix}logvar"])


@register_loss_by_fn("face_kl")
def loss_kl(preds, batch=None, prefix: str = "face_embs_"):
    return kl_loss(preds[f"{prefix}mu"], preds[f"{prefix}logvar"])


@register_loss_by_fn("pose_shadow_l2")
def pose_to_shadow_l2_loss(preds, batch=None):
    return (preds["pose_shadow_map"] - preds["shadow_map"].detach()).pow(2.0).mean()


@register_loss_by_fn("bound_primscale")
def loss_bound_primscale(
    preds,
    batch=None,
    key: str = "primscale_preclip",
    min_scale: float = 0.1,
    max_scale: float = 20.0,
):
    primscale = preds[key]
    return th.where(
        primscale < min_scale,
        1.0 / primscale.clamp(1e-7, th.inf),
        th.where(primscale > max_scale, (primscale - max_scale) ** 2, 0.0),
    ).mean()


@register_loss_by_fn("negcolor")
def loss_negcolor(preds, batch=None, key: str = "diff_color"):
    return preds[key].clamp(max=0.0).pow(2).mean()


@register_loss_by_fn("l2_reg")
def loss_l2_reg(preds, batch=None, key: str = "spec_dnml"):
    return preds[key].pow(2).mean()


@register_loss_by_fn("backlit_reg")
def loss_backlight_reg(
    preds,
    batch=None,
    key: str = "color_rand",
    cos_key: str = "cos_weight",
):
    weight = F.relu(-preds[cos_key]) ** 2
    return (weight * F.relu(preds[key])).sum() / (1.0 + weight.sum())


@register_loss_by_fn("primvolsum")
def loss_primvolsum(preds, batch=None, primscale_ref: float = 100.0):
    primscale = preds["primscale"]
    return th.mean(th.sum(th.prod(primscale_ref / primscale, dim=-1), dim=-1))


@register_loss_by_fn("alphaprior")
def loss_alphaprior(
    preds,
    batch=None,
    key: str = "alpha",
):
    alpha = preds[key]
    B = alpha.shape[0]

    return th.mean(
        th.log(0.1 + alpha.view(B, -1))
        + th.log(0.1 + 1.0 - alpha.view(B, -1))
        - -2.20727
    )
