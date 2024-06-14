# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch as th
import torch.nn.functional as thf

try:
    from . import sgutilslib
except:
    import sgutilslib

class EvaluateGaussian(th.autograd.Function):
    """Custom Function for raymarching Mixture of Volumetric Primitives."""
    @staticmethod
    def forward(ctx, lobe_dirs, lobe_sigmas, light_values, light_pts, prim_pts, n_lights, w_type):
        N, D = lobe_dirs.shape[:2]
        C = light_values.shape[-1]
        assert C == 3
        assert w_type in [0, 1, 2, 3]

        integral = th.empty(N, D, C, device=lobe_dirs.device)
        sgutilslib.evaluate_gaussian_fwd(
            lobe_dirs, lobe_sigmas, light_values, light_pts, prim_pts, n_lights, integral, w_type
        )
        ctx.mark_non_differentiable(light_pts, prim_pts, n_lights)
        ctx.save_for_backward(lobe_dirs, lobe_sigmas, light_values, light_pts, prim_pts, n_lights)
        ctx.w_type = w_type
        return integral

    @staticmethod
    def backward(ctx, grad_integral):
        lobe_dirs, lobe_sigmas, light_values, light_pts, prim_pts, n_lights = ctx.saved_tensors
        w_type = ctx.w_type
        N, D = lobe_dirs.shape[:2]
        L, C = light_values.shape[-2:]
        assert C == 3

        grad_lobe_dirs = th.zeros(lobe_dirs.shape, device=light_pts.device)
        grad_lobe_sigmas = th.zeros(lobe_sigmas.shape, device=light_pts.device)
        grad_light_values: Optional[th.Tensor] = None
        if light_values.requires_grad:
            grad_light_values = th.zeros(light_values.shape, device=light_pts.device)

        # TODO: Make the CUDA code able to handle strided inputs w/o contiguous() calls.
        sgutilslib.evaluate_gaussian_bwd(
            lobe_dirs,
            lobe_sigmas,
            light_values,
            light_pts,
            prim_pts,
            n_lights,
            grad_integral.contiguous(),
            grad_lobe_dirs,
            grad_lobe_sigmas,
            grad_light_values,
            w_type
        )
        return grad_lobe_dirs, grad_lobe_sigmas, grad_light_values, None, None, None, None

def evaluate_gaussian(
    lobe_dirs,
    lobe_sigmas,
    light_values,
    light_pts,
    prim_pts,
    n_lights,
    w_type: int = 0,
    normalize_lobe_dirs: bool = True
):
    if normalize_lobe_dirs:
        lobe_dirs = thf.normalize(lobe_dirs, dim=-1)

    assert lobe_dirs.shape[-1] == 3, lobe_dirs.shape[-1]
    assert (lobe_sigmas.dim() == 2 or lobe_sigmas.shape[2] == 1), lobe_sigmas.dim()
    assert light_pts.shape[-1] == 3, light_pts.shape[-1]
    assert light_pts.dim() == 3, light_pts.dim()
    assert prim_pts.shape[-1] == 3, prim_pts.shape[-1]
    assert prim_pts.dim() == 3, prim_pts.dim()

    if th.jit.is_scripting():
        N, D = lobe_dirs.shape[:2]
        C = light_values.shape[-1]
        assert C == 3

        integral = th.empty(N, D, C, device=lobe_dirs.device)
        sgutilslib.evaluate_gaussian_fwd(
            lobe_dirs, lobe_sigmas, light_values, light_pts, prim_pts, n_lights, integral, w_type
        )
        return integral
    else:
        return EvaluateGaussian.apply(
            lobe_dirs, lobe_sigmas, light_values, light_pts, prim_pts, n_lights, w_type
        )
