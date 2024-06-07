from typing import Optional

import torch as th
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians

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
    means3D = primpos.view(-1, 3).contiguous()
    scales = primscale.view(-1, 3).contiguous()
    rotations = primqvec.view(-1, 4).contiguous()
    opacity = opacity.view(-1, 1).contiguous()
    colors = colors.view(-1, 3).contiguous()

    if bg_color is None:
        bg_color = th.zeros(3, device=Rt.device)

    (
        xys,
        depths,
        radii,
        conics,
        compensation,
        num_tiles_hit,
        cov3d,
    ) = project_gaussians(
        means3D,
        scales,
        global_scale,
        rotations,
        Rt,
        fx,
        fy,
        cx,
        cy,
        cam_img_h,
        cam_img_w,
        block_width,
        z_near,
    )

    out_img, alpha = rasterize_gaussians(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                colors,
                opacity * compensation[:, None],
                cam_img_h,
                cam_img_w,
                block_width,
                bg_color,
                return_alpha=True
            )
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
        out_depth = rasterize_gaussians(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                depths[:, None].expand(-1, 3).contiguous(),
                opacity * compensation[:, None],
                cam_img_h,
                cam_img_w,
                block_width,
                bg_color,
                return_alpha=True
            )[0]
        depth = out_depth[..., 0]
        out["depth"] = depth[None]

    return out
