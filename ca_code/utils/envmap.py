# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import List, Tuple

import cv2
import math
import numpy as np
import torch as th
import torch.nn.functional as thf
from torch.special import erfinv, erf


# Rodrigues Vectors
def rvec_to_R(rvec: th.Tensor) -> th.Tensor:
    """Computes the rotation matrix R from a tensor of Rodrigues vectors.

    n = ||rvec||
    rn = rvec/||rvec||
    N = [rn]_x = [[0, -rz, ry], [rz, 0, -rx], [-ry, rx, 0]]
    R = I + sin(n)*N + (1-cos(n))*N*N
    """
    n = rvec.norm(dim=-1, p=2).clamp(min=1e-6)[..., None, None]
    rn = rvec / n[..., :, 0]
    zero = th.zeros_like(n[..., 0, 0])
    N = th.stack(
        (
            zero,
            -rn[..., 2],
            rn[..., 1],
            rn[..., 2],
            zero,
            -rn[..., 0],
            -rn[..., 1],
            rn[..., 0],
            zero,
        ),
        -1,
    ).view(rvec.shape[:-1] + (3, 3))
    R = (
        th.eye(3, dtype=n.dtype, device=n.device).view([1] * (rvec.dim() - 1) + [3, 3])
        + th.sin(n) * N
        + ((1 - th.cos(n)) * N) @ N
    )
    return R


def rotx(theta: float) -> np.ndarray:
    """
    Produces a counter-clockwise 3D rotation matrix around axis X with angle `theta` in radians.
    """
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]], dtype='float32')


def roty(theta: float) -> np.ndarray:
    """
    Produces a counter-clockwise 3D rotation matrix around axis Y with angle `theta` in radians.
    """
    return np.array([[np.cos(theta), 0, -np.sin(theta)],
                     [0, 1, 0],
                     [np.sin(theta), 0, np.cos(theta)]], dtype='float32')


def rotz(theta: float) -> np.ndarray:
    """
    Produces a counter-clockwise 3D rotation matrix around axis Z with angle `theta` in radians.
    """
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]], dtype='float32')

def rotate_envmap(
    image: th.Tensor, rot_x: float = 0, rot_y: float = 0, rot_z: float = 0
) -> th.Tensor:
    # image: envmap, tensor 3 x H x W 
    # rot_x, rot_y, rot_z: in radians
    # return: 3 x H x W

    height = image.shape[1]
    width = image.shape[2]
    theta, phi = th.meshgrid(
            (th.arange(height, dtype=th.float32) + 0.5) * 3.1415926 / height,
            (th.arange(-width//2, width//2, dtype=th.float32) + 0.5) * 3.1415926 * 2 / width)

    # H x W x 3 
    vec = th.stack([th.sin(theta) * th.sin(phi), th.cos(theta), th.sin(theta) * th.cos(phi)], dim=-1).to(image.device)
    rot_mat = rotz(rot_z).dot(roty(rot_y)).dot(rotx(rot_x)) 
    rot_mat = th.from_numpy(rot_mat).T
    rot_mat = rot_mat.contiguous().to(image.device)

    # rot vec 
    vec = th.matmul(vec, rot_mat[None, :, :])
    vec = th.clamp(vec, -1, 1)

    u = (1 / np.pi) * th.atan2(vec[:, :, 0], vec[:, :, 2])  # range: [-1, 1]
    v = (1 / np.pi) * th.acos(vec[:, :, 1]) # range: [0, 1]
    v = 2 * v - 1.0

    coords = th.stack([u, v], -1)

    new_image = thf.grid_sample(image[None, :, :, :], coords[None, :, :, :], padding_mode="border") 
    new_image = new_image[0]

    return new_image

def rotate_envmap_vec(image: th.Tensor, rot_vec: th.Tensor) -> th.Tensor:
    # image: 3 x H x W tensor 
    # rot_vec: (4,), rotation_axis, rotation_angle 

    height = image.shape[1]
    width = image.shape[2]
    theta, phi = th.meshgrid(
          (th.arange(height, dtype=th.float32) + 0.5) * 3.1415926 / height,
          (th.arange(-width//2, width//2, dtype=th.float32) + 0.5) * 3.1415926 * 2 / width)
 
    # H x W x 3 
    vec = th.stack([th.sin(theta) * th.sin(phi), th.cos(theta), th.sin(theta) * th.cos(phi)], dim=-1).to(image.device)

    rot_vec = thf.normalize(rot_vec[0:3], dim=0) * rot_vec[3]
    rot_mat = rvec_to_R(rot_vec).t().contiguous()
    vec = th.matmul(vec, rot_mat[None, :, :])
    vec = th.clamp(vec, -1, 1)

    u = (1 / np.pi) * th.atan2(vec[:, :, 0], vec[:, :, 2])  # range: [-1, 1]
    v = (1 / np.pi) * th.acos(vec[:, :, 1]) # range: [0, 1]
    v = 2 * v - 1.0

    coords = th.stack([u, v], -1)
    new_image = thf.grid_sample(image[None, :, :, :], coords[None, :, :, :], padding_mode="border") 
    new_image = new_image[0]

    return new_image

def rotate_envmap_mat(image: th.Tensor, rot_mat: th.Tensor) -> th.Tensor:
    # image: 3 x H x W tensor 
    # rot_mat: (3,3), rotation matrix 

    height = image.shape[1]
    width = image.shape[2]
    theta, phi = th.meshgrid(
          (th.arange(height, dtype=th.float32) + 0.5) * 3.1415926 / height,
          (th.arange(-width//2, width//2, dtype=th.float32) + 0.5) * 3.1415926 * 2 / width)

    # H x W x 3 
    vec = th.stack([th.sin(theta) * th.sin(phi), th.cos(theta), th.sin(theta) * th.cos(phi)], dim=-1).to(image.device)

    rot_mat = rot_mat.T.float().contiguous()
    vec = th.matmul(vec, rot_mat[None, :, :])
    vec = th.clamp(vec, -1, 1)

    u = (1 / np.pi) * th.atan2(vec[:, :, 0], vec[:, :, 2])  # range: [-1, 1]
    v = (1 / np.pi) * th.acos(vec[:, :, 1]) # range: [0, 1]
    v = 2 * v - 1.0

    coords = th.stack([u, v], -1)
    new_image = thf.grid_sample(image[None, :, :, :], coords[None, :, :, :], padding_mode="border") 
    new_image = new_image[0]

    return new_image


def envmap_to_image(
    w: int,
    h: int,
    envbg: th.Tensor,
    princpt: th.Tensor,
    focal: th.Tensor,
    camrot: th.Tensor = None,
    focal_scale: float = 0.2,
    blurbg: bool = True,
    D: th.Tensor = None
) -> th.Tensor:
    py, px = th.meshgrid(th.arange(0, h), th.arange(0, w))
    pixelcoords = th.stack([px, py], -1)[None].expand(princpt.shape[0], -1, -1, -1).to(envbg.device)
    
    if D is not None:
        focal_np = focal.float().data.cpu().numpy()
        princpt_np = princpt.float().data.cpu().numpy()
        D_np = D.float().data.cpu().numpy()
        pix_coords = pixelcoords[0].float().data.cpu().numpy()

        pixel_coords = []
        for bi in range(princpt.shape[0]):
            Ki = np.concatenate([focal_np[bi], princpt_np[bi][:, None]], -1)
            Ki = np.concatenate([Ki, np.array([[0, 0, 1]])], -2)
            Di = D_np[bi]
            
            pixelcoords_undistorted = cv2.fisheye.undistortPoints(
                pix_coords,
                Ki,
                Di,
                P=Ki
            ) # h * w * 2

        pixel_coords.append(pixelcoords_undistorted)
        pixelcoords = th.from_numpy(np.stack(pixel_coords)).contiguous().to(envbg.device)
    
    # convert pixel coords into directions
    raydir = (pixelcoords - princpt[:, None, None, :])
    raydir[..., 0] /= focal[:, None, None, 0, 0] * focal_scale
    raydir[..., 1] /= focal[:, None, None, 1, 1] * focal_scale
    raydir = th.cat([raydir, th.ones_like(raydir[:, :, :, 0:1])], dim=-1)
    if camrot is not None:
        raydir = th.einsum("bxy,bhwx->bhwy", camrot, raydir)
    raydir = thf.normalize(raydir, dim=-1)
    u = (1 / np.pi) * th.atan2(raydir[..., 0], raydir[..., 2])  # range: [-1, 1]
    v = (1 / np.pi) * th.acos(raydir[..., 1]) # range: [0, 1]
    v = 2 * v - 1.0
    uv = th.stack([u, v], -1)
    envbg = thf.grid_sample(envbg, uv, mode='bicubic', padding_mode="border", align_corners=True)

    if blurbg:
        blurkernel = th.exp(-th.linspace(-4., 4., 101) ** 2)
        blurkernel = (blurkernel[:, None] * blurkernel[None, :])
        blurkernel = blurkernel / th.sum(blurkernel)
        blurkernel = blurkernel[None, None, :, :].repeat(3, 1, 1, 1).to(princpt.device)
        envbg = thf.conv2d(envbg, weight=blurkernel, stride=1, padding=50, groups=3)
        envbg = thf.interpolate(envbg, size=(h, w))

    return envbg


def envmap_to_mirrorball(w, h, env, camrot=None):
    py, px = th.meshgrid(th.linspace(-1.0, 1.0, h), th.linspace(-1.0, 1.0, w))
    pixelcoords = th.stack([px, py], -1)[None].expand(env.shape[0], -1, -1, -1).to(env.device)
    zsq = pixelcoords.pow(2).sum(-1, keepdim=True)
    mask = (zsq < 1.0).float()[:, None, :, :, 0]
    nz = -(1.0 - zsq).clamp(min=0.0).sqrt()
    nml = th.cat([pixelcoords, nz], -1)
    ref = - 2.0 * nz * nml
    ref[..., 2] = 1.0 + ref[..., 2]
    if camrot is not None:
        ref = th.einsum("bxy,bhwx->bhwy", camrot, ref)
    envball = 255.0 * (0.5 * ref.permute(0, 3, 1, 2) + 0.5)
    u = (1 / np.pi) * th.atan2(ref[..., 0], ref[..., 2])  # range: [-1, 1]
    v = (1 / np.pi) * th.acos(ref[..., 1]) # range: [0, 1]
    v = 2 * v - 1.0
    uv = th.stack([u, v], -1)
    envball = thf.grid_sample(env, uv, mode='bicubic', padding_mode="border", align_corners=True)
    
    return th.cat([envball, mask], 1)


def importance_sample_sg(
    Xi: th.Tensor, n: th.Tensor, sigma: float, dim: int = 1
) -> Tuple[th.Tensor, th.Tensor]:
    o = (slice(None),) * dim
    dim_x = o + (slice(0, 1),)
    dim_y = o + (slice(1, 2),)
    dim_z = o + (slice(2, 3),)

    phi = Xi[dim_x]
    phi = 2.0 * np.pi * phi
    sqrt2sigma = math.sqrt(2.0) * sigma
    theta_new = sqrt2sigma * erfinv(Xi[dim_y] * math.erf(np.pi / sqrt2sigma))
    cos_theta = th.cos(theta_new)
    sin_theta = th.sin(theta_new)
    
    # from spherical coordinates to cartesian coordinates
    H = th.cat([th.cos(phi) * sin_theta, th.sin(phi) * sin_theta, cos_theta], dim=dim)
    
    pdf = math.sqrt(2.0) * np.pi ** (-0.5) / (sigma * math.erf(np.pi / sqrt2sigma)) * th.exp(-0.5 * (theta_new / sigma) ** 2)

    # from tangent-space vector to world-space sample vector
    m = n[dim_z] < 0.999
    up = th.zeros_like(n)
    up[dim_x][~m] = 1
    up[dim_z][m] = 1

    tangent = thf.normalize(th.cross(up, n, dim=dim), dim=dim)
    bitangent = th.cross(n, tangent, dim=dim)

    sample_vec = tangent * H[dim_x] + bitangent * H[dim_y] + n * H[dim_z]
    return thf.normalize(sample_vec, dim=dim), pdf


def dir2uv(d: th.Tensor, dim: int=1) -> th.Tensor:
    x = d.narrow(dim, 0, 1)
    y = d.narrow(dim, 1, 1)
    z = d.narrow(dim, 2, 1)
    
    u = (1 / np.pi) * th.atan2(x, z)  # range: [-1, 1]
    v = (1 / np.pi) * th.acos(y)  # range: [0, 1]
    v = 2 * v - 1.0
    return th.stack([u.squeeze(dim), v.squeeze(dim)], -1)


def sample_uv(d: th.tensor, img: th.tensor) -> th.Tensor:
    '''
    args:
        d: (B, 3, H, W)
        img: (B, 3, H', W')
    '''
    uv = dir2uv(d, 1)
    return thf.grid_sample(img, uv, padding_mode="border", align_corners=False)


def prefilterEnvmapSG(
    sigma: float, v: th.Tensor, env_tex: th.Tensor, num_samples: int=1
) -> th.tensor:
    # v: [B, 3, H, W]
    # env_tex: [B, 3, H, W]
    acc = None
    for i in range(num_samples):
        x_i = th.rand_like(v[:, :2])

        v_sample, pdf = importance_sample_sg(x_i, v, sigma)

        sample_color = sample_uv(v_sample, env_tex)

        if acc is None:
            out_shape = list(v.shape)
            acc = th.zeros(*out_shape, device=v.device, dtype=env_tex.dtype)
        acc += sample_color
        
    return acc / float(num_samples)

def compose_envmap(render, alpha, envbg, K, Rt):
    env_mirror = envmap_to_mirrorball(200, 200, envbg, Rt[:, :3, :3])
    # to offset mugsy color correction
    env_mirror[:, 0] = env_mirror[:, 0]
    env_mirror[:, 2] = env_mirror[:, 2]
    
    mirror_img = th.zeros_like(render)
    mirror_alpha = th.zeros_like(alpha)
    mirror_alpha[:, :, -200:, -200:] = env_mirror[:, 3:]
    mirror_img[:, :, -200:, -200:] = env_mirror[:, :3]
    
    envbg = envmap_to_image(
        render.shape[-1], render.shape[-2], envbg, K[:, :2, 2], K, Rt[:, :3, :3]
    )
    # to offset mugsy color correction
    envbg[:, 0] = envbg[:, 0]
    envbg[:, 2] = envbg[:, 2]
    render = render + (1.0 - alpha) * envbg.clamp(0, 1.0)
    render = (1.0 - mirror_alpha) * render + mirror_alpha * mirror_img
    
    return render