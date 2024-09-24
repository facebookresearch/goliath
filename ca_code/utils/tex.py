# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os

import cv2

import torch as th
from tqdm import tqdm
from argparse import ArgumentParser

from ca_code.utils.render_drtk import RenderLayer
from ca_code.utils.geom import compute_view_texture, make_uv_vert_index, make_uv_barys, vert_normals, index_image_impaint
from ca_code.utils.torchutils import index

def get_tex_rl(rl, image, ply, extrin, intrin, face_index, index_image, bary_image):
    '''
        image - [B, 3, rl.height, rl.width]
        ply - tuple (vert, faces)
        extrin - [B, 3, 4]
        instrin - [B, 3, 3]
        face_index - [uv_size, uv_size]
        index_image - [uv_size, uv_size, 3]
        bary_image - [uv_size, uv_size, 3]
    '''
    assert image.shape[0] == 1
    B = 1
    extrin = extrin[0]
    intrin = intrin[0]
    geom = ply[0]
    faces = ply[1]
    tex_tmp = th.zeros(1, 3, 1024, 1024).to(geom)
    R = extrin[:3, :3]
    t = extrin[:3, 3]
    camrot = R
    campos = R.T @ (-t)

    renders = rl(
                    geom,
                    tex_tmp,
                    K=intrin[None],
                    Rt=extrin[None],
                )

    tex_img, tex_mask = compute_view_texture(
        geom,
        faces,
        image,
        renders["index_img"],
        None,
        intrin[None],
        extrin[None],
        index_image,
        bary_image,
        face_index,
        intensity_threshold=None,
        normal_threshold=0.1
    )
    
    return tex_img, tex_mask