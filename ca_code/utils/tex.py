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

import care.data.io.typed as typed
from ca_code.utils.geom import compute_view_texture, make_uv_vert_index, make_uv_barys, vert_normals, index_image_impaint

def get_tex_rl(rl, image, ply, extrin, intrin, face_index, index_image, bary_image):
    geom = ply[0].float().cuda()[None]
    faces = ply[1].int().cuda()
    tex_tmp = th.zeros(1, 3, 1024, 1024).float().cuda()
    R = extrin[:3, :3]
    t = extrin[:3, 3]
    camrot = R
    campos = R.T @ (-t)

    cam_verts = (camrot[None, None] @ (geom - campos[None, None])[..., None])[..., 0]
    vn_cam = vert_normals(cam_verts, rl.vi.expand(geom.shape[0], -1, -1).long())

    ds = 2
    if ds > 1:
        intrin[:2, :2] /= ds
        intrin[:2, 2] = (intrin[:2, 2] + 0.5) / ds - 0.5
    renders = rl(
                    geom,
                    tex_tmp,
                    K=intrin[None],
                    Rt=extrin[None],
                    vn=vn_cam,
                    output_filters=['mask', 'vn_img', 'index_img'],
                )

    tex_img, tex_mask = compute_view_texture(
        geom,
        faces,
        image,
        renders["index_img"],
        renders["vn_img"],
        intrin[None],
        extrin[None],
        index_image,
        bary_image,
        face_index,
        intensity_threshold=None,
        normal_threshold=0.1
    )
    
    return tex_img, tex_mask


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--cid_list", type=str, required=True)
    args = parser.parse_args()
    cid_list = args.cid_list
    with open(cid_list, 'r') as f:
        capture_ids = f.readlines()
        capture_ids = [x[:-1] for x in capture_ids]

    for capture_id in capture_ids:
        print("Generating texture mean for {}......".format(capture_id))
        KRT, frame_list, mesh_template = None, None, None
        ply_path, image_path = None, None

        vt = th.as_tensor(mesh_template["vt"]).cuda()
        vi = th.as_tensor(mesh_template["vi"]).cuda()
        vti = th.as_tensor(mesh_template["vti"]).cuda()

        uv_size = (1024, 1024)
        inpaint_threshold = 100.0
        index_image = make_uv_vert_index(
            vt, vi, vti, uv_shape=uv_size, flip_uv=True
        ).cpu()
        face_index, bary_image = make_uv_barys(vt, vti, uv_shape=uv_size, flip_uv=True)

        # inpaint index uv images
        index_image, bary_image = index_image_impaint(
                        index_image, bary_image, inpaint_threshold
                    )
        face_index = index_image_impaint(face_index, distance_threshold=inpaint_threshold)

        rl = RenderLayer(
            2048,
            1334,
            mesh_template['vt'],
            mesh_template['vi'],
            mesh_template['vti'],
            flip_uvs=False,
        ).cuda()
        tex_total = th.zeros(1, 3, 1024, 1024).float().cuda()
        tex_cnt = th.zeros(1, 3, 1024, 1024).float().cuda()
        cameras = list(KRT.keys())

        for i, fid in enumerate(frame_list):
            if i > 4:
                break
            for camera in tqdm(cameras):
                img_path = image_path.format(capture_id=capture_id, camera=camera, frame=int(fid))
                ply_path_tmp = ply_path.format(capture_id=capture_id, frame=int(fid))
                if not (os.path.isfile(img_path) and os.path.isfile(ply_path_tmp)):
                    continue
                img_tmp = typed.load(img_path)
                img_tmp = th.from_numpy(img_tmp).permute(2, 0, 1).float().cuda()[None]

                ply_tmp = typed.load(ply_path_tmp)
                extrin = th.from_numpy(KRT[camera]["extrin"]).float().cuda()
                intrin = th.from_numpy(KRT[camera]["intrin"]).float().cuda()

                tex_img, tex_mask = get_tex_rl(rl, img_tmp, ply_tmp, extrin, intrin, face_index, index_image, bary_image)

                tex_total += tex_img
                tex_cnt += tex_mask.float()
        tex_total /= tex_cnt + 1e-5
        tex_mean = th.flip(tex_total[0].permute(1, 2, 0), (0,)).cpu().numpy(d)
        # cv2.imwrite(target_texmean_path, tex_mean[..., [2,1,0]])