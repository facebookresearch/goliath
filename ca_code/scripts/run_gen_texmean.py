import logging
import os
import sys
import cv2
import torch as th
from tqdm import tqdm
from typing import List
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig, OmegaConf
from addict import Dict as AttrDict

from ca_code.utils.torchutils import to_device
from ca_code.utils.render_drtk import RenderLayer
from ca_code.utils.dataloader import BodyDataset, collate_fn
from ca_code.utils.tex import get_tex_rl
from ca_code.utils.geom import GeometryModule, index_image_impaint, make_uv_vert_index, make_uv_barys
from ca_code.utils.lbs import LBSModule
from ca_code.utils.tex import get_tex_rl

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s][%(name)s]:%(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

def main(config: DictConfig):
    device = th.device(f"cuda:0")
    # force to get fully lit frames for tex mean extraction
    config.data.fully_lit_only = True
    config.data.partially_lit_only = False
    train_dataset = BodyDataset(**config.data)
    assets = AttrDict(train_dataset.static_assets)
    geo_fn = GeometryModule(
        th.LongTensor(assets.topology.vi),
        assets.topology.vt,
        assets.topology.vti,
        assets.topology.v2uv,
        uv_size=1024,
        impaint=True,
    ).to(device)
    lbs_fn = LBSModule(
        assets.lbs_model_json,
        assets.lbs_config_dict,
        assets.template_mesh_unscaled[None],
        assets.skeleton_scales,
        global_scaling=[10.0, 10.0, 10.0],  # meter
    ).to(device)

    vt = geo_fn.vt
    vi = geo_fn.vi
    vti = geo_fn.vti

    rl = RenderLayer(
        h=config.model.renderer.image_height,
        w=config.model.renderer.image_width,
        vt=vt,
        vi=vi.int(),
        vti=vti,
        flip_uvs=False,
    ).to(device)

    uv_size = (1024, 1024)
    inpaint_threshold = 100.0
    index_image = make_uv_vert_index(
        vt, vi, vti, uv_shape=uv_size, flip_uv=True
    ).cpu()
    face_index, bary_image = make_uv_barys(vt, vti, uv_shape=uv_size, flip_uv=True)

    # inpaint index uv images
    index_image, bary_image = index_image_impaint(index_image, bary_image, inpaint_threshold)
    face_index = index_image_impaint(face_index, distance_threshold=inpaint_threshold)

    frame_list = train_dataset.get_frame_list(
        fully_lit_only=train_dataset.fully_lit_only,
        partially_lit_only=train_dataset.partially_lit_only,
    )
    # we suggest to use occlusion-free 5 consecutive frames for tex_mean extraction
    num_frames = 5
    frame_list = frame_list[:num_frames]
    camera_list = train_dataset.get_camera_list()
    tex_total = th.zeros(1, 3, 1024, 1024).float().to(device)
    tex_cnt = th.zeros(1, 3, 1024, 1024).float().to(device)
    for fid in frame_list:
        for cid in tqdm(camera_list):
            try:
                current_data = train_dataset.get(fid, cid)
            except:
                logger.warning(f"error when loading cam{cid} frame{fid}, skipping")
                continue
            current_data = to_device(current_data, device)
            img = current_data['image'][None].float()
            extrin = current_data['Rt'][None]
            intrin = current_data['K'][None]
            pose = current_data['pose'][None]
            mesh_world = lbs_fn.pose(th.zeros_like(lbs_fn.lbs_template_verts), pose)
            faces = geo_fn.vi
            ply = (mesh_world, faces)
            tex_img, tex_mask = get_tex_rl(rl, img, ply, extrin, intrin, face_index, index_image, bary_image)

            tex_total += tex_img
            tex_cnt += tex_mask.float()

    tex_total /= tex_cnt + 1e-5
    tex_mean = th.flip(tex_total[0].permute(1, 2, 0), (0,)).cpu().numpy()
    tex_mean_path = os.path.join(config.data.root_path, "uv_image", "color_mean.png")
    cv2.imwrite(tex_mean_path, tex_mean[..., [2, 1, 0]])

if __name__ == "__main__":
    config_path: str = sys.argv[1]
    console_commands: List[str] = sys.argv[2:]

    config = OmegaConf.load(config_path)
    config_cli = OmegaConf.from_cli(args_list=console_commands)
    if config_cli:
        logger.info("Overriding with the following args values:")
        logger.info(f"{OmegaConf.to_yaml(config_cli)}")
        config = OmegaConf.merge(config, config_cli)

    main(config)