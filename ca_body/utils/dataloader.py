import argparse
import json
import zipfile
import logging

from enum import Enum
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

import pandas as pd
import pillow_avif
import torch
from PIL import Image
from pytorch3d.io import load_ply, save_ply
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

# There are a lot of frame-wise assets. Avoid re-fetching those when we
# switch cameras
CACHE_LENGTH = 160


class CaptureType(Enum):
    BODY = 1
    HEAD = 2
    HAND = 3


# Head and hand capture types only have assets for fully lit frames.
ASSETS_ONLY_FOR_FULLY_LIT_FRAMES = [CaptureType.HEAD, CaptureType.HAND]

root = logging.getLogger()
root.setLevel(logging.DEBUG)

def get_capture_type(capture_name: str) -> CaptureType:
    if "Head" in capture_name:
        return CaptureType.HEAD
    if "Hand" in capture_name:
        return CaptureType.HAND
    if "Body" in capture_name:
        return CaptureType.BODY
    raise ValueError(
        f"Could not determine capture type from capture name: {capture_name}"
    )


class BodyDataset(IterableDataset):
    def __init__(self, root_path: Path, split: str, fully_lit_only: bool = False):
        assert split in ["train", "test"]
        self.root_path: Path = Path(root_path)
        self.split: str = split
        self.fully_lit_only: bool = fully_lit_only

        self.capture_type: CaptureType = get_capture_type(self.root_path.name)
        self._get_fn: Callable[[int, int], Any] = None

        if self.capture_type == CaptureType.BODY:
            self._get_fn = self._get_for_body
        if self.capture_type == CaptureType.HEAD:
            self._get_fn = self._get_for_head
        if self.capture_type == CaptureType.HAND:
            self._get_fn = self._get_for_hand

    def asset_exists(self, frame: int) -> bool:
        if self.capture_type not in ASSETS_ONLY_FOR_FULLY_LIT_FRAMES:
            # Assets exist for every frame
            return True
        # Assets only exist if this frame is fully lit
        return frame in self.get_frame_set(fully_lit_only=True)

    @lru_cache(maxsize=1)
    def get_camera_calibration(self) -> Dict[str, Any]:
        with open(self.root_path / "camera_calibration.json", "r") as f:
            camera_calibration = json.load(f)
        return camera_calibration

    @lru_cache(maxsize=1)
    def get_camera_list(self) -> List[int]:
        return [int(j["cameraId"]) for j in self.get_camera_calibration()["KRT"]]

    @lru_cache(maxsize=2)
    def get_frame_set(self, fully_lit_only: bool = False) -> Set[int]:
        df = pd.read_csv(self.root_path / f"frame_splits_list.csv")
        frame_list = df[df.split == self.split].frame.tolist()

        if not fully_lit_only or self.capture_type is CaptureType.BODY:
            # All frames in Body captures are fully lit
            return set(frame_list)

        fully_lit = {frame for frame, index in self.load_light_pattern() if index == 0}
        return {f for f in fully_lit if f in frame_list}

    @lru_cache(maxsize=CACHE_LENGTH)
    def load_3d_keypoints(self, frame: int):
        if not self.asset_exists(frame):
            return None

        zip_path = self.root_path / "keypoints_3d" / "keypoints_3d.zip"
        with zipfile.ZipFile(zip_path, "r") as zipf:
            with zipf.open(f"{frame:06d}.json", "r") as json_file:
                return json.load(json_file)

        # kpts_path = self.root_path / "keypoints_3d" / f"{frame:06d}.json"
        # with open(kpts_path, "r") as f:
        #     content = json.loads(f.read())
        # return content

    def load_segmentation_parts(self, frame: int, camera: int):
        if not self.asset_exists(frame):
            return None

        zip_path = self.root_path / "segmentation_parts" / f"cam{camera:06d}.zip"
        with zipfile.ZipFile(zip_path, "r") as zipf:
            with zipf.open(f"cam{camera:06d}/{frame:06d}.png", "r") as png_file:
                return Image.open(BytesIO(png_file.read()))

        # png_path = (
        #     self.root_path
        #     / "segmentation_parts"
        #     / f"cam{camera:06d}"
        #     / f"{frame:06d}.png"
        # )
        # return Image.open(png_path)

    def load_segmentation_fgbg(self, frame: int, camera: int):
        if not self.asset_exists(frame):
            return None

        zip_path = self.root_path / "segmentation_fgbg" / f"cam{camera:06d}.zip"
        with zipfile.ZipFile(zip_path, "r") as zipf:
            with zipf.open(f"cam{camera:06d}/{frame:06d}.png", "r") as png_file:
                return Image.open(BytesIO(png_file.read()))

        # png_path = (
        #     self.root_path
        #     / "segmentation_fgbg"
        #     / f"cam{camera:06d}"
        #     / f"{frame:06d}.png"
        # )
        # return Image.open(png_path)

    def load_image(self, frame: int, camera: int) -> Image:
        zip_path = self.root_path / "image" / f"cam{camera:06d}.zip"

        with zipfile.ZipFile(zip_path, "r") as zipf:
            with zipf.open(f"cam{camera:06d}/{frame:06d}.avif", "r") as avif_file:
                avif_image = Image.open(BytesIO(avif_file.read()))
                return avif_image

        # avif_path = self.root_path / "image" / f"cam{camera:06d}" / f"{frame:06d}.avif"
        # return Image.open(avif_path)

    @lru_cache(maxsize=CACHE_LENGTH)
    def load_registration_vertices(self, frame: int):
        if not self.asset_exists(frame):
            return None

        zip_path = self.root_path / "kinematic_tracking" / "registration_vertices.zip"
        with zipfile.ZipFile(zip_path, "r") as zipf:
            with zipf.open(f"registration_vertices/{frame:06d}.ply", "r") as ply_file:
                vertices, _ = load_ply(BytesIO(ply_file.read()))
                return vertices

        # verts_path = (
        #     self.root_path
        #     / "kinematic_tracking"
        #     / "registration_vertices"
        #     / f"{frame:06d}.ply"
        # )
        # with open(verts_path, "rb") as f:
        #     # No faces are included
        #     verticies, _ = load_ply(f)
        # return verticies

    @lru_cache(maxsize=1)
    def load_registration_vertices_mean(self):
        mean_path = (
            self.root_path / "kinematic_tracking" / "registration_vertices_mean.npy"
        )
        return np.load(mean_path)

    @lru_cache(maxsize=1)
    def load_registration_vertices_variance(self) -> float:
        verts_path = (
            self.root_path / "kinematic_tracking" / "registration_vertices_variance.txt"
        )
        with open(verts_path, "r") as f:
            variance = float(f.read())
        return variance

    @lru_cache(maxsize=CACHE_LENGTH)
    def load_pose(self, frame: int):
        if not self.asset_exists(frame):
            return None

        zip_path = self.root_path / "kinematic_tracking" / "pose.zip"
        with zipfile.ZipFile(zip_path, "r") as zipf:
            with zipf.open(f"pose/{frame:06d}.txt", "r") as f:
                pose_arr = np.array([float(i) for i in f.read().splitlines()])
                return pose_arr

        # pose_path = self.root_path / "kinematic_tracking" / "pose" / f"{frame:06d}.txt"
        # with open(pose_path, "r") as f:
        #     pose_arr = np.array([float(i) for i in f.read().splitlines()])
        # return pose_arr

    @lru_cache(maxsize=1)
    def load_template_mesh(self):
        mesh_path = self.root_path / "kinematic_tracking" / "template_mesh.ply"
        with open(mesh_path, "rb") as f:
            verticies, faces = load_ply(f)
        return verticies, faces

    @lru_cache(maxsize=1)
    def load_template_mesh_unscaled(self):
        mesh_path = self.root_path / "kinematic_tracking" / "template_mesh_unscaled.ply"
        with open(mesh_path, "rb") as f:
            verticies, faces = load_ply(f)
        return verticies, faces

    @lru_cache(maxsize=1)
    def load_skeleton_scales(self):
        scales_path = self.root_path / "kinematic_tracking" / "skeleton_scales.txt"
        with open(scales_path, "r") as f:
            scales_arr = np.array([float(i) for i in f.read().splitlines()])
        return scales_arr

    @lru_cache(maxsize=CACHE_LENGTH)
    def load_ambient_occlusion(self, frame: int) -> Image:
        if not self.asset_exists(frame):
            return None
        zip_path = self.root_path / "uv_image" / "ambient_occlusion.zip"
        with zipfile.ZipFile(zip_path, "r") as zipf:
            with zipf.open(f"ambient_occlusion/{frame:06d}.png", "r") as png_file:
                return Image.open(BytesIO(png_file.read()))
        # png_path = (
        #     self.root_path / "uv_image" / "ambient_occlusion" / f"{frame:06d}.png"
        # )
        # return Image.open(png_path)

    @lru_cache(maxsize=1)
    def load_ambient_occlusion_mean(self) -> Image:
        png_path = self.root_path / "uv_image" / "ambient_occlusion_mean.png"
        return Image.open(png_path)

    @lru_cache(maxsize=1)
    def load_color_mean(self) -> Image:
        png_path = self.root_path / "uv_image" / "color_mean.png"
        return Image.open(png_path)

    @lru_cache(maxsize=1)
    def load_color_variance(self) -> float:
        color_var_path = self.root_path / "uv_image" / "color_variance.txt"
        with open(color_var_path, "r") as f:
            color_var = float(f.read())
        return color_var

    @lru_cache(maxsize=1)
    def load_color(self, frame: int) -> Image:
        if not self.asset_exists(frame):
            return None

        zip_path = self.root_path / "uv_image" / "color.zip"
        with zipfile.ZipFile(zip_path, "r") as zipf:
            with zipf.open(f"color/{frame:06d}.png", "r") as png_file:
                return Image.open(BytesIO(png_file.read()))

        # color_png_path = self.root_path / "uv_image" / "color" / f"{frame:06d}.png"
        # return Image.open(color_png_path)

    @lru_cache(maxsize=CACHE_LENGTH)
    def load_scan_mesh(self, frame: int):
        if not self.asset_exists(frame):
            return None

        zip_path = self.root_path / "scan_mesh" / "scan_mesh.zip"
        with zipfile.ZipFile(zip_path, "r") as zipf:
            with zipf.open(f"{frame:06d}.ply", "r") as ply_file:
                vertices, faces = load_ply(BytesIO(ply_file.read()))
                return vertices, faces

        # ply_path = self.root_path / "scan_mesh" / f"{frame:06d}.ply"
        # with open(ply_path, "rb") as f:
        #     verticies, faces = load_ply(f)
        # return verticies, faces

    @lru_cache(maxsize=CACHE_LENGTH)
    def load_head_pose(self, frame: int):
        zip_path = self.root_path / "head_pose" / "head_pose.zip"
        with zipfile.ZipFile(zip_path, "r") as zipf:
            with zipf.open(f"{frame:06d}.txt", "r") as txt_file:
                lines = txt_file.read().decode("utf-8").splitlines()
                rows = [line.split(" ") for line in lines]
                return np.array([[float(i) for i in row] for row in rows])

        # pose_path = self.root_path / "head_pose" / f"{frame:06d}.txt"
        # with open(pose_path, "r") as f:
        #     pose_arr = np.array([float(i) for i in f.read().splitlines()])
        # return pose_arr

    @lru_cache(maxsize=CACHE_LENGTH)
    def load_background(self, camera: int):
        zip_path = self.root_path / "per_view_background" / "per_view_background.zip"
        with zipfile.ZipFile(zip_path, "r") as zipf:
            with zipf.open(f"{camera:05d}.png", "r") as png_file:
                return Image.open(BytesIO(png_file.read()))

        # background_png_path = (
        #     self.root_path / "per_view_background" / f"{camera:06d}.png"
        # )
        # return Image.open(background_png_path)

    @lru_cache(maxsize=1)
    def load_light_pattern(self) -> List[Tuple[int]]:
        light_pattern_path = self.root_path / "lights" / "light_pattern_per_frame.json"
        with open(light_pattern_path, "r") as f:
            return json.load(f)

    @lru_cache(maxsize=1)
    def load_light_pattern_meta(self):
        light_pattern_path = self.root_path / "lights" / "light_pattern_metadata.json"
        with open(light_pattern_path, "r") as f:
            return json.load(f)

    def _get_for_body(self, frame: int, camera: int) -> Dict[str, Any]:
        template_mesh = self.load_template_mesh()
        skeleton_scales = self.load_skeleton_scales()
        ambient_occlusion = self.load_ambient_occlusion(frame)
        ambient_occlusion_mean = self.load_ambient_occlusion_mean()
        color_mean = self.load_color_mean()
        kpts = self.load_3d_keypoints(frame)
        registration_vertices = self.load_registration_vertices(frame)
        pose = self.load_pose(frame)
        scan_mesh = self.load_scan_mesh(frame)
        image = self.load_image(frame, camera)
        segmentation_parts = self.load_segmentation_parts(frame, camera)

        row = {
            "camera_id": camera,
            "frame_id": frame,
            "image": pil_to_tensor(image),
            "keypoints_3d": kpts,
            "registration_vertices": registration_vertices,
            "segmentation_parts": pil_to_tensor(segmentation_parts),
            "pose": pose,
            "template_mesh": template_mesh,
            "skeleton_scales": skeleton_scales,
            "ambient_occlusion_mean": pil_to_tensor(ambient_occlusion_mean),
            "color_mean": pil_to_tensor(color_mean),
            "scan_mesh": scan_mesh,
        }
        return row

    def _get_for_head(self, frame: int, camera: int) -> Dict[str, Any]:
        is_fully_lit_frame: bool = frame in self.get_frame_set(fully_lit_only=True)
        # head_pose = self.load_head_pose(frame)
        image = self.load_image(frame, camera)
        kpts = self.load_3d_keypoints(frame)
        reg_verts = self.load_registration_vertices(frame)
        reg_verts_mean = self.load_registration_vertices_mean()
        reg_verts_var = self.load_registration_vertices_variance()
        template_mesh = self.load_template_mesh()
        light_pattern = self.load_light_pattern()
        light_pattern_meta = self.load_light_pattern_meta()
        segmentation_parts = self.load_segmentation_parts(frame, camera)
        color_mean = self.load_color_mean()
        color_var = self.load_color_variance()
        color = self.load_color(frame)
        scan_mesh = self.load_scan_mesh(frame)
        background = self.load_background(camera)

        row = {
            "camera_id": camera,
            "frame_id": frame,
            "is_fully_lit_frame": is_fully_lit_frame,
            # "head_pose": head_pose,
            "image": image,
            "keypoints_3d": kpts,
            "registration_vertices": reg_verts,
            "registration_vertices_mean": reg_verts_mean,
            "registration_vertices_variance": reg_verts_var,
            "template_mesh": template_mesh,
            "light_pattern": light_pattern,
            "light_pattern_meta": light_pattern_meta,
            "segmentation_parts": segmentation_parts,
            "color_mean": color_mean,
            "color_variance": color_var,
            "color": color,
            "scan_mesh": scan_mesh,
            "background": background,
        }
        for key in row:
            if row[key] is None:
                row[key] = torch.zeros(1)
            if isinstance(row[key], Image.Image):
                row[key] = pil_to_tensor(row[key])
        return row

    def _get_for_hand(self, frame: int, camera: int) -> Dict[str, Any]:
        is_fully_lit_frame: bool = frame in self.get_frame_set(fully_lit_only=True)
        image = self.load_image(frame, camera)
        kpts = self.load_3d_keypoints(frame)
        skeleton_scales = self.load_skeleton_scales()
        pose = self.load_pose(frame)
        reg_verts = self.load_registration_vertices(frame)
        template_mesh = self.load_template_mesh()
        template_mesh_unscaled = self.load_template_mesh_unscaled()
        light_pattern = self.load_light_pattern()
        light_pattern_meta = self.load_light_pattern_meta()
        segmentation_parts = self.load_segmentation_parts(frame, camera)
        segmentation_fgbg = self.load_segmentation_fgbg(frame, camera)
        ambient_occlusion = self.load_ambient_occlusion(frame)
        ambient_occlusion_mean = self.load_ambient_occlusion_mean()
        scan_mesh = self.load_scan_mesh(frame)

        row = {
            "camera_id": camera,
            "frame_id": frame,
            "is_fully_lit_frame": is_fully_lit_frame,
            "image": image,
            "keypoints_3d": kpts,
            "skeleton_scales": skeleton_scales,
            "pose": pose,
            "registration_vertices": reg_verts,
            "template_mesh": template_mesh,
            "template_mesh_unscaled": template_mesh_unscaled,
            "light_pattern": light_pattern,
            "light_pattern_meta": light_pattern_meta,
            "segmentation_parts": segmentation_parts,
            "segmentation_fgbg": segmentation_fgbg,
            "ambient_occlusion": ambient_occlusion,
            "ambient_occlusion_mean": ambient_occlusion_mean,
            "scan_mesh": scan_mesh,
        }
        for key in row:
            if row[key] is None:
                row[key] = torch.zeros(1)
            if isinstance(row[key], Image.Image):
                row[key] = pil_to_tensor(row[key])
        return row

    def __iter__(self):
        for frame in self.get_frame_set(self.fully_lit_only):
            for camera in self.get_camera_list():

                # yield self._get_fn(frame, camera)

                try:
                    yield self._get_fn(frame, camera)
                except Exception as e:
                    print(e)
                    continue

    def __len__(self):
        return len(self.get_frame_set(self.fully_lit_only)) * len(
            self.get_camera_list()
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, help="Root path to capture data")
    parser.add_argument("-s", "--split", type=str, choices=["train", "test"])
    args = parser.parse_args()

    dataset = BodyDataset(root_path=args.input, split=args.split)
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=4,
    # )

    for row in tqdm(dataset):
        continue
