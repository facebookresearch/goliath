import json
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pillow_avif
from PIL import Image
from pytorch3d.io import load_ply, save_ply
from torch.utils.data import DataLoader, Dataset

CACHE_LENGTH = 256


class BodyDataset(Dataset):
    def __init__(self, root_path: Path, split: str):
        assert split in ["train", "test"]
        self.root_path = Path(root_path)
        self.split = split
        self.image_root_path = self.root_path / "image"
        self.keypoints_3d_root_path = self.root_path / "keypoints_3d"
        self.segmentation_parts_root_path = self.root_path / "segmentation_parts"
        self.kinematic_tracking_root_path = self.root_path / "kinematic_tracking"
        # self.registration_vertices_root_path = (
        #     self.kinematic_tracking_root_path / "registration_vertices"
        # )
        self.registration_vertices_root_path = (
            self.kinematic_tracking_root_path / "registration_mesh"
        )
        self.pose_root_path = self.kinematic_tracking_root_path / "pose"
        self.uv_image_root_path = self.root_path / "uv_image"

    @lru_cache(maxsize=CACHE_LENGTH)
    def get_camera_calibration(self) -> Dict[str, Any]:
        with open(self.root_path / "camera_calibration.json", "r") as f:
            camera_calibration = json.load(f)
        return camera_calibration

    @lru_cache(maxsize=CACHE_LENGTH)
    def get_camera_list(self) -> List[int]:
        return [int(j["cameraId"]) for j in self.get_camera_calibration()["KRT"]]

    @lru_cache(maxsize=CACHE_LENGTH)
    def get_full_frame_list(self) -> pd.DataFrame:
        return pd.read_csv(self.root_path / f"frame_splits_list.csv")

    @lru_cache(maxsize=CACHE_LENGTH)
    def get_frame_list(self):
        df = self.get_full_frame_list()
        return df[df.split == self.split].frame.tolist()

    @lru_cache(maxsize=CACHE_LENGTH)
    def load_3d_keypoints(self, frame: int):
        kpts_path = self.keypoints_3d_root_path / f"{frame:06d}.json"
        with open(kpts_path, "r") as f:
            content = json.loads(f.read())
        return content

    @lru_cache(maxsize=CACHE_LENGTH)
    def load_segmentation_parts(self, frame: int, camera: int):
        png_path = (
            self.segmentation_parts_root_path / f"cam{camera:06d}" / f"{frame:06d}.png"
        )
        return Image.open(png_path)

    @lru_cache(maxsize=CACHE_LENGTH)
    def load_image(self, frame: int, camera: int):
        avif_path = self.image_root_path / f"cam{camera:06d}" / f"{frame:06d}.avif"
        return Image.open(avif_path)

    @lru_cache(maxsize=CACHE_LENGTH)
    def load_registration_vertices(self, frame: int):
        verts_path = self.registration_vertices_root_path / f"{frame:06d}.ply"
        with open(verts_path, "rb") as f:
            verticies, _ = load_ply(f)
        return verticies

    @lru_cache(maxsize=CACHE_LENGTH)
    def load_pose(self, frame: int):
        pose_path = self.pose_root_path / f"{frame:06d}.txt"

    @lru_cache(maxsize=CACHE_LENGTH)
    def load_template_mesh(self):
        mesh_path = self.kinematic_tracking_root_path / "template_mesh.ply"
        with open(mesh_path, "rb") as f:
            verticies, faces = load_ply(f)
        return verticies, faces

    @lru_cache(maxsize=CACHE_LENGTH)
    def load_skeleton_scales(self):
        scales_path = self.kinematic_tracking_root_path / "skeleton_scales.txt"
        with open(scales_path, "r") as f:
            content = f.read()
        return content

    @lru_cache(maxsize=CACHE_LENGTH)
    def load_ambient_occlusion(self, frame: int) -> Image:
        png_path = self.uv_image_root_path / "ambient_occlusion" / f"{frame:06d}.png"
        return Image.open(png_path)

    @lru_cache(maxsize=CACHE_LENGTH)
    def load_ambient_occlusion_mean(self) -> Image:
        png_path = self.uv_image_root_path / "ambient_occlusion_mean.png"
        return Image.open(png_path)

    @lru_cache(maxsize=CACHE_LENGTH)
    def load_color_mean(self) -> Image:
        png_path = self.uv_image_root_path / "color_mean.png"
        return Image.open(png_path)

    @lru_cache(maxsize=CACHE_LENGTH)
    def load_scan_mesh(self, frame: int):
        ply_path = self.root_path / "scan_mesh" / f"{frame:06d}.ply"
        with open(ply_path, "rb") as f:
            verticies, faces = load_ply(f)
        return verticies, faces

    def __getitem__(self, frame: int, camera: int):
        template_mesh = self.load_template_mesh()
        skeleton_scales = self.load_skeleton_scales()
        ambient_occlusion_mean = self.load_ambient_occlusion_mean()
        color_mean = self.load_color_mean()
        kpts = self.load_3d_keypoints(frame)
        registration_vertices = self.load_registration_vertices(frame)
        pose = self.load_pose(frame)
        ambient_occlusion = self.load_ambient_occlusion(frame)
        scan_mesh = self.load_scan_mesh(frame)
        image = self.load_image(frame, camera)
        segmentation_parts = self.load_segmentation_parts(frame, camera)
        return {
            "camera_id": camera,
            "frame_id": frame,
            "image": image,
            "keypoints_3d": kpts,
            "registration_vertices": registration_vertices,
            "segmentation_parts": segmentation_parts,
            "pose": pose,
            "template_mesh": template_mesh,
            "skeleton_scales": skeleton_scales,
            "ambient_occlusion_mean": ambient_occlusion_mean,
            "color_mean": color_mean,
            "scan_mesh": scan_mesh,
        }

    def __iter__(self):
        for frame in self.get_frame_list():
            for camera in self.get_camera_list():
                yield self.__getitem__(frame, camera)


if __name__ == "__main__":
    dataset = BodyDataset(
        root_path="/home/yitb/goliath_oss/root_dir_unzipped", split="train"
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )

    for row in dataset:
        print(row)
