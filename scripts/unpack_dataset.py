import argparse
import multiprocessing as mp
import shutil
import zipfile
from itertools import repeat
from pathlib import Path

from tqdm import tqdm

N_PARALLEL_PROC = 8


def unzip_zip(zip_path, output_path):
    with zipfile.ZipFile(zip_path, "r") as zipf:
        zipf.extractall(output_path)


def unzip_star(args):
    return unzip_zip(*args)


def unzip_images(input_root_path: Path, output_root_path: Path):
    input_images_root = input_root_path / "image"
    print(f"Extracting all of the zips under {input_images_root}")
    output_images_root = output_root_path / "image"
    output_images_root.mkdir(exist_ok=True)
    cam_zips = list(input_images_root.glob("*.zip"))
    assert len(cam_zips) > 0

    tasks = zip(cam_zips, repeat(output_images_root))
    with mp.Pool(processes=N_PARALLEL_PROC) as pool:
        for _ in tqdm(
            pool.imap_unordered(unzip_star, tasks), desc="Unzipping image zips"
        ):
            continue


def unzip_keypoints_3d(input_root_path: Path, output_root_path: Path):
    input_zip_path = input_root_path / "keypoints_3d" / "keypoints_3d.zip"
    output_path = output_root_path / "keypoints_3d"
    output_path.mkdir(exist_ok=True)
    print(f"Extracting {input_zip_path} into {output_path}")
    unzip_zip(input_zip_path, output_path)


def unzip_per_view_background(input_root_path: Path, output_root_path: Path):
    input_zip_path = input_root_path / "per_view_background" / "per_view_background.zip"
    output_path = output_root_path / "per_view_background"
    output_path.mkdir(exist_ok=True)
    print(f"Extracting {input_zip_path} into {output_path}")
    unzip_zip(input_zip_path, output_path)


def unzip_kinematic_tracking(input_root_path: Path, output_root_path: Path):
    input_path = input_root_path / "kinematic_tracking"
    output_path = output_root_path / "kinematic_tracking"
    output_path.mkdir(exist_ok=True)

    files = []
    zips = []
    if "Body" in input_root_path.name:
        files = [
            "skeleton_scales.txt",
            "template_mesh.ply",
        ]
        zips = [
            "registration_vertices.zip",
            "pose.zip",
        ]
    if "Head" in input_root_path.name:
        files = [
            "registration_vertices_mean.npy",
            "registration_vertices_variance.txt",
            "template_mesh.ply",
        ]
        zips = [
            "registration_vertices.zip",
        ]

    if "Hand" in input_root_path.name:
        files = [
            "skeleton_scales.txt",
            "template_mesh.ply",
            "template_mesh_unscaled.ply",
        ]
        zips = ["pose.zip", "registration_vertices.zip"]

    for f in files:
        print(f"Copying file {input_path / f} to {output_path / f}")
        shutil.copy(input_path / f, output_path / f)
    for f in zips:
        print(f"Unzipping {input_path / f} into {output_path}")
        unzip_zip(input_path / f, output_path)


def unzip_segmentation_parts(input_root_path: Path, output_root_path: Path):
    input_images_root = input_root_path / "segmentation_parts"
    print(f"Extracting all of the zips under {input_images_root}")
    output_images_root = output_root_path / "segmentation_parts"
    output_images_root.mkdir(exist_ok=True)
    cam_zips = list(input_images_root.glob("*.zip"))
    assert len(cam_zips) > 0

    tasks = zip(cam_zips, repeat(output_images_root))
    with mp.Pool(processes=N_PARALLEL_PROC) as pool:
        for _ in tqdm(
            pool.imap_unordered(unzip_star, tasks),
            desc="Unzipping segmentation parts zips",
        ):
            continue


def unzip_uv_image(input_root_path: Path, output_root_path: Path):
    input_path = input_root_path / "uv_image"
    output_path = output_root_path / "uv_image"
    output_path.mkdir(exist_ok=True)
    files = []
    zips = []

    if "Body" in input_root_path.name:
        files = [
            "ambient_occlusion_mean.png",
            "color_mean.png",
        ]
        zips = ["ambient_occlusion.zip"]

    if "Head" in input_root_path.name:
        files = [
            "color_mean.png",
            "color_variance.txt",
        ]
        zips = ["color.zip"]

    if "Hand" in input_root_path.name:
        files = [
            "ambient_occlusion_mean.png",
        ]
        zips = ["ambient_occlusion.zip"]

    for f in files:
        print(f"Copying file {input_path / f} to {output_path / f}")
        shutil.copy(input_path / f, output_path / f)
    for f in zips:
        print(f"Unzipping {input_path / f} into {output_path}")
        unzip_zip(input_path / f, output_path)


def unzip_scan_mesh(input_root_path: Path, output_root_path: Path):
    input_path = input_root_path / "scan_mesh" / "scan_mesh.zip"
    output_path = output_root_path / "scan_mesh"
    print(f"Unzipping {input_path} into {output_path}")
    output_path.mkdir(exist_ok=True)
    unzip_zip(input_path, output_path)


def copy_misc(input_root_path: Path, output_root_path: Path):
    files = [
        "camera_calibration.json",
        "frame_splits_list.csv",
        "frame_segments_list.csv",
    ]
    for f in files:
        print(f"Copying file {input_root_path / f} to {output_root_path / f}")
        shutil.copy(input_root_path / f, output_root_path / f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("-p", "--processes", type=int, default=8)
    args = parser.parse_args()

    if args.output is None:
        # Do it in-place
        args.output = args.input

    input_root_path = Path(args.input)
    output_root_path = Path(args.output)
    output_root_path.mkdir(exist_ok=True)

    copy_misc(input_root_path, output_root_path)
    unzip_scan_mesh(input_root_path, output_root_path)
    unzip_uv_image(input_root_path, output_root_path)
    unzip_segmentation_parts(input_root_path, output_root_path)
    unzip_kinematic_tracking(input_root_path, output_root_path)
    unzip_keypoints_3d(input_root_path, output_root_path)
    unzip_images(input_root_path, output_root_path)

    if "Head" in input_root_path.name:
        unzip_per_view_background(input_root_path, output_root_path)


if __name__ == "__main__":
    main()
