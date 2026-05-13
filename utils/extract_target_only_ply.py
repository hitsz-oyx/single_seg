#!/usr/bin/env python3
"""从 single_seg 输出中提取融合后的纯目标点云（无背景点），便于观察假阳性。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from single_seg.single_object_segmenter import write_ply, write_ply_with_normals


def read_ply_points_and_colors(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取 PLY 文件的 xyz 和 rgb。"""
    with open(path, "rb") as f:
        header_bytes = b""
        while True:
            line = f.readline()
            header_bytes += line
            if line.startswith(b"end_header"):
                break
        header = header_bytes.decode("ascii")
        has_normal = False
        for line in header.strip().split("\n"):
            if line.startswith("property float nx"):
                has_normal = True

        if has_normal:
            dtype = np.dtype(
                [
                    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                    ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                    ("r", "u1"), ("g", "u1"), ("b", "u1"),
                ]
            )
            data = np.frombuffer(f.read(), dtype=dtype)
            return (
                np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32),
                np.stack([data["r"], data["g"], data["b"]], axis=1).astype(np.uint8),
            )
        else:
            dtype = np.dtype(
                [
                    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                    ("r", "u1"), ("g", "u1"), ("b", "u1"),
                ]
            )
            data = np.frombuffer(f.read(), dtype=dtype)
            return (
                np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32),
                np.stack([data["r"], data["g"], data["b"]], axis=1).astype(np.uint8),
            )


def read_ply_labels(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        header_bytes = b""
        while True:
            line = f.readline()
            header_bytes += line
            if line.startswith(b"end_header"):
                break
        header = header_bytes.decode("ascii")
        has_normal = any(line.startswith("property float nx") for line in header.strip().split("\n"))
        if has_normal:
            dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                              ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                              ("label", "<i4")])
        else:
            dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("label", "<i4")])
        data = np.frombuffer(f.read(), dtype=dtype)
        return data["label"].astype(np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="提取融合后的纯目标点云（无背景点）。"
    )
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="frame_outputs 目录路径，如 tests/outputs/.../frame_outputs")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="输出目录，默认为 input-dir 同级的目标点云目录 target_only_clouds")
    parser.add_argument("--color", type=str, default="original",
                        choices=["original", "blue", "cyan"],
                        help="目标点着色方式: original(原始RGB), blue(深蓝), cyan(青色)")
    parser.add_argument("--save-normal", type=int, default=0)
    parser.add_argument("--voxel-size", type=float, default=0.002)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_dir.parent / "target_only_clouds"
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_files = sorted(input_dir.glob("*_scene_rgb.ply"))
    if not scene_files:
        raise FileNotFoundError(f"no *_scene_rgb.ply files found in {input_dir}")

    color_map = {
        "original": None,
        "blue": np.array([30, 60, 180], dtype=np.uint8),
        "cyan": np.array([0, 200, 200], dtype=np.uint8),
    }
    target_color = color_map[args.color]

    frame_summaries = []
    for scene_path in scene_files:
        stem = scene_path.stem.replace("_scene_rgb", "")
        label_path = input_dir / f"{stem}_instance_label.ply"
        if not label_path.is_file():
            label_path = input_dir / f"{stem}_label.ply"
        if not label_path.is_file():
            print(f"  skipping {stem}: label file not found")
            continue

        points, colors = read_ply_points_and_colors(scene_path)
        labels = read_ply_labels(label_path)

        target_mask = labels > 0
        num_target = int(np.count_nonzero(target_mask))
        if num_target == 0:
            print(f"  {stem}: 0 target points, skipping")
            continue

        target_points = points[target_mask]
        target_colors = colors[target_mask]

        if target_color is not None:
            target_colors = np.tile(target_color, (num_target, 1))

        ply_name = f"{stem}_target_only_rgb.ply"
        if args.save_normal:
            from single_seg.single_object_segmenter import estimate_normals_towards_cameras
            normals = estimate_normals_towards_cameras(
                target_points,
                camera_centers=[],
                voxel_size=float(args.voxel_size),
            )
            write_ply_with_normals(output_dir / ply_name, target_points, target_colors, normals)
        else:
            write_ply(output_dir / ply_name, target_points, target_colors)

        frame_summaries.append({
            "frame": stem,
            "target_points": num_target,
            "ply_file": ply_name,
        })
        print(f"  {stem}: {num_target} target points -> {output_dir / ply_name}")

    print(f"\nDone. {len(frame_summaries)} frames extracted to {output_dir}")


if __name__ == "__main__":
    main()
