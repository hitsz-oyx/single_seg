#!/usr/bin/env python3
"""Aggregate multi-target offline segmentation results into per-frame labeled point clouds.

Reads label PLY files and confidence scores from offline_{target_name}/frame_XXXXX/
directories and merges them into a single point cloud per frame. When multiple targets
claim the same voxel, the target with the highest confidence score wins.

Example:
    python utils/aggregate_offline_labels.py \
        --config configs/offline_replay.yaml

    python utils/aggregate_offline_labels.py \
        --base-output-dir tests/outputs/demo0_1_ds3_offline \
        --targets-json assets/prompts/targets.json \
        --targets redcup bowl \
        --voxel-size 0.003 \
        --output-dir tests/outputs/demo0_1_ds3_aggregated
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from single_seg.single_object_segmenter import write_label_ply


def write_ply_with_label(path: Path, points: np.ndarray, colors: np.ndarray, labels: np.ndarray) -> None:
    if points.shape[0] != colors.shape[0] or points.shape[0] != labels.shape[0]:
        raise ValueError("points, colors, and labels must have the same length")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property int label\n"
        "end_header\n"
    ).encode("ascii")
    verts = np.empty(
        points.shape[0],
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("label", "<i4")],
    )
    verts["x"], verts["y"], verts["z"] = points[:, 0], points[:, 1], points[:, 2]
    verts["red"], verts["green"], verts["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    verts["label"] = labels.astype(np.int32, copy=False)
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(verts.tobytes())


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_ply_header(path: Path) -> tuple[int, list[str], int]:
    """Parse PLY header, return (num_vertices, property_names, header_bytes)."""
    with open(path, "rb") as f:
        header_bytes = 0
        num_vertices = 0
        properties: list[str] = []
        while True:
            line = f.readline()
            header_bytes += len(line)
            text = line.decode("ascii").strip()
            if line.startswith(b"element vertex"):
                num_vertices = int(text.split()[2])
            elif line.startswith(b"property"):
                properties.append(text.split()[2])
            if line.startswith(b"end_header"):
                break
    return num_vertices, properties, header_bytes


_PROP_DTYPE: dict[str, tuple[str, int]] = {
    "x": ("<f4", 4), "y": ("<f4", 4), "z": ("<f4", 4),
    "nx": ("<f4", 4), "ny": ("<f4", 4), "nz": ("<f4", 4),
    "red": ("u1", 1), "green": ("u1", 1), "blue": ("u1", 1),
    "label": ("<i4", 4),
}


def read_ply_xyz_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    num_vertices, properties, header_bytes = _parse_ply_header(path)
    dtype = np.dtype([(p, _PROP_DTYPE[p][0]) for p in properties if p in _PROP_DTYPE])
    with open(path, "rb") as f:
        f.seek(header_bytes)
        data = np.frombuffer(f.read(), dtype=dtype, count=num_vertices)
    points = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    labels = data["label"].astype(np.int32)
    return points, labels


def read_ply_scene_rgb(path: Path) -> np.ndarray:
    num_vertices, properties, header_bytes = _parse_ply_header(path)
    dtype = np.dtype([(p, _PROP_DTYPE[p][0]) for p in properties if p in _PROP_DTYPE])
    with open(path, "rb") as f:
        f.seek(header_bytes)
        data = np.frombuffer(f.read(), dtype=dtype, count=num_vertices)
    colors = np.stack([data["red"], data["green"], data["blue"]], axis=1).astype(np.uint8)
    return colors


def voxel_merge_by_confidence(
    all_points: list[np.ndarray],
    all_labels: list[np.ndarray],
    all_colors: list[np.ndarray],
    all_confidences: list[np.ndarray],
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge points from multiple targets using voxel grid, picking the label with highest confidence per voxel."""
    if not all_points:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int32), np.empty((0, 3), dtype=np.uint8)

    points = np.concatenate(all_points, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    confidences = np.concatenate(all_confidences, axis=0)

    if points.shape[0] == 0 or voxel_size <= 0.0:
        return points, labels, colors

    voxel_keys = np.floor(points / voxel_size).astype(np.int64)
    voxel_keys -= voxel_keys.min(axis=0, keepdims=True)

    inverse = np.unique(voxel_keys, axis=0, return_inverse=True)[1]
    counts = np.bincount(inverse, minlength=int(inverse.max()) + 1)

    point_sum = np.zeros((counts.shape[0], 3), dtype=np.float64)
    np.add.at(point_sum, inverse, points.astype(np.float64))
    down_points = (point_sum / counts[:, None]).astype(np.float32)

    order = np.lexsort((-confidences, inverse))
    _, first_idx = np.unique(inverse[order], return_index=True)
    best_indices = order[first_idx]

    best_labels = labels[best_indices].astype(np.int32)
    best_colors = colors[best_indices].astype(np.uint8)

    return down_points, best_labels, best_colors


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate multi-target offline segmentation results")
    parser.add_argument("--config", type=Path, default=None, help="YAML config file")
    parser.add_argument("--base-output-dir", type=Path, default=None,
                        help="Base output dir containing offline_{target}/frame_XXXXX/ subdirs")
    parser.add_argument("--targets-json", type=Path, default=None,
                        help="Path to targets.json mapping target names to IDs and colors")
    parser.add_argument("--targets", nargs="+", default=None,
                        help="List of target names to aggregate (e.g. redcup bowl)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for aggregated PLY files")
    parser.add_argument("--voxel-size", type=float, default=0.003,
                        help="Voxel size for merging overlapping points from different targets")
    args = parser.parse_args()

    if args.config:
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        base_output_dir = Path(cfg["output_dir"]).expanduser().resolve()
        output_dir = base_output_dir.parent / (base_output_dir.name + "_aggregated")
        targets = cfg.get("targets", [])
        targets_json = Path(cfg.get("targets_json", "assets/prompts/targets.json")).expanduser().resolve()
        voxel_size = float(cfg.get("voxel_size", args.voxel_size))
    else:
        if args.base_output_dir is None or args.targets is None or args.targets_json is None:
            raise ValueError("Must provide --config OR all of --base-output-dir, --targets, --targets-json")
        base_output_dir = args.base_output_dir.expanduser().resolve()
        output_dir = args.output_dir.expanduser().resolve() if args.output_dir else None
        targets = args.targets
        targets_json = args.targets_json.expanduser().resolve()
        voxel_size = args.voxel_size
        if output_dir is None:
            output_dir = base_output_dir.parent / (base_output_dir.name + "_aggregated")

    target_info = load_json(targets_json)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_dirs: set[str] = set()
    for target_name in targets:
        target_dir = base_output_dir / f"offline_{target_name}"
        if not target_dir.is_dir():
            print(f"WARNING: {target_dir} not found, skipping")
            continue
        for fd in target_dir.iterdir():
            if fd.is_dir():
                frame_dirs.add(fd.name)

    frame_dirs_sorted = sorted(frame_dirs)
    print(f"Aggregating {len(targets)} targets x {len(frame_dirs_sorted)} frames")
    print(f"Voxel size: {voxel_size:.3f}")
    print(f"Output directory: {output_dir}")

    total_frames = len(frame_dirs_sorted)
    for frame_idx, frame_name in enumerate(frame_dirs_sorted):
        all_points: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        all_vis_colors: list[np.ndarray] = []
        all_orig_colors: list[np.ndarray] = []
        all_confidences: list[np.ndarray] = []

        full_all_points: list[np.ndarray] = []
        full_all_labels: list[np.ndarray] = []
        full_all_confidences: list[np.ndarray] = []

        scene_rgb_loaded = False
        scene_colors: np.ndarray | None = None

        for target_name in targets:
            info = target_info[target_name]
            target_id = info["id"]
            target_color = np.array(info["color"], dtype=np.uint8)

            frame_dir = base_output_dir / f"offline_{target_name}" / frame_name / "frame_outputs"
            label_ply = frame_dir / f"{frame_name}_instance_label.ply"
            scene_rgb_ply = frame_dir / f"{frame_name}_scene_rgb.ply"
            confidence_npy = frame_dir / f"{frame_name}_instance_confidence.npy"
            if not label_ply.is_file():
                print(f"  [{frame_idx+1}/{total_frames}] {frame_name}/{target_name}: label file not found, skipping")
                continue

            if not confidence_npy.is_file():
                raise FileNotFoundError(
                    f"Confidence file not found: {confidence_npy}. "
                    "Re-run offline_replay.py for this target to generate confidence scores."
                )

            points, labels = read_ply_xyz_labels(label_ply)
            confidences = np.load(str(confidence_npy)).astype(np.float32, copy=False)
            if confidences.shape[0] != labels.shape[0]:
                print(
                    f"ERROR: {target_name}/{frame_name}: confidence has {confidences.shape[0]} points "
                    f"but PLY has {labels.shape[0]} points. "
                )
                print(f"  Delete {label_ply.parent} and re-run target '{target_name}'")
                raise SystemExit(1)

            if not scene_rgb_loaded and scene_rgb_ply.is_file():
                scene_colors = read_ply_scene_rgb(scene_rgb_ply)
                scene_rgb_loaded = True

            full_all_points.append(points)
            full_all_labels.append(labels)
            full_conf = confidences.copy()
            full_conf[labels == 0] = 0.0
            full_all_confidences.append(full_conf)

            valid_mask = labels == target_id
            if not np.any(valid_mask):
                print(f"  [{frame_idx+1}/{total_frames}] {frame_name}/{target_name}: no valid points (label={target_id}), skipping")
                continue

            valid_points = points[valid_mask]
            valid_labels = labels[valid_mask]
            valid_vis_colors = np.tile(target_color, (valid_points.shape[0], 1))
            valid_orig_colors = scene_colors[valid_mask] if scene_colors is not None else valid_vis_colors
            valid_confidences = confidences[valid_mask]

            all_points.append(valid_points)
            all_labels.append(valid_labels)
            all_vis_colors.append(valid_vis_colors)
            all_orig_colors.append(valid_orig_colors)
            all_confidences.append(valid_confidences)

        if not all_points:
            print(f"  [{frame_idx+1}/{total_frames}] {frame_name}: no valid points from any target, skipping")
            continue

        combined_points, combined_labels, combined_vis_colors = voxel_merge_by_confidence(
            all_points, all_labels, all_vis_colors, all_confidences, voxel_size,
        )

        label_path = output_dir / f"{frame_name}_aggregated.ply"
        write_label_ply(label_path, combined_points, combined_labels)
        print(f"  [{frame_idx+1}/{total_frames}] {frame_name}: {combined_points.shape[0]} points -> {label_path.name}")

        rgb_path = output_dir / f"{frame_name}_aggregated_instance_rgb.ply"
        write_ply_with_label(rgb_path, combined_points, combined_vis_colors, combined_labels)
        print(f"  [{frame_idx+1}/{total_frames}] {frame_name}: {combined_points.shape[0]} points -> {rgb_path.name}")

        _, _, combined_orig_colors = voxel_merge_by_confidence(
            all_points, all_labels, all_orig_colors, all_confidences, voxel_size,
        )
        orig_path = output_dir / f"{frame_name}_aggregated_original_rgb.ply"
        write_ply_with_label(orig_path, combined_points, combined_orig_colors, combined_labels)
        print(f"  [{frame_idx+1}/{total_frames}] {frame_name}: {combined_points.shape[0]} points -> {orig_path.name}")

        if full_all_points and scene_colors is not None:
            full_points, full_labels, full_colors = voxel_merge_by_confidence(
                full_all_points, full_all_labels,
                [scene_colors] * len(full_all_points),
                full_all_confidences, voxel_size,
            )
            full_path = output_dir / f"{frame_name}_full_scene.ply"
            write_ply_with_label(full_path, full_points, full_colors, full_labels)
            print(f"  [{frame_idx+1}/{total_frames}] {frame_name}: {full_points.shape[0]} points -> {full_path.name}")

    print(f"\nAll done. Aggregated {total_frames} frames to {output_dir}")


if __name__ == "__main__":
    main()
