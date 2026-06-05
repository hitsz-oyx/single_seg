#!/usr/bin/env python3
"""Aggregate multi-target offline segmentation results into per-frame labeled point clouds.

Reads instance_label.ply (labels only) and scene_rgb.ply (points + colors) from
offline_{target_name}/frame_XXXXX/ directories. All targets share the same scene
point cloud with different label annotations (0=background, N=target_id).
Labels are merged by taking the per-point maximum across all targets, then each
target's labeled points are clustered (DBSCAN) and only the dominant cluster is
retained — outlier points are relabeled as background.

Outputs per frame:
  - frame_XXXXX_instance_rgb.ply   : target-colored points (label > 0 only)
  - frame_XXXXX_original_rgb.ply   : original-RGB points (label > 0 only)
  - frame_XXXXX_full_scene.ply     : all scene points with merged labels

Example:
    python utils/aggregate_offline_labels.py \
        --config configs/offline_replay.yaml

    python utils/aggregate_offline_labels.py \
        --base-output-dir tests/outputs/demo0_1_ds3_offline \
        --targets-json assets/prompts/targets.json \
        --targets redcup bowl \
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
from scipy.spatial import KDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def detect_targets_from_dir(base_output_dir: Path, target_info: dict[str, Any]) -> list[str]:
    targets: list[str] = []
    if not base_output_dir.is_dir():
        return targets
    for item in sorted(base_output_dir.iterdir()):
        if item.is_dir() and item.name.startswith("offline_"):
            target_name = item.name[len("offline_"):]
            if target_name in target_info:
                targets.append(target_name)
            else:
                print(f"WARNING: Found directory {item.name} but target '{target_name}' not in targets.json, skipping")
    return targets


def refine_labels_by_cluster(
    points: np.ndarray,
    labels: np.ndarray,
    eps: float = 0.03,
    min_samples: int = 10,
    knn_iter: int = 2,
) -> np.ndarray:
    """多轮质心距离 + KNN 双重检查修正标签。

    每轮迭代：
    1. 重新计算当前 label 下各目标的质心和半径
    2. 对每个标记点检查：
       - 如果离自己质心过远 → 噪声，标为 0
       - 最近质心属于别的目标，且 KNN 多数也属于该目标 → 修正
    3. 修正后的 label 影响下一轮的质心和 KNN 计算
    """
    current = labels.copy()

    for iteration in range(knn_iter):
        unique_targets = np.unique(current[current > 0])
        if len(unique_targets) == 0:
            break

        centroids: dict[int, np.ndarray] = {}
        radii: dict[int, float] = {}

        for tid in unique_targets:
            mask = current == tid
            pts = points[mask]
            centroid = pts.mean(axis=0)
            centroids[int(tid)] = centroid
            dists = np.sqrt(((pts - centroid) ** 2).sum(axis=1))
            radii[int(tid)] = float(np.percentile(dists, 95))

        tid_list = np.array(list(centroids.keys()), dtype=np.int32)
        centroid_arr = np.array([centroids[t] for t in tid_list])

        labeled_mask = current > 0
        labeled_idx = np.where(labeled_mask)[0]
        n_labeled = len(labeled_idx)

        if n_labeled <= min_samples:
            break

        tree = KDTree(points[labeled_idx])
        k = min(min_samples, n_labeled - 1)
        _, indices = tree.query(points[labeled_idx], k=k + 1)

        n_relabeled = 0
        n_noise = 0
        changes = np.zeros(len(current), dtype=bool)

        for i, pt_idx in enumerate(labeled_idx):
            pt = points[pt_idx]
            current_label = int(current[pt_idx])

            dists = np.sqrt(((centroid_arr - pt) ** 2).sum(axis=1))
            nearest_idx = int(np.argmin(dists))
            nearest_tid = int(tid_list[nearest_idx])

            own_radius = radii[current_label]
            own_dist = float(dists[tid_list == current_label][0])

            if own_dist > eps + own_radius:
                current[pt_idx] = 0
                changes[pt_idx] = True
                n_noise += 1
                continue

            if nearest_tid == current_label:
                continue

            neighbor_labels = current[labeled_idx[indices[i, 1:]]]
            values, counts = np.unique(neighbor_labels, return_counts=True)
            majority_label = int(values[np.argmax(counts)])

            if majority_label == nearest_tid and majority_label != current_label:
                current[pt_idx] = nearest_tid
                changes[pt_idx] = True
                n_relabeled += 1

        if n_relabeled > 0 or n_noise > 0:
            print(f"    iter {iteration + 1}: relabeled {n_relabeled}, removed {n_noise} noise")
        if not changes.any():
            break

    return current


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate multi-target offline segmentation results")
    parser.add_argument("--config", type=Path, default=None, help="YAML config file")
    parser.add_argument("--base-output-dir", type=Path, default=None,
                        help="Base output dir containing offline_{target}/frame_XXXXX/ subdirs")
    parser.add_argument("--targets-json", type=Path, default=None,
                        help="Path to targets.json mapping target names to IDs and colors")
    parser.add_argument("--targets", nargs="*", default=None,
                        help="List of target names to aggregate (e.g. redcup bowl); if omitted, auto-detect from base_output_dir")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for aggregated PLY files")
    parser.add_argument("--cluster-eps", type=float, default=0.03,
                        help="DBSCAN 聚类半径(m)，用于离群点过滤（默认0.03）")
    parser.add_argument("--cluster-min-samples", type=int, default=10,
                        help="DBSCAN/KNN 最小样本数（默认10）")
    parser.add_argument("--knn-iter", type=int, default=2,
                        help="KNN 迭代次数（默认2，多次迭代逐渐修正边界点）")
    args = parser.parse_args()
    cfg: dict[str, Any] = {}

    if args.config:
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        base_output_dir = Path(cfg["output_dir"]).expanduser().resolve()
        output_dir = base_output_dir.parent / (base_output_dir.name + "_aggregated")
        targets = cfg.get("targets", [])
        targets_json = Path(cfg.get("targets_json", "assets/prompts/targets.json")).expanduser().resolve()
    else:
        if args.base_output_dir is None or args.targets_json is None:
            raise ValueError("Must provide --config OR --base-output-dir and --targets-json")
        base_output_dir = args.base_output_dir.expanduser().resolve()
        output_dir = args.output_dir.expanduser().resolve() if args.output_dir else None
        targets = args.targets or []
        targets_json = args.targets_json.expanduser().resolve()
        if output_dir is None:
            output_dir = base_output_dir.parent / (base_output_dir.name + "_aggregated")

    target_info = load_json(targets_json)
    if not targets:
        detected = detect_targets_from_dir(base_output_dir, target_info)
        if not detected:
            print(f"ERROR: No targets specified and no offline_* directories found in {base_output_dir}")
            raise SystemExit(1)
        targets = detected
        print(f"Auto-detected targets: {targets}")
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
    print(f"Output directory: {output_dir}")

    total_frames = len(frame_dirs_sorted)
    for frame_idx, frame_name in enumerate(frame_dirs_sorted):
        all_labels: list[np.ndarray] = []
        scene_points: np.ndarray | None = None
        scene_colors: np.ndarray | None = None
        first_target = targets[0]

        for target_name in targets:
            frame_dir = base_output_dir / f"offline_{target_name}" / frame_name / "frame_outputs"
            label_ply = frame_dir / f"{frame_name}_instance_label.ply"
            if not label_ply.is_file():
                print(f"  [{frame_idx+1}/{total_frames}] {frame_name}/{target_name}: label file not found, skipping")
                continue

            points, labels = read_ply_xyz_labels(label_ply)
            all_labels.append(labels)

            if scene_points is None:
                scene_points = points

        if not all_labels:
            print(f"  [{frame_idx+1}/{total_frames}] {frame_name}: no valid labels from any target, skipping")
            continue

        cluster_eps = args.cluster_eps or (cfg.get("cluster_eps", 0.03) if args.config else 0.03)
        cluster_min_samples = args.cluster_min_samples or (cfg.get("cluster_min_samples", 10) if args.config else 10)

        merged_labels = np.zeros_like(all_labels[0])
        for labels in all_labels:
            np.maximum(merged_labels, labels, out=merged_labels)

        if cluster_eps > 0 and scene_points is not None:
            merged_labels = refine_labels_by_cluster(
                scene_points, merged_labels,
                eps=float(cluster_eps),
                min_samples=int(cluster_min_samples),
                knn_iter=int(args.knn_iter or (cfg.get("knn_iter", 2) if args.config else 2)),
            )

        scene_rgb_ply = (base_output_dir / f"offline_{first_target}" / frame_name
                         / "frame_outputs" / f"{frame_name}_scene_rgb.ply")
        if scene_rgb_ply.is_file():
            scene_colors = read_ply_scene_rgb(scene_rgb_ply)

        frame_out_dir = output_dir / frame_name
        frame_out_dir.mkdir(parents=True, exist_ok=True)

        full_path = frame_out_dir / f"{frame_name}_full_scene.ply"
        write_ply_with_label(full_path, scene_points, scene_colors, merged_labels)
        print(f"  [{frame_idx+1}/{total_frames}] {frame_name}: {scene_points.shape[0]} points -> {full_path}")

        valid_mask = merged_labels > 0
        if np.any(valid_mask):
            valid_labels = merged_labels[valid_mask]
            valid_vis_colors = np.zeros((valid_labels.shape[0], 3), dtype=np.uint8)
            for target_name in targets:
                info = target_info[target_name]
                target_id = int(info["id"])
                target_mask = valid_labels == target_id
                valid_vis_colors[target_mask] = np.array(info["color"], dtype=np.uint8)

            valid_orig_colors = scene_colors[valid_mask] if scene_colors is not None else valid_vis_colors
            valid_points = scene_points[valid_mask]

            rgb_path = frame_out_dir / f"{frame_name}_instance_rgb.ply"
            write_ply_with_label(rgb_path, valid_points, valid_vis_colors, valid_labels)
            print(f"  [{frame_idx+1}/{total_frames}] {frame_name}: {valid_points.shape[0]} points -> {rgb_path}")

            orig_path = frame_out_dir / f"{frame_name}_original_rgb.ply"
            write_ply_with_label(orig_path, valid_points, valid_orig_colors, valid_labels)
            print(f"  [{frame_idx+1}/{total_frames}] {frame_name}: {valid_points.shape[0]} points -> {orig_path}")

    print(f"\nAll done. Aggregated {total_frames} frames to {output_dir}")


if __name__ == "__main__":
    main()