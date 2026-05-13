#!/usr/bin/env python3
"""Replay a RealSense live debug dump through Fast-FoundationStereo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from single_seg.realsense_rgbd_segmenter import (  # noqa: E402
    FAST_STEREO_DEFAULT_MODEL,
    FastFoundationStereoRunner,
    align_rectified_depth_to_color_torch,
    filter_depth_edges_torch,
)
from single_seg.single_object_segmenter import (  # noqa: E402
    backproject_scene_points_with_labels,
    camera_center_from_pose_record,
    estimate_normals_towards_cameras,
    filter_target_mask_by_depth_band,
    filter_target_labels_by_dominant_plane,
    fuse_scene_geometry,
    write_label_ply,
    write_label_ply_with_normals,
    write_live_debug_target_object_cloud,
    write_ply,
    write_ply_with_normals,
)


DEFAULT_INPUT_DIR = REPO_ROOT / "tests" / "outputs" / "realsense_live_redcup_three_cam_fast"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tests" / "outputs" / "fast_replay_redcup"
TARGET_COLOR = np.array([255, 70, 70], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute Fast-FoundationStereo depth from live_rgbd_debug and rebuild PLYs from saved 2D masks.",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=1)
    parser.add_argument("--camera-ids", default="", help="Comma-separated camera IDs; empty means all cameras.")
    parser.add_argument("--depth-min", type=float, default=0.1)
    parser.add_argument("--depth-max", type=float, default=3.0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--frame-voxel-size", type=float, default=0.002)
    parser.add_argument("--target-3d-mask-erode-kernel", type=int, default=0)
    parser.add_argument("--target-plane-filter-enabled", type=int, default=0)
    parser.add_argument("--target-plane-filter-distance-m", type=float, default=0.004)
    parser.add_argument("--target-plane-filter-min-points", type=int, default=80)
    parser.add_argument("--target-plane-filter-min-inlier-ratio", type=float, default=0.25)
    parser.add_argument("--target-plane-filter-max-inlier-ratio", type=float, default=0.85)
    parser.add_argument("--target-plane-filter-max-planes", type=int, default=1)
    parser.add_argument("--target-plane-filter-ransac-iterations", type=int, default=256)
    parser.add_argument("--target-depth-band-filter-enabled", type=int, default=0)
    parser.add_argument("--target-depth-band-filter-range-m", type=float, default=0.015)
    parser.add_argument("--target-depth-band-filter-min-valid-pixels", type=int, default=50)
    parser.add_argument("--target-depth-band-filter-min-keep-pixels", type=int, default=20)
    parser.add_argument("--fast-model-path", type=Path, default=FAST_STEREO_DEFAULT_MODEL)
    parser.add_argument("--fast-valid-iters", type=int, default=12)
    parser.add_argument("--fast-max-disp", type=int, default=192)
    parser.add_argument("--fast-scale", type=float, default=1.0)
    parser.add_argument("--fast-remove-invisible", type=int, default=1)
    parser.add_argument("--fast-hiera", type=int, default=0)
    parser.add_argument("--fast-optimize-build-volume", choices=("pytorch1", "triton"), default="pytorch1")
    parser.add_argument("--depth-edge-filter-enabled", type=int, default=0)
    parser.add_argument("--depth-edge-filter-threshold-m", type=float, default=0.5)
    parser.add_argument("--depth-edge-filter-stage", choices=("rectified", "aligned"), default="rectified")
    parser.add_argument("--save-depth-debug", type=int, default=1)
    parser.add_argument("--save-normal", type=int, default=1)
    parser.add_argument("--save-camera-target-clouds", type=int, default=1)
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_debug_roots(input_dir: Path) -> tuple[Path, Path]:
    input_dir = input_dir.expanduser().resolve()
    if (input_dir / "live_rgbd_debug").is_dir():
        return input_dir, input_dir / "live_rgbd_debug"
    if input_dir.name == "live_rgbd_debug" and input_dir.is_dir():
        return input_dir.parent, input_dir
    raise FileNotFoundError(f"{input_dir} is not an output root or live_rgbd_debug directory")


def selected_frames(debug_root: Path, frame_index: int, max_frames: int) -> list[Path]:
    frames = sorted(path for path in debug_root.glob("frame_*") if path.is_dir())
    if not frames:
        raise RuntimeError(f"no frame_* directories found under {debug_root}")
    if frame_index < 0 or frame_index >= len(frames):
        raise IndexError(f"--frame-index {frame_index} is outside available range 0..{len(frames) - 1}")
    end = None if max_frames <= 0 else frame_index + max_frames
    return frames[frame_index:end]


def resolve_camera_ids(frame_dir: Path, requested: str) -> list[str]:
    if requested.strip():
        camera_ids = [item.strip() for item in requested.split(",") if item.strip()]
    else:
        camera_ids = sorted(path.name for path in frame_dir.iterdir() if path.is_dir())
    if not camera_ids:
        raise RuntimeError(f"no camera directories found under {frame_dir}")
    missing = [camera_id for camera_id in camera_ids if not (frame_dir / camera_id).is_dir()]
    if missing:
        raise FileNotFoundError(f"missing camera directories under {frame_dir}: {missing}")
    return camera_ids


def load_mask(source_root: Path, frame_name: str, camera_id: str, rgb_shape: tuple[int, int, int]) -> np.ndarray:
    mask_path = source_root / "masks_2d" / frame_name / camera_id / "semantic_label.png"
    if not mask_path.is_file():
        return np.zeros(rgb_shape[:2], dtype=np.uint8)
    mask_image = np.asarray(Image.open(mask_path))
    if mask_image.ndim == 2:
        mask = mask_image > 0
    else:
        mask = np.any(mask_image[..., :3] > 0, axis=2)
    return mask.astype(np.uint8)


def erode_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = int(kernel_size)
    if kernel <= 1:
        return mask.astype(np.uint8, copy=False)
    if kernel % 2 == 0:
        kernel += 1
    footprint = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    return cv2.erode(mask.astype(np.uint8, copy=False), footprint, iterations=1)


def read_ir_pair(camera_dir: Path, payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    left_path = camera_dir / str(payload.get("ir_left_rect_file", "ir_left_rect.png"))
    right_path = camera_dir / str(payload.get("ir_right_rect_file", "ir_right_rect.png"))
    if not left_path.is_file() or not right_path.is_file():
        raise FileNotFoundError(f"missing rectified IR images under {camera_dir}")
    return (
        np.asarray(Image.open(left_path).convert("L"), dtype=np.uint8),
        np.asarray(Image.open(right_path).convert("L"), dtype=np.uint8),
    )


def depth_to_vis(depth: np.ndarray, depth_min: float, depth_max: float) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    vis = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        denom = max(float(depth_max) - float(depth_min), 1e-6)
        norm = np.clip((depth - float(depth_min)) / denom, 0.0, 1.0)
        vis = np.round((1.0 - norm) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)


def disparity_to_vis(disparity: np.ndarray) -> np.ndarray:
    finite = np.isfinite(disparity) & (disparity > 0)
    vis = np.zeros(disparity.shape, dtype=np.uint8)
    if np.any(finite):
        lo, hi = np.percentile(disparity[finite], [2.0, 98.0])
        if hi > lo:
            norm = np.clip((disparity - lo) / (hi - lo), 0.0, 1.0)
            vis = np.round(norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)


def z_stats(points: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    target = np.asarray(labels) > 0
    if not np.any(target):
        return {"count": 0, "percentiles": []}
    z = np.asarray(points, dtype=np.float32)[target, 2]
    percentiles = np.percentile(z, [0, 1, 5, 25, 50, 75, 95, 99, 100]).astype(float).tolist()
    return {"count": int(z.shape[0]), "percentiles": percentiles}


def save_depth_debug(
    *,
    output_dir: Path,
    frame_name: str,
    camera_id: str,
    depth: np.ndarray,
    disparity: np.ndarray,
    depth_min: float,
    depth_max: float,
) -> None:
    camera_dir = output_dir / "fast_depth_debug" / frame_name / camera_id
    camera_dir.mkdir(parents=True, exist_ok=True)
    np.save(camera_dir / "depth_aligned_m.npy", depth.astype(np.float32, copy=False))
    np.save(camera_dir / "disparity.npy", disparity.astype(np.float32, copy=False))
    Image.fromarray(depth_to_vis(depth, depth_min, depth_max)).save(camera_dir / "depth_aligned_vis.png")
    Image.fromarray(disparity_to_vis(disparity)).save(camera_dir / "disparity_vis.png")


def process_frame(
    *,
    frame_dir: Path,
    source_root: Path,
    output_dir: Path,
    camera_ids: list[str],
    stereo_runner: FastFoundationStereoRunner,
    depth_min: float,
    depth_max: float,
    stride: int,
    frame_voxel_size: float,
    target_3d_mask_erode_kernel: int,
    target_plane_filter_enabled: bool,
    target_plane_filter_distance_m: float,
    target_plane_filter_min_points: int,
    target_plane_filter_min_inlier_ratio: float,
    target_plane_filter_max_inlier_ratio: float,
    target_plane_filter_max_planes: int,
    target_plane_filter_ransac_iterations: int,
    target_depth_band_filter_enabled: bool,
    target_depth_band_filter_range_m: float,
    target_depth_band_filter_min_valid_pixels: int,
    target_depth_band_filter_min_keep_pixels: int,
    depth_edge_filter_enabled: bool,
    depth_edge_filter_threshold_m: float,
    depth_edge_filter_stage: str,
    save_depth_debug_enabled: bool,
    save_normal: bool,
    save_camera_target_clouds: bool,
) -> dict[str, Any]:
    point_chunks: list[np.ndarray] = []
    color_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    camera_summaries: list[dict[str, Any]] = []
    camera_centers: list[np.ndarray] = []

    for camera_id in camera_ids:
        camera_dir = frame_dir / camera_id
        payload = load_json(camera_dir / "camera_payload.json")
        rgb = np.asarray(Image.open(camera_dir / "rgb.png").convert("RGB"), dtype=np.uint8)
        left_ir, right_ir = read_ir_pair(camera_dir, payload)
        mask = load_mask(source_root, frame_dir.name, camera_id, rgb.shape)
        mask_for_3d = erode_mask(mask, int(target_3d_mask_erode_kernel))
        color_intrinsics = dict(payload["color_intrinsics"])
        color_intrinsics["width"] = int(rgb.shape[1])
        color_intrinsics["height"] = int(rgb.shape[0])

        t0 = time.perf_counter()
        stereo_output = stereo_runner.infer_depth(
            left_image=left_ir,
            right_image=right_ir,
            rectified_k=np.asarray(payload["rectified_k"], dtype=np.float32),
            baseline_m=float(payload["baseline_m"]),
            return_torch=True,
            include_input_images=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        infer_time = time.perf_counter() - t0

        depth_rect = stereo_output["depth_m"].to(dtype=torch.float32)
        depth_rect = torch.where(
            torch.isfinite(depth_rect) & (depth_rect >= float(depth_min)) & (depth_rect <= float(depth_max)),
            depth_rect,
            torch.zeros((), dtype=torch.float32, device=depth_rect.device),
        )
        depth_color_before_edge: torch.Tensor | None = None
        if depth_edge_filter_enabled and depth_edge_filter_stage == "rectified":
            depth_color_before_edge = align_rectified_depth_to_color_torch(
                depth_rect,
                rectified_intrinsics=stereo_output["rectified_intrinsics"],
                rectified_to_color=np.asarray(payload["rectified_to_color"], dtype=np.float64),
                color_intrinsics=color_intrinsics,
                color_shape=rgb.shape[:2],
            )
            depth_color_before_edge = torch.where(
                torch.isfinite(depth_color_before_edge)
                & (depth_color_before_edge >= float(depth_min))
                & (depth_color_before_edge <= float(depth_max)),
                depth_color_before_edge.to(dtype=torch.float32),
                torch.zeros((), dtype=torch.float32, device=depth_color_before_edge.device),
            )
            depth_rect = filter_depth_edges_torch(depth_rect, threshold_m=float(depth_edge_filter_threshold_m))
        depth_color = align_rectified_depth_to_color_torch(
            depth_rect,
            rectified_intrinsics=stereo_output["rectified_intrinsics"],
            rectified_to_color=np.asarray(payload["rectified_to_color"], dtype=np.float64),
            color_intrinsics=color_intrinsics,
            color_shape=rgb.shape[:2],
        )
        depth_color = torch.where(
            torch.isfinite(depth_color) & (depth_color >= float(depth_min)) & (depth_color <= float(depth_max)),
            depth_color.to(dtype=torch.float32),
            torch.zeros((), dtype=torch.float32, device=depth_color.device),
        )
        stats_depth_before = depth_color if depth_color_before_edge is None else depth_color_before_edge
        valid_before_edge = int(torch.count_nonzero(stats_depth_before > 0).item())
        mask_t = torch.as_tensor(mask > 0, dtype=torch.bool, device=depth_color.device)
        target_valid_before_edge = int(torch.count_nonzero((stats_depth_before > 0) & mask_t).item())
        if depth_edge_filter_enabled and depth_edge_filter_stage == "aligned":
            depth_color = filter_depth_edges_torch(depth_color, threshold_m=float(depth_edge_filter_threshold_m))
        valid_after_edge = int(torch.count_nonzero(depth_color > 0).item())
        target_valid_after_edge = int(torch.count_nonzero((depth_color > 0) & mask_t).item())

        depth_np = depth_color.detach().cpu().numpy().astype(np.float32, copy=False)
        disparity_np = stereo_output["disparity"].detach().cpu().numpy().astype(np.float32, copy=False)
        pose_record = dict(payload["pose_record"])
        camera_center = camera_center_from_pose_record(pose_record)
        if camera_center is not None:
            camera_centers.append(camera_center)
        mask_for_3d, target_depth_band_summary = filter_target_mask_by_depth_band(
            mask_for_3d,
            depth_np,
            enabled=bool(target_depth_band_filter_enabled),
            range_m=float(target_depth_band_filter_range_m),
            min_valid_pixels=int(target_depth_band_filter_min_valid_pixels),
            min_keep_pixels=int(target_depth_band_filter_min_keep_pixels),
        )
        points, colors, labels = backproject_scene_points_with_labels(
            rgb,
            depth_np,
            mask_for_3d,
            np.asarray(pose_record["cam2world_4x4"], dtype=np.float64),
            color_intrinsics,
            None,
            float(depth_min),
            float(depth_max),
            int(stride),
        )
        labels, target_plane_filter_summary = filter_target_labels_by_dominant_plane(
            points,
            labels,
            enabled=bool(target_plane_filter_enabled),
            distance_m=float(target_plane_filter_distance_m),
            min_points=int(target_plane_filter_min_points),
            min_inlier_ratio=float(target_plane_filter_min_inlier_ratio),
            max_inlier_ratio=float(target_plane_filter_max_inlier_ratio),
            max_planes=int(target_plane_filter_max_planes),
            ransac_iterations=int(target_plane_filter_ransac_iterations),
        )
        point_chunks.append(points)
        color_chunks.append(colors)
        label_chunks.append(labels)

        target_points = labels > 0
        if save_camera_target_clouds:
            write_live_debug_target_object_cloud(
                live_debug_root=output_dir / "live_rgbd_debug",
                frame_name=f"{frame_dir.name}.png",
                camera_id=camera_id,
                target_name=str(source_root.name),
                points=points[target_points],
                colors=colors[target_points],
                save_normal=bool(save_normal),
                camera_center=camera_center,
                voxel_size=float(frame_voxel_size),
                target_pixels=int(np.count_nonzero(mask)),
            )
        if save_depth_debug_enabled:
            save_depth_debug(
                output_dir=output_dir,
                frame_name=frame_dir.name,
                camera_id=camera_id,
                depth=depth_np,
                disparity=disparity_np,
                depth_min=float(depth_min),
                depth_max=float(depth_max),
            )

        camera_summaries.append(
            {
                "camera_id": camera_id,
                "infer_time_sec": infer_time,
                "valid_depth_pixels_before_edge": valid_before_edge,
                "valid_depth_pixels_after_edge": valid_after_edge,
                "removed_depth_pixels_by_edge": int(max(valid_before_edge - valid_after_edge, 0)),
                "target_valid_pixels_before_edge": target_valid_before_edge,
                "target_valid_pixels_after_edge": target_valid_after_edge,
                "removed_target_valid_pixels_by_edge": int(
                    max(target_valid_before_edge - target_valid_after_edge, 0)
                ),
                "depth_edge_filter_stage": str(depth_edge_filter_stage),
                "target_pixels_2d": int(np.count_nonzero(mask)),
                "target_pixels_3d_mask": int(np.count_nonzero(mask_for_3d)),
                "target_depth_band_filter": target_depth_band_summary,
                "backprojected_points": int(points.shape[0]),
                "target_points": int(np.count_nonzero(target_points)),
                "target_plane_filter": target_plane_filter_summary,
                "target_z_stats": z_stats(points, labels),
            }
        )

    points_xyz, colors_rgb, labels = fuse_scene_geometry(
        point_chunks,
        color_chunks,
        label_chunks,
        float(frame_voxel_size),
    )
    instance_colors = colors_rgb.copy()
    instance_colors[labels > 0] = TARGET_COLOR
    frame_output_dir = output_dir / "frame_outputs"
    frame_output_dir.mkdir(parents=True, exist_ok=True)
    normals: np.ndarray | None = None
    if save_normal:
        normals = estimate_normals_towards_cameras(
            points_xyz,
            camera_centers=camera_centers,
            voxel_size=float(frame_voxel_size),
        )
        write_ply_with_normals(frame_output_dir / f"{frame_dir.name}_scene_rgb.ply", points_xyz, colors_rgb, normals)
        write_ply_with_normals(
            frame_output_dir / f"{frame_dir.name}_instance_rgb.ply",
            points_xyz,
            instance_colors,
            normals,
        )
        write_label_ply_with_normals(frame_output_dir / f"{frame_dir.name}_label.ply", points_xyz, labels, normals)
    else:
        write_ply(frame_output_dir / f"{frame_dir.name}_scene_rgb.ply", points_xyz, colors_rgb)
        write_ply(frame_output_dir / f"{frame_dir.name}_instance_rgb.ply", points_xyz, instance_colors)
        write_label_ply(frame_output_dir / f"{frame_dir.name}_label.ply", points_xyz, labels)

    meta = {
        "frame_name": frame_dir.name,
        "num_points": int(points_xyz.shape[0]),
        "num_labeled_points": int(np.count_nonzero(labels > 0)),
        "target_z_stats": z_stats(points_xyz, labels),
        "has_normals": bool(normals is not None),
        "camera_summaries": camera_summaries,
    }
    (frame_output_dir / f"{frame_dir.name}_instance_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meta


def main() -> None:
    args = parse_args()
    source_root, debug_root = resolve_debug_roots(args.input_dir)
    frames = selected_frames(debug_root, int(args.frame_index), int(args.max_frames))
    camera_ids = resolve_camera_ids(frames[0], str(args.camera_ids))
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and args.overwrite_output:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = FastFoundationStereoRunner(
        model_path=Path(args.fast_model_path),
        valid_iters=int(args.fast_valid_iters),
        max_disp=int(args.fast_max_disp),
        scale=float(args.fast_scale),
        remove_invisible=bool(args.fast_remove_invisible),
        hiera=bool(args.fast_hiera),
        optimize_build_volume=str(args.fast_optimize_build_volume),
    )
    summary: dict[str, Any] = {
        "input_dir": str(source_root),
        "debug_root": str(debug_root),
        "output_dir": str(output_dir),
        "camera_ids": camera_ids,
        "frames": [],
        "fast_stereo": {
            "model_path": str(Path(args.fast_model_path).expanduser().resolve()),
            "valid_iters": int(args.fast_valid_iters),
            "max_disp": int(args.fast_max_disp),
            "scale": float(args.fast_scale),
            "remove_invisible": bool(args.fast_remove_invisible),
            "hiera": bool(args.fast_hiera),
            "optimize_build_volume": str(args.fast_optimize_build_volume),
            "depth_edge_filter_enabled": bool(args.depth_edge_filter_enabled),
            "depth_edge_filter_threshold_m": float(args.depth_edge_filter_threshold_m),
            "depth_edge_filter_stage": str(args.depth_edge_filter_stage),
        },
        "segmenter": {
            "depth_min": float(args.depth_min),
            "depth_max": float(args.depth_max),
            "stride": int(args.stride),
            "frame_voxel_size": float(args.frame_voxel_size),
            "target_3d_mask_erode_kernel": int(args.target_3d_mask_erode_kernel),
            "target_plane_filter_enabled": bool(args.target_plane_filter_enabled),
            "target_plane_filter_distance_m": float(args.target_plane_filter_distance_m),
            "target_plane_filter_min_points": int(args.target_plane_filter_min_points),
            "target_plane_filter_min_inlier_ratio": float(args.target_plane_filter_min_inlier_ratio),
            "target_plane_filter_max_inlier_ratio": float(args.target_plane_filter_max_inlier_ratio),
            "target_plane_filter_max_planes": int(args.target_plane_filter_max_planes),
            "target_plane_filter_ransac_iterations": int(args.target_plane_filter_ransac_iterations),
            "target_depth_band_filter_enabled": bool(args.target_depth_band_filter_enabled),
            "target_depth_band_filter_range_m": float(args.target_depth_band_filter_range_m),
            "target_depth_band_filter_min_valid_pixels": int(args.target_depth_band_filter_min_valid_pixels),
            "target_depth_band_filter_min_keep_pixels": int(args.target_depth_band_filter_min_keep_pixels),
            "save_depth_debug": bool(args.save_depth_debug),
            "save_normal": bool(args.save_normal),
            "save_camera_target_clouds": bool(args.save_camera_target_clouds),
        },
    }
    t0 = time.perf_counter()
    for frame_dir in frames:
        frame_summary = process_frame(
            frame_dir=frame_dir,
            source_root=source_root,
            output_dir=output_dir,
            camera_ids=camera_ids,
            stereo_runner=runner,
            depth_min=float(args.depth_min),
            depth_max=float(args.depth_max),
            stride=int(args.stride),
            frame_voxel_size=float(args.frame_voxel_size),
            target_3d_mask_erode_kernel=int(args.target_3d_mask_erode_kernel),
            target_plane_filter_enabled=bool(args.target_plane_filter_enabled),
            target_plane_filter_distance_m=float(args.target_plane_filter_distance_m),
            target_plane_filter_min_points=int(args.target_plane_filter_min_points),
            target_plane_filter_min_inlier_ratio=float(args.target_plane_filter_min_inlier_ratio),
            target_plane_filter_max_inlier_ratio=float(args.target_plane_filter_max_inlier_ratio),
            target_plane_filter_max_planes=int(args.target_plane_filter_max_planes),
            target_plane_filter_ransac_iterations=int(args.target_plane_filter_ransac_iterations),
            target_depth_band_filter_enabled=bool(args.target_depth_band_filter_enabled),
            target_depth_band_filter_range_m=float(args.target_depth_band_filter_range_m),
            target_depth_band_filter_min_valid_pixels=int(args.target_depth_band_filter_min_valid_pixels),
            target_depth_band_filter_min_keep_pixels=int(args.target_depth_band_filter_min_keep_pixels),
            depth_edge_filter_enabled=bool(args.depth_edge_filter_enabled),
            depth_edge_filter_threshold_m=float(args.depth_edge_filter_threshold_m),
            depth_edge_filter_stage=str(args.depth_edge_filter_stage),
            save_depth_debug_enabled=bool(args.save_depth_debug),
            save_normal=bool(args.save_normal),
            save_camera_target_clouds=bool(args.save_camera_target_clouds),
        )
        summary["frames"].append(frame_summary)
        infer_total = sum(float(item["infer_time_sec"]) for item in frame_summary["camera_summaries"])
        removed_target = sum(
            int(item["removed_target_valid_pixels_by_edge"]) for item in frame_summary["camera_summaries"]
        )
        print(
            f"{frame_dir.name} points={frame_summary['num_points']} labeled={frame_summary['num_labeled_points']} "
            f"stereo_infer={infer_total:.3f}s removed_target_depth={removed_target}"
        )
    summary["elapsed_sec"] = time.perf_counter() - t0
    (output_dir / "fast_replay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved outputs under {output_dir}")


if __name__ == "__main__":
    main()
