#!/usr/bin/env python3
"""Replay SAM3 single-object segmentation from a RealSense live_rgbd_debug dump."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from single_seg.single_object_segmenter import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    SingleObjectPointCloudSegmenter,
)


DEFAULT_INPUT_DIR = REPO_ROOT / "tests" / "outputs" / "realsense_live_redcup_three_cam_fast"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tests" / "outputs" / "sam3_replay_redcup"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM3 segmentation on frames saved by --save-live-debug 1."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--prompt-task-info", type=Path, required=True)
    parser.add_argument("--prompt-image-root", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=1)
    parser.add_argument("--camera-ids", default="", help="Comma-separated camera IDs; empty means all cameras.")
    parser.add_argument("--depth-device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--tracker-image-size", type=int, default=896)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--mask-threshold", type=float, default=0.6)
    parser.add_argument("--prompt-keep-score-threshold", type=float, default=0.2)
    parser.add_argument("--video-mask-prob-threshold", type=float, default=0.95)
    parser.add_argument("--depth-min", type=float, default=0.1)
    parser.add_argument("--depth-max", type=float, default=3.0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--frame-voxel-size", type=float, default=0.002)
    parser.add_argument("--target-cluster-filter-enabled", type=int, default=0)
    parser.add_argument("--target-cluster-radius-m", type=float, default=0.013)
    parser.add_argument("--target-cluster-min-points", type=int, default=45)
    parser.add_argument("--target-cluster-keep-largest", type=int, default=1)
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
    parser.add_argument("--target-3d-mask-erode-kernel", type=int, default=0)
    parser.add_argument("--save-ply", type=int, default=1)
    parser.add_argument("--save-normal", type=int, default=0)
    parser.add_argument("--save-debug-2d", type=int, default=1)
    parser.add_argument("--save-live-debug", type=int, default=1)
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument(
        "--target-vis-color",
        type=str,
        default=None,
        help="目标可视化颜色 R,G,B，默认 (255,70,70) 红色。例如 --target-vis-color 30,60,180 深蓝",
    )
    return parser.parse_args()


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


def resolve_depth_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--depth-device cuda requested, but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_camera_input(camera_dir: Path, camera_id: str, depth_device: torch.device) -> dict[str, object]:
    payload = load_json(camera_dir / "camera_payload.json")
    rgb = np.asarray(Image.open(camera_dir / "rgb.png").convert("RGB"), dtype=np.uint8)
    depth_np = np.load(camera_dir / "depth_aligned_m.npy").astype(np.float32, copy=False)
    depth_m: np.ndarray | torch.Tensor
    if depth_device.type == "cuda":
        depth_m = torch.as_tensor(depth_np, dtype=torch.float32, device=depth_device)
    else:
        depth_m = depth_np
    intrinsics = dict(payload.get("depth_intrinsics") or payload.get("color_intrinsics") or {})
    intrinsics["width"] = int(rgb.shape[1])
    intrinsics["height"] = int(rgb.shape[0])
    pose_record = dict(payload["pose_record"])
    pose_record.setdefault("camera_id", str(camera_id))
    return {
        "rgb": rgb,
        "depth_m": depth_m,
        "intrinsics": intrinsics,
        "pose_record": pose_record,
        "fovy_deg": None,
    }


def _parse_color(color_str: str | None) -> tuple[int, int, int] | None:
    if color_str is None:
        return None
    parts = [int(c.strip()) for c in color_str.split(",")]
    if len(parts) != 3:
        raise ValueError(f"--target-vis-color must be R,G,B, got {color_str!r}")
    return (parts[0], parts[1], parts[2])


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SAM3 tracker replay requires CUDA; torch.cuda.is_available() is false")
    source_root, debug_root = resolve_debug_roots(args.input_dir)
    frames = selected_frames(debug_root, int(args.frame_index), int(args.max_frames))
    camera_ids = resolve_camera_ids(frames[0], str(args.camera_ids))
    depth_device = resolve_depth_device(str(args.depth_device))
    live_debug_root = Path(args.output_dir).expanduser().resolve() / "live_rgbd_debug" if args.save_live_debug else None

    print(f"input={source_root}")
    print(f"debug_root={debug_root}")
    print(f"output={Path(args.output_dir).expanduser().resolve()}")
    print(f"frames={len(frames)} cameras={camera_ids} depth_device={depth_device}")

    with SingleObjectPointCloudSegmenter(
        target_name=str(args.target_name),
        prompt_task_info=Path(args.prompt_task_info).expanduser().resolve(),
        prompt_image_root=Path(args.prompt_image_root).expanduser().resolve(),
        checkpoint_path=Path(args.checkpoint_path).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        overwrite_output=bool(args.overwrite_output),
        confidence=float(args.confidence),
        mask_threshold=float(args.mask_threshold),
        prompt_keep_score_threshold=float(args.prompt_keep_score_threshold),
        video_mask_prob_threshold=float(args.video_mask_prob_threshold),
        depth_scale=1.0,
        depth_min=float(args.depth_min),
        depth_max=float(args.depth_max),
        stride=int(args.stride),
        frame_voxel_size=float(args.frame_voxel_size),
        target_cluster_filter_enabled=bool(args.target_cluster_filter_enabled),
        target_cluster_radius_m=float(args.target_cluster_radius_m),
        target_cluster_min_points=int(args.target_cluster_min_points),
        target_cluster_keep_largest=bool(args.target_cluster_keep_largest),
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
        target_3d_mask_erode_kernel=int(args.target_3d_mask_erode_kernel),
        save_ply=bool(args.save_ply),
        save_normal=bool(args.save_normal),
        save_debug_2d=bool(args.save_debug_2d),
        tracker_image_size=int(args.tracker_image_size),
        target_vis_color=_parse_color(args.target_vis_color),
    ) as segmenter:
        for frame_dir in frames:
            camera_inputs = {
                camera_id: load_camera_input(frame_dir / camera_id, camera_id, depth_device)
                for camera_id in camera_ids
            }
            result = segmenter.process_frame(
                frame_name=f"{frame_dir.name}.png",
                camera_inputs=camera_inputs,
                live_debug_root=live_debug_root,
            )
            print(
                f"{frame_dir.name} points={int(result['points_xyz'].shape[0])} "
                f"labeled={int(torch.count_nonzero(result['instance_labels'] > 0).item())}"
            )

    print(f"saved outputs under {Path(args.output_dir).expanduser().resolve()}")


if __name__ == "__main__":
    main()
