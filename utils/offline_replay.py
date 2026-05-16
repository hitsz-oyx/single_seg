#!/usr/bin/env python3
"""Unified offline replay: Fast-Stereo + SAM3 on a pre-converted live_rgbd_debug directory.

Use convert_demo0_to_live_debug.py (or equivalent) first to produce live_rgbd_debug format.
Then this script runs Fast-Stereo depth estimation + SAM3 single-object segmentation.

No dependency on replay_fast_debug_dump.py or replay_sam3_segmenter_debug_dump.py.

Example:
    # Step 1: convert raw data to live_rgbd_debug format
    python utils/convert_demo0_to_live_debug.py \\
        --demo0-dir /path/to/raw/data \\
        --realsense-para-dir /path/to/realsense_para \\
        --output-dir tests/outputs/my_data_live

    # Step 2: Fast-Stereo + SAM3
    python utils/offline_replay.py --config configs/offline_replay.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from single_seg.realsense_rgbd_segmenter import (
    FastFoundationStereoRunner,
    align_rectified_depth_to_color_torch,
    filter_depth_edges_torch,
)
from single_seg.single_object_segmenter import (
    SingleObjectPointCloudSegmenter,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Fast-Stereo depth estimation (overwrites depth_aligned_m.npy in-place)
# ---------------------------------------------------------------------------

def run_fast_stereo(
    input_dir: Path,
    frame_name: str,
    stereo_runner: FastFoundationStereoRunner,
    depth_min: float,
    depth_max: float,
    depth_edge_filter_enabled: bool,
    depth_edge_filter_threshold_m: float,
) -> None:
    frame_dir = input_dir / "live_rgbd_debug" / frame_name
    camera_ids = sorted([d.name for d in frame_dir.iterdir() if d.is_dir()])

    for camera_id in camera_ids:
        camera_dir = frame_dir / camera_id
        payload = load_json(camera_dir / "camera_payload.json")
        rgb = np.asarray(Image.open(camera_dir / "rgb.png").convert("RGB"), dtype=np.uint8)

        left_ir = np.asarray(
            Image.open(camera_dir / "ir_left_rect.png").convert("L"), dtype=np.uint8
        )
        right_ir = np.asarray(
            Image.open(camera_dir / "ir_right_rect.png").convert("L"), dtype=np.uint8
        )

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
            torch.isfinite(depth_rect)
            & (depth_rect >= depth_min)
            & (depth_rect <= depth_max),
            depth_rect,
            torch.zeros((), dtype=torch.float32, device=depth_rect.device),
        )

        if depth_edge_filter_enabled:
            depth_rect = filter_depth_edges_torch(depth_rect, threshold_m=depth_edge_filter_threshold_m)

        depth_color = align_rectified_depth_to_color_torch(
            depth_rect,
            rectified_intrinsics=stereo_output["rectified_intrinsics"],
            rectified_to_color=np.asarray(payload["rectified_to_color"], dtype=np.float64),
            color_intrinsics=color_intrinsics,
            color_shape=rgb.shape[:2],
        )
        depth_color = torch.where(
            torch.isfinite(depth_color)
            & (depth_color >= depth_min)
            & (depth_color <= depth_max),
            depth_color.to(dtype=torch.float32),
            torch.zeros((), dtype=torch.float32, device=depth_color.device),
        )

        depth_np = depth_color.detach().cpu().numpy().astype(np.float32)
        np.save(camera_dir / "depth_aligned_m.npy", depth_np)

        valid = depth_np > 0
        print(
            f"  [{camera_id}] Fast depth: range=[{depth_np[valid].min():.3f}, "
            f"{depth_np[valid].max():.3f}]  infer={infer_time:.2f}s"
        )


# ---------------------------------------------------------------------------
# SAM3 segmentation
# ---------------------------------------------------------------------------

def load_camera_input(
    camera_dir: Path, camera_id: str, device: torch.device
) -> dict[str, Any]:
    payload = load_json(camera_dir / "camera_payload.json")
    rgb = np.asarray(Image.open(camera_dir / "rgb.png").convert("RGB"), dtype=np.uint8)
    depth_np = np.load(camera_dir / "depth_aligned_m.npy").astype(np.float32, copy=False)
    depth_m: np.ndarray | torch.Tensor
    if device.type == "cuda":
        depth_m = torch.as_tensor(depth_np, dtype=torch.float32, device=device)
    else:
        depth_m = depth_np
    intrinsics = dict(payload["color_intrinsics"])
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


def run_sam3_segmentation(
    input_dir: Path,
    output_dir: Path,
    frame_name: str,
    cfg: dict[str, Any],
) -> None:
    frame_dir = input_dir / "live_rgbd_debug" / frame_name
    camera_ids = sorted([d.name for d in frame_dir.iterdir() if d.is_dir()])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    live_debug_root = input_dir / "live_rgbd_debug" if cfg.get("save_live_debug") else None

    segmenter = SingleObjectPointCloudSegmenter(
        target_name=cfg["target_name"],
        prompt_task_info=Path(cfg["prompt_task_info"]).expanduser().resolve(),
        prompt_image_root=Path(cfg["prompt_image_root"]).expanduser().resolve(),
        checkpoint_path=Path(cfg["checkpoint_path"]).expanduser().resolve(),
        output_dir=output_dir,
        source_dir=input_dir,
        overwrite_output=bool(cfg.get("overwrite_output")),
        confidence=float(cfg["confidence"]),
        mask_threshold=float(cfg["mask_threshold"]),
        prompt_keep_score_threshold=float(cfg["prompt_keep_score_threshold"]),
        video_mask_prob_threshold=float(cfg["video_mask_prob_threshold"]),
        depth_scale=1.0,
        depth_min=float(cfg["depth_min"]),
        depth_max=float(cfg["depth_max"]),
        stride=int(cfg["stride"]),
        frame_voxel_size=float(cfg["frame_voxel_size"]),
        target_cluster_filter_enabled=bool(cfg["target_cluster_filter_enabled"]),
        target_cluster_radius_m=float(cfg["target_cluster_radius_m"]),
        target_cluster_min_points=int(cfg["target_cluster_min_points"]),
        target_cluster_keep_largest=bool(cfg["target_cluster_keep_largest"]),
        target_plane_filter_enabled=bool(cfg["target_plane_filter_enabled"]),
        target_plane_filter_distance_m=float(cfg["target_plane_filter_distance_m"]),
        target_plane_filter_min_points=int(cfg["target_plane_filter_min_points"]),
        target_plane_filter_min_inlier_ratio=float(cfg["target_plane_filter_min_inlier_ratio"]),
        target_plane_filter_max_inlier_ratio=float(cfg["target_plane_filter_max_inlier_ratio"]),
        target_plane_filter_max_planes=int(cfg["target_plane_filter_max_planes"]),
        target_plane_filter_ransac_iterations=int(cfg["target_plane_filter_ransac_iterations"]),
        target_depth_band_filter_enabled=bool(cfg["target_depth_band_filter_enabled"]),
        target_depth_band_filter_range_m=float(cfg["target_depth_band_filter_range_m"]),
        target_depth_band_filter_min_valid_pixels=int(cfg["target_depth_band_filter_min_valid_pixels"]),
        target_depth_band_filter_min_keep_pixels=int(cfg["target_depth_band_filter_min_keep_pixels"]),
        target_3d_mask_erode_kernel=int(cfg["target_3d_mask_erode_kernel"]),
        save_ply=bool(cfg.get("save_ply", True)),
        save_normal=bool(cfg.get("save_normal", False)),
        save_debug_2d=bool(cfg.get("save_debug_2d", True)),
        tracker_image_size=int(cfg.get("tracker_image_size", 896)),
        target_vis_color=tuple(cfg["target_vis_color"]) if cfg.get("target_vis_color") else None,
    )

    with segmenter:
        camera_inputs = {
            cid: load_camera_input(frame_dir / cid, cid, device)
            for cid in camera_ids
        }
        result = segmenter.process_frame(
            frame_name=f"{frame_name}.png",
            camera_inputs=camera_inputs,
            live_debug_root=live_debug_root,
        )
        print(
            f"  {frame_name}: points={int(result['points_xyz'].shape[0])} "
            f"labeled={int(torch.count_nonzero(result['instance_labels'] > 0).item())}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified offline replay: Fast-Stereo + SAM3")
    parser.add_argument("--config", type=Path, required=True, help="YAML config file")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    input_dir = Path(cfg["input_dir"]).expanduser().resolve()
    output_dir = Path(cfg["output_dir"]).expanduser().resolve()
    frame_name = cfg["frame_name"]

    if not (input_dir / "live_rgbd_debug").is_dir():
        raise FileNotFoundError(f"live_rgbd_debug not found under {input_dir}")

    print("=" * 60)
    print("Step 1: Fast-Stereo depth estimation")
    print("=" * 60)
    fs_cfg = cfg.get("fast_stereo", {})
    stereo_runner = FastFoundationStereoRunner(
        model_path=Path(fs_cfg["model_path"]).expanduser().resolve(),
        valid_iters=int(fs_cfg.get("valid_iters", 12)),
        max_disp=int(fs_cfg.get("max_disp", 192)),
        scale=float(fs_cfg.get("scale", 1.0)),
        remove_invisible=bool(fs_cfg.get("remove_invisible", True)),
        hiera=bool(fs_cfg.get("hiera", False)),
        optimize_build_volume=str(fs_cfg.get("optimize_build_volume", "pytorch1")),
    )
    run_fast_stereo(
        input_dir=input_dir,
        frame_name=frame_name,
        stereo_runner=stereo_runner,
        depth_min=float(cfg["depth_min"]),
        depth_max=float(cfg["depth_max"]),
        depth_edge_filter_enabled=bool(fs_cfg.get("depth_edge_filter_enabled", False)),
        depth_edge_filter_threshold_m=float(fs_cfg.get("depth_edge_filter_threshold_m", 0.5)),
    )

    print()
    print("=" * 60)
    print("Step 2: SAM3 segmentation")
    print("=" * 60)
    run_sam3_segmentation(
        input_dir=input_dir,
        output_dir=output_dir,
        frame_name=frame_name,
        cfg=cfg,
    )

    print()
    print(f"All done. Output: {output_dir}")


if __name__ == "__main__":
    main()
