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
    align_color_to_rectified_depth_torch,
    filter_depth_edges_torch,
    resolve_depth_pose_record_from_payload,
)
from single_seg.single_object_segmenter import (
    SingleObjectPointCloudSegmenter,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_camera_poses(poses_json_path: Path) -> dict[str, dict[str, Any]]:
    raw = load_json(poses_json_path)
    if "cameras" in raw:
        return {cam["camera_id"]: cam for cam in raw["cameras"]}
    return dict(raw)


def load_targets(targets_json_path: Path) -> dict[str, int]:
    return load_json(targets_json_path)


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
) -> dict[str, dict[str, Any]]:
    frame_dir = input_dir / "live_rgbd_debug" / frame_name
    camera_ids = sorted([d.name for d in frame_dir.iterdir() if d.is_dir()])
    all_stereo_intrinsics: dict[str, dict[str, Any]] = {}

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

        rgb_aligned_t = align_color_to_rectified_depth_torch(
            rgb,
            depth_rect,
            rectified_intrinsics=stereo_output["rectified_intrinsics"],
            rectified_to_color=np.asarray(payload["rectified_to_color"], dtype=np.float64),
            color_intrinsics=color_intrinsics,
        )

        depth_rect_np = depth_rect.detach().cpu().numpy().astype(np.float32)
        np.save(camera_dir / "depth_aligned_m.npy", depth_rect_np)

        rgb_aligned_np = rgb_aligned_t.detach().cpu().numpy()
        Image.fromarray(rgb_aligned_np).save(camera_dir / "rgb_aligned.png")

        all_stereo_intrinsics[camera_id] = stereo_output["rectified_intrinsics"]
        with open(camera_dir / "stereo_intrinsics.json", "w", encoding="utf-8") as f:
            json.dump(stereo_output["rectified_intrinsics"], f)

        valid = depth_rect_np > 0
        print(
            f"  [{camera_id}] Fast depth: range=[{depth_rect_np[valid].min():.3f}, "
            f"{depth_rect_np[valid].max():.3f}]  infer={infer_time:.2f}s"
        )

    return all_stereo_intrinsics


# ---------------------------------------------------------------------------
# SAM3 segmentation
# ---------------------------------------------------------------------------

def load_camera_input(
    camera_dir: Path,
    camera_id: str,
    device: torch.device,
    stereo_intrinsics: dict[str, Any],
    camera_poses: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    payload = load_json(camera_dir / "camera_payload.json")
    rgb_aligned_path = camera_dir / "rgb_aligned.png"
    if rgb_aligned_path.exists():
        rgb = np.asarray(Image.open(rgb_aligned_path).convert("RGB"), dtype=np.uint8)
    else:
        rgb = np.asarray(Image.open(camera_dir / "rgb.png").convert("RGB"), dtype=np.uint8)
    depth_np = np.load(camera_dir / "depth_aligned_m.npy").astype(np.float32, copy=False)
    depth_m: np.ndarray | torch.Tensor
    if device.type == "cuda":
        depth_m = torch.as_tensor(depth_np, dtype=torch.float32, device=device)
    else:
        depth_m = depth_np
    intrinsics = dict(stereo_intrinsics)
    intrinsics["width"] = int(depth_np.shape[1])
    intrinsics["height"] = int(depth_np.shape[0])
    if camera_id in camera_poses:
        payload = dict(payload)
        payload["pose_record"] = camera_poses[camera_id]
    pose_record = resolve_depth_pose_record_from_payload(
        payload,
        coordinate_frame="rectified_depth",
    )
    return {
        "rgb": rgb,
        "depth_m": depth_m,
        "intrinsics": intrinsics,
        "pose_record": pose_record,
        "fovy_deg": None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified offline replay: Fast-Stereo + SAM3")
    parser.add_argument("--config", type=Path, required=True, help="YAML config file")
    parser.add_argument("--target", type=str, default=None,
                        help="Single target name to process (overrides config targets list)")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    input_dir = Path(cfg["input_dir"]).expanduser().resolve()
    base_output_dir = Path(cfg["output_dir"]).expanduser().resolve()

    live_debug_dir = input_dir / "live_rgbd_debug"
    if not live_debug_dir.is_dir():
        raise FileNotFoundError(f"live_rgbd_debug not found under {input_dir}")

    all_frames = sorted([d.name for d in live_debug_dir.iterdir() if d.is_dir()])
    frame_start = int(cfg.get("frame_start", 0))
    frame_end = int(cfg["frame_end"]) if cfg.get("frame_end") is not None else len(all_frames)
    frames_to_process = all_frames[frame_start:frame_end]

    targets = [args.target] if args.target else cfg.get("targets", [])
    if isinstance(targets, str):
        raise TypeError(f"Config 'targets' must be a list (e.g. targets: [redcup]), got a string: {targets!r}")
    if not targets:
        raise ValueError("Must provide --target or config 'targets' list")
    target_name = targets[0]
    if len(targets) > 1:
        print(f"WARNING: multiple targets in config, but only processing first: {target_name}. Use --target to specify one.")

    targets_json_path = Path(cfg.get("targets_json", "assets/prompts/targets.json")).expanduser().resolve()
    target_id_map = load_targets(targets_json_path)
    target_id = target_id_map[target_name]["id"]
    print(f"Target: {target_name} (id={target_id})")
    print(f"Frames to process: {len(frames_to_process)} ({frame_start} to {frame_end})")

    camera_poses_json = cfg.get("camera_poses_json")
    if camera_poses_json:
        camera_poses = load_camera_poses(Path(camera_poses_json).expanduser().resolve())
        print(f"Loaded external camera poses from: {camera_poses_json}")
    else:
        camera_poses = {}
        print("WARNING: No camera_poses_json configured, using payload pose_record")

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

    prompt_task_info = Path(cfg["prompts_root"]) / target_name / "task_info.json"
    prompt_image_root = Path(cfg["prompts_root"]) / target_name
    first_output_dir = base_output_dir / f"offline_{target_name}" / frames_to_process[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    live_debug_root = input_dir / "live_rgbd_debug" if cfg.get("save_live_debug") else None

    segmenter = SingleObjectPointCloudSegmenter(
        target_name=target_name,
        prompt_task_info=prompt_task_info.expanduser().resolve(),
        prompt_image_root=prompt_image_root.expanduser().resolve(),
        checkpoint_path=Path(cfg["checkpoint_path"]).expanduser().resolve(),
        output_dir=first_output_dir,
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
        target_id=target_id,
    )

    with segmenter:
        for frame_name in frames_to_process:
            print()
            print("=" * 60)
            print(f"Processing frame: {frame_name}")
            print("=" * 60)

            print()
            print("Step 1: Fast-Stereo depth estimation")
            print("-" * 40)
            run_fast_stereo(
                input_dir=input_dir,
                frame_name=frame_name,
                stereo_runner=stereo_runner,
                depth_min=float(cfg["depth_min"]),
                depth_max=float(cfg["depth_max"]),
                depth_edge_filter_enabled=bool(fs_cfg.get("depth_edge_filter_enabled", False)),
                depth_edge_filter_threshold_m=float(fs_cfg.get("depth_edge_filter_threshold_m", 0.5)),
            )

            frame_dir = live_debug_dir / frame_name
            camera_ids = sorted([d.name for d in frame_dir.iterdir() if d.is_dir()])
            stereo_intrinsics: dict[str, dict[str, Any]] = {}
            for cid in camera_ids:
                with open(frame_dir / cid / "stereo_intrinsics.json") as f:
                    stereo_intrinsics[cid] = json.load(f)

            print()
            print("Step 2: SAM3 segmentation")
            print("-" * 40)
            target_output_dir = base_output_dir / f"offline_{target_name}" / frame_name
            segmenter.output_dir = target_output_dir
            segmenter.frame_output_dir = target_output_dir / "frame_outputs"
            segmenter.frame_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"  [{target_name} (id={target_id})]")

            camera_inputs = {
                cid: load_camera_input(frame_dir / cid, cid, device, stereo_intrinsics[cid], camera_poses)
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
            print(f"    -> Output: {target_output_dir}")

    print()
    print(f"All done. Processed {len(frames_to_process)} frames for target '{target_name}'.")


if __name__ == "__main__":
    main()
