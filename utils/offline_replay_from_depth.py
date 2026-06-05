#!/usr/bin/env python3
"""使用已有深度数据的离线分割脚本（跳过Fast-Stereo，直接读已有深度）。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from single_seg.single_object_segmenter import SingleObjectPointCloudSegmenter, write_ply
from single_seg.realsense_rgbd_segmenter import resolve_depth_pose_record_from_payload


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_camera_inputs(
    frame_dir: Path,
    extrinsics_data: dict,
    device: torch.device,
) -> dict[str, dict]:
    """从 rollout render/predicted 格式构建 camera_inputs。"""
    cameras = extrinsics_data.get("cameras", [])
    camera_inputs = {}

    for cam in cameras:
        camera_id = cam["camera_id"]
        intrinsics = dict(cam["intrinsics"])

        cam_dir = frame_dir / camera_id
        rgb_path = cam_dir / "rgb.npy"
        depth_path = cam_dir / "depth_m.npy"

        if not rgb_path.exists() or not depth_path.exists():
            print(f"  跳过 {camera_id}: 文件不存在")
            continue

        rgb = np.load(rgb_path).astype(np.uint8)
        depth_np = np.load(depth_path).astype(np.float32)

        h, w = depth_np.shape
        intrinsics["width"] = w
        intrinsics["height"] = h

        if device.type == "cuda":
            depth_m = torch.as_tensor(depth_np, dtype=torch.float32, device=device)
        else:
            depth_m = depth_np

        payload = {"pose_record": cam}
        pose_record = resolve_depth_pose_record_from_payload(
            payload, coordinate_frame="rectified_depth",
        )

        camera_inputs[camera_id] = {
            "rgb": rgb,
            "depth_m": depth_m,
            "intrinsics": intrinsics,
            "pose_record": pose_record,
            "fovy_deg": None,
        }

    return camera_inputs


def main():
    parser = argparse.ArgumentParser(description="离线分割（使用已有深度）")
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML config file（可选，提供后从文件读取参数）")
    parser.add_argument("--input", type=Path, default=None,
                        help="输入目录，如 rollout1/xxx/render/depth_frame_000005/predicted")
    parser.add_argument("--target", type=str, default=None,
                        help="目标物体名称，如 stove_burner_virtual")
    parser.add_argument("--output", type=Path, default=None,
                        help="输出目录")
    parser.add_argument("--prompts-root", type=Path, default=None,
                        help="prompts根目录")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="SAM3 checkpoint路径")
    parser.add_argument("--depth-min", type=float, default=None)
    parser.add_argument("--depth-max", type=float, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--frame-voxel-size", type=float, default=None)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--mask-threshold", type=float, default=None)
    parser.add_argument("--prompt-keep-score-threshold", type=float, default=None)
    parser.add_argument("--video-mask-prob-threshold", type=float, default=None)
    args = parser.parse_args()

    cfg: dict = {}
    if args.config is not None:
        raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if raw is not None:
            cfg = raw

    def _val(cli_val, key: str, default):
        if cli_val is not None:
            return cli_val
        if key in cfg:
            return cfg[key]
        return default

    input_dir = _val(args.input, "input_dir", None)
    target = _val(args.target, "target", None)
    output_dir = _val(args.output, "output_dir", None)

    if input_dir is None:
        parser.error("--input is required (provide via CLI or config file)")
    if target is None:
        parser.error("--target is required (provide via CLI or config file)")
    if output_dir is None:
        parser.error("--output is required (provide via CLI or config file)")

    input_dir = Path(input_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    if not input_dir.is_dir():
        print(f"错误: 输入目录不存在: {input_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frame_name = "frame0"
    target_root = output_dir / f"offline_{target}"
    frame_output_dir = target_root / frame_name

    print(f"设备: {device}")
    print(f"输入: {input_dir}")
    print(f"输出根目录: {output_dir}")
    print(f"目标目录: {target_root}")
    print(f"目标: {target}")
    print()

    extrinsics_path = input_dir / "camera_extrinsics.json"
    if not extrinsics_path.exists():
        print(f"错误: 未找到 camera_extrinsics.json: {extrinsics_path}")
        return

    extrinsics_data = load_json(extrinsics_path)

    camera_inputs = build_camera_inputs(input_dir, extrinsics_data, device)
    if not camera_inputs:
        print("错误: 没有有效的相机数据")
        return

    print(f"加载了 {len(camera_inputs)} 个相机:")
    for cid, data in camera_inputs.items():
        print(f"  {cid}: {data['rgb'].shape}, depth {data['depth_m'].shape}")

    prompts_root = _val(args.prompts_root, "prompts_root", "assets/prompts")
    prompts_root = Path(prompts_root).expanduser().resolve()
    prompt_task_info = prompts_root / target / "task_info.json"
    prompt_image_root = prompts_root / target

    if not prompt_task_info.exists():
        print(f"错误: 未找到任务提示文件: {prompt_task_info}")
        return

    checkpoint = _val(args.checkpoint, "checkpoint_path", "checkpoints/sam3.pt")
    checkpoint = Path(checkpoint).expanduser().resolve()

    targets_json = _val(None, "targets_json", "assets/prompts/targets.json")
    target_id_map = load_json(Path(targets_json).expanduser().resolve())
    target_id = int(target_id_map[target]["id"])
    print(f"目标ID: {target_id}")

    depth_min = _val(args.depth_min, "depth_min", 0.1)
    depth_max = _val(args.depth_max, "depth_max", 3.0)
    stride = _val(args.stride, "stride", 1)
    frame_voxel_size = _val(args.frame_voxel_size, "frame_voxel_size", 0)
    confidence = _val(args.confidence, "confidence", 0.25)
    mask_threshold = _val(args.mask_threshold, "mask_threshold", 0.6)
    prompt_keep_score_threshold = _val(args.prompt_keep_score_threshold,
                                       "prompt_keep_score_threshold", 0.2)
    video_mask_prob_threshold = _val(args.video_mask_prob_threshold,
                                     "video_mask_prob_threshold", 0.95)

    frame_output_dir.mkdir(parents=True, exist_ok=True)

    segmenter = SingleObjectPointCloudSegmenter(
        target_name=target,
        prompt_task_info=prompt_task_info,
        prompt_image_root=prompt_image_root,
        checkpoint_path=checkpoint,
        output_dir=frame_output_dir,
        overwrite_output=bool(cfg.get("overwrite_output", True)),
        target_id=target_id,
        confidence=confidence,
        mask_threshold=mask_threshold,
        prompt_keep_score_threshold=prompt_keep_score_threshold,
        video_mask_prob_threshold=video_mask_prob_threshold,
        depth_scale=1.0,
        depth_min=depth_min,
        depth_max=depth_max,
        stride=stride,
        frame_voxel_size=frame_voxel_size,
        target_cluster_filter_enabled=bool(cfg.get("target_cluster_filter_enabled", False)),
        target_cluster_radius_m=float(cfg.get("target_cluster_radius_m", 0.025)),
        target_cluster_min_points=int(cfg.get("target_cluster_min_points", 35)),
        target_cluster_keep_largest=bool(cfg.get("target_cluster_keep_largest", True)),
        target_plane_filter_enabled=bool(cfg.get("target_plane_filter_enabled", False)),
        target_plane_filter_distance_m=float(cfg.get("target_plane_filter_distance_m", 0.004)),
        target_plane_filter_min_points=int(cfg.get("target_plane_filter_min_points", 80)),
        target_plane_filter_min_inlier_ratio=float(cfg.get("target_plane_filter_min_inlier_ratio", 0.25)),
        target_plane_filter_max_inlier_ratio=float(cfg.get("target_plane_filter_max_inlier_ratio", 0.85)),
        target_plane_filter_max_planes=int(cfg.get("target_plane_filter_max_planes", 1)),
        target_plane_filter_ransac_iterations=int(cfg.get("target_plane_filter_ransac_iterations", 256)),
        target_depth_band_filter_enabled=bool(cfg.get("target_depth_band_filter_enabled", True)),
        target_depth_band_filter_range_m=float(cfg.get("target_depth_band_filter_range_m", 0.08)),
        target_depth_band_filter_min_valid_pixels=int(cfg.get("target_depth_band_filter_min_valid_pixels", 50)),
        target_depth_band_filter_min_keep_pixels=int(cfg.get("target_depth_band_filter_min_keep_pixels", 20)),
        target_3d_mask_erode_kernel=int(cfg.get("target_3d_mask_erode_kernel", 0)),
        single_object_mode_enabled=bool(cfg.get("single_object_mode_enabled", False)),
        single_object_cluster_radius_m=float(cfg.get("single_object_cluster_radius_m", 0.05)),
        single_object_cluster_min_points=int(cfg.get("single_object_cluster_min_points", 50)),
        single_object_cluster_max_points=int(cfg.get("single_object_cluster_max_points", 500)),
        single_object_camera_distance_ratio=float(cfg.get("single_object_camera_distance_ratio", 3.0)),
        save_ply=bool(cfg.get("save_ply", True)),
        save_normal=bool(cfg.get("save_normal", False)),
        save_debug_2d=bool(cfg.get("save_debug_2d", True)),
        tracker_image_size=int(cfg.get("tracker_image_size", 896)),
        target_vis_color=tuple(cfg["target_vis_color"]) if cfg.get("target_vis_color") else None,
    )

    print(f"\n开始分割: {target}")
    seg_t0 = time.perf_counter()
    result = segmenter.process_frame(
        frame_name=f"{frame_name}",
        camera_inputs=camera_inputs,
        live_debug_root=frame_output_dir / "debug",
        view_root=frame_output_dir,
    )
    seg_time = time.perf_counter() - seg_t0
    print(f"分割耗时: {seg_time:.2f}s")
    print()

    points_xyz = result.get("points_xyz")
    raw_colors = result.get("raw_colors")
    instance_labels = result.get("instance_labels")

    if points_xyz is not None and raw_colors is not None:
        pts = points_xyz.cpu().numpy() if torch.is_tensor(points_xyz) else points_xyz
        cols = raw_colors.cpu().numpy() if torch.is_tensor(raw_colors) else raw_colors

        scene_ply = target_root / "full_scene.ply"
        write_ply(scene_ply, pts, cols)
        print(f"场景点云: {scene_ply} ({len(pts)} 点)")

        if instance_labels is not None:
            lbls = instance_labels.cpu().numpy() if torch.is_tensor(instance_labels) else instance_labels
            target_mask = lbls == segmenter.target_id
            if target_mask.any():
                target_ply = target_root / "target.ply"
                write_ply(target_ply, pts[target_mask], cols[target_mask])
                print(f"目标点云: {target_ply} ({target_mask.sum()} 点)")

    print("\n完成!")


if __name__ == "__main__":
    main()
