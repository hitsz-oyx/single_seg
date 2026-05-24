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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import multiprocessing
import os
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


def run_single_camera_inference(
    camera_id: str,
    camera_dir: Path,
    model_path: str,
    valid_iters: int,
    max_disp: int,
    scale: float,
    depth_min: float,
    depth_max: float,
    depth_edge_filter_enabled: bool,
    depth_edge_filter_threshold_m: float,
) -> tuple[str, dict[str, Any], dict[str, float], np.ndarray, np.ndarray]:
    import torch
    from single_seg.realsense_rgbd_segmenter import (
        FastFoundationStereoRunner,
        align_color_to_rectified_depth_torch,
        filter_depth_edges_torch,
    )
    
    timing: dict[str, float] = {}
    
    t0 = time.perf_counter()
    payload = load_json(camera_dir / "camera_payload.json")
    timing["load_payload_sec"] = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    rgb = np.asarray(Image.open(camera_dir / "rgb.png").convert("RGB"), dtype=np.uint8)
    timing["load_rgb_sec"] = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    left_ir = np.asarray(
        Image.open(camera_dir / "ir_left_rect.png").convert("L"), dtype=np.uint8
    )
    timing["load_left_ir_sec"] = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    right_ir = np.asarray(
        Image.open(camera_dir / "ir_right_rect.png").convert("L"), dtype=np.uint8
    )
    timing["load_right_ir_sec"] = time.perf_counter() - t0
    
    t0_model = time.perf_counter()
    runner = FastFoundationStereoRunner(
        model_path=Path(model_path),
        valid_iters=valid_iters,
        max_disp=max_disp,
        scale=scale,
        remove_invisible=True,
        hiera=False,
        optimize_build_volume="pytorch1",
    )
    timing["model_load_sec"] = time.perf_counter() - t0_model
    
    color_intrinsics = dict(payload["color_intrinsics"])
    color_intrinsics["width"] = int(rgb.shape[1])
    color_intrinsics["height"] = int(rgb.shape[0])
    
    t0 = time.perf_counter()
    stereo_output = runner.infer_depth(
        left_image=left_ir,
        right_image=right_ir,
        rectified_k=np.asarray(payload["rectified_k"], dtype=np.float32),
        baseline_m=float(payload["baseline_m"]),
        return_torch=True,
        include_input_images=False,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timing["stereo_infer_sec"] = time.perf_counter() - t0
    
    depth_rect = stereo_output["depth_m"].to(dtype=torch.float32)
    depth_rect = torch.where(
        torch.isfinite(depth_rect)
        & (depth_rect >= depth_min)
        & (depth_rect <= depth_max),
        depth_rect,
        torch.zeros((), dtype=torch.float32, device=depth_rect.device),
    )
    
    t0 = time.perf_counter()
    if depth_edge_filter_enabled:
        depth_rect = filter_depth_edges_torch(depth_rect, threshold_m=depth_edge_filter_threshold_m)
    timing["depth_filter_sec"] = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    rgb_aligned_t = align_color_to_rectified_depth_torch(
        rgb,
        depth_rect,
        rectified_intrinsics=stereo_output["rectified_intrinsics"],
        rectified_to_color=np.asarray(payload["rectified_to_color"], dtype=np.float64),
        color_intrinsics=color_intrinsics,
    )
    timing["align_color_sec"] = time.perf_counter() - t0
    
    depth_rect_np = depth_rect.detach().cpu().numpy().astype(np.float32)
    rgb_aligned_np = rgb_aligned_t.detach().cpu().numpy()
    
    intrinsics = stereo_output["rectified_intrinsics"]
    valid = depth_rect_np > 0
    print(
        f"  [{camera_id}] Fast depth: range=[{depth_rect_np[valid].min():.3f}, "
        f"{depth_rect_np[valid].max():.3f}]  infer={timing['stereo_infer_sec']:.2f}s"
    )
    
    del runner
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    return camera_id, intrinsics, timing, rgb_aligned_np, depth_rect_np


def _load_camera_data(camera_id: str, camera_dir: Path) -> tuple[str, dict, np.ndarray, np.ndarray, np.ndarray]:
    payload = load_json(camera_dir / "camera_payload.json")
    rgb = np.asarray(Image.open(camera_dir / "rgb.png").convert("RGB"), dtype=np.uint8)
    left_ir = np.asarray(Image.open(camera_dir / "ir_left_rect.png").convert("L"), dtype=np.uint8)
    right_ir = np.asarray(Image.open(camera_dir / "ir_right_rect.png").convert("L"), dtype=np.uint8)
    return camera_id, payload, rgb, left_ir, right_ir


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
    save_debug_files: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, float], dict[str, dict[str, float]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    frame_dir = input_dir / "live_rgbd_debug" / frame_name
    camera_ids = sorted([d.name for d in frame_dir.iterdir() if d.is_dir()])
    all_stereo_intrinsics: dict[str, dict[str, Any]] = {}
    per_camera_times: dict[str, float] = {}
    per_camera_timing: dict[str, dict[str, float]] = {}
    aligned_rgb_by_camera: dict[str, np.ndarray] = {}
    aligned_depth_by_camera: dict[str, np.ndarray] = {}

    for camera_id in camera_ids:
        camera_dir = frame_dir / camera_id
        timing: dict[str, float] = {}
        
        t0 = time.perf_counter()
        payload = load_json(camera_dir / "camera_payload.json")
        timing["load_payload_sec"] = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        rgb = np.asarray(Image.open(camera_dir / "rgb.png").convert("RGB"), dtype=np.uint8)
        timing["load_rgb_sec"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        left_ir = np.asarray(
            Image.open(camera_dir / "ir_left_rect.png").convert("L"), dtype=np.uint8
        )
        timing["load_left_ir_sec"] = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        right_ir = np.asarray(
            Image.open(camera_dir / "ir_right_rect.png").convert("L"), dtype=np.uint8
        )
        timing["load_right_ir_sec"] = time.perf_counter() - t0

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
        timing["stereo_infer_sec"] = time.perf_counter() - t0

        depth_rect = stereo_output["depth_m"].to(dtype=torch.float32)
        depth_rect = torch.where(
            torch.isfinite(depth_rect)
            & (depth_rect >= depth_min)
            & (depth_rect <= depth_max),
            depth_rect,
            torch.zeros((), dtype=torch.float32, device=depth_rect.device),
        )

        t0 = time.perf_counter()
        if depth_edge_filter_enabled:
            depth_rect = filter_depth_edges_torch(depth_rect, threshold_m=depth_edge_filter_threshold_m)
        timing["depth_filter_sec"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        rgb_aligned_t = align_color_to_rectified_depth_torch(
            rgb,
            depth_rect,
            rectified_intrinsics=stereo_output["rectified_intrinsics"],
            rectified_to_color=np.asarray(payload["rectified_to_color"], dtype=np.float64),
            color_intrinsics=color_intrinsics,
        )
        timing["align_color_sec"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        depth_rect_np = depth_rect.detach().cpu().numpy().astype(np.float32)
        if save_debug_files:
            np.save(camera_dir / "depth_aligned_m.npy", depth_rect_np)
        aligned_depth_by_camera[camera_id] = depth_rect_np
        timing["save_depth_sec"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        rgb_aligned_np = rgb_aligned_t.detach().cpu().numpy()
        if save_debug_files:
            Image.fromarray(rgb_aligned_np).save(camera_dir / "rgb_aligned.png")
        aligned_rgb_by_camera[camera_id] = rgb_aligned_np
        timing["save_rgb_aligned_sec"] = time.perf_counter() - t0

        all_stereo_intrinsics[camera_id] = stereo_output["rectified_intrinsics"]
        
        t0 = time.perf_counter()
        with open(camera_dir / "stereo_intrinsics.json", "w", encoding="utf-8") as f:
            json.dump(stereo_output["rectified_intrinsics"], f)
        timing["save_intrinsics_sec"] = time.perf_counter() - t0

        valid = depth_rect_np > 0
        infer_time = timing["stereo_infer_sec"]
        per_camera_times[camera_id] = infer_time
        per_camera_timing[camera_id] = timing
        print(
            f"  [{camera_id}] Fast depth: range=[{depth_rect_np[valid].min():.3f}, "
            f"{depth_rect_np[valid].max():.3f}]  infer={infer_time:.2f}s"
        )

    return all_stereo_intrinsics, per_camera_times, per_camera_timing, aligned_rgb_by_camera, aligned_depth_by_camera


def _load_camera_data(camera_id: str, camera_dir: Path) -> tuple[str, dict, np.ndarray, np.ndarray, np.ndarray]:
    payload = load_json(camera_dir / "camera_payload.json")
    rgb = np.asarray(Image.open(camera_dir / "rgb.png").convert("RGB"), dtype=np.uint8)
    left_ir = np.asarray(Image.open(camera_dir / "ir_left_rect.png").convert("L"), dtype=np.uint8)
    right_ir = np.asarray(Image.open(camera_dir / "ir_right_rect.png").convert("L"), dtype=np.uint8)
    return camera_id, payload, rgb, left_ir, right_ir


def run_fast_stereo_parallel(
    input_dir: Path,
    frame_name: str,
    stereo_runner: FastFoundationStereoRunner,
    depth_min: float,
    depth_max: float,
    depth_edge_filter_enabled: bool,
    depth_edge_filter_threshold_m: float,
    save_debug_files: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, float], dict[str, dict[str, float]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    frame_dir = input_dir / "live_rgbd_debug" / frame_name
    camera_ids = sorted([d.name for d in frame_dir.iterdir() if d.is_dir()])
    
    t0 = time.perf_counter()
    camera_data: dict[str, tuple] = {}
    with ThreadPoolExecutor(max_workers=len(camera_ids)) as executor:
        futures = {
            executor.submit(_load_camera_data, cid, frame_dir / cid): cid 
            for cid in camera_ids
        }
        for future in as_completed(futures):
            cid = futures[future]
            camera_data[cid] = future.result()
    load_data_time = time.perf_counter() - t0
    
    all_stereo_intrinsics: dict[str, dict[str, Any]] = {}
    per_camera_times: dict[str, float] = {}
    per_camera_timing: dict[str, dict[str, float]] = {}
    aligned_rgb_by_camera: dict[str, np.ndarray] = {}
    aligned_depth_by_camera: dict[str, np.ndarray] = {}
    
    for camera_id in camera_ids:
        camera_dir = frame_dir / camera_id
        cid, payload, rgb, left_ir, right_ir = camera_data[camera_id]
        timing: dict[str, float] = {}
        
        timing["load_payload_sec"] = 0.0
        timing["load_rgb_sec"] = 0.0
        timing["load_left_ir_sec"] = 0.0
        timing["load_right_ir_sec"] = 0.0
        
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
        timing["stereo_infer_sec"] = time.perf_counter() - t0
        
        depth_rect = stereo_output["depth_m"].to(dtype=torch.float32)
        depth_rect = torch.where(
            torch.isfinite(depth_rect)
            & (depth_rect >= depth_min)
            & (depth_rect <= depth_max),
            depth_rect,
            torch.zeros((), dtype=torch.float32, device=depth_rect.device),
        )
        
        t0 = time.perf_counter()
        if depth_edge_filter_enabled:
            depth_rect = filter_depth_edges_torch(depth_rect, threshold_m=depth_edge_filter_threshold_m)
        timing["depth_filter_sec"] = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        rgb_aligned_t = align_color_to_rectified_depth_torch(
            rgb,
            depth_rect,
            rectified_intrinsics=stereo_output["rectified_intrinsics"],
            rectified_to_color=np.asarray(payload["rectified_to_color"], dtype=np.float64),
            color_intrinsics=color_intrinsics,
        )
        timing["align_color_sec"] = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        if save_debug_files:
            depth_rect_np = depth_rect.detach().cpu().numpy().astype(np.float32)
            np.save(camera_dir / "depth_aligned_m.npy", depth_rect_np)
        else:
            depth_rect_np = depth_rect.detach().cpu().numpy().astype(np.float32)
        aligned_depth_by_camera[camera_id] = depth_rect_np
        timing["save_depth_sec"] = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        if save_debug_files:
            rgb_aligned_np = rgb_aligned_t.detach().cpu().numpy()
            Image.fromarray(rgb_aligned_np).save(camera_dir / "rgb_aligned.png")
            aligned_rgb_by_camera[camera_id] = rgb_aligned_np
        else:
            aligned_rgb_by_camera[camera_id] = rgb_aligned_t
        timing["save_rgb_aligned_sec"] = time.perf_counter() - t0
        
        all_stereo_intrinsics[camera_id] = stereo_output["rectified_intrinsics"]
        
        t0 = time.perf_counter()
        with open(camera_dir / "stereo_intrinsics.json", "w", encoding="utf-8") as f:
            json.dump(stereo_output["rectified_intrinsics"], f)
        timing["save_intrinsics_sec"] = time.perf_counter() - t0
        
        valid = depth_rect_np > 0
        infer_time = timing["stereo_infer_sec"]
        per_camera_times[camera_id] = infer_time
        per_camera_timing[camera_id] = timing
        print(
            f"  [{camera_id}] Fast depth: range=[{depth_rect_np[valid].min():.3f}, "
            f"{depth_rect_np[valid].max():.3f}]  infer={infer_time:.2f}s"
        )
    
    for cid in camera_ids:
        per_camera_timing[cid]["parallel_load_data_sec"] = load_data_time
    
    return all_stereo_intrinsics, per_camera_times, per_camera_timing, aligned_rgb_by_camera, aligned_depth_by_camera


def run_fast_stereo_multistream(
    input_dir: Path,
    frame_name: str,
    stereo_runner: FastFoundationStereoRunner,
    depth_min: float,
    depth_max: float,
    depth_edge_filter_enabled: bool,
    depth_edge_filter_threshold_m: float,
    save_debug_files: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, float], dict[str, dict[str, float]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    frame_dir = input_dir / "live_rgbd_debug" / frame_name
    camera_ids = sorted([d.name for d in frame_dir.iterdir() if d.is_dir()])
    
    t0_load = time.perf_counter()
    camera_data: dict[str, tuple] = {}
    with ThreadPoolExecutor(max_workers=len(camera_ids)) as executor:
        futures = {
            executor.submit(_load_camera_data, cid, frame_dir / cid): cid 
            for cid in camera_ids
        }
        for future in as_completed(futures):
            cid = futures[future]
            camera_data[cid] = future.result()
    load_data_time = time.perf_counter() - t0_load
    
    all_stereo_intrinsics: dict[str, dict[str, Any]] = {}
    per_camera_times: dict[str, float] = {}
    per_camera_timing: dict[str, dict[str, float]] = {}
    aligned_rgb_by_camera: dict[str, np.ndarray] = {}
    aligned_depth_by_camera: dict[str, np.ndarray] = {}
    
    num_cameras = len(camera_ids)
    streams = [torch.cuda.Stream() for _ in range(num_cameras)]
    events = [torch.cuda.Event() for _ in range(num_cameras)]
    results: dict[str, tuple] = {}
    
    t0_infer = time.perf_counter()
    
    for i, camera_id in enumerate(camera_ids):
        cid, payload, rgb, left_ir, right_ir = camera_data[camera_id]
        stream = streams[i]
        
        with torch.cuda.stream(stream):
            timing: dict[str, float] = {}
            timing["load_payload_sec"] = 0.0
            timing["load_rgb_sec"] = 0.0
            timing["load_left_ir_sec"] = 0.0
            timing["load_right_ir_sec"] = 0.0
            
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
            stream.synchronize()
            timing["stereo_infer_sec"] = time.perf_counter() - t0
            
            depth_rect = stereo_output["depth_m"].to(dtype=torch.float32)
            depth_rect = torch.where(
                torch.isfinite(depth_rect)
                & (depth_rect >= depth_min)
                & (depth_rect <= depth_max),
                depth_rect,
                torch.zeros((), dtype=torch.float32, device=depth_rect.device),
            )
            
            t0 = time.perf_counter()
            if depth_edge_filter_enabled:
                depth_rect = filter_depth_edges_torch(depth_rect, threshold_m=depth_edge_filter_threshold_m)
            timing["depth_filter_sec"] = time.perf_counter() - t0
            
            t0 = time.perf_counter()
            rgb_aligned_t = align_color_to_rectified_depth_torch(
                rgb,
                depth_rect,
                rectified_intrinsics=stereo_output["rectified_intrinsics"],
                rectified_to_color=np.asarray(payload["rectified_to_color"], dtype=np.float64),
                color_intrinsics=color_intrinsics,
            )
            timing["align_color_sec"] = time.perf_counter() - t0
            
            depth_rect_np = depth_rect.detach().cpu().numpy().astype(np.float32)
            rgb_aligned_np = rgb_aligned_t.detach().cpu().numpy()
            
            results[camera_id] = (payload, color_intrinsics, stereo_output, depth_rect_np, rgb_aligned_np, timing)
            events[i].record(stream)
    
    torch.cuda.synchronize()
    inference_time = time.perf_counter() - t0_infer
    
    for i, camera_id in enumerate(camera_ids):
        camera_dir = frame_dir / camera_id
        payload, color_intrinsics, stereo_output, depth_rect_np, rgb_aligned_np, timing = results[camera_id]
        
        t0 = time.perf_counter()
        if save_debug_files:
            np.save(camera_dir / "depth_aligned_m.npy", depth_rect_np)
        timing["save_depth_sec"] = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        if save_debug_files:
            Image.fromarray(rgb_aligned_np).save(camera_dir / "rgb_aligned.png")
        timing["save_rgb_aligned_sec"] = time.perf_counter() - t0
        
        all_stereo_intrinsics[camera_id] = stereo_output["rectified_intrinsics"]
        
        t0 = time.perf_counter()
        with open(camera_dir / "stereo_intrinsics.json", "w", encoding="utf-8") as f:
            json.dump(stereo_output["rectified_intrinsics"], f)
        timing["save_intrinsics_sec"] = time.perf_counter() - t0
        
        valid = depth_rect_np > 0
        per_camera_times[camera_id] = timing["stereo_infer_sec"]
        per_camera_timing[camera_id] = timing
        print(
            f"  [{camera_id}] Fast depth: range=[{depth_rect_np[valid].min():.3f}, "
            f"{depth_rect_np[valid].max():.3f}]  infer={timing['stereo_infer_sec']:.2f}s"
        )
        
        aligned_depth_by_camera[camera_id] = depth_rect_np
        aligned_rgb_by_camera[camera_id] = rgb_aligned_np
    
    for cid in camera_ids:
        per_camera_timing[cid]["parallel_load_data_sec"] = load_data_time
        per_camera_timing[cid]["multistream_inference_sec"] = inference_time
    
    return all_stereo_intrinsics, per_camera_times, per_camera_timing, aligned_rgb_by_camera, aligned_depth_by_camera


def run_fast_stereo_multiprocess(
    input_dir: Path,
    frame_name: str,
    stereo_runner: FastFoundationStereoRunner,
    depth_min: float,
    depth_max: float,
    depth_edge_filter_enabled: bool,
    depth_edge_filter_threshold_m: float,
    save_debug_files: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, float], dict[str, dict[str, float]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    frame_dir = input_dir / "live_rgbd_debug" / frame_name
    camera_ids = sorted([d.name for d in frame_dir.iterdir() if d.is_dir()])
    
    model_path = str(stereo_runner.model_path)
    valid_iters = int(stereo_runner.args.valid_iters)
    max_disp = int(stereo_runner.args.max_disp)
    scale = float(stereo_runner.args.scale)
    
    all_stereo_intrinsics: dict[str, dict[str, Any]] = {}
    per_camera_times: dict[str, float] = {}
    per_camera_timing: dict[str, dict[str, float]] = {}
    aligned_rgb_by_camera: dict[str, np.ndarray] = {}
    aligned_depth_by_camera: dict[str, np.ndarray] = {}
    
    t0_total = time.perf_counter()
    
    mp_context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=len(camera_ids), mp_context=mp_context) as executor:
        futures = {
            executor.submit(
                run_single_camera_inference,
                cid,
                frame_dir / cid,
                model_path,
                valid_iters,
                max_disp,
                scale,
                depth_min,
                depth_max,
                depth_edge_filter_enabled,
                depth_edge_filter_threshold_m,
            ): cid
            for cid in camera_ids
        }
        
        for future in as_completed(futures):
            cid = futures[future]
            try:
                result = future.result()
                camera_id, intrinsics, timing, rgb_aligned_np, depth_rect_np = result
                
                camera_dir = frame_dir / camera_id
                if save_debug_files:
                    np.save(camera_dir / "depth_aligned_m.npy", depth_rect_np.astype(np.float32))
                    Image.fromarray(rgb_aligned_np).save(camera_dir / "rgb_aligned.png")
                
                with open(camera_dir / "stereo_intrinsics.json", "w", encoding="utf-8") as f:
                    json.dump(intrinsics, f)
                
                all_stereo_intrinsics[camera_id] = intrinsics
                per_camera_times[camera_id] = timing["stereo_infer_sec"]
                per_camera_timing[camera_id] = timing
                aligned_rgb_by_camera[camera_id] = rgb_aligned_np
                aligned_depth_by_camera[camera_id] = depth_rect_np
                
            except Exception as e:
                print(f"Error processing {cid}: {e}")
                raise
    
    multiprocess_total_time = time.perf_counter() - t0_total
    for cid in camera_ids:
        per_camera_timing[cid]["multiprocess_total_sec"] = multiprocess_total_time
    
    return all_stereo_intrinsics, per_camera_times, per_camera_timing, aligned_rgb_by_camera, aligned_depth_by_camera


# ---------------------------------------------------------------------------
# SAM3 segmentation
# ---------------------------------------------------------------------------

def load_camera_input(
    camera_dir: Path,
    camera_id: str,
    device: torch.device,
    stereo_intrinsics: dict[str, Any],
    camera_poses: dict[str, dict[str, Any]],
    aligned_rgb: np.ndarray | torch.Tensor | None = None,
    aligned_depth: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    timing = {}
    
    t0 = time.perf_counter()
    payload = load_json(camera_dir / "camera_payload.json")
    timing["load_payload_time_sec"] = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    if aligned_rgb is not None:
        rgb = aligned_rgb
    else:
        rgb_aligned_path = camera_dir / "rgb_aligned.png"
        if rgb_aligned_path.exists():
            rgb = np.asarray(Image.open(rgb_aligned_path).convert("RGB"), dtype=np.uint8)
        else:
            rgb = np.asarray(Image.open(camera_dir / "rgb.png").convert("RGB"), dtype=np.uint8)
    timing["load_rgb_time_sec"] = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    if aligned_depth is not None:
        depth_np = aligned_depth.astype(np.float32, copy=False)
    else:
        depth_np = np.load(camera_dir / "depth_aligned_m.npy").astype(np.float32, copy=False)
    timing["load_depth_time_sec"] = time.perf_counter() - t0
    
    depth_m: np.ndarray | torch.Tensor
    t0 = time.perf_counter()
    if device.type == "cuda":
        depth_m = torch.as_tensor(depth_np, dtype=torch.float32, device=device)
    else:
        depth_m = depth_np
    timing["convert_depth_time_sec"] = time.perf_counter() - t0
    
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
    }, timing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified offline replay: Fast-Stereo + SAM3")
    parser.add_argument("--config", type=Path, required=True, help="YAML config file")
    parser.add_argument("--target", type=str, default=None,
                        help="Single target name to process (overrides config targets list)")
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="Input directory with live_rgbd_debug (overrides config input_dir)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for results (overrides config output_dir)")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    input_dir = (args.input_dir if args.input_dir else Path(cfg["input_dir"])).expanduser().resolve()
    base_output_dir = (args.output_dir if args.output_dir else Path(cfg["output_dir"])).expanduser().resolve()

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
        target_id=target_id,
    )
    
    if bool(cfg.get("overwrite_output", False)):
        print("\n[overwrite_output=true] Cleaning up output directories...")
        target_base_dir = base_output_dir / f"offline_{target_name}"
        for frame_name in frames_to_process:
            frame_output_dir = target_base_dir / frame_name
            if frame_output_dir.exists():
                shutil.rmtree(frame_output_dir)
        print(f"[overwrite_output=true] Cleaned {len(frames_to_process)} frame directories.\n")
    
    with segmenter:
        for frame_name in frames_to_process:
            print()
            print("=" * 60)
            print(f"Processing frame: {frame_name}")
            print("=" * 60)

            frame_total_t0 = time.perf_counter()

            stereo_mode = cfg.get("stereo_mode", "parallel_loading")
            
            print()
            if stereo_mode == "multistream":
                print("Step 1: Fast-Stereo depth estimation (CUDA Streams 并行)")
            elif stereo_mode == "multiprocess":
                print("Step 1: Fast-Stereo depth estimation (multiprocess)")
            elif stereo_mode == "parallel_loading":
                print("Step 1: Fast-Stereo depth estimation (parallel loading)")
            else:
                print("Step 1: Fast-Stereo depth estimation (serial)")
            print("-" * 40)
            
            if stereo_mode == "multistream":
                (stereo_intrinsics, per_camera_stereo_times, per_camera_stereo_timing, 
                 aligned_rgb_by_camera, aligned_depth_by_camera) = run_fast_stereo_multistream(
                    input_dir=input_dir,
                    frame_name=frame_name,
                    stereo_runner=stereo_runner,
                    depth_min=float(cfg["depth_min"]),
                    depth_max=float(cfg["depth_max"]),
                    depth_edge_filter_enabled=bool(fs_cfg.get("depth_edge_filter_enabled", False)),
                    depth_edge_filter_threshold_m=float(fs_cfg.get("depth_edge_filter_threshold_m", 0.5)),
                    save_debug_files=bool(cfg.get("save_debug_2d", False)),
                )
            elif stereo_mode == "multiprocess":
                (stereo_intrinsics, per_camera_stereo_times, per_camera_stereo_timing, 
                 aligned_rgb_by_camera, aligned_depth_by_camera) = run_fast_stereo_multiprocess(
                    input_dir=input_dir,
                    frame_name=frame_name,
                    stereo_runner=stereo_runner,
                    depth_min=float(cfg["depth_min"]),
                    depth_max=float(cfg["depth_max"]),
                    depth_edge_filter_enabled=bool(fs_cfg.get("depth_edge_filter_enabled", False)),
                    depth_edge_filter_threshold_m=float(fs_cfg.get("depth_edge_filter_threshold_m", 0.5)),
                    save_debug_files=bool(cfg.get("save_debug_2d", False)),
                )
            elif stereo_mode == "parallel_loading":
                (stereo_intrinsics, per_camera_stereo_times, per_camera_stereo_timing, 
                 aligned_rgb_by_camera, aligned_depth_by_camera) = run_fast_stereo_parallel(
                    input_dir=input_dir,
                    frame_name=frame_name,
                    stereo_runner=stereo_runner,
                    depth_min=float(cfg["depth_min"]),
                    depth_max=float(cfg["depth_max"]),
                    depth_edge_filter_enabled=bool(fs_cfg.get("depth_edge_filter_enabled", False)),
                    depth_edge_filter_threshold_m=float(fs_cfg.get("depth_edge_filter_threshold_m", 0.5)),
                    save_debug_files=bool(cfg.get("save_debug_2d", False)),
                )
            else:
                (stereo_intrinsics, per_camera_stereo_times, per_camera_stereo_timing, 
                 aligned_rgb_by_camera, aligned_depth_by_camera) = run_fast_stereo(
                    input_dir=input_dir,
                    frame_name=frame_name,
                    stereo_runner=stereo_runner,
                    depth_min=float(cfg["depth_min"]),
                    depth_max=float(cfg["depth_max"]),
                    depth_edge_filter_enabled=bool(fs_cfg.get("depth_edge_filter_enabled", False)),
                    depth_edge_filter_threshold_m=float(fs_cfg.get("depth_edge_filter_threshold_m", 0.5)),
                    save_debug_files=bool(cfg.get("save_debug_2d", False)),
                )

            frame_dir = live_debug_dir / frame_name
            camera_ids = sorted([d.name for d in frame_dir.iterdir() if d.is_dir()])
            
            other_t0 = time.perf_counter()
            frame_stereo_intrinsics: dict[str, dict[str, Any]] = {}
            for cid in camera_ids:
                with open(frame_dir / cid / "stereo_intrinsics.json") as f:
                    frame_stereo_intrinsics[cid] = json.load(f)
            load_intrinsics_time = time.perf_counter() - other_t0

            print()
            print("Step 2: SAM3 segmentation")
            print("-" * 40)
            target_output_dir = base_output_dir / f"offline_{target_name}" / frame_name
            
            segmenter.output_dir = target_output_dir
            segmenter.frame_output_dir = target_output_dir / "frame_outputs"
            
            mkdir_t0 = time.perf_counter()
            segmenter.frame_output_dir.mkdir(parents=True, exist_ok=True)
            mkdir_time = time.perf_counter() - mkdir_t0
            
            print(f"  [{target_name} (id={target_id})]")

            per_camera_load_timing = {}
            camera_inputs = {}
            for cid in camera_ids:
                input_data, timing = load_camera_input(
                    frame_dir / cid, cid, device, frame_stereo_intrinsics[cid], camera_poses,
                    aligned_rgb=aligned_rgb_by_camera.get(cid),
                    aligned_depth=aligned_depth_by_camera.get(cid),
                )
                camera_inputs[cid] = input_data
                per_camera_load_timing[cid] = timing
            
            load_camera_inputs_time = sum(sum(t.values()) for t in per_camera_load_timing.values())
            
            sam3_t0 = time.perf_counter()
            result = segmenter.process_frame(
                frame_name=f"{frame_name}.png",
                camera_inputs=camera_inputs,
                live_debug_root=live_debug_root,
            )
            sam3_time = time.perf_counter() - sam3_t0
            total_frame_time = time.perf_counter() - frame_total_t0
            
            per_camera_load_summary = {
                cid: {
                    "load_payload_time_sec": t["load_payload_time_sec"],
                    "load_rgb_time_sec": t["load_rgb_time_sec"],
                    "load_depth_time_sec": t["load_depth_time_sec"],
                    "convert_depth_time_sec": t["convert_depth_time_sec"],
                    "total_time_sec": sum(t.values()),
                }
                for cid, t in per_camera_load_timing.items()
            }
            
            load_payload_total = sum(t["load_payload_time_sec"] for t in per_camera_load_timing.values())
            load_rgb_total = sum(t["load_rgb_time_sec"] for t in per_camera_load_timing.values())
            load_depth_total = sum(t["load_depth_time_sec"] for t in per_camera_load_timing.values())
            convert_depth_total = sum(t["convert_depth_time_sec"] for t in per_camera_load_timing.values())
            
            stereo_load_payload = sum(t.get("load_payload_sec", 0) for t in per_camera_stereo_timing.values())
            stereo_load_rgb = sum(t.get("load_rgb_sec", 0) for t in per_camera_stereo_timing.values())
            stereo_load_left_ir = sum(t.get("load_left_ir_sec", 0) for t in per_camera_stereo_timing.values())
            stereo_load_right_ir = sum(t.get("load_right_ir_sec", 0) for t in per_camera_stereo_timing.values())
            stereo_infer = sum(t.get("stereo_infer_sec", 0) for t in per_camera_stereo_timing.values())
            stereo_depth_filter = sum(t.get("depth_filter_sec", 0) for t in per_camera_stereo_timing.values())
            stereo_align = sum(t.get("align_color_sec", 0) for t in per_camera_stereo_timing.values())
            stereo_save_depth = sum(t.get("save_depth_sec", 0) for t in per_camera_stereo_timing.values())
            stereo_save_rgb = sum(t.get("save_rgb_aligned_sec", 0) for t in per_camera_stereo_timing.values())
            stereo_save_intrinsics = sum(t.get("save_intrinsics_sec", 0) for t in per_camera_stereo_timing.values())
            stereo_parallel_load = list(per_camera_stereo_timing.values())[0].get("parallel_load_data_sec", 0) if per_camera_stereo_timing else 0
            
            remaining_other = (total_frame_time 
                - stereo_load_payload - stereo_load_rgb - stereo_load_left_ir - stereo_load_right_ir
                - stereo_infer - stereo_depth_filter - stereo_align 
                - stereo_save_depth - stereo_save_rgb - stereo_save_intrinsics
                - stereo_parallel_load
                - sam3_time - load_intrinsics_time - mkdir_time 
                - load_payload_total - load_rgb_total - load_depth_total - convert_depth_total)

            frame_metadata = {
                "stereo_per_camera_time_sec": per_camera_stereo_times,
                "stereo_per_camera_timing_sec": per_camera_stereo_timing,
                "stereo_load_payload_time_sec": stereo_load_payload,
                "stereo_load_rgb_time_sec": stereo_load_rgb,
                "stereo_load_left_ir_time_sec": stereo_load_left_ir,
                "stereo_load_right_ir_time_sec": stereo_load_right_ir,
                "stereo_infer_time_sec": stereo_infer,
                "stereo_depth_filter_time_sec": stereo_depth_filter,
                "stereo_align_time_sec": stereo_align,
                "stereo_save_depth_time_sec": stereo_save_depth,
                "stereo_save_rgb_time_sec": stereo_save_rgb,
                "stereo_save_intrinsics_time_sec": stereo_save_intrinsics,
                "stereo_parallel_load_time_sec": stereo_parallel_load,
                "stereo_total_time_sec": sum(per_camera_stereo_times.values()),
                "stereo_wall_clock_time_sec": per_camera_stereo_timing.get(camera_ids[0], {}).get("multiprocess_total_sec") or sum(per_camera_stereo_times.values()),
                "load_intrinsics_time_sec": load_intrinsics_time,
                "mkdir_time_sec": mkdir_time,
                "load_camera_per_camera_time_sec": per_camera_load_summary,
                "load_payload_time_sec": load_payload_total,
                "load_rgb_time_sec": load_rgb_total,
                "load_depth_time_sec": load_depth_total,
                "convert_depth_time_sec": convert_depth_total,
                "load_camera_inputs_time_sec": load_camera_inputs_time,
                "other_time_sec": max(0.0, remaining_other),
                "sam3_time_sec": sam3_time,
                "total_frame_time_sec": total_frame_time,
            }
            print(
                f"  {frame_name}: points={int(result['points_xyz'].shape[0])} "
                f"labeled={int(torch.count_nonzero(result['instance_labels'] > 0).item())}"
            )
            print(f"    -> Output: {target_output_dir}")
            wall_time = per_camera_stereo_timing.get(camera_ids[0], {}).get("multiprocess_total_sec") or sum(per_camera_stereo_times.values())
            print(f"    -> Total: {total_frame_time:.3f}s (stereo={wall_time:.3f}s, load_in={load_intrinsics_time:.3f}s, mkdir={mkdir_time:.3f}s, load_cam={load_camera_inputs_time:.3f}s, sam3={sam3_time:.3f}s, other={max(0.0, remaining_other):.3f}s)")
            
            segmenter.update_frame_metadata(frame_metadata)

    print()
    print(f"All done. Processed {len(frames_to_process)} frames for target '{target_name}'.")


if __name__ == "__main__":
    main()
