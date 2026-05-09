#!/usr/bin/env python3
"""Replay a RealSense live debug dump through the original FoundationStereo."""

from __future__ import annotations

import argparse
import json
import logging
import os
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
FOUNDATION_STEREO_ROOT = REPO_ROOT / "third_party" / "foundationstereo"
DEFAULT_CKPT_PATH = FOUNDATION_STEREO_ROOT / "pretrained_models" / "23-51-11" / "model_best_bp2.pth"
DEFAULT_INPUT_DIR = REPO_ROOT / "tests" / "outputs" / "realsense_live_three_cam_fast_nofilter"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tests" / "outputs" / "foundationstereo_replay"

# The original repo can run without xformers through PyTorch SDPA. Set this before
# importing its vendored DINOv2 modules so missing xformers is not treated as active.
os.environ.setdefault("XFORMERS_DISABLED", "1")

if str(FOUNDATION_STEREO_ROOT) not in sys.path:
    sys.path.insert(0, str(FOUNDATION_STEREO_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from single_seg.single_object_segmenter import (  # noqa: E402
    backproject_scene_points_with_labels,
    fuse_scene_geometry,
    write_label_ply,
    write_ply,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute stereo depth with the original FoundationStereo from a "
            "single-seg live_rgbd_debug dump, then rebuild scene/instance PLYs."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Output root containing live_rgbd_debug/ and masks_2d/, or live_rgbd_debug itself.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ckpt-path", type=Path, default=DEFAULT_CKPT_PATH)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=1)
    parser.add_argument(
        "--camera-ids",
        default="",
        help="Comma-separated camera IDs. Empty means all cameras found in the first selected frame.",
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--valid-iters", type=int, default=32)
    parser.add_argument("--hiera", type=int, default=0)
    parser.add_argument("--remove-invisible", type=int, default=1)
    parser.add_argument("--depth-min", type=float, default=0.1)
    parser.add_argument("--depth-max", type=float, default=3.0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--frame-voxel-size", type=float, default=0.003)
    parser.add_argument("--save-depth-debug", type=int, default=1)
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


def intrinsics_from_payload(payload: dict[str, Any], rgb_shape: tuple[int, int, int]) -> dict[str, float]:
    intrinsics = dict(payload["color_intrinsics"])
    intrinsics["width"] = int(rgb_shape[1])
    intrinsics["height"] = int(rgb_shape[0])
    return intrinsics


def rectified_intrinsics_from_k(rectified_k: np.ndarray, shape: tuple[int, int]) -> dict[str, float]:
    height, width = shape
    return {
        "fx": float(rectified_k[0, 0]),
        "fy": float(rectified_k[1, 1]),
        "cx": float(rectified_k[0, 2]),
        "cy": float(rectified_k[1, 2]),
        "width": int(width),
        "height": int(height),
    }


def project_points_to_depth_image_torch(
    points_src: torch.Tensor,
    src_to_dst: np.ndarray | torch.Tensor,
    dst_intrinsics: dict[str, float],
    dst_shape: tuple[int, int],
) -> torch.Tensor:
    height, width = int(dst_shape[0]), int(dst_shape[1])
    device = points_src.device
    depth_out = torch.full((height * width,), float("inf"), dtype=torch.float32, device=device)
    if points_src.numel() == 0:
        depth_out[~torch.isfinite(depth_out)] = 0.0
        return depth_out.reshape(height, width)

    transform = torch.as_tensor(src_to_dst, dtype=torch.float32, device=device)
    ones = torch.ones((points_src.shape[0], 1), dtype=torch.float32, device=device)
    points_h = torch.cat([points_src.to(torch.float32), ones], dim=1)
    points_dst = (transform @ points_h.T).T[:, :3]
    z = points_dst[:, 2]
    valid_z = z > 0
    if not bool(valid_z.any().item()):
        depth_out[~torch.isfinite(depth_out)] = 0.0
        return depth_out.reshape(height, width)

    points_dst = points_dst[valid_z]
    z = z[valid_z]
    fx = float(dst_intrinsics["fx"])
    fy = float(dst_intrinsics["fy"])
    cx = float(dst_intrinsics["cx"])
    cy = float(dst_intrinsics["cy"])
    u = torch.round((points_dst[:, 0] * fx / z) + cx).to(torch.int64)
    v = torch.round((points_dst[:, 1] * fy / z) + cy).to(torch.int64)
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    linear = (v[in_bounds] * width) + u[in_bounds]
    depth_out.scatter_reduce_(0, linear, z[in_bounds].to(torch.float32), reduce="amin", include_self=True)
    depth_out[~torch.isfinite(depth_out)] = 0.0
    return depth_out.reshape(height, width)


def align_rectified_depth_to_color_torch(
    depth_rect_m: torch.Tensor,
    *,
    rectified_intrinsics: dict[str, float],
    rectified_to_color: np.ndarray,
    color_intrinsics: dict[str, float],
    color_shape: tuple[int, int],
) -> torch.Tensor:
    depth = depth_rect_m.to(dtype=torch.float32)
    height, width = depth.shape
    device = depth.device
    valid = torch.isfinite(depth) & (depth > 0)
    if not bool(valid.any().item()):
        return torch.zeros(color_shape, dtype=torch.float32, device=device)

    fx = float(rectified_intrinsics["fx"])
    fy = float(rectified_intrinsics["fy"])
    cx = float(rectified_intrinsics["cx"])
    cy = float(rectified_intrinsics["cy"])
    vv, uu = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    depth_valid = depth[valid]
    x = ((uu[valid] - cx) / fx) * depth_valid
    y = ((vv[valid] - cy) / fy) * depth_valid
    points_rect = torch.stack([x, y, depth_valid], dim=1)
    return project_points_to_depth_image_torch(
        points_rect,
        rectified_to_color,
        color_intrinsics,
        color_shape,
    )


def read_ir_pair(camera_dir: Path, payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    left_name = str(payload.get("ir_left_rect_file", "ir_left_rect.png"))
    right_name = str(payload.get("ir_right_rect_file", "ir_right_rect.png"))
    left_path = camera_dir / left_name
    right_path = camera_dir / right_name
    if not left_path.is_file() or not right_path.is_file():
        raise FileNotFoundError(
            f"missing rectified IR files under {camera_dir}; "
            "rerun live capture with --save-live-debug 1 and fast depth, or use a dump that already has IR images"
        )
    left = np.asarray(Image.open(left_path).convert("L"), dtype=np.uint8)
    right = np.asarray(Image.open(right_path).convert("L"), dtype=np.uint8)
    return left, right


def ir_to_model_tensor(image: np.ndarray, scale: float, device: torch.device) -> torch.Tensor:
    if scale <= 0.0 or scale > 1.0:
        raise ValueError("--scale must be in (0, 1]")
    if scale != 1.0:
        image = cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    image_rgb = np.repeat(image[..., None], 3, axis=2)
    return torch.as_tensor(np.ascontiguousarray(image_rgb), device=device, dtype=torch.float32)[None].permute(0, 3, 1, 2)


def disparity_to_vis(disparity: np.ndarray) -> np.ndarray:
    finite = np.isfinite(disparity) & (disparity > 0)
    vis = np.zeros(disparity.shape, dtype=np.uint8)
    if np.any(finite):
        lo, hi = np.percentile(disparity[finite], [2.0, 98.0])
        if hi <= lo:
            hi = float(disparity[finite].max())
            lo = float(disparity[finite].min())
        if hi > lo:
            norm = np.clip((disparity - lo) / (hi - lo), 0.0, 1.0)
            vis = np.round(norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)


def depth_to_vis(depth: np.ndarray, depth_min: float, depth_max: float) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    vis = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        norm = np.clip((depth - float(depth_min)) / (float(depth_max) - float(depth_min)), 0.0, 1.0)
        vis = np.round((1.0 - norm) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)


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


def prepare_model_args(ckpt_path: Path, scale: float, valid_iters: int, hiera: int, remove_invisible: int) -> Any:
    from omegaconf import OmegaConf

    cfg_path = ckpt_path.parent / "cfg.yaml"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"missing FoundationStereo checkpoint: {ckpt_path}")
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing FoundationStereo config next to checkpoint: {cfg_path}")
    cfg = OmegaConf.load(str(cfg_path))
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    cfg["ckpt_dir"] = str(ckpt_path)
    cfg["scale"] = float(scale)
    cfg["valid_iters"] = int(valid_iters)
    cfg["hiera"] = int(hiera)
    cfg["remove_invisible"] = int(remove_invisible)
    cfg["get_pc"] = 0
    return OmegaConf.create(cfg)


def load_model(ckpt_path: Path, args: Any, device: torch.device) -> FoundationStereo:
    from core.foundation_stereo import FoundationStereo

    logging.info("loading FoundationStereo checkpoint: %s", ckpt_path)
    model = FoundationStereo(args)
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        logging.info("checkpoint global_step=%s epoch=%s", ckpt.get("global_step"), ckpt.get("epoch"))
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def infer_foundation_depth(
    *,
    model: FoundationStereo,
    model_args: Any,
    left_ir: np.ndarray,
    right_ir: np.ndarray,
    rectified_k: np.ndarray,
    baseline_m: float,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray, dict[str, float]]:
    from core.utils.utils import InputPadder

    left = ir_to_model_tensor(left_ir, float(model_args.scale), device)
    right = ir_to_model_tensor(right_ir, float(model_args.scale), device)
    height, width = int(left.shape[-2]), int(left.shape[-1])
    rectified_k_scaled = rectified_k.astype(np.float32, copy=True)
    rectified_k_scaled[:2] *= float(model_args.scale)

    padder = InputPadder(left.shape, divis_by=32, force_square=False)
    left, right = padder.pad(left, right)
    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        if not int(model_args.hiera):
            disparity = model.forward(left, right, iters=int(model_args.valid_iters), test_mode=True)
        else:
            disparity = model.run_hierachical(
                left,
                right,
                iters=int(model_args.valid_iters),
                test_mode=True,
                small_ratio=0.5,
            )
    disparity = padder.unpad(disparity.float()).reshape(height, width)
    if int(model_args.remove_invisible):
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=device),
            torch.arange(width, dtype=torch.float32, device=device),
            indexing="ij",
        )
        disparity = torch.where((xx - disparity) < 0, torch.full_like(disparity, float("inf")), disparity)
    valid_disp = torch.isfinite(disparity) & (disparity > 0)
    depth = torch.where(
        valid_disp,
        float(rectified_k_scaled[0, 0]) * float(baseline_m) / disparity,
        torch.zeros_like(disparity),
    )
    rectified_intrinsics = rectified_intrinsics_from_k(rectified_k_scaled, (height, width))
    return depth.to(torch.float32), disparity.detach().cpu().numpy(), rectified_intrinsics


def process_frame(
    *,
    frame_dir: Path,
    source_root: Path,
    output_dir: Path,
    camera_ids: list[str],
    model: FoundationStereo,
    model_args: Any,
    device: torch.device,
    depth_min: float,
    depth_max: float,
    stride: int,
    frame_voxel_size: float,
    save_depth_debug: bool,
) -> dict[str, Any]:
    frame_output_dir = output_dir / "frame_outputs"
    debug_output_dir = output_dir / "foundation_depth_debug" / frame_dir.name
    point_chunks: list[np.ndarray] = []
    color_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    camera_summaries: list[dict[str, Any]] = []

    for camera_id in camera_ids:
        camera_dir = frame_dir / camera_id
        payload = load_json(camera_dir / "camera_payload.json")
        rgb = np.asarray(Image.open(camera_dir / "rgb.png").convert("RGB"), dtype=np.uint8)
        left_ir, right_ir = read_ir_pair(camera_dir, payload)
        rectified_k = np.asarray(payload["rectified_k"], dtype=np.float32)
        color_intrinsics = intrinsics_from_payload(payload, rgb.shape)
        color_shape = (int(rgb.shape[0]), int(rgb.shape[1]))

        t0 = time.perf_counter()
        depth_rect, disparity, rect_intr = infer_foundation_depth(
            model=model,
            model_args=model_args,
            left_ir=left_ir,
            right_ir=right_ir,
            rectified_k=rectified_k,
            baseline_m=float(payload["baseline_m"]),
            device=device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_time = time.perf_counter() - t0

        depth_color = align_rectified_depth_to_color_torch(
            depth_rect,
            rectified_intrinsics=rect_intr,
            rectified_to_color=np.asarray(payload["rectified_to_color"], dtype=np.float64),
            color_intrinsics=color_intrinsics,
            color_shape=color_shape,
        )
        depth_color = torch.where(
            torch.isfinite(depth_color) & (depth_color >= float(depth_min)) & (depth_color <= float(depth_max)),
            depth_color.to(torch.float32),
            torch.zeros((), dtype=torch.float32, device=depth_color.device),
        )
        depth_np = depth_color.detach().cpu().numpy().astype(np.float32, copy=False)
        mask = load_mask(source_root, frame_dir.name, camera_id, rgb.shape)
        pose_record = payload["pose_record"]
        cam2world_gl = np.asarray(pose_record["cam2world_4x4"], dtype=np.float64)
        points, colors, labels = backproject_scene_points_with_labels(
            rgb,
            depth_np,
            mask,
            cam2world_gl,
            color_intrinsics,
            None,
            float(depth_min),
            float(depth_max),
            int(stride),
        )
        point_chunks.append(points)
        color_chunks.append(colors)
        label_chunks.append(labels)

        valid_depth = int(np.count_nonzero(depth_np > 0))
        target_points = int(np.count_nonzero(labels > 0))
        camera_summaries.append(
            {
                "camera_id": camera_id,
                "infer_time_sec": infer_time,
                "valid_depth_pixels": valid_depth,
                "backprojected_points": int(points.shape[0]),
                "target_points": target_points,
            }
        )
        if save_depth_debug:
            camera_debug_dir = debug_output_dir / camera_id
            camera_debug_dir.mkdir(parents=True, exist_ok=True)
            np.save(camera_debug_dir / "depth_aligned_m.npy", depth_np)
            np.save(camera_debug_dir / "disparity.npy", disparity.astype(np.float32, copy=False))
            Image.fromarray(disparity_to_vis(disparity)).save(camera_debug_dir / "disparity_vis.png")
            Image.fromarray(depth_to_vis(depth_np, depth_min, depth_max)).save(camera_debug_dir / "depth_aligned_vis.png")

    points_xyz, colors_rgb, labels = fuse_scene_geometry(
        point_chunks,
        color_chunks,
        label_chunks,
        float(frame_voxel_size),
    )
    target_mask = labels > 0
    instance_colors = colors_rgb.copy()
    instance_colors[target_mask] = np.array([255, 0, 0], dtype=np.uint8)

    frame_output_dir.mkdir(parents=True, exist_ok=True)
    frame_stem = frame_dir.name
    write_ply(frame_output_dir / f"{frame_stem}_scene_rgb.ply", points_xyz, colors_rgb)
    write_ply(frame_output_dir / f"{frame_stem}_instance_rgb.ply", points_xyz, instance_colors)
    write_label_ply(frame_output_dir / f"{frame_stem}_label.ply", points_xyz, labels)
    return {
        "frame_name": frame_dir.name,
        "num_points": int(points_xyz.shape[0]),
        "num_labeled_points": int(np.count_nonzero(labels > 0)),
        "camera_summaries": camera_summaries,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("FoundationStereo replay requires CUDA for practical inference")
    source_root, debug_root = resolve_debug_roots(args.input_dir)
    frames = selected_frames(debug_root, int(args.frame_index), int(args.max_frames))
    camera_ids = resolve_camera_ids(frames[0], str(args.camera_ids))
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and args.overwrite_output:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_args = prepare_model_args(
        args.ckpt_path.expanduser().resolve(),
        float(args.scale),
        int(args.valid_iters),
        int(args.hiera),
        int(args.remove_invisible),
    )
    device = torch.device("cuda")
    model = load_model(args.ckpt_path.expanduser().resolve(), model_args, device)

    summary: dict[str, Any] = {
        "input_dir": str(source_root),
        "debug_root": str(debug_root),
        "output_dir": str(output_dir),
        "ckpt_path": str(args.ckpt_path.expanduser().resolve()),
        "foundationstereo_root": str(FOUNDATION_STEREO_ROOT),
        "scale": float(args.scale),
        "valid_iters": int(args.valid_iters),
        "hiera": int(args.hiera),
        "remove_invisible": bool(args.remove_invisible),
        "depth_min": float(args.depth_min),
        "depth_max": float(args.depth_max),
        "stride": int(args.stride),
        "frame_voxel_size": float(args.frame_voxel_size),
        "camera_ids": camera_ids,
        "frames": [],
    }
    t0 = time.perf_counter()
    for frame_dir in frames:
        frame_summary = process_frame(
            frame_dir=frame_dir,
            source_root=source_root,
            output_dir=output_dir,
            camera_ids=camera_ids,
            model=model,
            model_args=model_args,
            device=device,
            depth_min=float(args.depth_min),
            depth_max=float(args.depth_max),
            stride=int(args.stride),
            frame_voxel_size=float(args.frame_voxel_size),
            save_depth_debug=bool(args.save_depth_debug),
        )
        summary["frames"].append(frame_summary)
        infer_total = sum(float(item["infer_time_sec"]) for item in frame_summary["camera_summaries"])
        logging.info(
            "%s points=%d labeled=%d stereo_infer=%.3fs",
            frame_summary["frame_name"],
            frame_summary["num_points"],
            frame_summary["num_labeled_points"],
            infer_total,
        )
    summary["elapsed_sec"] = time.perf_counter() - t0
    (output_dir / "foundationstereo_replay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logging.info("saved outputs under %s", output_dir)


if __name__ == "__main__":
    main()
