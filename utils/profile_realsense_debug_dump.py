#!/usr/bin/env python3
"""Profile single-seg inference from a RealSense live_rgbd_debug dump."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from single_seg.single_object_segmenter import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_PROMPT_IMAGE_ROOT,
    DEFAULT_PROMPT_TASK_INFO,
    SingleObjectPointCloudSegmenter,
)
from single_seg.realsense_rgbd_segmenter import (  # noqa: E402
    FAST_STEREO_DEFAULT_MODEL,
    FastFoundationStereoRunner,
    align_rectified_depth_to_color_torch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run timing-only single-seg inference on frames saved under live_rgbd_debug/."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "tests" / "outputs" / "realsense_live_fast_tuned" / "live_rgbd_debug",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "tests" / "outputs" / "realsense_live_fast_tuned_profile_gpu_depth",
    )
    parser.add_argument("--target-name", default="plate")
    parser.add_argument("--prompt-task-info", type=Path, default=DEFAULT_PROMPT_TASK_INFO)
    parser.add_argument("--prompt-image-root", type=Path, default=DEFAULT_PROMPT_IMAGE_ROOT)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--camera-id", default="cam_00")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--depth-source",
        choices=("saved", "fast", "auto"),
        default="saved",
        help="saved reads depth_aligned_m.npy; fast recomputes depth from IR images and camera_payload.json.",
    )
    parser.add_argument("--depth-device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--sync-timing", type=int, default=1)
    parser.add_argument("--tracker-image-size", type=int, default=896)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--frame-voxel-size", type=float, default=0.003)
    parser.add_argument("--depth-min", type=float, default=0.1)
    parser.add_argument("--depth-max", type=float, default=3.0)
    parser.add_argument("--fx", type=float, default=910.0)
    parser.add_argument("--fy", type=float, default=910.0)
    parser.add_argument("--cx", type=float, default=639.5)
    parser.add_argument("--cy", type=float, default=359.5)
    parser.add_argument("--fast-model-path", type=Path, default=FAST_STEREO_DEFAULT_MODEL)
    parser.add_argument("--fast-valid-iters", type=int, default=8)
    parser.add_argument("--fast-max-disp", type=int, default=192)
    parser.add_argument("--fast-scale", type=float, default=1.0)
    parser.add_argument("--fast-remove-invisible", type=int, default=1)
    parser.add_argument("--fast-hiera", type=int, default=0)
    return parser.parse_args()


def resolve_depth_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--depth-device cuda requested, but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mean(items: list[dict[str, object]], key: str) -> float:
    if not items:
        return 0.0
    return float(sum(float(item[key]) for item in items) / len(items))


def load_camera_payload(camera_dir: Path) -> dict[str, object] | None:
    payload_path = camera_dir / "camera_payload.json"
    if not payload_path.is_file():
        return None
    return json.loads(payload_path.read_text(encoding="utf-8"))


def fallback_intrinsics(args: argparse.Namespace, rgb: np.ndarray) -> dict[str, float]:
    return {
        "fx": float(args.fx),
        "fy": float(args.fy),
        "cx": float(args.cx),
        "cy": float(args.cy),
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
    }


def fallback_pose_record(camera_id: str) -> dict[str, object]:
    cam2world = np.eye(4, dtype=np.float64)
    return {
        "camera_id": str(camera_id),
        "cam2world_4x4": cam2world.tolist(),
        "world2cam_4x4": np.linalg.inv(cam2world).tolist(),
    }


def resolve_frame_depth_source(
    *,
    requested: str,
    camera_payload: dict[str, object] | None,
    camera_dir: Path,
) -> str:
    if requested != "auto":
        return requested
    if (
        camera_payload is not None
        and str(camera_payload.get("depth_source", "")).lower() == "fast"
        and (camera_dir / str(camera_payload.get("ir_left_rect_file", "ir_left_rect.png"))).is_file()
        and (camera_dir / str(camera_payload.get("ir_right_rect_file", "ir_right_rect.png"))).is_file()
    ):
        return "fast"
    return "saved"


def build_fast_depth_from_payload(
    *,
    camera_dir: Path,
    camera_payload: dict[str, object],
    stereo_runner: FastFoundationStereoRunner,
    depth_min: float,
    depth_max: float,
    color_shape: tuple[int, int],
) -> torch.Tensor:
    required = ("rectified_k", "rectified_to_color", "baseline_m", "color_intrinsics")
    missing = [key for key in required if key not in camera_payload]
    if missing:
        raise ValueError(f"{camera_dir / 'camera_payload.json'} is missing required fast fields: {missing}")

    left_path = camera_dir / str(camera_payload.get("ir_left_rect_file", "ir_left_rect.png"))
    right_path = camera_dir / str(camera_payload.get("ir_right_rect_file", "ir_right_rect.png"))
    if not left_path.is_file() or not right_path.is_file():
        raise FileNotFoundError(f"missing IR debug images under {camera_dir}")
    ir_left = np.asarray(Image.open(left_path).convert("L"), dtype=np.uint8)
    ir_right = np.asarray(Image.open(right_path).convert("L"), dtype=np.uint8)
    stereo_output = stereo_runner.infer_depth(
        left_image=ir_left,
        right_image=ir_right,
        rectified_k=np.asarray(camera_payload["rectified_k"], dtype=np.float32),
        baseline_m=float(camera_payload["baseline_m"]),
        return_torch=True,
    )
    depth = align_rectified_depth_to_color_torch(
        stereo_output["depth_m"],
        rectified_intrinsics=stereo_output["rectified_intrinsics"],
        rectified_to_color=np.asarray(camera_payload["rectified_to_color"], dtype=np.float64),
        color_intrinsics=dict(camera_payload["color_intrinsics"]),
        color_shape=color_shape,
    )
    return torch.where(
        torch.isfinite(depth) & (depth >= float(depth_min)) & (depth <= float(depth_max)),
        depth.to(dtype=torch.float32),
        torch.zeros((), dtype=torch.float32, device=depth.device),
    )


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    frames = sorted(path for path in input_dir.glob("frame_*") if path.is_dir())
    if int(args.max_frames) > 0:
        frames = frames[: int(args.max_frames)]
    if not frames:
        raise RuntimeError(f"no frame_* directories found under {input_dir}")

    depth_device = resolve_depth_device(str(args.depth_device))
    first_payload = load_camera_payload(frames[0] / str(args.camera_id))
    first_depth_source = resolve_frame_depth_source(
        requested=str(args.depth_source),
        camera_payload=first_payload,
        camera_dir=frames[0] / str(args.camera_id),
    )
    stereo_runner: FastFoundationStereoRunner | None = None
    if first_depth_source == "fast":
        stereo_runner = FastFoundationStereoRunner(
            model_path=Path(args.fast_model_path),
            valid_iters=int(args.fast_valid_iters),
            max_disp=int(args.fast_max_disp),
            scale=float(args.fast_scale),
            remove_invisible=bool(args.fast_remove_invisible),
            hiera=bool(args.fast_hiera),
        )

    print(f"input={input_dir}")
    print(
        f"frames={len(frames)} depth_source={args.depth_source}->{first_depth_source} "
        f"depth_device={depth_device} output={Path(args.output_dir).resolve()}"
    )
    with SingleObjectPointCloudSegmenter(
        target_name=str(args.target_name),
        prompt_task_info=Path(args.prompt_task_info).resolve(),
        prompt_image_root=Path(args.prompt_image_root).resolve(),
        checkpoint_path=Path(args.checkpoint_path).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        overwrite_output=True,
        confidence=0.25,
        mask_threshold=0.6,
        prompt_keep_score_threshold=0.2,
        video_mask_prob_threshold=0.95,
        depth_scale=1.0,
        depth_min=float(args.depth_min),
        depth_max=float(args.depth_max),
        stride=int(args.stride),
        frame_voxel_size=float(args.frame_voxel_size),
        save_ply=False,
        save_debug_2d=False,
        tracker_image_size=int(args.tracker_image_size),
    ) as segmenter:
        segmenter.sync_timing = bool(args.sync_timing)
        t0 = time.perf_counter()
        for frame_dir in frames:
            camera_dir = frame_dir / str(args.camera_id)
            rgb = np.asarray(Image.open(camera_dir / "rgb.png").convert("RGB"), dtype=np.uint8)
            camera_payload = load_camera_payload(camera_dir)
            frame_depth_source = resolve_frame_depth_source(
                requested=str(args.depth_source),
                camera_payload=camera_payload,
                camera_dir=camera_dir,
            )
            if frame_depth_source == "fast":
                if camera_payload is None:
                    raise FileNotFoundError(f"--depth-source fast requires {camera_dir / 'camera_payload.json'}")
                if stereo_runner is None:
                    raise RuntimeError("Fast-FoundationStereo runner was not initialized")
                depth = build_fast_depth_from_payload(
                    camera_dir=camera_dir,
                    camera_payload=camera_payload,
                    stereo_runner=stereo_runner,
                    depth_min=float(args.depth_min),
                    depth_max=float(args.depth_max),
                    color_shape=rgb.shape[:2],
                )
            else:
                depth_np = np.load(camera_dir / "depth_aligned_m.npy").astype(np.float32, copy=False)
                depth = (
                    torch.as_tensor(depth_np, dtype=torch.float32, device=depth_device)
                    if depth_device.type == "cuda"
                    else depth_np
                )
            frame_intrinsics = (
                dict(camera_payload["color_intrinsics"])
                if camera_payload is not None and camera_payload.get("color_intrinsics") is not None
                else fallback_intrinsics(args, rgb)
            )
            frame_intrinsics["width"] = int(rgb.shape[1])
            frame_intrinsics["height"] = int(rgb.shape[0])
            pose_record = (
                dict(camera_payload["pose_record"])
                if camera_payload is not None and camera_payload.get("pose_record") is not None
                else fallback_pose_record(str(args.camera_id))
            )
            result = segmenter.process_frame(
                frame_name=f"{frame_dir.name}.png",
                camera_inputs={
                    str(args.camera_id): {
                        "rgb": rgb,
                        "depth_m": depth,
                        "intrinsics": frame_intrinsics,
                        "pose_record": pose_record,
                        "fovy_deg": None,
                    }
                },
            )
            item = segmenter.timeline[-1]
            breakdown = item["residual_breakdown_sec"]
            print(
                f"frame={int(item['frame_index']):03d} "
                f"runtime={float(item['frame_runtime_sec']):.4f}s "
                f"propagate={float(item['propagate_time_sec']):.4f}s "
                f"mask_post={float(breakdown['mask_postprocess_time_sec']):.4f}s "
                f"backproject={float(item['backproject_time_sec']):.4f}s "
                f"fuse={float(item['fuse_time_sec']):.4f}s "
                f"points={int(result['points_xyz'].shape[0])}"
            )
        elapsed = time.perf_counter() - t0
        print(f"processed={len(frames)} elapsed={elapsed:.4f}s")

    summary_path = Path(args.output_dir).resolve() / "single_object_timeline.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    timeline = list(summary.get("timeline", []))
    later = timeline[1:] if len(timeline) > 1 else timeline
    print(
        "aggregate "
        f"all_mean={mean(timeline, 'frame_runtime_sec'):.4f}s "
        f"later_mean={mean(later, 'frame_runtime_sec'):.4f}s "
        f"later_fps={(1.0 / mean(later, 'frame_runtime_sec')) if later else 0.0:.2f}"
    )


if __name__ == "__main__":
    main()
