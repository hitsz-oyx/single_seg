from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from single_seg.single_object_segmenter import (
    DEFAULT_CHECKPOINT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROMPT_IMAGE_ROOT,
    DEFAULT_PROMPT_TASK_INFO,
    REPO_ROOT,
    SingleSegConfig,
    SingleObjectPointCloudSegmenter,
    resolve_repo_path,
)


FAST_STEREO_ROOT = REPO_ROOT / "third_party" / "fastfoundationstereo"
FAST_STEREO_DEFAULT_MODEL = (
    FAST_STEREO_ROOT / "weights" / "23-36-37" / "model_best_bp2_serialize.pth"
)
DEPTH_SOURCE_CHOICES = ("fast", "native")

if str(FAST_STEREO_ROOT) not in sys.path:
    sys.path.insert(0, str(FAST_STEREO_ROOT))

try:  # noqa: E402
    from Utils import set_logging_format, set_seed
except ImportError:  # pragma: no cover - only used when Fast-FoundationStereo deps are absent

    def set_logging_format() -> None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    def set_seed(seed: int) -> None:
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

try:  # pragma: no cover - import availability depends on host env
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - handled at runtime by run_live()
    rs = None


DISTORTION_TO_OPENCV = (
    {
        rs.distortion.none: np.zeros(5, dtype=np.float64),
        rs.distortion.brown_conrady: None,
        rs.distortion.modified_brown_conrady: None,
        rs.distortion.inverse_brown_conrady: None,
    }
    if rs is not None
    else {}
)


def intrinsics_to_matrix(intr: rs.intrinsics) -> np.ndarray:
    return np.array(
        [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def intrinsics_to_payload(intr: rs.intrinsics) -> dict[str, float]:
    return {
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "cx": float(intr.ppx),
        "cy": float(intr.ppy),
        "width": int(intr.width),
        "height": int(intr.height),
    }


def intrinsics_to_distortion(intr: rs.intrinsics) -> np.ndarray:
    coeffs = np.array(intr.coeffs[:5], dtype=np.float64)
    if intr.model in DISTORTION_TO_OPENCV:
        if DISTORTION_TO_OPENCV[intr.model] is not None:
            return DISTORTION_TO_OPENCV[intr.model].copy()
        return coeffs
    raise RuntimeError(f"Unsupported distortion model for OpenCV projection: {intr.model}")


def extrinsics_to_matrix(extr: rs.extrinsics) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = np.asarray(extr.rotation, dtype=np.float64).reshape(3, 3)
    mat[:3, 3] = np.asarray(extr.translation, dtype=np.float64)
    return mat


def latest_frames(pipeline: rs.pipeline, timeout_ms: int) -> rs.composite_frame:
    frames = pipeline.wait_for_frames(timeout_ms=timeout_ms)
    while True:
        ok, newer = pipeline.try_wait_for_frames(timeout_ms=1)
        if not ok:
            break
        frames = newer
    return frames


def ensure_three_channels(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] >= 3:
        return np.ascontiguousarray(image[..., :3])
    raise ValueError(f"unsupported image shape: {image.shape}")


def normalize_depth_source(depth_source: object) -> str:
    source = str(depth_source).strip().lower()
    if source not in DEPTH_SOURCE_CHOICES:
        raise ValueError(f"depth_source must be one of {DEPTH_SOURCE_CHOICES}, got {depth_source!r}")
    return source


def build_rectification(
    left_intr: rs.intrinsics,
    right_intr: rs.intrinsics,
    left_to_right: np.ndarray,
    *,
    image_size: tuple[int, int],
    alpha: float = 0.0,
) -> dict[str, np.ndarray]:
    k1 = intrinsics_to_matrix(left_intr)
    d1 = intrinsics_to_distortion(left_intr)
    k2 = intrinsics_to_matrix(right_intr)
    d2 = intrinsics_to_distortion(right_intr)
    r = left_to_right[:3, :3]
    t = left_to_right[:3, 3:4]
    r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
        k1,
        d1,
        k2,
        d2,
        image_size,
        r,
        t,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=float(alpha),
    )
    map1_l, map2_l = cv2.initUndistortRectifyMap(k1, d1, r1, p1, image_size, cv2.CV_32FC1)
    map1_r, map2_r = cv2.initUndistortRectifyMap(k2, d2, r2, p2, image_size, cv2.CV_32FC1)
    rectified_to_left = np.eye(4, dtype=np.float64)
    rectified_to_left[:3, :3] = r1.T
    return {
        "map1_l": map1_l,
        "map2_l": map2_l,
        "map1_r": map1_r,
        "map2_r": map2_r,
        "rectified_k": p1[:3, :3].astype(np.float32),
        "rectified_to_left": rectified_to_left,
        "baseline_m": float(np.linalg.norm(t)),
    }


def build_undistort_maps(
    intr: rs.intrinsics,
    *,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    k = intrinsics_to_matrix(intr)
    d = intrinsics_to_distortion(intr)
    return cv2.initUndistortRectifyMap(
        k,
        d,
        np.eye(3, dtype=np.float64),
        k,
        image_size,
        cv2.CV_32FC1,
    )


def backproject_depth_to_points(
    depth_m: np.ndarray,
    intrinsics: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth_m.shape
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    vv, uu = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.int32)
    depth_valid = depth_m[valid].astype(np.float32, copy=False)
    uu_valid = uu[valid].astype(np.float32, copy=False)
    vv_valid = vv[valid].astype(np.float32, copy=False)
    x = (uu_valid - cx) / fx * depth_valid
    y = (vv_valid - cy) / fy * depth_valid
    points = np.stack([x, y, depth_valid], axis=1).astype(np.float32, copy=False)
    pixels = np.stack([uu_valid.astype(np.int32), vv_valid.astype(np.int32)], axis=1)
    return points, pixels


def project_points_to_depth_image(
    points_src: np.ndarray,
    src_to_dst: np.ndarray,
    dst_intrinsics: dict[str, float],
    dst_shape: tuple[int, int],
) -> np.ndarray:
    height, width = dst_shape
    depth_out = np.full((height, width), np.inf, dtype=np.float32)
    if points_src.size == 0:
        depth_out[~np.isfinite(depth_out)] = 0.0
        return depth_out
    rot = src_to_dst[:3, :3].astype(np.float32, copy=False)
    trans = src_to_dst[:3, 3].astype(np.float32, copy=False)
    points_dst = (points_src @ rot.T) + trans
    z = points_dst[:, 2]
    valid = np.isfinite(z) & (z > 0)
    if not np.any(valid):
        depth_out[~np.isfinite(depth_out)] = 0.0
        return depth_out
    points_dst = points_dst[valid]
    z = z[valid]
    fx = float(dst_intrinsics["fx"])
    fy = float(dst_intrinsics["fy"])
    cx = float(dst_intrinsics["cx"])
    cy = float(dst_intrinsics["cy"])
    u = np.rint((points_dst[:, 0] * fx / z) + cx).astype(np.int32)
    v = np.rint((points_dst[:, 1] * fy / z) + cy).astype(np.int32)
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(in_bounds):
        depth_out[~np.isfinite(depth_out)] = 0.0
        return depth_out
    u = u[in_bounds]
    v = v[in_bounds]
    z = z[in_bounds].astype(np.float32, copy=False)
    np.minimum.at(depth_out, (v, u), z)
    depth_out[~np.isfinite(depth_out)] = 0.0
    return depth_out


def align_rectified_depth_to_color(
    depth_rect_m: np.ndarray,
    *,
    rectified_intrinsics: dict[str, float],
    rectified_to_color: np.ndarray,
    color_intrinsics: dict[str, float],
    color_shape: tuple[int, int],
) -> np.ndarray:
    points_rect, _ = backproject_depth_to_points(depth_rect_m, rectified_intrinsics)
    return project_points_to_depth_image(
        points_rect,
        rectified_to_color,
        color_intrinsics,
        color_shape,
    )


def project_points_to_depth_image_torch(
    points_src: torch.Tensor,
    src_to_dst: np.ndarray | torch.Tensor,
    dst_intrinsics: dict[str, float],
    dst_shape: tuple[int, int],
) -> torch.Tensor:
    height, width = (int(dst_shape[0]), int(dst_shape[1]))
    device = points_src.device
    depth_out = torch.full((height * width,), float("inf"), dtype=torch.float32, device=device)
    if points_src.numel() == 0:
        return torch.zeros((height, width), dtype=torch.float32, device=device)

    transform = torch.as_tensor(src_to_dst, dtype=torch.float32, device=device)
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    points_dst = (points_src.to(torch.float32) @ rot.T) + trans
    z = points_dst[:, 2]
    valid = torch.isfinite(z) & (z > 0)
    points_dst = points_dst[valid]
    z = z[valid]
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
    rectified_to_color: np.ndarray | torch.Tensor,
    color_intrinsics: dict[str, float],
    color_shape: tuple[int, int],
) -> torch.Tensor:
    depth = depth_rect_m.to(dtype=torch.float32)
    height, width = depth.shape
    device = depth.device
    valid = torch.isfinite(depth) & (depth > 0)

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


def to_jsonable(value: object) -> object:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def build_live_debug_camera_payload(
    *,
    payload: dict[str, object],
    depth_source: str,
    depth_min: float,
    depth_max: float,
) -> dict[str, object]:
    debug_payload: dict[str, object] = {
        "camera_id": str(payload["camera_id"]),
        "serial_number": str(payload.get("serial_number", "")),
        "depth_source": str(depth_source),
        "rgb_file": "rgb.png",
        "depth_aligned_file": "depth_aligned_m.npy",
        "color_intrinsics": to_jsonable(payload.get("color_intrinsics")),
        "pose_record": to_jsonable(payload.get("pose_record")),
        "depth_min": float(depth_min),
        "depth_max": float(depth_max),
    }
    if depth_source == "fast":
        debug_payload.update(
            {
                "ir_left_rect_file": "ir_left_rect.png",
                "ir_right_rect_file": "ir_right_rect.png",
                "rectified_k": to_jsonable(payload["rectified_k"]),
                "rectified_to_color": to_jsonable(payload["rectified_to_color"]),
                "baseline_m": float(payload["baseline_m"]),
            }
        )
    return debug_payload


@dataclass(frozen=True)
class LiveCameraPose:
    camera_id: str
    serial_number: str
    cam2world_4x4: np.ndarray


def load_live_camera_pose_map(pose_path: Path | None) -> dict[str, LiveCameraPose]:
    if pose_path is None:
        return {}
    payload = json.loads(Path(pose_path).read_text(encoding="utf-8"))
    cameras = payload.get("cameras", payload)
    if not isinstance(cameras, list):
        raise ValueError("camera pose file must contain a list of cameras")
    pose_map: dict[str, LiveCameraPose] = {}
    for index, camera in enumerate(cameras):
        if not isinstance(camera, dict):
            raise ValueError(f"camera pose record at index {index} must be a dict")
        camera_id = str(camera.get("camera_id", f"cam_{index:02d}"))
        serial_number = str(camera.get("serial_number", camera_id))
        cam2world = np.asarray(camera.get("cam2world_4x4"), dtype=np.float64)
        if cam2world.shape != (4, 4):
            raise ValueError(f"camera {camera_id} must define cam2world_4x4")
        pose = LiveCameraPose(
            camera_id=camera_id,
            serial_number=serial_number,
            cam2world_4x4=cam2world,
        )
        pose_map[camera_id] = pose
        pose_map[serial_number] = pose
    return pose_map


def resolve_live_pose(
    *,
    camera_index: int,
    serial_number: str,
    pose_map: dict[str, LiveCameraPose],
    camera_count: int,
) -> LiveCameraPose:
    if serial_number in pose_map:
        return pose_map[serial_number]
    camera_id = f"cam_{camera_index:02d}"
    if camera_id in pose_map:
        return pose_map[camera_id]
    if camera_count != 1:
        raise ValueError(
            "camera_poses_json is required for multi-camera fusion so each D435 has a valid cam2world_4x4"
        )
    return LiveCameraPose(
        camera_id=camera_id,
        serial_number=serial_number,
        cam2world_4x4=np.eye(4, dtype=np.float64),
    )


def pose_record_from_cam2world(camera_id: str, cam2world_4x4: np.ndarray) -> dict[str, object]:
    world2cam = np.linalg.inv(cam2world_4x4)
    return {
        "camera_id": camera_id,
        "cam2world_4x4": cam2world_4x4.tolist(),
        "world2cam_4x4": world2cam.tolist(),
    }


class FastFoundationStereoRunner:
    def __init__(
        self,
        *,
        model_path: Path,
        valid_iters: int,
        max_disp: int,
        scale: float,
        remove_invisible: bool,
        hiera: bool,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Fast-FoundationStereo checkpoint not found: {self.model_path}")
        with (self.model_path.parent / "cfg.yaml").open("r", encoding="utf-8") as handle:
            cfg: dict[str, Any] = yaml.safe_load(handle)
        cfg.update(
            {
                "model_dir": str(self.model_path),
                "valid_iters": int(valid_iters),
                "max_disp": int(max_disp),
                "scale": float(scale),
                "remove_invisible": int(remove_invisible),
                "hiera": int(hiera),
            }
        )
        self.args = OmegaConf.create(cfg)
        self.model = torch.load(str(self.model_path), map_location="cpu", weights_only=False)
        self.model.args.valid_iters = int(valid_iters)
        self.model.args.max_disp = int(max_disp)
        self.model.cuda().eval()

    def infer_depth(
        self,
        *,
        left_image: np.ndarray,
        right_image: np.ndarray,
        rectified_k: np.ndarray,
        baseline_m: float,
        return_torch: bool = False,
    ) -> dict[str, object]:
        from Utils import AMP_DTYPE  # noqa: PLC0415
        from core.utils.utils import InputPadder  # noqa: PLC0415

        left_rgb = ensure_three_channels(left_image)
        right_rgb = ensure_three_channels(right_image)
        scale = float(self.args.scale)
        if scale != 1.0:
            left_rgb = cv2.resize(left_rgb, fx=scale, fy=scale, dsize=None, interpolation=cv2.INTER_AREA)
            right_rgb = cv2.resize(
                right_rgb,
                dsize=(left_rgb.shape[1], left_rgb.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        k_model = rectified_k.astype(np.float32, copy=True)
        k_model[:2] *= scale
        height, width = left_rgb.shape[:2]
        img0 = torch.as_tensor(left_rgb, dtype=torch.float32, device="cuda")[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(right_rgb, dtype=torch.float32, device="cuda")[None].permute(0, 3, 1, 2)
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)
        with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
            if not int(self.args.hiera):
                disp = self.model.forward(
                    img0,
                    img1,
                    iters=int(self.args.valid_iters),
                    test_mode=True,
                    optimize_build_volume="pytorch1",
                )
            else:
                disp = self.model.run_hierachical(
                    img0,
                    img1,
                    iters=int(self.args.valid_iters),
                    test_mode=True,
                    small_ratio=0.5,
                )
        disp = padder.unpad(disp.float()).detach().reshape(height, width).clamp_min_(0)
        if int(self.args.remove_invisible):
            xx = torch.arange(width, dtype=disp.dtype, device=disp.device)[None, :].expand(height, width)
            invalid = (xx - disp) < 0
            disp = disp.clone()
            disp[invalid] = float("inf")
        depth_m = float(k_model[0, 0]) * float(baseline_m) / disp
        output: dict[str, object] = {
            "rectified_intrinsics": {
                "fx": float(k_model[0, 0]),
                "fy": float(k_model[1, 1]),
                "cx": float(k_model[0, 2]),
                "cy": float(k_model[1, 2]),
                "width": int(width),
                "height": int(height),
            }
        }
        if return_torch:
            output.update(
                {
                    "left_rgb": left_rgb,
                    "right_rgb": right_rgb,
                    "disparity": disp,
                    "depth_m": depth_m,
                }
            )
            return output
        disp_np = disp.detach().cpu().numpy()
        depth_np = depth_m.detach().cpu().numpy()
        output.update(
            {
                "left_rgb": left_rgb,
                "right_rgb": right_rgb,
                "disparity": disp_np.astype(np.float32, copy=False),
                "depth_m": depth_np.astype(np.float32, copy=False),
            }
        )
        return output


class RealSenseRgbdCamera:
    def __init__(
        self,
        *,
        camera_id: str,
        serial_number: str,
        cam2world_4x4: np.ndarray,
        color_width: int,
        color_height: int,
        stereo_width: int,
        stereo_height: int,
        fps: int,
        alpha: float,
        wait_timeout_ms: int,
        depth_source: str,
    ) -> None:
        self.camera_id = str(camera_id)
        self.serial_number = str(serial_number)
        self.cam2world_4x4 = np.asarray(cam2world_4x4, dtype=np.float64)
        self.color_width = int(color_width)
        self.color_height = int(color_height)
        self.stereo_width = int(stereo_width)
        self.stereo_height = int(stereo_height)
        self.fps = int(fps)
        self.alpha = float(alpha)
        self.wait_timeout_ms = int(wait_timeout_ms)
        self.depth_source = normalize_depth_source(depth_source)
        self.pipeline = rs.pipeline()
        self.profile: rs.pipeline_profile | None = None
        self.color_intrinsics: dict[str, float] | None = None
        self.color_map1: np.ndarray | None = None
        self.color_map2: np.ndarray | None = None
        self.rectification: dict[str, np.ndarray] | None = None
        self.rectified_to_color: np.ndarray | None = None
        self.align_to_color: object | None = None
        self.depth_scale = 0.001
        self.pose_record: dict[str, object] = pose_record_from_cam2world(self.camera_id, self.cam2world_4x4)

    def start(self) -> None:
        config = rs.config()
        config.enable_device(self.serial_number)
        config.enable_stream(rs.stream.color, self.color_width, self.color_height, rs.format.bgr8, self.fps)
        if self.depth_source == "fast":
            config.enable_stream(rs.stream.infrared, 1, self.stereo_width, self.stereo_height, rs.format.y8, self.fps)
            config.enable_stream(rs.stream.infrared, 2, self.stereo_width, self.stereo_height, rs.format.y8, self.fps)
        else:
            config.enable_stream(rs.stream.depth, self.stereo_width, self.stereo_height, rs.format.z16, self.fps)
        self.profile = self.pipeline.start(config)
        color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        color_intr = color_profile.get_intrinsics()
        self.color_intrinsics = intrinsics_to_payload(color_intr)
        if self.depth_source == "native":
            self.depth_scale = float(self.profile.get_device().first_depth_sensor().get_depth_scale())
            self.align_to_color = rs.align(rs.stream.color)
            return

        left_profile = self.profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        right_profile = self.profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
        left_intr = left_profile.get_intrinsics()
        right_intr = right_profile.get_intrinsics()
        left_to_right = extrinsics_to_matrix(left_profile.get_extrinsics_to(right_profile))
        left_to_color = extrinsics_to_matrix(left_profile.get_extrinsics_to(color_profile))
        self.rectification = build_rectification(
            left_intr,
            right_intr,
            left_to_right,
            image_size=(self.stereo_width, self.stereo_height),
            alpha=self.alpha,
        )
        self.color_map1, self.color_map2 = build_undistort_maps(
            color_intr,
            image_size=(self.color_width, self.color_height),
        )
        rectified_to_left = self.rectification["rectified_to_left"]
        self.rectified_to_color = left_to_color @ rectified_to_left

    def warmup(self, num_frames: int) -> None:
        for _ in range(max(int(num_frames), 0)):
            self.pipeline.wait_for_frames(timeout_ms=self.wait_timeout_ms)

    def capture(self) -> dict[str, object]:
        frames = latest_frames(self.pipeline, timeout_ms=self.wait_timeout_ms)
        if self.depth_source == "native":
            if self.align_to_color is None:
                raise RuntimeError("RealSense native depth alignment was not initialized")
            aligned_frames = self.align_to_color.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                raise RuntimeError(f"missing color/depth frame from RealSense camera {self.camera_id}")
            color_raw_bgr = np.asanyarray(color_frame.get_data())
            depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * float(self.depth_scale)
            return {
                "camera_id": self.camera_id,
                "serial_number": self.serial_number,
                "depth_source": self.depth_source,
                "rgb": cv2.cvtColor(color_raw_bgr, cv2.COLOR_BGR2RGB),
                "depth_m": depth_m,
                "color_intrinsics": self.color_intrinsics,
                "pose_record": self.pose_record,
            }

        if (
            self.rectification is None
            or self.rectified_to_color is None
            or self.color_map1 is None
            or self.color_map2 is None
        ):
            raise RuntimeError("Fast stereo rectification was not initialized")
        color_raw_bgr = np.asanyarray(frames.get_color_frame().get_data())
        left_raw = np.asanyarray(frames.get_infrared_frame(1).get_data())
        right_raw = np.asanyarray(frames.get_infrared_frame(2).get_data())
        color_undistorted_bgr = cv2.remap(
            color_raw_bgr,
            self.color_map1,
            self.color_map2,
            interpolation=cv2.INTER_LINEAR,
        )
        left_rect = cv2.remap(
            left_raw,
            self.rectification["map1_l"],
            self.rectification["map2_l"],
            interpolation=cv2.INTER_LINEAR,
        )
        right_rect = cv2.remap(
            right_raw,
            self.rectification["map1_r"],
            self.rectification["map2_r"],
            interpolation=cv2.INTER_LINEAR,
        )
        return {
            "camera_id": self.camera_id,
            "serial_number": self.serial_number,
            "depth_source": self.depth_source,
            "rgb": cv2.cvtColor(color_undistorted_bgr, cv2.COLOR_BGR2RGB),
            "ir_left_rect": left_rect,
            "ir_right_rect": right_rect,
            "rectified_k": self.rectification["rectified_k"],
            "rectified_to_color": self.rectified_to_color,
            "baseline_m": self.rectification["baseline_m"],
            "color_intrinsics": self.color_intrinsics,
            "pose_record": self.pose_record,
        }

    def stop(self) -> None:
        with contextlib.suppress(Exception):
            self.pipeline.stop()


def write_live_debug(
    *,
    output_dir: Path,
    frame_index: int,
    camera_id: str,
    depth_source: str,
    rgb: np.ndarray,
    ir_left: np.ndarray | None,
    ir_right: np.ndarray | None,
    depth_aligned_m: np.ndarray | torch.Tensor,
    camera_payload: dict[str, object] | None = None,
) -> None:
    frame_dir = output_dir / "live_rgbd_debug" / f"frame_{frame_index:05d}" / camera_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    (frame_dir / "depth_source.txt").write_text(depth_source + "\n", encoding="utf-8")
    if camera_payload is not None:
        (frame_dir / "camera_payload.json").write_text(
            json.dumps(to_jsonable(camera_payload), indent=2),
            encoding="utf-8",
        )
    cv2.imwrite(str(frame_dir / "rgb.png"), rgb[..., ::-1])
    if ir_left is not None:
        cv2.imwrite(str(frame_dir / "ir_left_rect.png"), ir_left)
    if ir_right is not None:
        cv2.imwrite(str(frame_dir / "ir_right_rect.png"), ir_right)
    if torch.is_tensor(depth_aligned_m):
        depth_aligned_m = depth_aligned_m.detach().cpu().numpy()
    np.save(frame_dir / "depth_aligned_m.npy", depth_aligned_m)
    valid = depth_aligned_m > 0
    depth_vis = np.zeros((*depth_aligned_m.shape, 3), dtype=np.uint8)
    if np.any(valid):
        lo = float(np.percentile(depth_aligned_m[valid], 2))
        hi = float(np.percentile(depth_aligned_m[valid], 98))
        if hi > lo:
            scaled = np.clip((depth_aligned_m - lo) / (hi - lo), 0.0, 1.0)
            depth_vis = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            depth_vis[~valid] = 0
    cv2.imwrite(str(frame_dir / "depth_aligned_vis.png"), depth_vis)


def parse_serials(serials_text: str | None) -> list[str]:
    if serials_text is None:
        return []
    return [item.strip() for item in serials_text.split(",") if item.strip()]


def enumerate_device_serials() -> list[str]:
    ctx = rs.context()
    serials: list[str] = []
    for device in ctx.query_devices():
        serials.append(str(device.get_info(rs.camera_info.serial_number)))
    return serials


def select_serials(*, requested_serials: list[str], camera_count: int) -> list[str]:
    available = enumerate_device_serials()
    if requested_serials:
        missing = [serial for serial in requested_serials if serial not in available]
        if missing:
            raise RuntimeError(f"requested RealSense serials not found: {missing}; available={available}")
        return requested_serials[: int(camera_count)]
    if len(available) < int(camera_count):
        raise RuntimeError(f"requested {camera_count} cameras but only found {len(available)}: {available}")
    return available[: int(camera_count)]


def maybe_apply_low_bandwidth_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if not bool(args.low_bandwidth_mode):
        return args
    args.fps = 6
    args.color_width = 640
    args.color_height = 480
    args.stereo_width = 480
    args.stereo_height = 270
    args.camera_warmup_frames = max(int(args.camera_warmup_frames), 5)
    args.wait_timeout_ms = max(int(args.wait_timeout_ms), 6000)
    return args


def normalize_serials_value(serials: object) -> str | None:
    if serials is None:
        return None
    if isinstance(serials, str):
        stripped = serials.strip()
        return stripped or None
    if isinstance(serials, (list, tuple)):
        parts = [str(item).strip() for item in serials if str(item).strip()]
        return ",".join(parts) if parts else None
    return str(serials)


def load_live_arg_defaults(config_path: Path | str | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    resolved_config_path = resolve_repo_path(config_path)
    with resolved_config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    defaults: dict[str, Any] = {}
    segmenter = SingleSegConfig.from_mapping(payload, base_dir=REPO_ROOT)
    defaults.update(segmenter.to_segmenter_kwargs())

    realsense_cfg = payload.get("realsense", {})
    if not isinstance(realsense_cfg, dict):
        raise ValueError("realsense config section must be a mapping")
    fast_cfg = payload.get("fast_stereo", {})
    if not isinstance(fast_cfg, dict):
        raise ValueError("fast_stereo config section must be a mapping")

    realsense_values = {
        "camera_count": realsense_cfg.get("camera_count"),
        "camera_serials": normalize_serials_value(realsense_cfg.get("camera_serials")),
        "camera_poses_json": realsense_cfg.get("camera_poses_json"),
        "camera_warmup_frames": realsense_cfg.get("camera_warmup_frames"),
        "wait_timeout_ms": realsense_cfg.get("wait_timeout_ms"),
        "fps": realsense_cfg.get("fps"),
        "color_width": realsense_cfg.get("color_width"),
        "color_height": realsense_cfg.get("color_height"),
        "stereo_width": realsense_cfg.get("stereo_width"),
        "stereo_height": realsense_cfg.get("stereo_height"),
        "stereo_alpha": realsense_cfg.get("stereo_alpha"),
        "depth_source": realsense_cfg.get("depth_source"),
        "low_bandwidth_mode": realsense_cfg.get("low_bandwidth_mode"),
        "max_frames": realsense_cfg.get("max_frames"),
        "save_live_debug": realsense_cfg.get("save_live_debug"),
    }
    for key, value in realsense_values.items():
        if value is None:
            continue
        if key == "camera_poses_json":
            defaults[key] = resolve_repo_path(value)
        elif key == "depth_source":
            defaults[key] = normalize_depth_source(value)
        elif key in {"low_bandwidth_mode", "save_live_debug"}:
            defaults[key] = int(bool(value))
        else:
            defaults[key] = value

    fast_values = {
        "fast_model_path": fast_cfg.get("model_path", fast_cfg.get("fast_model_path")),
        "fast_valid_iters": fast_cfg.get("valid_iters", fast_cfg.get("fast_valid_iters")),
        "fast_max_disp": fast_cfg.get("max_disp", fast_cfg.get("fast_max_disp")),
        "fast_scale": fast_cfg.get("scale", fast_cfg.get("fast_scale")),
        "fast_remove_invisible": fast_cfg.get(
            "remove_invisible",
            fast_cfg.get("fast_remove_invisible"),
        ),
        "fast_hiera": fast_cfg.get("hiera", fast_cfg.get("fast_hiera")),
    }
    for key, value in fast_values.items():
        if value is None:
            continue
        if key == "fast_model_path":
            defaults[key] = resolve_repo_path(value)
        elif key in {"fast_remove_invisible", "fast_hiera"}:
            defaults[key] = int(bool(value))
        else:
            defaults[key] = value
    return defaults


def build_arg_parser(defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run single-seg on live D435 cameras with Fast-FoundationStereo or native RealSense depth."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--target-name", default="plate")
    parser.add_argument("--prompt-task-info", type=Path, default=DEFAULT_PROMPT_TASK_INFO)
    parser.add_argument("--prompt-image-root", type=Path, default=DEFAULT_PROMPT_IMAGE_ROOT)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "realsense_live")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--save-ply", action="store_true", default=False)
    parser.add_argument("--save-debug-2d", action="store_true", default=False)
    parser.add_argument("--max-frames", type=int, default=1)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--frame-voxel-size", type=float, default=0.003)
    parser.add_argument("--depth-min", type=float, default=0.1)
    parser.add_argument("--depth-max", type=float, default=3.0)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--mask-threshold", type=float, default=0.6)
    parser.add_argument("--prompt-keep-score-threshold", type=float, default=0.2)
    parser.add_argument("--video-mask-prob-threshold", type=float, default=0.95)
    parser.add_argument("--tracker-image-size", type=int, default=896)
    parser.add_argument("--camera-count", type=int, default=1)
    parser.add_argument("--camera-serials", type=str, default=None)
    parser.add_argument("--camera-poses-json", type=Path, default=None)
    parser.add_argument("--camera-warmup-frames", type=int, default=5)
    parser.add_argument("--wait-timeout-ms", type=int, default=6000)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--color-width", type=int, default=640)
    parser.add_argument("--color-height", type=int, default=480)
    parser.add_argument("--stereo-width", type=int, default=480)
    parser.add_argument("--stereo-height", type=int, default=270)
    parser.add_argument("--stereo-alpha", type=float, default=0.0)
    parser.add_argument("--depth-source", choices=DEPTH_SOURCE_CHOICES, default="fast")
    parser.add_argument("--low-bandwidth-mode", type=int, default=1)
    parser.add_argument("--fast-model-path", type=Path, default=FAST_STEREO_DEFAULT_MODEL)
    parser.add_argument("--fast-valid-iters", type=int, default=8)
    parser.add_argument("--fast-max-disp", type=int, default=192)
    parser.add_argument("--fast-scale", type=float, default=1.0)
    parser.add_argument("--fast-remove-invisible", type=int, default=1)
    parser.add_argument("--fast-hiera", type=int, default=0)
    parser.add_argument("--save-live-debug", type=int, default=1)
    if defaults:
        parser.set_defaults(**defaults)
    return parser


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    parser = build_arg_parser(load_live_arg_defaults(pre_args.config))
    return maybe_apply_low_bandwidth_defaults(parser.parse_args())


def build_camera_inputs_from_live_frames(
    *,
    captured_frames: list[dict[str, object]],
    stereo_runner: FastFoundationStereoRunner | None,
    depth_min: float,
    depth_max: float,
    output_dir: Path,
    frame_index: int,
    write_debug_images: bool,
) -> dict[str, dict[str, object]]:
    camera_inputs: dict[str, dict[str, object]] = {}
    for payload in captured_frames:
        camera_id = str(payload["camera_id"])
        depth_source = normalize_depth_source(payload.get("depth_source", "fast"))
        logging.info(f"Building RGBD for {camera_id} using depth_source={depth_source}")
        rgb = np.asarray(payload["rgb"], dtype=np.uint8)
        ir_left_rect: np.ndarray | None = None
        ir_right_rect: np.ndarray | None = None
        if depth_source == "native":
            depth_aligned_m = np.asarray(payload["depth_m"], dtype=np.float32)
        else:
            if stereo_runner is None:
                raise RuntimeError("depth_source='fast' requires a Fast-FoundationStereo runner")
            ir_left_rect = np.asarray(payload["ir_left_rect"], dtype=np.uint8)
            ir_right_rect = np.asarray(payload["ir_right_rect"], dtype=np.uint8)
            stereo_output = stereo_runner.infer_depth(
                left_image=ir_left_rect,
                right_image=ir_right_rect,
                rectified_k=np.asarray(payload["rectified_k"], dtype=np.float32),
                baseline_m=float(payload["baseline_m"]),
                return_torch=True,
            )
            depth_aligned_m = align_rectified_depth_to_color_torch(
                stereo_output["depth_m"],
                rectified_intrinsics=stereo_output["rectified_intrinsics"],
                rectified_to_color=np.asarray(payload["rectified_to_color"], dtype=np.float64),
                color_intrinsics=dict(payload["color_intrinsics"]),
                color_shape=rgb.shape[:2],
            )
        if torch.is_tensor(depth_aligned_m):
            depth_aligned_m = depth_aligned_m.to(dtype=torch.float32)
            depth_aligned_m = torch.where(
                torch.isfinite(depth_aligned_m)
                & (depth_aligned_m >= float(depth_min))
                & (depth_aligned_m <= float(depth_max)),
                depth_aligned_m,
                torch.zeros((), dtype=torch.float32, device=depth_aligned_m.device),
            )
            depth_valid_ratio = float((depth_aligned_m > 0).float().mean().item())
        else:
            depth_aligned_m = np.asarray(depth_aligned_m, dtype=np.float32).copy()
            depth_aligned_m[~np.isfinite(depth_aligned_m)] = 0.0
            depth_aligned_m[(depth_aligned_m < float(depth_min)) | (depth_aligned_m > float(depth_max))] = 0.0
            depth_valid_ratio = float((depth_aligned_m > 0).mean())
        camera_inputs[camera_id] = {
            "rgb": rgb,
            "depth_m": depth_aligned_m,
            "intrinsics": dict(payload["color_intrinsics"]),
            "pose_record": dict(payload["pose_record"]),
            "fovy_deg": None,
        }
        if write_debug_images:
            write_live_debug(
                output_dir=output_dir,
                frame_index=frame_index,
                camera_id=camera_id,
                depth_source=depth_source,
                rgb=rgb,
                ir_left=ir_left_rect,
                ir_right=ir_right_rect,
                depth_aligned_m=depth_aligned_m,
                camera_payload=build_live_debug_camera_payload(
                    payload=payload,
                    depth_source=depth_source,
                    depth_min=float(depth_min),
                    depth_max=float(depth_max),
                ),
            )
        logging.info(
            f"Built RGBD for {camera_id}: source={depth_source} rgb={rgb.shape} "
            f"depth_valid_ratio={depth_valid_ratio:.3f}"
        )
    return camera_inputs


def run_live(args: argparse.Namespace) -> None:
    if rs is None:
        raise RuntimeError("pyrealsense2 is required for the RealSense live runner")
    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    depth_source = normalize_depth_source(args.depth_source)
    requested_serials = parse_serials(args.camera_serials)
    serials = select_serials(requested_serials=requested_serials, camera_count=int(args.camera_count))
    pose_map = load_live_camera_pose_map(args.camera_poses_json)
    cameras: list[RealSenseRgbdCamera] = []
    try:
        for index, serial in enumerate(serials):
            pose = resolve_live_pose(
                camera_index=index,
                serial_number=serial,
                pose_map=pose_map,
                camera_count=int(args.camera_count),
            )
            camera = RealSenseRgbdCamera(
                camera_id=pose.camera_id,
                serial_number=serial,
                cam2world_4x4=pose.cam2world_4x4,
                color_width=int(args.color_width),
                color_height=int(args.color_height),
                stereo_width=int(args.stereo_width),
                stereo_height=int(args.stereo_height),
                fps=int(args.fps),
                alpha=float(args.stereo_alpha),
                wait_timeout_ms=int(args.wait_timeout_ms),
                depth_source=depth_source,
            )
            camera.start()
            camera.warmup(int(args.camera_warmup_frames))
            cameras.append(camera)
            logging.info(f"Started camera {camera.camera_id} serial={serial} depth_source={depth_source}")

        stereo_runner: FastFoundationStereoRunner | None = None
        if depth_source == "fast":
            stereo_runner = FastFoundationStereoRunner(
                model_path=Path(args.fast_model_path),
                valid_iters=int(args.fast_valid_iters),
                max_disp=int(args.fast_max_disp),
                scale=float(args.fast_scale),
                remove_invisible=bool(args.fast_remove_invisible),
                hiera=bool(args.fast_hiera),
            )
            logging.info("Fast-FoundationStereo runner loaded")
        else:
            logging.info("Using RealSense native depth aligned to color; Fast-FoundationStereo runner not loaded")
        with SingleObjectPointCloudSegmenter(
            target_name=str(args.target_name),
            prompt_task_info=Path(args.prompt_task_info).resolve(),
            prompt_image_root=Path(args.prompt_image_root).resolve(),
            checkpoint_path=Path(args.checkpoint_path).resolve(),
            output_dir=Path(args.output_dir).resolve(),
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
            save_ply=bool(args.save_ply),
            save_debug_2d=bool(args.save_debug_2d),
            tracker_image_size=int(args.tracker_image_size),
        ) as segmenter:
            logging.info("SingleObjectPointCloudSegmenter loaded")
            frame_limit = None if int(args.max_frames) <= 0 else int(args.max_frames)
            frame_index = 0
            while frame_limit is None or frame_index < frame_limit:
                logging.info(f"Capturing live frame {frame_index:05d}")
                captured_frames = [camera.capture() for camera in cameras]
                camera_inputs = build_camera_inputs_from_live_frames(
                    captured_frames=captured_frames,
                    stereo_runner=stereo_runner,
                    depth_min=float(args.depth_min),
                    depth_max=float(args.depth_max),
                    output_dir=Path(args.output_dir).resolve(),
                    frame_index=frame_index,
                    write_debug_images=bool(args.save_live_debug),
                )
                frame_name = f"frame_{frame_index:05d}.png"
                logging.info(f"Running single-seg for {frame_name}")
                result = segmenter.process_frame(
                    frame_name=frame_name,
                    camera_inputs=camera_inputs,
                )
                logging.info(
                    f"[frame {result['frame_index']:03d}] {frame_name} "
                    f"points={result['points_xyz'].shape[0]} cameras={len(camera_inputs)}"
                )
                frame_index += 1
    finally:
        for camera in cameras:
            camera.stop()


def main() -> None:
    run_live(parse_args())


if __name__ == "__main__":
    main()
