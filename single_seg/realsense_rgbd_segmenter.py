from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from typing import Any

import cv2
import numpy as np
import open3d as o3d
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
STEREO_RECTIFICATION_CHOICES = ("opencv", "passthrough")
DEPTH_EDGE_FILTER_STAGE_CHOICES = ("rectified", "aligned")
FAST_ALIGN_BACKEND_CHOICES = ("torch", "open3d", "librealsense")
CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)
_RECTIFIED_RAY_GRID_CACHE: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor]] = {}

if str(FAST_STEREO_ROOT) not in sys.path:
    sys.path.insert(0, str(FAST_STEREO_ROOT))

ICP_DIR = REPO_ROOT / "icp"
if str(ICP_DIR) not in sys.path:
    sys.path.insert(0, str(ICP_DIR))


def _transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    return (points @ T[:3, :3].T + T[:3, 3][None, :]).astype(np.float64)


def _invert_transform(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def _rotation_matrix_to_quaternion(R: np.ndarray) -> tuple:
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return w, x, y, z


def _format_icp_pose(R: np.ndarray, t: np.ndarray) -> str:
    qw, qx, qy, qz = _rotation_matrix_to_quaternion(R)
    return (
        f"  position: [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]\n"
        f"  orientation (R):\n"
        f"    [{R[0,0]:.6f}, {R[0,1]:.6f}, {R[0,2]:.6f}]\n"
        f"    [{R[1,0]:.6f}, {R[1,1]:.6f}, {R[1,2]:.6f}]\n"
        f"    [{R[2,0]:.6f}, {R[2,1]:.6f}, {R[2,2]:.6f}]\n"
        f"  orientation (quat): [qw={qw:.6f}, qx={qx:.6f}, qy={qy:.6f}, qz={qz:.6f}]"
    )


def _sample_obj_surface_points(obj_path: Path, sample_points: int = 10000) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(str(obj_path), enable_post_processing=True)
    if mesh.is_empty():
        raise ValueError(f"Mesh is empty: {obj_path}")
    sampled = mesh.sample_points_uniformly(number_of_points=sample_points)
    points = np.asarray(sampled.points, dtype=np.float64)
    logging.info(f"Sampled mesh {obj_path.name}: {len(points)} points")
    return points


class _PointCloudViewer:
    def __init__(self) -> None:
        import queue
        import threading

        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._closed = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update(self, points: np.ndarray, colors: np.ndarray | None) -> None:
        if self._closed:
            return
        import queue as _queue_mod
        try:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except _queue_mod.Empty:
                    break
            self._queue.put_nowait((points, colors))
        except _queue_mod.Full:
            pass

    def _run(self) -> None:
        import queue as _queue_mod
        import open3d as o3d
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Single-Seg Live", width=1280, height=720)
        pcd = o3d.geometry.PointCloud()
        first_frame = True
        while not self._closed:
            try:
                points, colors = self._queue.get(timeout=0.1)
            except _queue_mod.Empty:
                vis.poll_events()
                if hasattr(vis, "update_renderer"):
                    vis.update_renderer()
                continue
            if points.shape[0] == 0:
                continue
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            if colors is not None and colors.shape[0] == points.shape[0]:
                pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
            if first_frame:
                vis.add_geometry(pcd)
                first_frame = False
                opt = vis.get_render_option()
                opt.point_size = 2.0
            else:
                vis.update_geometry(pcd)
            vis.poll_events()
            if hasattr(vis, "update_renderer"):
                vis.update_renderer()
            elif hasattr(vis, "render"):
                vis.render()

    def close(self) -> None:
        self._closed = True
        self._thread.join(timeout=2.0)


def sync_cuda_if_needed(enabled: bool) -> None:
    """Synchronize CUDA so wall-clock timing reflects queued GPU work."""
    if bool(enabled) and torch.cuda.is_available():
        torch.cuda.synchronize()


def get_rectified_ray_grid(
    *,
    height: int,
    width: int,
    intrinsics: dict[str, float],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cached per-pixel x/z and y/z factors for a rectified depth image."""
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    key = (
        int(height),
        int(width),
        round(fx, 6),
        round(fy, 6),
        round(cx, 6),
        round(cy, 6),
        device.type,
        device.index,
    )
    cached = _RECTIFIED_RAY_GRID_CACHE.get(key)
    if cached is not None:
        return cached
    v = torch.arange(int(height), dtype=torch.float32, device=device)
    u = torch.arange(int(width), dtype=torch.float32, device=device)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    x_over_z = ((uu - cx) / fx).reshape(-1)
    y_over_z = ((vv - cy) / fy).reshape(-1)
    cached = (x_over_z, y_over_z)
    _RECTIFIED_RAY_GRID_CACHE[key] = cached
    return cached

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


def intrinsics_payload_from_k(k: np.ndarray, *, width: int, height: int) -> dict[str, float]:
    matrix = np.asarray(k, dtype=np.float64).reshape(3, 3)
    return {
        "fx": float(matrix[0, 0]),
        "fy": float(matrix[1, 1]),
        "cx": float(matrix[0, 2]),
        "cy": float(matrix[1, 2]),
        "width": int(width),
        "height": int(height),
    }


def intrinsics_payload_to_rs_intrinsics(
    intrinsics: dict[str, float],
    *,
    width: int,
    height: int,
) -> rs.intrinsics:
    intr = rs.intrinsics()
    intr.width = int(width)
    intr.height = int(height)
    intr.fx = float(intrinsics["fx"])
    intr.fy = float(intrinsics["fy"])
    intr.ppx = float(intrinsics["cx"])
    intr.ppy = float(intrinsics["cy"])
    intr.model = rs.distortion.none
    intr.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
    return intr


def intrinsics_to_distortion(intr: rs.intrinsics) -> np.ndarray:
    coeffs = np.array(intr.coeffs[:5], dtype=np.float64)
    if intr.model in DISTORTION_TO_OPENCV:
        if DISTORTION_TO_OPENCV[intr.model] is not None:
            return DISTORTION_TO_OPENCV[intr.model].copy()
        return coeffs
    raise RuntimeError(f"Unsupported distortion model for OpenCV projection: {intr.model}")


def extrinsics_to_matrix(extr: rs.extrinsics) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = np.asarray(extr.rotation, dtype=np.float64).reshape(3, 3).T
    mat[:3, 3] = np.asarray(extr.translation, dtype=np.float64)
    return mat


def matrix_to_rs_extrinsics(mat: np.ndarray | torch.Tensor) -> rs.extrinsics:
    if torch.is_tensor(mat):
        matrix = mat.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        matrix = np.asarray(mat, dtype=np.float32)
    extr = rs.extrinsics()
    extr.rotation = matrix[:3, :3].T.reshape(-1).tolist()
    extr.translation = matrix[:3, 3].tolist()
    return extr


def depth_gl_to_color_gl_transform(depth_to_color_cv: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert an OpenCV depth->color transform into single_seg GL camera coordinates."""
    if torch.is_tensor(depth_to_color_cv):
        depth_to_color = depth_to_color_cv.detach().cpu().numpy()
    else:
        depth_to_color = np.asarray(depth_to_color_cv, dtype=np.float64)
    return CV_TO_GL @ depth_to_color.reshape(4, 4) @ CV_TO_GL


def depth_cam2world_from_color_pose(
    color_cam2world_gl: np.ndarray,
    depth_to_color_cv: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Compose RGB/color cam2world with depth->color to get depth cam2world in GL convention."""
    color_cam2world = np.asarray(color_cam2world_gl, dtype=np.float64).reshape(4, 4)
    return color_cam2world @ depth_gl_to_color_gl_transform(depth_to_color_cv)


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


def image_to_cuda_chw(image: np.ndarray) -> torch.Tensor:
    """Convert a 2D/3D image array to a CUDA NCHW float tensor."""
    image = np.asarray(image)
    if not image.flags.c_contiguous or not image.flags.writeable:
        image = np.array(image, copy=True, order="C")
    if image.ndim == 2:
        image_t = torch.as_tensor(image, dtype=torch.float32, device="cuda")
        return image_t[None, None].expand(1, 3, image.shape[0], image.shape[1]).contiguous()
    if image.ndim == 3 and image.shape[2] >= 3:
        image_t = torch.as_tensor(np.ascontiguousarray(image[..., :3]), dtype=torch.float32, device="cuda")
        return image_t[None].permute(0, 3, 1, 2).contiguous()
    raise ValueError(f"unsupported image shape: {image.shape}")


def resize_model_image(
    image: np.ndarray,
    *,
    scale: float,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    image = np.asarray(image)
    if target_size is not None:
        return cv2.resize(image, dsize=target_size, interpolation=cv2.INTER_AREA)
    if float(scale) != 1.0:
        return cv2.resize(image, fx=float(scale), fy=float(scale), dsize=None, interpolation=cv2.INTER_AREA)
    return image


def normalize_depth_source(depth_source: object) -> str:
    source = str(depth_source).strip().lower()
    if source not in DEPTH_SOURCE_CHOICES:
        raise ValueError(f"depth_source must be one of {DEPTH_SOURCE_CHOICES}, got {depth_source!r}")
    return source


def normalize_stereo_rectification_mode(mode: object) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in STEREO_RECTIFICATION_CHOICES:
        raise ValueError(f"stereo_rectification_mode must be one of {STEREO_RECTIFICATION_CHOICES}, got {mode!r}")
    return normalized


def normalize_depth_edge_filter_stage(stage: object) -> str:
    normalized = str(stage).strip().lower()
    if normalized not in DEPTH_EDGE_FILTER_STAGE_CHOICES:
        raise ValueError(f"depth_edge_filter_stage must be one of {DEPTH_EDGE_FILTER_STAGE_CHOICES}, got {stage!r}")
    return normalized


def normalize_fast_align_backend(backend: object) -> str:
    normalized = str(backend).strip().lower()
    if normalized not in FAST_ALIGN_BACKEND_CHOICES:
        raise ValueError(f"fast_align_backend must be one of {FAST_ALIGN_BACKEND_CHOICES}, got {backend!r}")
    return normalized


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


def build_passthrough_rectification(left_intr: rs.intrinsics, left_to_right: np.ndarray) -> dict[str, object]:
    """Use RealSense IR frames directly when the device already returns rectified stereo streams."""
    translation = np.asarray(left_to_right[:3, 3], dtype=np.float64)
    baseline = abs(float(translation[0]))
    if baseline <= 0.0:
        baseline = float(np.linalg.norm(translation))
    return {
        "map1_l": None,
        "map2_l": None,
        "map1_r": None,
        "map2_r": None,
        "rectified_k": intrinsics_to_matrix(left_intr).astype(np.float32),
        "rectified_to_left": np.eye(4, dtype=np.float64),
        "baseline_m": baseline,
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
    out_height, out_width = (int(color_shape[0]), int(color_shape[1]))
    depth_out = torch.full(
        (out_height * out_width,),
        float("inf"),
        dtype=torch.float32,
        device=device,
    )

    depth_flat = depth.reshape(-1)
    valid = torch.isfinite(depth_flat) & (depth_flat > 0)
    depth_valid = depth_flat[valid]
    if depth_valid.numel() == 0:
        return torch.zeros((out_height, out_width), dtype=torch.float32, device=device)

    x_over_z, y_over_z = get_rectified_ray_grid(
        height=int(height),
        width=int(width),
        intrinsics=rectified_intrinsics,
        device=device,
    )
    x_rect = x_over_z[valid] * depth_valid
    y_rect = y_over_z[valid] * depth_valid

    transform = torch.as_tensor(rectified_to_color, dtype=torch.float32, device=device)
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    x_color = rot[0, 0] * x_rect + rot[0, 1] * y_rect + rot[0, 2] * depth_valid + trans[0]
    y_color = rot[1, 0] * x_rect + rot[1, 1] * y_rect + rot[1, 2] * depth_valid + trans[1]
    z_color = rot[2, 0] * x_rect + rot[2, 1] * y_rect + rot[2, 2] * depth_valid + trans[2]

    valid_z = torch.isfinite(z_color) & (z_color > 0)
    x_color = x_color[valid_z]
    y_color = y_color[valid_z]
    z_color = z_color[valid_z]
    if z_color.numel() == 0:
        return torch.zeros((out_height, out_width), dtype=torch.float32, device=device)

    fx = float(color_intrinsics["fx"])
    fy = float(color_intrinsics["fy"])
    cx = float(color_intrinsics["cx"])
    cy = float(color_intrinsics["cy"])
    u = torch.round((x_color * fx / z_color) + cx).to(torch.int64)
    v = torch.round((y_color * fy / z_color) + cy).to(torch.int64)
    in_bounds = (u >= 0) & (u < out_width) & (v >= 0) & (v < out_height)
    linear = (v[in_bounds] * out_width) + u[in_bounds]
    if linear.numel() > 0:
        depth_out.scatter_reduce_(
            0,
            linear,
            z_color[in_bounds].to(torch.float32),
            reduce="amin",
            include_self=True,
        )
    depth_out[~torch.isfinite(depth_out)] = 0.0
    return depth_out.reshape(out_height, out_width)


def align_color_to_rectified_depth_torch(
    color_image: np.ndarray | torch.Tensor,
    depth_rect_m: torch.Tensor,
    *,
    rectified_intrinsics: dict[str, float],
    rectified_to_color: np.ndarray | torch.Tensor,
    color_intrinsics: dict[str, float],
) -> torch.Tensor:
    """Sample an RGB image onto the rectified depth grid without resampling depth."""
    depth = depth_rect_m.to(dtype=torch.float32)
    height, width = depth.shape
    device = depth.device
    if torch.is_tensor(color_image):
        color = color_image.to(device=device, dtype=torch.uint8, non_blocking=True)
    else:
        color = torch.as_tensor(np.ascontiguousarray(color_image), dtype=torch.uint8, device=device)
    if color.ndim != 3 or color.shape[2] < 3:
        raise ValueError(f"color_image must be HxWx3, got {tuple(color.shape)}")
    color = color[:, :, :3]
    color_height, color_width = int(color.shape[0]), int(color.shape[1])
    output = torch.zeros((height * width, 3), dtype=torch.uint8, device=device)

    depth_flat = depth.reshape(-1)
    valid = torch.isfinite(depth_flat) & (depth_flat > 0)
    depth_valid = depth_flat[valid]
    if depth_valid.numel() == 0:
        return output.reshape(height, width, 3)

    x_over_z, y_over_z = get_rectified_ray_grid(
        height=int(height),
        width=int(width),
        intrinsics=rectified_intrinsics,
        device=device,
    )
    x_rect = x_over_z[valid] * depth_valid
    y_rect = y_over_z[valid] * depth_valid

    transform = torch.as_tensor(rectified_to_color, dtype=torch.float32, device=device)
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    x_color = rot[0, 0] * x_rect + rot[0, 1] * y_rect + rot[0, 2] * depth_valid + trans[0]
    y_color = rot[1, 0] * x_rect + rot[1, 1] * y_rect + rot[1, 2] * depth_valid + trans[1]
    z_color = rot[2, 0] * x_rect + rot[2, 1] * y_rect + rot[2, 2] * depth_valid + trans[2]
    valid_z = torch.isfinite(z_color) & (z_color > 0)
    if not bool(valid_z.any().item()):
        return output.reshape(height, width, 3)

    fx = float(color_intrinsics["fx"])
    fy = float(color_intrinsics["fy"])
    cx = float(color_intrinsics["cx"])
    cy = float(color_intrinsics["cy"])
    u = torch.round((x_color[valid_z] * fx / z_color[valid_z]) + cx).to(torch.int64)
    v = torch.round((y_color[valid_z] * fy / z_color[valid_z]) + cy).to(torch.int64)
    in_bounds = (u >= 0) & (u < color_width) & (v >= 0) & (v < color_height)
    if not bool(in_bounds.any().item()):
        return output.reshape(height, width, 3)

    depth_indices = torch.nonzero(valid, as_tuple=False).flatten()[valid_z]
    output[depth_indices[in_bounds]] = color[v[in_bounds], u[in_bounds]]
    return output.reshape(height, width, 3)


def align_rectified_depth_to_color_open3d(
    depth_rect_m: torch.Tensor,
    *,
    rectified_intrinsics: dict[str, float],
    rectified_to_color: np.ndarray | torch.Tensor,
    color_intrinsics: dict[str, float],
    color_shape: tuple[int, int],
    depth_scale: float = 10000.0,
    depth_max_m: float = 3.0,
) -> torch.Tensor:
    """Depth-to-color alignment via Open3D tensor point projection."""
    try:
        import open3d as o3d  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on optional Open3D install
        raise RuntimeError("fast_align_backend='open3d' requires open3d") from exc

    depth = depth_rect_m.to(dtype=torch.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth_rect_m must be HxW, got {tuple(depth.shape)}")
    height, width = (int(depth.shape[0]), int(depth.shape[1]))
    out_height, out_width = (int(color_shape[0]), int(color_shape[1]))
    device = depth.device
    if device.type == "cuda":
        if not o3d.core.cuda.is_available():
            raise RuntimeError("fast_align_backend='open3d' requires an Open3D build with CUDA support")
        cuda_index = device.index if device.index is not None else torch.cuda.current_device()
        o3d_device = o3d.core.Device(f"CUDA:{int(cuda_index)}")
    else:
        o3d_device = o3d.core.Device("CPU:0")

    valid_depth = torch.where(
        torch.isfinite(depth) & (depth > 0.0),
        depth,
        torch.zeros((), dtype=torch.float32, device=device),
    )
    depth_u16 = torch.clamp(torch.round(valid_depth * float(depth_scale)), 0, 65535).to(torch.uint16).contiguous()
    depth_tensor = o3d.core.Tensor.from_dlpack(
        torch.utils.dlpack.to_dlpack(depth_u16.reshape(height, width, 1))
    )
    depth_image = o3d.t.geometry.Image(depth_tensor)
    rectified_k = o3d.core.Tensor(
        [
            [float(rectified_intrinsics["fx"]), 0.0, float(rectified_intrinsics["cx"])],
            [0.0, float(rectified_intrinsics["fy"]), float(rectified_intrinsics["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=o3d.core.Dtype.Float32,
        device=o3d_device,
    )
    color_k = o3d.core.Tensor(
        [
            [float(color_intrinsics["fx"]), 0.0, float(color_intrinsics["cx"])],
            [0.0, float(color_intrinsics["fy"]), float(color_intrinsics["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=o3d.core.Dtype.Float32,
        device=o3d_device,
    )
    if torch.is_tensor(rectified_to_color):
        transform_np = rectified_to_color.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        transform_np = np.asarray(rectified_to_color, dtype=np.float32)
    transform = o3d.core.Tensor(transform_np, dtype=o3d.core.Dtype.Float32, device=o3d_device)

    pointcloud = o3d.t.geometry.PointCloud.create_from_depth_image(
        depth_image,
        rectified_k,
        depth_scale=float(depth_scale),
        depth_max=float(depth_max_m),
        stride=1,
        with_normals=False,
    )
    pointcloud.transform(transform)
    projected = pointcloud.project_to_depth_image(
        out_width,
        out_height,
        color_k,
        depth_scale=float(depth_scale),
        depth_max=float(depth_max_m),
    )
    projected_t = torch.utils.dlpack.from_dlpack(projected.as_tensor().to_dlpack()).reshape(out_height, out_width)
    return projected_t.to(device=device, dtype=torch.float32) / float(depth_scale)


class LibrealsenseSoftwareAligner:
    """Depth/color aligner using librealsense software frames."""

    def __init__(
        self,
        *,
        rectified_intrinsics: dict[str, float],
        rectified_to_color: np.ndarray | torch.Tensor,
        color_intrinsics: dict[str, float],
        depth_shape: tuple[int, int],
        color_shape: tuple[int, int],
        depth_units: float = 0.0001,
        fps: int = 30,
        align_to: str = "color",
    ) -> None:
        self.depth_units = float(depth_units)
        self.depth_height, self.depth_width = (int(depth_shape[0]), int(depth_shape[1]))
        self.color_height, self.color_width = (int(color_shape[0]), int(color_shape[1]))
        align_to_normalized = str(align_to).strip().lower()
        if align_to_normalized not in {"color", "depth"}:
            raise ValueError("align_to must be 'color' or 'depth'")
        self.align_to = align_to_normalized
        self.frame_number = 0
        self._pixel_refs: list[np.ndarray] = []

        self.device = rs.software_device()
        self.depth_sensor = self.device.add_sensor("Depth")
        self.color_sensor = self.device.add_sensor("Color")

        option_range = rs.option_range()
        option_range.min = 0.00001
        option_range.max = 0.1
        option_range.step = 0.00001
        option_range.default = self.depth_units
        self.depth_sensor.add_option(rs.option.depth_units, option_range, False)

        depth_intrinsics = intrinsics_payload_to_rs_intrinsics(
            rectified_intrinsics,
            width=self.depth_width,
            height=self.depth_height,
        )
        depth_stream = rs.video_stream()
        depth_stream.type = rs.stream.depth
        depth_stream.width = self.depth_width
        depth_stream.height = self.depth_height
        depth_stream.fps = int(fps)
        depth_stream.bpp = 2
        depth_stream.fmt = rs.format.z16
        depth_stream.intrinsics = depth_intrinsics
        depth_stream.index = 0
        depth_stream.uid = 1
        self.depth_profile = self.depth_sensor.add_video_stream(depth_stream)

        color_intrinsics_rs = intrinsics_payload_to_rs_intrinsics(
            color_intrinsics,
            width=self.color_width,
            height=self.color_height,
        )
        color_stream = rs.video_stream()
        color_stream.type = rs.stream.color
        color_stream.width = self.color_width
        color_stream.height = self.color_height
        color_stream.fps = int(fps)
        color_stream.bpp = 3
        color_stream.fmt = rs.format.rgb8
        color_stream.intrinsics = color_intrinsics_rs
        color_stream.index = 0
        color_stream.uid = 2
        self.color_profile = self.color_sensor.add_video_stream(color_stream)

        depth_to_color = matrix_to_rs_extrinsics(rectified_to_color)
        self.depth_profile.register_extrinsics_to(self.color_profile, depth_to_color)

        self.syncer = rs.syncer()
        self.depth_sensor.open(self.depth_profile)
        self.color_sensor.open(self.color_profile)
        self.depth_sensor.start(self.syncer)
        self.color_sensor.start(self.syncer)
        self.rs_align = rs.align(rs.stream.color if self.align_to == "color" else rs.stream.depth)
        self._prime_syncer()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.depth_sensor.stop()
        with contextlib.suppress(Exception):
            self.color_sensor.stop()
        with contextlib.suppress(Exception):
            self.depth_sensor.close()
        with contextlib.suppress(Exception):
            self.color_sensor.close()

    def _depth_to_z16(self, depth_m: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth_m, dtype=np.float32)
        valid = np.isfinite(depth) & (depth > 0.0)
        z16 = np.zeros(depth.shape, dtype=np.uint16)
        scaled = np.rint(depth[valid] / self.depth_units)
        z16[valid] = np.clip(scaled, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        return np.ascontiguousarray(z16)

    def _push_frames(self, depth_z16: np.ndarray, color_rgb: np.ndarray) -> None:
        frame_number = int(self.frame_number)
        self.frame_number += 1
        timestamp_ms = frame_number * (1000.0 / 30.0)
        depth_pixels = np.ascontiguousarray(depth_z16)
        color_pixels = np.ascontiguousarray(color_rgb)
        self._pixel_refs.extend([depth_pixels, color_pixels])
        if len(self._pixel_refs) > 12:
            del self._pixel_refs[:-12]

        depth_frame = rs.software_video_frame()
        depth_frame.stride = self.depth_width * 2
        depth_frame.bpp = 2
        depth_frame.timestamp = timestamp_ms
        depth_frame.pixels = depth_pixels
        depth_frame.domain = rs.timestamp_domain.hardware_clock
        depth_frame.frame_number = frame_number
        depth_frame.profile = self.depth_profile.as_video_stream_profile()
        depth_frame.depth_units = self.depth_units
        self.depth_sensor.on_video_frame(depth_frame)

        color_frame = rs.software_video_frame()
        color_frame.stride = self.color_width * 3
        color_frame.bpp = 3
        color_frame.timestamp = timestamp_ms
        color_frame.pixels = color_pixels
        color_frame.domain = rs.timestamp_domain.hardware_clock
        color_frame.frame_number = frame_number
        color_frame.profile = self.color_profile.as_video_stream_profile()
        self.color_sensor.on_video_frame(color_frame)

    def _wait_for_frameset(self, timeout_ms: int = 1000) -> rs.composite_frame:
        for _ in range(4):
            frames = self.syncer.wait_for_frames(timeout_ms)
            if frames.get_depth_frame() and frames.get_color_frame():
                return frames
        raise RuntimeError("librealsense software aligner did not receive a synchronized depth/color frameset")

    def _prime_syncer(self) -> None:
        depth = np.zeros((self.depth_height, self.depth_width), dtype=np.uint16)
        color = np.zeros((self.color_height, self.color_width, 3), dtype=np.uint8)
        self._push_frames(depth, color)
        with contextlib.suppress(Exception):
            self.syncer.wait_for_frames(100)

    def align_depth_to_color(self, depth_m: np.ndarray, color_rgb: np.ndarray) -> np.ndarray:
        depth_z16 = self._depth_to_z16(depth_m)
        self._push_frames(depth_z16, color_rgb)
        frames = self._wait_for_frameset()
        aligned_frames = self.rs_align.process(frames)
        aligned_depth = aligned_frames.get_depth_frame()
        if not aligned_depth:
            raise RuntimeError("librealsense align did not produce an aligned depth frame")
        aligned_z16 = np.asanyarray(aligned_depth.get_data()).astype(np.float32, copy=False)
        return aligned_z16 * self.depth_units

    def align_color_to_depth(self, depth_m: np.ndarray, color_rgb: np.ndarray) -> np.ndarray:
        depth_z16 = self._depth_to_z16(depth_m)
        self._push_frames(depth_z16, color_rgb)
        frames = self._wait_for_frameset()
        aligned_frames = self.rs_align.process(frames)
        aligned_color = aligned_frames.get_color_frame()
        if not aligned_color:
            raise RuntimeError("librealsense align did not produce an aligned color frame")
        return np.ascontiguousarray(np.asanyarray(aligned_color.get_data())[..., :3])

    def align(self, depth_m: np.ndarray, color_rgb: np.ndarray) -> np.ndarray:
        if self.align_to == "depth":
            return self.align_color_to_depth(depth_m, color_rgb)
        return self.align_depth_to_color(depth_m, color_rgb)


def filter_depth_edges_torch(depth_m: torch.Tensor, *, threshold_m: float) -> torch.Tensor:
    """Remove depth pixels around strong local depth jumps with a Sobel filter."""
    threshold = float(threshold_m)
    depth = depth_m.to(dtype=torch.float32)
    valid = torch.isfinite(depth) & (depth > 0)
    depth_clean = torch.where(valid, depth, torch.zeros((), dtype=torch.float32, device=depth.device))
    if threshold <= 0.0 or depth_clean.numel() == 0:
        return depth_clean
    if depth_clean.ndim != 2:
        raise ValueError(f"depth edge filter expects a 2D depth map, got shape={tuple(depth_clean.shape)}")
    kernel_x = depth_clean.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
    ).reshape(1, 1, 3, 3)
    kernel_y = depth_clean.new_tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
    ).reshape(1, 1, 3, 3)
    depth_4d = depth_clean.reshape(1, 1, *depth_clean.shape)
    depth_padded = torch.nn.functional.pad(depth_4d, (1, 1, 1, 1), mode="replicate")
    grad_x = torch.nn.functional.conv2d(depth_padded, kernel_x).reshape_as(depth_clean).abs()
    grad_y = torch.nn.functional.conv2d(depth_padded, kernel_y).reshape_as(depth_clean).abs()
    edge = (grad_x > threshold) | (grad_y > threshold)
    return torch.where(edge, torch.zeros((), dtype=torch.float32, device=depth.device), depth_clean)


def filter_depth_edges_numpy(depth_m: np.ndarray, *, threshold_m: float) -> np.ndarray:
    """NumPy/OpenCV equivalent of filter_depth_edges_torch for tests or CPU paths."""
    threshold = float(threshold_m)
    depth = np.asarray(depth_m, dtype=np.float32).copy()
    depth[~np.isfinite(depth) | (depth <= 0)] = 0.0
    if threshold <= 0.0 or depth.size == 0:
        return depth
    grad_x = np.abs(cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3))
    grad_y = np.abs(cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3))
    depth[(grad_x > threshold) | (grad_y > threshold)] = 0.0
    return depth


def to_jsonable(value: object) -> object:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
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
        "depth_intrinsics": to_jsonable(payload.get("depth_intrinsics")),
        "pose_record": to_jsonable(payload.get("pose_record")),
        "color_pose_record": to_jsonable(payload.get("color_pose_record")),
        "depth_pose_record": to_jsonable(payload.get("depth_pose_record")),
        "pointcloud_frame": str(payload.get("pointcloud_frame", "color")),
        "emitter_enabled": to_jsonable(payload.get("emitter_enabled")),
        "depth_min": float(depth_min),
        "depth_max": float(depth_max),
    }
    if depth_source == "fast":
        debug_payload.update(
            {
                "stereo_rectification_mode": str(payload.get("stereo_rectification_mode", "opencv")),
                "fast_align_backend": str(payload.get("fast_align_backend", "torch")),
                "fast_alignment_direction": str(payload.get("fast_alignment_direction", "depth_to_color")),
                "ir_left_raw_file": "ir_left_raw.png",
                "ir_right_raw_file": "ir_right_raw.png",
                "ir_left_rect_file": "ir_left_rect.png",
                "ir_right_rect_file": "ir_right_rect.png",
                "rectified_k": to_jsonable(payload["rectified_k"]),
                "rectified_to_color": to_jsonable(payload["rectified_to_color"]),
                "depth_to_color_4x4": to_jsonable(payload.get("depth_to_color_4x4")),
                "baseline_m": float(payload["baseline_m"]),
                "left_ir_intrinsics": to_jsonable(payload.get("left_ir_intrinsics")),
                "right_ir_intrinsics": to_jsonable(payload.get("right_ir_intrinsics")),
                "left_to_right_4x4": to_jsonable(payload.get("left_to_right_4x4")),
            }
        )
    return debug_payload


def load_source_config_for_snapshot(config_path: Path | None) -> dict[str, object] | None:
    """加载原始配置文件内容，用于 live debug 输出留档。"""
    if config_path is None:
        return None
    resolved_path = resolve_repo_path(config_path)
    snapshot: dict[str, object] = {"path": str(resolved_path)}
    if not resolved_path.is_file():
        snapshot["error"] = "config file not found"
        return snapshot
    try:
        with resolved_path.open("r", encoding="utf-8") as handle:
            snapshot["config"] = yaml.safe_load(handle) or {}
    except Exception as exc:  # noqa: BLE001 - debug snapshot should not stop a run
        snapshot["error"] = f"{type(exc).__name__}: {exc}"
    return snapshot


def resolved_path_text(value: object | None) -> str | None:
    if value is None:
        return None
    return str(Path(value).expanduser().resolve())


def build_effective_live_config(args: argparse.Namespace, *, serials: list[str]) -> dict[str, object]:
    """Build a runnable YAML config from the final argparse namespace after CLI overrides."""
    segmenter = {
        "target_name": str(args.target_name),
        "prompt_task_info": resolved_path_text(args.prompt_task_info),
        "prompt_image_root": resolved_path_text(args.prompt_image_root),
        "checkpoint_path": resolved_path_text(args.checkpoint_path),
        "output_dir": resolved_path_text(args.output_dir),
        "overwrite_output": bool(args.overwrite_output),
        "confidence": float(args.confidence),
        "mask_threshold": float(args.mask_threshold),
        "prompt_keep_score_threshold": float(args.prompt_keep_score_threshold),
        "video_mask_prob_threshold": float(args.video_mask_prob_threshold),
        "depth_scale": 1.0,
        "depth_min": float(args.depth_min),
        "depth_max": float(args.depth_max),
        "stride": int(args.stride),
        "frame_voxel_size": float(args.frame_voxel_size),
        "target_cluster_filter_enabled": bool(args.target_cluster_filter_enabled),
        "target_cluster_radius_m": float(args.target_cluster_radius_m),
        "target_cluster_min_points": int(args.target_cluster_min_points),
        "target_cluster_keep_largest": bool(args.target_cluster_keep_largest),
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
        "target_3d_mask_erode_kernel": int(args.target_3d_mask_erode_kernel),
        "save_ply": bool(args.save_ply),
        "save_normal": bool(args.save_normal),
        "save_debug_2d": bool(args.save_debug_2d),
        "tracker_image_size": None if args.tracker_image_size is None else int(args.tracker_image_size),
    }
    realsense = {
        "camera_count": int(len(serials) if serials else args.camera_count),
        "camera_serials": ",".join(str(serial) for serial in serials) if serials else normalize_serials_value(args.camera_serials),
        "camera_poses_json": resolved_path_text(args.camera_poses_json),
        "max_frames": int(args.max_frames),
        "camera_warmup_frames": int(args.camera_warmup_frames),
        "wait_timeout_ms": int(args.wait_timeout_ms),
        "fps": int(args.fps),
        "color_width": int(args.color_width),
        "color_height": int(args.color_height),
        "stereo_width": int(args.stereo_width),
        "stereo_height": int(args.stereo_height),
        "stereo_alpha": float(args.stereo_alpha),
        "stereo_rectification_mode": normalize_stereo_rectification_mode(args.stereo_rectification_mode),
        "emitter_enabled": None if args.emitter_enabled is None else int(args.emitter_enabled),
        "depth_source": normalize_depth_source(args.depth_source),
        "low_bandwidth_mode": bool(args.low_bandwidth_mode),
        "save_live_debug": bool(args.save_live_debug),
        "compute_depth_valid_ratio": bool(args.compute_depth_valid_ratio),
    }
    fast_stereo = {
        "model_path": resolved_path_text(args.fast_model_path),
        "valid_iters": int(args.fast_valid_iters),
        "max_disp": int(args.fast_max_disp),
        "scale": float(args.fast_scale),
        "remove_invisible": bool(args.fast_remove_invisible),
        "depth_edge_filter_enabled": bool(args.fast_depth_edge_filter_enabled),
        "depth_edge_filter_threshold_m": float(args.fast_depth_edge_filter_threshold_m),
        "depth_edge_filter_stage": normalize_depth_edge_filter_stage(args.fast_depth_edge_filter_stage),
        "align_backend": normalize_fast_align_backend(args.fast_align_backend),
        "hiera": bool(args.fast_hiera),
        "optimize_build_volume": str(args.fast_optimize_build_volume),
    }
    return {
        "segmenter": segmenter,
        "realsense": realsense,
        "fast_stereo": fast_stereo,
    }


def write_live_debug_config_snapshot(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    serials: list[str],
    cameras: list["RealSenseRgbdCamera"],
    depth_source: str,
) -> None:
    """保存本次 live 运行的原始配置和实际生效参数，便于后续复现 debug。"""
    camera_records = [
        {
            "camera_id": camera.camera_id,
            "serial_number": camera.serial_number,
            "color_width": camera.color_width,
            "color_height": camera.color_height,
            "stereo_width": camera.stereo_width,
            "stereo_height": camera.stereo_height,
            "fps": camera.fps,
            "depth_source": camera.depth_source,
            "stereo_rectification_mode": camera.stereo_rectification_mode,
            "requested_emitter_enabled": camera.emitter_enabled,
            "applied_emitter_enabled": camera.applied_emitter_enabled,
            "color_intrinsics": to_jsonable(camera.color_intrinsics),
            "depth_intrinsics": to_jsonable(camera.depth_intrinsics),
            "left_ir_intrinsics": to_jsonable(camera.left_ir_intrinsics),
            "right_ir_intrinsics": to_jsonable(camera.right_ir_intrinsics),
            "left_to_right_4x4": to_jsonable(camera.left_to_right_4x4),
            "depth_to_color_4x4": to_jsonable(camera.depth_to_color_4x4),
            "pose_record": to_jsonable(camera.pose_record),
            "depth_pose_record": to_jsonable(camera.depth_pose_record),
        }
        for camera in cameras
    ]
    effective_config = build_effective_live_config(args, serials=serials)
    snapshot = {
        "source_config": load_source_config_for_snapshot(args.config),
        "effective_config": effective_config,
        "effective_config_file": "effective_config.yaml",
        "effective_args": to_jsonable(vars(args)),
        "resolved_camera_serials": [str(serial) for serial in serials],
        "depth_source": str(depth_source),
        "started_cameras": camera_records,
        "command_argv": list(sys.argv),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "effective_config.yaml").write_text(
        yaml.safe_dump(to_jsonable(effective_config), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (output_dir / "live_debug_config.yaml").write_text(
        yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


@dataclass(frozen=True)
class LiveCameraPose:
    camera_id: str
    serial_number: str
    cam2world_4x4: np.ndarray
    depth_cam2world_4x4: np.ndarray | None = None
    pose_record: dict[str, object] | None = None


def load_live_camera_pose_map(pose_path: Path | None) -> dict[str, LiveCameraPose]:
    if pose_path is None:
        return {}
    resolved_pose_path = Path(pose_path)
    if not resolved_pose_path.is_file():
        logging.warning(f"camera pose file not found, falling back to identity pose when allowed: {resolved_pose_path}")
        return {}
    payload = json.loads(resolved_pose_path.read_text(encoding="utf-8"))
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
        depth_payload = camera.get("depth_cam2world_4x4", camera.get("rectified_depth_cam2world_4x4"))
        depth_cam2world = None
        if depth_payload is not None:
            depth_cam2world_arr = np.asarray(depth_payload, dtype=np.float64)
            if depth_cam2world_arr.shape != (4, 4):
                raise ValueError(f"camera {camera_id} depth_cam2world_4x4 must be 4x4")
            depth_cam2world = depth_cam2world_arr
        pose = LiveCameraPose(
            camera_id=camera_id,
            serial_number=serial_number,
            cam2world_4x4=cam2world,
            depth_cam2world_4x4=depth_cam2world,
            pose_record=dict(camera),
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
        depth_cam2world_4x4=None,
        pose_record=None,
    )


def pose_record_from_cam2world(
    camera_id: str,
    cam2world_4x4: np.ndarray,
    *,
    coordinate_frame: str = "color",
) -> dict[str, object]:
    world2cam = np.linalg.inv(cam2world_4x4)
    return {
        "camera_id": camera_id,
        "coordinate_frame": str(coordinate_frame),
        "cam2world_4x4": cam2world_4x4.tolist(),
        "world2cam_4x4": world2cam.tolist(),
    }


def resolve_depth_pose_record_from_payload(
    payload: dict[str, object],
    *,
    coordinate_frame: str,
) -> dict[str, object]:
    depth_pose = payload.get("depth_pose_record")
    if isinstance(depth_pose, dict) and depth_pose.get("cam2world_4x4") is not None:
        return dict(depth_pose)
    color_pose = payload.get("pose_record")
    if isinstance(color_pose, dict) and color_pose.get("cam2world_4x4") is not None:
        depth_to_color = payload.get("depth_to_color_4x4", payload.get("rectified_to_color"))
        if depth_to_color is not None:
            depth_cam2world = depth_cam2world_from_color_pose(
                np.asarray(color_pose["cam2world_4x4"], dtype=np.float64),
                np.asarray(depth_to_color, dtype=np.float64),
            )
            record = pose_record_from_cam2world(
                str(payload.get("camera_id", color_pose.get("camera_id", "cam_00"))),
                depth_cam2world,
                coordinate_frame=coordinate_frame,
            )
            record.update(
                {
                    "depth_cam2world_4x4": depth_cam2world.tolist(),
                    "world2depth_4x4": np.linalg.inv(depth_cam2world).tolist(),
                    "color_cam2world_4x4": np.asarray(color_pose["cam2world_4x4"], dtype=np.float64).tolist(),
                    "depth_to_color_4x4": np.asarray(depth_to_color, dtype=np.float64).tolist(),
                }
            )
            return record
    return dict(color_pose) if isinstance(color_pose, dict) else {"camera_id": str(payload.get("camera_id", "cam_00"))}


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
        optimize_build_volume: str = "pytorch1",
    ) -> None:
        if optimize_build_volume not in {"pytorch1", "triton"}:
            raise ValueError("optimize_build_volume must be 'pytorch1' or 'triton'")
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
                "optimize_build_volume": optimize_build_volume,
            }
        )
        self.args = OmegaConf.create(cfg)
        self.model = torch.load(str(self.model_path), map_location="cpu", weights_only=False)
        self.model.args.valid_iters = int(valid_iters)
        self.model.args.max_disp = int(max_disp)
        self.model.requires_grad_(False)
        self.model.cuda().eval()

    def infer_depth(
        self,
        *,
        left_image: np.ndarray,
        right_image: np.ndarray,
        rectified_k: np.ndarray,
        baseline_m: float,
        return_torch: bool = False,
        include_input_images: bool = True,
    ) -> dict[str, object]:
        from Utils import AMP_DTYPE  # noqa: PLC0415
        from core.utils.utils import InputPadder  # noqa: PLC0415

        scale = float(self.args.scale)
        left_model = resize_model_image(left_image, scale=scale)
        right_model = resize_model_image(
            right_image,
            scale=scale,
            target_size=(int(left_model.shape[1]), int(left_model.shape[0])) if scale != 1.0 else None,
        )
        k_model = rectified_k.astype(np.float32, copy=True)
        k_model[:2] *= scale
        height, width = left_model.shape[:2]
        img0 = image_to_cuda_chw(left_model)
        img1 = image_to_cuda_chw(right_model)
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
            if not int(self.args.hiera):
                disp = self.model.forward(
                    img0,
                    img1,
                    iters=int(self.args.valid_iters),
                    test_mode=True,
                    optimize_build_volume=str(self.args.optimize_build_volume),
                )
            else:
                disp = self.model.run_hierachical(
                    img0,
                    img1,
                    iters=int(self.args.valid_iters),
                    test_mode=True,
                    small_ratio=0.5,
                )
        disp = padder.unpad(disp.float()).detach().reshape(height, width).clamp_min(0)
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
        if include_input_images:
            output.update(
                {
                    "left_rgb": ensure_three_channels(left_model),
                    "right_rgb": ensure_three_channels(right_model),
                }
            )
        if return_torch:
            output.update(
                {
                    "disparity": disp,
                    "depth_m": depth_m,
                }
            )
            return output
        disp_np = disp.detach().cpu().numpy()
        depth_np = depth_m.detach().cpu().numpy()
        output.update(
            {
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
        depth_cam2world_4x4: np.ndarray | None = None,
        source_pose_record: dict[str, object] | None = None,
        color_width: int,
        color_height: int,
        stereo_width: int,
        stereo_height: int,
        fps: int,
        alpha: float,
        wait_timeout_ms: int,
        depth_source: str,
        stereo_rectification_mode: str = "opencv",
        emitter_enabled: int | None = None,
    ) -> None:
        self.camera_id = str(camera_id)
        self.serial_number = str(serial_number)
        self.cam2world_4x4 = np.asarray(cam2world_4x4, dtype=np.float64)
        self.configured_depth_cam2world_4x4 = (
            None if depth_cam2world_4x4 is None else np.asarray(depth_cam2world_4x4, dtype=np.float64)
        )
        self.source_pose_record = dict(source_pose_record or {})
        self.color_width = int(color_width)
        self.color_height = int(color_height)
        self.stereo_width = int(stereo_width)
        self.stereo_height = int(stereo_height)
        self.fps = int(fps)
        self.alpha = float(alpha)
        self.wait_timeout_ms = int(wait_timeout_ms)
        self.depth_source = normalize_depth_source(depth_source)
        self.stereo_rectification_mode = normalize_stereo_rectification_mode(stereo_rectification_mode)
        self.emitter_enabled = None if emitter_enabled is None else bool(int(emitter_enabled))
        self.pipeline = rs.pipeline()
        self.profile: rs.pipeline_profile | None = None
        self.color_intrinsics: dict[str, float] | None = None
        self.left_ir_intrinsics: dict[str, float] | None = None
        self.right_ir_intrinsics: dict[str, float] | None = None
        self.left_to_right_4x4: np.ndarray | None = None
        self.depth_intrinsics: dict[str, float] | None = None
        self.depth_to_color_4x4: np.ndarray | None = None
        self.depth_cam2world_4x4: np.ndarray | None = self.configured_depth_cam2world_4x4
        self.color_map1: np.ndarray | None = None
        self.color_map2: np.ndarray | None = None
        self.rectification: dict[str, object] | None = None
        self.rectified_to_color: np.ndarray | None = None
        self.align_to_color: object | None = None
        self.depth_scale = 0.001
        self.applied_emitter_enabled: bool | None = None
        self.pose_record: dict[str, object] = pose_record_from_cam2world(
            self.camera_id,
            self.cam2world_4x4,
            coordinate_frame="color",
        )
        self.depth_pose_record: dict[str, object] | None = None

    def _set_depth_geometry(
        self,
        *,
        depth_intrinsics: dict[str, float],
        depth_to_color_4x4: np.ndarray,
        coordinate_frame: str,
    ) -> None:
        self.depth_intrinsics = dict(depth_intrinsics)
        self.depth_to_color_4x4 = np.asarray(depth_to_color_4x4, dtype=np.float64)
        computed_depth_pose = depth_cam2world_from_color_pose(
            self.cam2world_4x4,
            self.depth_to_color_4x4,
        )
        self.depth_cam2world_4x4 = (
            self.configured_depth_cam2world_4x4
            if self.configured_depth_cam2world_4x4 is not None
            else computed_depth_pose
        )
        self.depth_pose_record = pose_record_from_cam2world(
            self.camera_id,
            self.depth_cam2world_4x4,
            coordinate_frame=coordinate_frame,
        )
        self.depth_pose_record.update(
            {
                "depth_cam2world_4x4": self.depth_cam2world_4x4.tolist(),
                "world2depth_4x4": np.linalg.inv(self.depth_cam2world_4x4).tolist(),
                "color_cam2world_4x4": self.cam2world_4x4.tolist(),
                "depth_to_color_4x4": self.depth_to_color_4x4.tolist(),
            }
        )

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
        depth_sensor = self.profile.get_device().first_depth_sensor()
        if self.emitter_enabled is not None:
            if depth_sensor.supports(rs.option.emitter_enabled):
                depth_sensor.set_option(rs.option.emitter_enabled, float(int(self.emitter_enabled)))
                self.applied_emitter_enabled = bool(int(depth_sensor.get_option(rs.option.emitter_enabled)))
                logging.info(
                    f"Set RealSense emitter_enabled={int(self.applied_emitter_enabled)} "
                    f"for camera {self.camera_id}"
                )
            else:
                logging.warning(f"RealSense emitter_enabled option is not supported by camera {self.camera_id}")
        color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        color_intr = color_profile.get_intrinsics()
        self.color_intrinsics = intrinsics_to_payload(color_intr)
        if self.depth_source == "native":
            depth_profile = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
            depth_intr = depth_profile.get_intrinsics()
            depth_to_color = extrinsics_to_matrix(depth_profile.get_extrinsics_to(color_profile))
            self._set_depth_geometry(
                depth_intrinsics=intrinsics_to_payload(depth_intr),
                depth_to_color_4x4=depth_to_color,
                coordinate_frame="native_depth",
            )
            self.depth_scale = float(depth_sensor.get_depth_scale())
            self.align_to_color = rs.align(rs.stream.depth)
            return

        left_profile = self.profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        right_profile = self.profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
        left_intr = left_profile.get_intrinsics()
        right_intr = right_profile.get_intrinsics()
        left_to_right = extrinsics_to_matrix(left_profile.get_extrinsics_to(right_profile))
        left_to_color = extrinsics_to_matrix(left_profile.get_extrinsics_to(color_profile))
        self.left_ir_intrinsics = intrinsics_to_payload(left_intr)
        self.right_ir_intrinsics = intrinsics_to_payload(right_intr)
        self.left_to_right_4x4 = left_to_right
        if self.stereo_rectification_mode == "passthrough":
            self.rectification = build_passthrough_rectification(left_intr, left_to_right)
        else:
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
        rectified_to_left = np.asarray(self.rectification["rectified_to_left"], dtype=np.float64)
        self.rectified_to_color = left_to_color @ rectified_to_left
        rectified_k = np.asarray(self.rectification["rectified_k"], dtype=np.float32)
        self._set_depth_geometry(
            depth_intrinsics=intrinsics_payload_from_k(
                rectified_k,
                width=self.stereo_width,
                height=self.stereo_height,
            ),
            depth_to_color_4x4=self.rectified_to_color,
            coordinate_frame="rectified_depth",
        )

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
                "emitter_enabled": self.applied_emitter_enabled,
                "rgb": cv2.cvtColor(color_raw_bgr, cv2.COLOR_BGR2RGB),
                "depth_m": depth_m,
                "color_intrinsics": self.color_intrinsics,
                "depth_intrinsics": self.depth_intrinsics,
                "depth_to_color_4x4": self.depth_to_color_4x4,
                "color_pose_record": self.pose_record,
                "depth_pose_record": self.depth_pose_record,
                "pose_record": self.depth_pose_record or self.pose_record,
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
        if self.stereo_rectification_mode == "passthrough":
            left_rect = np.ascontiguousarray(left_raw)
            right_rect = np.ascontiguousarray(right_raw)
        else:
            left_rect = cv2.remap(
                left_raw,
                np.asarray(self.rectification["map1_l"]),
                np.asarray(self.rectification["map2_l"]),
                interpolation=cv2.INTER_LINEAR,
            )
            right_rect = cv2.remap(
                right_raw,
                np.asarray(self.rectification["map1_r"]),
                np.asarray(self.rectification["map2_r"]),
                interpolation=cv2.INTER_LINEAR,
            )
        return {
            "camera_id": self.camera_id,
            "serial_number": self.serial_number,
            "depth_source": self.depth_source,
            "emitter_enabled": self.applied_emitter_enabled,
            "rgb": cv2.cvtColor(color_undistorted_bgr, cv2.COLOR_BGR2RGB),
            "ir_left_raw": left_raw,
            "ir_right_raw": right_raw,
            "ir_left_rect": left_rect,
            "ir_right_rect": right_rect,
            "stereo_rectification_mode": self.stereo_rectification_mode,
            "rectified_k": self.rectification["rectified_k"],
            "rectified_to_color": self.rectified_to_color,
            "depth_intrinsics": self.depth_intrinsics,
            "depth_to_color_4x4": self.depth_to_color_4x4,
            "baseline_m": self.rectification["baseline_m"],
            "left_ir_intrinsics": self.left_ir_intrinsics,
            "right_ir_intrinsics": self.right_ir_intrinsics,
            "left_to_right_4x4": self.left_to_right_4x4,
            "color_intrinsics": self.color_intrinsics,
            "color_pose_record": self.pose_record,
            "depth_pose_record": self.depth_pose_record,
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
    rgb: np.ndarray | torch.Tensor,
    ir_left: np.ndarray | None,
    ir_right: np.ndarray | None,
    depth_aligned_m: np.ndarray | torch.Tensor,
    ir_left_raw: np.ndarray | None = None,
    ir_right_raw: np.ndarray | None = None,
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
    if torch.is_tensor(rgb):
        rgb_np = rgb.detach().cpu().numpy()
    else:
        rgb_np = rgb
    cv2.imwrite(str(frame_dir / "rgb.png"), rgb_np[..., ::-1])
    if ir_left_raw is not None:
        cv2.imwrite(str(frame_dir / "ir_left_raw.png"), ir_left_raw)
    if ir_right_raw is not None:
        cv2.imwrite(str(frame_dir / "ir_right_raw.png"), ir_right_raw)
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
        "stereo_rectification_mode": realsense_cfg.get("stereo_rectification_mode"),
        "emitter_enabled": realsense_cfg.get("emitter_enabled"),
        "depth_source": realsense_cfg.get("depth_source"),
        "low_bandwidth_mode": realsense_cfg.get("low_bandwidth_mode"),
        "max_frames": realsense_cfg.get("max_frames"),
        "save_live_debug": realsense_cfg.get("save_live_debug"),
        "compute_depth_valid_ratio": realsense_cfg.get("compute_depth_valid_ratio"),
    }
    for key, value in realsense_values.items():
        if value is None:
            continue
        if key == "camera_poses_json":
            defaults[key] = resolve_repo_path(value)
        elif key == "depth_source":
            defaults[key] = normalize_depth_source(value)
        elif key == "stereo_rectification_mode":
            defaults[key] = normalize_stereo_rectification_mode(value)
        elif key in {"low_bandwidth_mode", "save_live_debug", "emitter_enabled", "compute_depth_valid_ratio"}:
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
        "fast_depth_edge_filter_enabled": fast_cfg.get(
            "depth_edge_filter_enabled",
            fast_cfg.get("fast_depth_edge_filter_enabled"),
        ),
        "fast_depth_edge_filter_threshold_m": fast_cfg.get(
            "depth_edge_filter_threshold_m",
            fast_cfg.get("fast_depth_edge_filter_threshold_m"),
        ),
        "fast_depth_edge_filter_stage": fast_cfg.get(
            "depth_edge_filter_stage",
            fast_cfg.get("fast_depth_edge_filter_stage"),
        ),
        "fast_align_backend": fast_cfg.get(
            "align_backend",
            fast_cfg.get("fast_align_backend"),
        ),
        "fast_hiera": fast_cfg.get("hiera", fast_cfg.get("fast_hiera")),
        "fast_optimize_build_volume": fast_cfg.get(
            "optimize_build_volume",
            fast_cfg.get("fast_optimize_build_volume"),
        ),
    }
    for key, value in fast_values.items():
        if value is None:
            continue
        if key == "fast_model_path":
            defaults[key] = resolve_repo_path(value)
        elif key == "fast_depth_edge_filter_stage":
            defaults[key] = normalize_depth_edge_filter_stage(value)
        elif key == "fast_align_backend":
            defaults[key] = normalize_fast_align_backend(value)
        elif key in {"fast_remove_invisible", "fast_hiera", "fast_depth_edge_filter_enabled"}:
            defaults[key] = int(bool(value))
        else:
            defaults[key] = value

    icp_cfg = payload.get("icp", {})
    if isinstance(icp_cfg, dict):
        if "use_icp" in icp_cfg:
            defaults["use_icp"] = bool(icp_cfg["use_icp"])
        if "icp_obj_path" in icp_cfg:
            defaults["icp_obj_path"] = resolve_repo_path(icp_cfg["icp_obj_path"])
        if "goicp_sample_points" in icp_cfg:
            defaults["icp_goicp_sample_points"] = int(icp_cfg["goicp_sample_points"])
        if "open3d_sample_points" in icp_cfg:
            defaults["icp_open3d_sample_points"] = int(icp_cfg["open3d_sample_points"])
        if "o3d_max_correspondence_distance" in icp_cfg:
            defaults["icp_o3d_max_corr_dist"] = float(icp_cfg["o3d_max_correspondence_distance"])
        if "o3d_max_iterations" in icp_cfg:
            defaults["icp_o3d_max_iterations"] = int(icp_cfg["o3d_max_iterations"])
        if "o3d_relative_rmse" in icp_cfg:
            defaults["icp_o3d_relative_rmse"] = float(icp_cfg["o3d_relative_rmse"])
        if "view" in icp_cfg:
            defaults["view"] = bool(icp_cfg["view"])
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
    parser.add_argument(
        "--save-normal",
        "--save-normals",
        dest="save_normal",
        type=int,
        nargs="?",
        const=1,
        default=0,
        help="保存 PLY 时是否写入估计法线，1 开启 / 0 关闭",
    )
    parser.add_argument("--save-debug-2d", action="store_true", default=False)
    parser.add_argument("--max-frames", type=int, default=1)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--frame-voxel-size", type=float, default=0.003)
    parser.add_argument("--target-cluster-filter-enabled", type=int, default=1)
    parser.add_argument("--target-cluster-radius-m", type=float, default=0.025)
    parser.add_argument("--target-cluster-min-points", type=int, default=35)
    parser.add_argument("--target-cluster-keep-largest", type=int, default=1)
    parser.add_argument("--target-plane-filter-enabled", type=int, default=0)
    parser.add_argument("--target-plane-filter-distance-m", type=float, default=0.004)
    parser.add_argument("--target-plane-filter-min-points", type=int, default=80)
    parser.add_argument("--target-plane-filter-min-inlier-ratio", type=float, default=0.25)
    parser.add_argument("--target-plane-filter-max-inlier-ratio", type=float, default=0.85)
    parser.add_argument("--target-plane-filter-max-planes", type=int, default=1)
    parser.add_argument("--target-plane-filter-ransac-iterations", type=int, default=256)
    parser.add_argument("--target-depth-band-filter-enabled", type=int, default=1)
    parser.add_argument("--target-depth-band-filter-range-m", type=float, default=0.08)
    parser.add_argument("--target-depth-band-filter-min-valid-pixels", type=int, default=50)
    parser.add_argument("--target-depth-band-filter-min-keep-pixels", type=int, default=20)
    parser.add_argument("--target-3d-mask-erode-kernel", type=int, default=0)
    parser.add_argument("--depth-min", type=float, default=0.1)
    parser.add_argument("--depth-max", type=float, default=3.0)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--mask-threshold", type=float, default=0.6)
    parser.add_argument("--prompt-keep-score-threshold", type=float, default=0.2)
    parser.add_argument("--video-mask-prob-threshold", type=float, default=0.95)
    parser.add_argument("--tracker-image-size", type=int, default=896)
    parser.add_argument("--target-vis-color", type=int, nargs=3, default=None, metavar=("R", "G", "B"),
                        help="目标点云高亮颜色 R G B，例如 30 60 180 为深蓝")
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
    parser.add_argument("--stereo-rectification-mode", choices=STEREO_RECTIFICATION_CHOICES, default="opencv")
    parser.add_argument("--emitter-enabled", type=int, choices=(0, 1), default=None)
    parser.add_argument("--depth-source", choices=DEPTH_SOURCE_CHOICES, default="fast")
    parser.add_argument("--low-bandwidth-mode", type=int, default=1)
    parser.add_argument("--compute-depth-valid-ratio", type=int, default=1,
                        help="计算深度有效比例并打印日志（需要 CUDA 同步），0 关闭可提速")
    parser.add_argument("--fast-model-path", type=Path, default=FAST_STEREO_DEFAULT_MODEL)
    parser.add_argument("--fast-valid-iters", type=int, default=8)
    parser.add_argument("--fast-max-disp", type=int, default=192)
    parser.add_argument("--fast-scale", type=float, default=1.0)
    parser.add_argument("--fast-remove-invisible", type=int, default=1)
    parser.add_argument("--fast-depth-edge-filter-enabled", type=int, default=0)
    parser.add_argument("--fast-depth-edge-filter-threshold-m", type=float, default=0.5)
    parser.add_argument("--fast-depth-edge-filter-stage", choices=DEPTH_EDGE_FILTER_STAGE_CHOICES, default="rectified")
    parser.add_argument("--fast-align-backend", choices=FAST_ALIGN_BACKEND_CHOICES, default="torch")
    parser.add_argument("--fast-hiera", type=int, default=0)
    parser.add_argument("--fast-optimize-build-volume", choices=("pytorch1", "triton"), default="pytorch1")
    parser.add_argument("--save-live-debug", type=int, default=1)
    parser.add_argument("--sync-timing",
        type=int,
        default=0,
        help="同步 CUDA 后再计时，1 开启会让耗时归因更准确，但会降低吞吐",
    )
    parser.add_argument("--use-icp", action="store_true", default=False,
                        help="启用 ICP 配准，实时计算物体位姿")
    parser.add_argument("--icp-obj-path", type=Path, default=None,
                        help="ICP 配准参考物体 OBJ 文件路径")
    parser.add_argument("--icp-goicp-sample-points", type=int, default=5000,
                        help="Go-ICP 配准使用的参考点云采样点数（第一帧初始化时使用）")
    parser.add_argument("--icp-open3d-sample-points", type=int, default=3000,
                        help="Open3D ICP 配准使用的参考点云采样点数（后续帧跟踪时使用）")
    parser.add_argument("--icp-o3d-max-corr-dist", type=float, default=0.02,
                        help="Open3D ICP 最大对应距离（米）")
    parser.add_argument("--icp-o3d-max-iterations", type=int, default=50,
                        help="Open3D ICP 最大迭代次数")
    parser.add_argument("--icp-o3d-relative-rmse", type=float, default=1e-4,
                        help="Open3D ICP 相对 RMSE 收敛阈值")
    parser.add_argument("--view", action="store_true", default=False,
                        help="启用 Open3D 实时点云可视化（需要 DISPLAY 环境变量）")
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
    fast_depth_edge_filter_enabled: bool = False,
    fast_depth_edge_filter_threshold_m: float = 0.5,
    fast_depth_edge_filter_stage: str = "rectified",
    output_dir: Path,
    frame_index: int,
    write_debug_images: bool,
    sync_timing: bool = False,
    timing: dict[str, object] | None = None,
    fast_align_backend: str = "torch",
    fast_aligners: dict[str, LibrealsenseSoftwareAligner] | None = None,
    compute_depth_valid_ratio: bool = True,
) -> dict[str, dict[str, object]]:
    total_t0 = time.perf_counter()
    camera_inputs: dict[str, dict[str, object]] = {}
    per_camera_timing: list[dict[str, object]] = []
    stereo_infer_total = 0.0
    depth_align_total = 0.0
    depth_to_cpu_total = 0.0
    open3d_align_total = 0.0
    librealsense_align_total = 0.0
    depth_filter_total = 0.0
    debug_write_total = 0.0
    edge_filter_stage = normalize_depth_edge_filter_stage(fast_depth_edge_filter_stage)
    align_backend = normalize_fast_align_backend(fast_align_backend)
    for payload in captured_frames:
        camera_t0 = time.perf_counter()
        camera_id = str(payload["camera_id"])
        depth_source = normalize_depth_source(payload.get("depth_source", "fast"))
        logging.info(f"Building RGBD for {camera_id} using depth_source={depth_source}")
        rgb = np.asarray(payload["rgb"], dtype=np.uint8)
        rgb_for_points = rgb
        intrinsics_for_points = dict(payload.get("color_intrinsics", {}))
        pose_record_for_points = dict(payload.get("pose_record", {"camera_id": camera_id}))
        pointcloud_frame = "color"
        fast_alignment_direction = "none"
        ir_left_rect: np.ndarray | None = None
        ir_right_rect: np.ndarray | None = None
        edge_filter_summary: dict[str, object] | None = None
        stereo_intrinsics: dict[str, float] | None = None
        stereo_time_sec = 0.0
        rectified_edge_filter_time_sec = 0.0
        depth_align_time_sec = 0.0
        depth_to_cpu_time_sec = 0.0
        open3d_align_time_sec = 0.0
        librealsense_align_time_sec = 0.0
        depth_range_filter_time_sec = 0.0
        aligned_edge_filter_time_sec = 0.0
        depth_valid_ratio_time_sec = 0.0
        live_debug_write_time_sec = 0.0
        if depth_source == "native":
            native_depth_t0 = time.perf_counter()
            depth_aligned_m = np.asarray(payload["depth_m"], dtype=np.float32)
            intrinsics_for_points = dict(payload.get("depth_intrinsics") or payload.get("color_intrinsics", {}))
            pose_record_for_points = dict(payload.get("depth_pose_record") or payload.get("pose_record", {}))
            pointcloud_frame = "native_depth"
            depth_align_time_sec = time.perf_counter() - native_depth_t0
        else:
            if stereo_runner is None:
                raise RuntimeError("depth_source='fast' requires a Fast-FoundationStereo runner")
            ir_left_rect = np.asarray(payload["ir_left_rect"], dtype=np.uint8)
            ir_right_rect = np.asarray(payload["ir_right_rect"], dtype=np.uint8)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            stereo_t0 = time.perf_counter()
            stereo_output = stereo_runner.infer_depth(
                left_image=ir_left_rect,
                right_image=ir_right_rect,
                rectified_k=np.asarray(payload["rectified_k"], dtype=np.float32),
                baseline_m=float(payload["baseline_m"]),
                return_torch=True,
                include_input_images=False,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            stereo_time_sec = time.perf_counter() - stereo_t0
            rectified_depth_m = stereo_output["depth_m"]
            stereo_intrinsics = dict(stereo_output["rectified_intrinsics"])
            if bool(fast_depth_edge_filter_enabled) and edge_filter_stage == "rectified":
                if not torch.is_tensor(rectified_depth_m):
                    raise RuntimeError("Fast rectified depth edge filtering requires torch depth output")
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                edge_t0 = time.perf_counter()
                valid_before = int(torch.count_nonzero(torch.isfinite(rectified_depth_m) & (rectified_depth_m > 0)).item())
                rectified_depth_m = filter_depth_edges_torch(
                    rectified_depth_m,
                    threshold_m=float(fast_depth_edge_filter_threshold_m),
                )
                valid_after = int(torch.count_nonzero(rectified_depth_m > 0).item())
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                rectified_edge_filter_time_sec = time.perf_counter() - edge_t0
                edge_filter_summary = {
                    "enabled": True,
                    "backend": "torch",
                    "stage": edge_filter_stage,
                    "threshold_m": float(fast_depth_edge_filter_threshold_m),
                    "valid_pixels_before": valid_before,
                    "valid_pixels_after": valid_after,
                    "removed_pixels": int(max(valid_before - valid_after, 0)),
                }
            if align_backend == "open3d":
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                align_t0 = time.perf_counter()
                depth_aligned_m = align_rectified_depth_to_color_open3d(
                    rectified_depth_m,
                    rectified_intrinsics=stereo_intrinsics,
                    rectified_to_color=np.asarray(payload["rectified_to_color"], dtype=np.float64),
                    color_intrinsics=dict(payload["color_intrinsics"]),
                    color_shape=rgb.shape[:2],
                    depth_max_m=float(depth_max),
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                open3d_align_time_sec = time.perf_counter() - align_t0
                depth_align_time_sec = open3d_align_time_sec
                fast_alignment_direction = "depth_to_color"
            else:
                depth_aligned_m = rectified_depth_m
                intrinsics_for_points = dict(stereo_intrinsics)
                pose_record_for_points = resolve_depth_pose_record_from_payload(
                    payload,
                    coordinate_frame="rectified_depth",
                )
                pointcloud_frame = "rectified_depth"
                fast_alignment_direction = "color_to_depth"
        if torch.is_tensor(depth_aligned_m):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            range_t0 = time.perf_counter()
            depth_aligned_m = depth_aligned_m.to(dtype=torch.float32)
            depth_aligned_m = torch.where(
                torch.isfinite(depth_aligned_m)
                & (depth_aligned_m >= float(depth_min))
                & (depth_aligned_m <= float(depth_max)),
                depth_aligned_m,
                torch.zeros((), dtype=torch.float32, device=depth_aligned_m.device),
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            depth_range_filter_time_sec = time.perf_counter() - range_t0
            if depth_source == "fast" and bool(fast_depth_edge_filter_enabled) and edge_filter_stage == "aligned":
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                aligned_edge_t0 = time.perf_counter()
                valid_before = int(torch.count_nonzero(depth_aligned_m > 0).item())
                depth_aligned_m = filter_depth_edges_torch(
                    depth_aligned_m,
                    threshold_m=float(fast_depth_edge_filter_threshold_m),
                )
                valid_after = int(torch.count_nonzero(depth_aligned_m > 0).item())
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                aligned_edge_filter_time_sec = time.perf_counter() - aligned_edge_t0
                edge_filter_summary = {
                    "enabled": True,
                    "backend": "torch",
                    "stage": edge_filter_stage,
                    "threshold_m": float(fast_depth_edge_filter_threshold_m),
                    "valid_pixels_before": valid_before,
                    "valid_pixels_after": valid_after,
                    "removed_pixels": int(max(valid_before - valid_after, 0)),
                }
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            valid_t0 = time.perf_counter()
            if compute_depth_valid_ratio:
                depth_valid_ratio = float((depth_aligned_m > 0).float().mean().item())
            else:
                depth_valid_ratio = 0.0
            depth_valid_ratio_time_sec = time.perf_counter() - valid_t0
        else:
            range_t0 = time.perf_counter()
            depth_aligned_m = np.asarray(depth_aligned_m, dtype=np.float32).copy()
            depth_aligned_m[~np.isfinite(depth_aligned_m)] = 0.0
            depth_aligned_m[(depth_aligned_m < float(depth_min)) | (depth_aligned_m > float(depth_max))] = 0.0
            depth_range_filter_time_sec = time.perf_counter() - range_t0
            if depth_source == "fast" and bool(fast_depth_edge_filter_enabled) and edge_filter_stage == "aligned":
                aligned_edge_t0 = time.perf_counter()
                valid_before = int(np.count_nonzero(depth_aligned_m > 0))
                depth_aligned_m = filter_depth_edges_numpy(
                    depth_aligned_m,
                    threshold_m=float(fast_depth_edge_filter_threshold_m),
                )
                valid_after = int(np.count_nonzero(depth_aligned_m > 0))
                aligned_edge_filter_time_sec = time.perf_counter() - aligned_edge_t0
                edge_filter_summary = {
                    "enabled": True,
                    "backend": "opencv",
                    "stage": edge_filter_stage,
                    "threshold_m": float(fast_depth_edge_filter_threshold_m),
                    "valid_pixels_before": valid_before,
                    "valid_pixels_after": valid_after,
                    "removed_pixels": int(max(valid_before - valid_after, 0)),
                }
            valid_t0 = time.perf_counter()
            if compute_depth_valid_ratio:
                depth_valid_ratio = float((depth_aligned_m > 0).mean())
            else:
                depth_valid_ratio = 0.0
            depth_valid_ratio_time_sec = time.perf_counter() - valid_t0

        if depth_source == "fast" and align_backend != "open3d":
            if stereo_intrinsics is None:
                raise RuntimeError("missing FastStereo rectified intrinsics")
            if align_backend == "librealsense":
                if fast_aligners is None:
                    raise RuntimeError("fast_align_backend='librealsense' requires a fast_aligners cache")
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                align_total_t0 = time.perf_counter()
                cpu_t0 = time.perf_counter()
                if torch.is_tensor(depth_aligned_m):
                    rectified_depth_np = depth_aligned_m.detach().cpu().numpy().astype(np.float32, copy=False)
                else:
                    rectified_depth_np = np.asarray(depth_aligned_m, dtype=np.float32)
                depth_to_cpu_time_sec = time.perf_counter() - cpu_t0
                aligner = fast_aligners.get(camera_id)
                if aligner is None:
                    aligner = LibrealsenseSoftwareAligner(
                        rectified_intrinsics=stereo_intrinsics,
                        rectified_to_color=np.asarray(payload["rectified_to_color"], dtype=np.float64),
                        color_intrinsics=dict(payload["color_intrinsics"]),
                        depth_shape=rectified_depth_np.shape,
                        color_shape=rgb.shape[:2],
                        align_to="depth",
                    )
                    fast_aligners[camera_id] = aligner
                librealsense_t0 = time.perf_counter()
                rgb_for_points = aligner.align_color_to_depth(rectified_depth_np, rgb)
                librealsense_align_time_sec = time.perf_counter() - librealsense_t0
                depth_align_time_sec = time.perf_counter() - align_total_t0
                depth_aligned_m = rectified_depth_np
            else:
                if not torch.is_tensor(depth_aligned_m):
                    depth_for_color = torch.as_tensor(np.ascontiguousarray(depth_aligned_m), dtype=torch.float32)
                else:
                    depth_for_color = depth_aligned_m
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                align_t0 = time.perf_counter()
                rgb_depth_t = align_color_to_rectified_depth_torch(
                    rgb,
                    depth_for_color,
                    rectified_intrinsics=stereo_intrinsics,
                    rectified_to_color=np.asarray(payload["rectified_to_color"], dtype=np.float64),
                    color_intrinsics=dict(payload["color_intrinsics"]),
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                depth_align_time_sec = time.perf_counter() - align_t0
                rgb_for_points = rgb_depth_t

        camera_inputs[camera_id] = {
            "rgb": rgb_for_points,
            "depth_m": depth_aligned_m,
            "intrinsics": intrinsics_for_points,
            "pose_record": pose_record_for_points,
            "fovy_deg": None,
        }
        if depth_source == "fast":
            camera_inputs[camera_id]["stereo_time_sec"] = stereo_time_sec
            camera_inputs[camera_id]["fast_align_backend"] = align_backend
            camera_inputs[camera_id]["fast_alignment_direction"] = fast_alignment_direction
            camera_inputs[camera_id]["pointcloud_frame"] = pointcloud_frame
        if edge_filter_summary is not None:
            camera_inputs[camera_id]["fast_depth_edge_filter"] = edge_filter_summary
        if write_debug_images:
            debug_t0 = time.perf_counter()
            camera_payload = build_live_debug_camera_payload(
                payload={
                    **payload,
                    "depth_intrinsics": intrinsics_for_points,
                    "pose_record": pose_record_for_points,
                    "pointcloud_frame": pointcloud_frame,
                    "fast_align_backend": align_backend,
                    "fast_alignment_direction": fast_alignment_direction,
                },
                depth_source=depth_source,
                depth_min=float(depth_min),
                depth_max=float(depth_max),
            )
            if edge_filter_summary is not None:
                camera_payload["fast_depth_edge_filter"] = edge_filter_summary
            write_live_debug(
                output_dir=output_dir,
                frame_index=frame_index,
                camera_id=camera_id,
                depth_source=depth_source,
                rgb=rgb_for_points,
                ir_left=ir_left_rect,
                ir_right=ir_right_rect,
                depth_aligned_m=depth_aligned_m,
                ir_left_raw=payload.get("ir_left_raw"),
                ir_right_raw=payload.get("ir_right_raw"),
                camera_payload=camera_payload,
            )
            live_debug_write_time_sec = time.perf_counter() - debug_t0
        camera_timing = {
            "camera_id": camera_id,
            "depth_source": depth_source,
            "total_time_sec": time.perf_counter() - camera_t0,
            "depth_align_backend": align_backend if depth_source == "fast" else "native",
            "alignment_direction": fast_alignment_direction if depth_source == "fast" else "color_to_depth",
            "pointcloud_frame": pointcloud_frame,
            "stereo_infer_time_sec": stereo_time_sec,
            "rectified_edge_filter_time_sec": rectified_edge_filter_time_sec,
            "depth_align_time_sec": depth_align_time_sec,
            "depth_to_cpu_time_sec": depth_to_cpu_time_sec,
            "open3d_align_time_sec": open3d_align_time_sec,
            "librealsense_align_time_sec": librealsense_align_time_sec,
            "depth_range_filter_time_sec": depth_range_filter_time_sec,
            "aligned_edge_filter_time_sec": aligned_edge_filter_time_sec,
            "depth_valid_ratio_time_sec": depth_valid_ratio_time_sec,
            "live_debug_write_time_sec": live_debug_write_time_sec,
            "depth_valid_ratio": depth_valid_ratio,
        }
        camera_inputs[camera_id]["rgbd_build_timing_sec"] = camera_timing
        per_camera_timing.append(camera_timing)
        stereo_infer_total += stereo_time_sec
        depth_align_total += depth_align_time_sec
        depth_to_cpu_total += depth_to_cpu_time_sec
        open3d_align_total += open3d_align_time_sec
        librealsense_align_total += librealsense_align_time_sec
        depth_filter_total += (
            rectified_edge_filter_time_sec
            + depth_range_filter_time_sec
            + aligned_edge_filter_time_sec
            + depth_valid_ratio_time_sec
        )
        debug_write_total += live_debug_write_time_sec
    if timing is not None:
        timing.clear()
        timing.update(
            {
                "total_time_sec": time.perf_counter() - total_t0,
                "sync_timing": bool(sync_timing),
                "camera_count": len(per_camera_timing),
                "fast_align_backend": align_backend,
                "fast_alignment_direction": (
                    "color_to_depth" if align_backend != "open3d" else "depth_to_color"
                ),
                "stereo_infer_time_sec": stereo_infer_total,
                "depth_align_time_sec": depth_align_total,
                "depth_to_cpu_time_sec": depth_to_cpu_total,
                "open3d_align_time_sec": open3d_align_total,
                "librealsense_align_time_sec": librealsense_align_total,
                "depth_filter_time_sec": depth_filter_total,
                "live_debug_write_time_sec": debug_write_total,
                "per_camera": per_camera_timing,
            }
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
    fast_aligners: dict[str, LibrealsenseSoftwareAligner] = {}
    viewer: _PointCloudViewer | None = None
    icp_log_file = None
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
                depth_cam2world_4x4=pose.depth_cam2world_4x4,
                source_pose_record=pose.pose_record,
                color_width=int(args.color_width),
                color_height=int(args.color_height),
                stereo_width=int(args.stereo_width),
                stereo_height=int(args.stereo_height),
                fps=int(args.fps),
                alpha=float(args.stereo_alpha),
                wait_timeout_ms=int(args.wait_timeout_ms),
                depth_source=depth_source,
                stereo_rectification_mode=str(args.stereo_rectification_mode),
                emitter_enabled=args.emitter_enabled,
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
                optimize_build_volume=str(args.fast_optimize_build_volume),
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
            target_vis_color=args.target_vis_color,
        ) as segmenter:
            segmenter.sync_timing = bool(args.sync_timing)
            logging.info("SingleObjectPointCloudSegmenter loaded")
            if bool(args.save_live_debug):
                write_live_debug_config_snapshot(
                    output_dir=segmenter.output_dir,
                    args=args,
                    serials=serials,
                    cameras=cameras,
                    depth_source=depth_source,
                )
                logging.info(f"Saved live debug config snapshot to {segmenter.output_dir / 'live_debug_config.yaml'}")
            frame_limit = None if int(args.max_frames) <= 0 else int(args.max_frames)
            frame_index = 0

            use_view = bool(args.view)
            if use_view:
                viewer = _PointCloudViewer()

            use_icp = bool(args.use_icp)
            icp_goicp_reference_points = None
            icp_open3d_reference_points = None
            icp_solver_ctx = None
            if use_icp:
                icp_obj_path = args.icp_obj_path
                if icp_obj_path is None:
                    icp_obj_path = resolve_repo_path("assets/icp_assets/book.obj")
                else:
                    icp_obj_path = Path(icp_obj_path)
                if not icp_obj_path.exists():
                    logging.warning(f"ICP 参考 OBJ 文件不存在: {icp_obj_path}，将跳过 ICP 配准")
                    use_icp = False
                else:
                    goicp_sample_points = int(args.icp_goicp_sample_points)
                    open3d_sample_points = int(args.icp_open3d_sample_points)
                    logging.info(f"Loading ICP reference OBJ: {icp_obj_path.resolve()}")
                    if goicp_sample_points == open3d_sample_points:
                        shared_points = _sample_obj_surface_points(
                            icp_obj_path.resolve(),
                            sample_points=goicp_sample_points,
                        )
                        icp_goicp_reference_points = shared_points
                        icp_open3d_reference_points = shared_points
                        logging.info(f"Sampled {shared_points.shape[0]} shared reference points for both Go-ICP and Open3D")
                    else:
                        icp_goicp_reference_points = _sample_obj_surface_points(
                            icp_obj_path.resolve(),
                            sample_points=goicp_sample_points,
                        )
                        icp_open3d_reference_points = _sample_obj_surface_points(
                            icp_obj_path.resolve(),
                            sample_points=open3d_sample_points,
                        )
                        logging.info(f"Sampled {icp_goicp_reference_points.shape[0]} points for Go-ICP")
                        logging.info(f"Sampled {icp_open3d_reference_points.shape[0]} points for Open3D")

                    icp_reference_points = icp_goicp_reference_points
                    if icp_reference_points.shape[0] < 4:
                        logging.warning("ICP 参考 OBJ 顶点数 < 4，将跳过 ICP 配准")
                        use_icp = False
                        icp_goicp_reference_points = None
                        icp_open3d_reference_points = None
                        icp_reference_points = None
                    else:
                        icp_initialized = False
                        T_cam_to_book = np.eye(4, dtype=np.float64)
                        logging.info(f"Loaded ICP reference: {icp_reference_points.shape[0]} vertices")

            icp_log_file = None
            if use_icp:
                icp_log_path = segmenter.output_dir / "icp_pose_log.csv"
                icp_log_file = open(icp_log_path, "w")
                icp_log_file.write("frame,type,fitness,rmse,time_sec,t_x,t_y,t_z,qw,qx,qy,qz\n")
                icp_log_file.flush()

            while frame_limit is None or frame_index < frame_limit:
                loop_t0 = time.perf_counter()
                logging.info(f"Capturing live frame {frame_index:05d}")
                capture_t0 = time.perf_counter()
                captured_frames = []
                capture_per_camera: list[dict[str, object]] = []
                for camera in cameras:
                    camera_capture_t0 = time.perf_counter()
                    captured = camera.capture()
                    captured_frames.append(captured)
                    capture_per_camera.append(
                        {
                            "camera_id": str(captured.get("camera_id", camera.camera_id)),
                            "capture_time_sec": time.perf_counter() - camera_capture_t0,
                        }
                    )
                capture_time = time.perf_counter() - capture_t0
                rgbd_build_timing: dict[str, object] = {}
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                build_t0 = time.perf_counter()
                camera_inputs = build_camera_inputs_from_live_frames(
                    captured_frames=captured_frames,
                    stereo_runner=stereo_runner,
                    depth_min=float(args.depth_min),
                    depth_max=float(args.depth_max),
                    fast_depth_edge_filter_enabled=bool(args.fast_depth_edge_filter_enabled),
                    fast_depth_edge_filter_threshold_m=float(args.fast_depth_edge_filter_threshold_m),
                    fast_depth_edge_filter_stage=str(args.fast_depth_edge_filter_stage),
                    output_dir=Path(args.output_dir).resolve(),
                    frame_index=frame_index,
                    write_debug_images=bool(args.save_live_debug),
                    sync_timing=bool(args.sync_timing),
                    timing=rgbd_build_timing,
                    fast_align_backend=str(args.fast_align_backend),
                    fast_aligners=fast_aligners,
                    compute_depth_valid_ratio=bool(args.compute_depth_valid_ratio),
                )
                build_camera_inputs_time = time.perf_counter() - build_t0
                frame_name = f"frame_{frame_index:05d}.png"
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                process_t0 = time.perf_counter()

                result = segmenter.process_frame(
                    frame_name=frame_name,
                    camera_inputs=camera_inputs,
                    live_debug_root=(
                        segmenter.output_dir / "live_rgbd_debug" if bool(args.save_live_debug) else None
                    ),
                    view_root=Path("/tmp") if use_view else None,
                )

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                process_wall_time = time.perf_counter() - process_t0

                view_time_sec = 0.0
                if use_view and viewer is not None:
                    view_t0 = time.perf_counter()
                    points_xyz_np = result["points_xyz"].cpu().numpy().astype(np.float32)
                    vis_colors_np: np.ndarray | None = result["vis_colors"]
                    if vis_colors_np is None:
                        raw_colors_np = result["raw_colors"].cpu().numpy().astype(np.uint8)
                        labels_np = result["instance_labels"].cpu().numpy()
                        vis_colors_np = raw_colors_np.copy()
                        target_mask_v = labels_np > 0
                        if target_mask_v.any():
                            vis_colors_np[target_mask_v] = np.array(
                                [0, 255, 0], dtype=np.uint8
                            )
                    viewer.update(points_xyz_np, vis_colors_np)
                    view_time_sec = time.perf_counter() - view_t0

                icp_time_sec = 0.0
                icp_result = None
                icp_type = "Open3D ICP"

                if use_icp and icp_goicp_reference_points is not None and icp_open3d_reference_points is not None:
                    points_xyz_t = result["points_xyz"]
                    labels_t = result["instance_labels"]
                    target_mask = labels_t > 0
                    target_count = int(torch.sum(target_mask).item())
                    if torch.any(target_mask):
                        icp_t0 = time.perf_counter()
                        target_points = points_xyz_t[target_mask].cpu().numpy().astype(np.float64)

                        if not icp_initialized:
                            logging.info(f"[Frame {frame_index}] Using Go-ICP for initial registration")
                            from icp.goicp import GoICPConfig, register_point_clouds
                            config = GoICPConfig()
                            config.goicp_mse_thresh = 0.01
                            config.goicp_epsilon = 0.001
                            config.goicp_quiet = True
                            try:
                                goicp_result, _ = register_point_clouds(
                                    moving_points=target_points,
                                    reference_points=icp_goicp_reference_points,
                                    config=config,
                                )
                                icp_result = goicp_result
                                icp_initialized = True
                                icp_type = "Go-ICP"
                            except Exception as e:
                                logging.warning(f"Go-ICP failed: {e}, falling back to Open3D ICP")
                                icp_initialized = True
                            icp_time_sec = time.perf_counter() - icp_t0
                        else:
                            logging.info(f"[Frame {frame_index}] Using Open3D ICP for tracking")
                            icp_type = "Open3D ICP"
                            import open3d as o3d

                            source_pcd = o3d.geometry.PointCloud()
                            source_pcd.points = o3d.utility.Vector3dVector(target_points)
                            reference_pcd = o3d.geometry.PointCloud()
                            reference_pcd.points = o3d.utility.Vector3dVector(icp_open3d_reference_points)

                            o3d_icp_result = o3d.pipelines.registration.registration_icp(
                                source=source_pcd,
                                target=reference_pcd,
                                max_correspondence_distance=float(args.icp_o3d_max_corr_dist),
                                init=T_cam_to_book,
                                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                                    max_iteration=int(args.icp_o3d_max_iterations),
                                    relative_rmse=float(args.icp_o3d_relative_rmse),
                                ),
                            )

                            icp_time_sec = time.perf_counter() - icp_t0

                            class FakeICPResult:
                                def __init__(self, data):
                                    self.transformation = data.transformation
                                    self.fitness = data.fitness
                                    self.inlier_rmse = data.inlier_rmse
                            icp_result = FakeICPResult(o3d_icp_result)


                loop_runtime = time.perf_counter() - loop_t0
                fps = 1.0 / loop_runtime if loop_runtime > 0 else float("inf")

                segmenter.update_frame_metadata({
                    "total_frame_time_sec": loop_runtime,
                    "capture_time_sec": capture_time,
                    "capture_per_camera": capture_per_camera,
                    "build_camera_inputs_time_sec": build_camera_inputs_time,
                    "rgbd_build_timing_sec": rgbd_build_timing,
                    "process_frame_time_sec": process_wall_time,
                    "view_time_sec": view_time_sec,
                    "icp_time_sec": icp_time_sec,
                    "other_time_sec": max(0.0, loop_runtime - capture_time - build_camera_inputs_time - process_wall_time - view_time_sec - icp_time_sec),
                    "loop_runtime_sec": loop_runtime,
                })

                if icp_result is not None and icp_result is not False:
                    T_cam_to_book = icp_result.transformation
                    T_book_in_world = _invert_transform(T_cam_to_book)
                    R_book = T_book_in_world[:3, :3]
                    t_book = T_book_in_world[:3, 3]

                    print(f"\n=== {icp_type} Pose [frame {result['frame_index']:03d}] ===")
                    print(f"  Time: {icp_time_sec:.4f}s  FPS: {fps:.1f}")
                    print(f"  Fitness: {icp_result.fitness:.6f}  RMSE: {icp_result.inlier_rmse:.6f}")
                    print(f"\n  Book pose in world:")
                    print(_format_icp_pose(R_book, t_book))
                    print(f"\n========================================\n")
                    if icp_log_file is not None:
                        qw, qx, qy, qz = _rotation_matrix_to_quaternion(R_book)
                        icp_log_file.write(
                            f"{result['frame_index']},{icp_type},{icp_result.fitness:.6f},"
                            f"{icp_result.inlier_rmse:.6f},{icp_time_sec:.6f},"
                            f"{t_book[0]:.6f},{t_book[1]:.6f},{t_book[2]:.6f},"
                            f"{qw:.6f},{qx:.6f},{qy:.6f},{qz:.6f}\n"
                        )
                        icp_log_file.flush()
                else:
                    print(f"\rFPS: {fps:.1f}  frame={result['frame_index']:03d}  loop={loop_runtime:.3f}s", end="")

                logging.info(
                    f"[frame {result['frame_index']:03d}] {frame_name} "
                    f"points={result['points_xyz'].shape[0]} cameras={len(camera_inputs)} "
                    f"loop={loop_runtime:.3f}s fps={fps:.1f} "
                    f"capture={capture_time:.3f}s rgbd={build_camera_inputs_time:.3f}s "
                    f"process={process_wall_time:.3f}s icp={icp_time_sec:.3f}s"
                )
                frame_index += 1
    finally:
        if icp_log_file is not None:
            icp_log_file.close()
        if viewer is not None:
            viewer.close()
        for aligner in fast_aligners.values():
            aligner.close()
        for camera in cameras:
            camera.stop()


def main() -> None:
    run_live(parse_args())


if __name__ == "__main__":
    main()
