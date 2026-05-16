#!/usr/bin/env python3
"""用 AprilTag 标定多台 RealSense 相机外参，输出 single_seg 可用的 camera_poses_json。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = REPO_ROOT / "configs" / "camera_poses_apriltag.json"
FURNITURE_BASE_TAG_SIZE_M = 0.048
FURNITURE_BASE_TAG_POSES: dict[int, np.ndarray] = {
    0: np.array([[-0.03, -0.03, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64),
    1: np.array([[0.03, -0.03, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64),
    2: np.array([[-0.03, 0.03, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64),
    3: np.array([[0.03, 0.03, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64),
}
CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


@dataclass(frozen=True)
class TagLayout:
    tag_size_m: float
    world_t_tag_by_id: dict[int, np.ndarray]
    name: str = "custom"


@dataclass
class CameraCalibration:
    camera_id: str
    serial_number: str
    intrinsics: dict[str, float]
    world_t_camera_cv: np.ndarray
    world_t_camera_single_seg: np.ndarray
    num_frames_used: int
    num_tag_observations: int
    tag_counts: dict[int, int]
    mean_pose_error: float | None
    mean_decision_margin: float | None
    depth_intrinsics: dict[str, float] | None = None
    depth_to_color_4x4: np.ndarray | None = None
    world_t_depth_cv: np.ndarray | None = None
    world_t_depth_single_seg: np.ndarray | None = None
    depth_frame: str | None = None
    baseline_m: float | None = None


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def rpy_to_matrix(rpy_rad: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(value) for value in np.asarray(rpy_rad, dtype=np.float64)]
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def make_transform(position: np.ndarray, rotation: np.ndarray | None = None) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    if rotation is not None:
        transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return transform


def transform_from_layout_record(record: Any) -> np.ndarray:
    if isinstance(record, dict):
        for matrix_key in ("world_T_tag", "world_t_tag", "transform", "matrix"):
            if matrix_key in record:
                matrix = np.asarray(record[matrix_key], dtype=np.float64)
                if matrix.shape != (4, 4):
                    raise ValueError(f"{matrix_key} must be 4x4")
                return matrix
        position = np.asarray(record.get("position", record.get("xyz", [0.0, 0.0, 0.0])), dtype=np.float64)
        if "rpy_deg" in record:
            rotation = rpy_to_matrix(np.deg2rad(np.asarray(record["rpy_deg"], dtype=np.float64)))
        elif "rpy_rad" in record:
            rotation = rpy_to_matrix(np.asarray(record["rpy_rad"], dtype=np.float64))
        elif "rotation" in record:
            rotation = np.asarray(record["rotation"], dtype=np.float64)
        else:
            rotation = np.eye(3, dtype=np.float64)
        return make_transform(position, rotation)
    matrix = np.asarray(record, dtype=np.float64)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape == (3,):
        return make_transform(matrix)
    if matrix.shape == (2, 3):
        return make_transform(matrix[0], rpy_to_matrix(matrix[1]))
    raise ValueError(f"unsupported tag transform record shape: {matrix.shape}")


def furniture_base_tag_layout() -> TagLayout:
    return TagLayout(
        tag_size_m=FURNITURE_BASE_TAG_SIZE_M,
        world_t_tag_by_id={
            tag_id: make_transform(position_rpy[0], rpy_to_matrix(position_rpy[1]))
            for tag_id, position_rpy in FURNITURE_BASE_TAG_POSES.items()
        },
        name="furniture_base",
    )


def load_tag_layout(path: Path | None) -> TagLayout:
    if path is None:
        return furniture_base_tag_layout()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("tag layout file must contain a mapping")
    tag_size = payload.get("tag_size_m", payload.get("tag_size", payload.get("base_tag_size")))
    if tag_size is None:
        raise ValueError("tag layout file must define tag_size_m")
    raw_tags = payload.get("tags", payload.get("world_t_tag_by_id", payload.get("tag_poses")))
    if raw_tags is None:
        raise ValueError("tag layout file must define tags")
    tags: dict[int, np.ndarray] = {}
    if isinstance(raw_tags, dict):
        for raw_id, record in raw_tags.items():
            tags[int(raw_id)] = transform_from_layout_record(record)
    elif isinstance(raw_tags, list):
        for item in raw_tags:
            if not isinstance(item, dict) or "id" not in item:
                raise ValueError("tag list records must be mappings with an id field")
            tags[int(item["id"])] = transform_from_layout_record(item)
    else:
        raise ValueError("tags must be a mapping or list")
    if not tags:
        raise ValueError("tag layout contains no tags")
    return TagLayout(
        tag_size_m=float(tag_size),
        world_t_tag_by_id=tags,
        name=str(payload.get("name", path.stem)),
    )


def rotation_matrix_to_quat_xyzw(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ],
            dtype=np.float64,
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ],
                dtype=np.float64,
            )
        elif axis == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.array(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ],
                dtype=np.float64,
            )
    norm = np.linalg.norm(quat)
    if norm <= 0.0:
        raise ValueError("invalid rotation matrix")
    return quat / norm


def quat_xyzw_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(value) for value in np.asarray(quat, dtype=np.float64)]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def average_transforms(transforms: list[np.ndarray]) -> np.ndarray:
    if not transforms:
        raise ValueError("cannot average an empty transform list")
    positions = np.stack([np.asarray(transform, dtype=np.float64)[:3, 3] for transform in transforms], axis=0)
    quats = np.stack([rotation_matrix_to_quat_xyzw(transform[:3, :3]) for transform in transforms], axis=0)
    reference = quats[0]
    for index in range(quats.shape[0]):
        if float(np.dot(reference, quats[index])) < 0.0:
            quats[index] *= -1.0
    accumulator = quats.T @ quats
    _, vectors = np.linalg.eigh(accumulator)
    avg_quat = vectors[:, -1]
    if float(np.dot(reference, avg_quat)) < 0.0:
        avg_quat *= -1.0
    avg_quat /= np.linalg.norm(avg_quat)
    return make_transform(positions.mean(axis=0), quat_xyzw_to_rotation_matrix(avg_quat))


def camera_to_world_from_detection(detection: Any, world_t_tag: np.ndarray) -> np.ndarray:
    camera_t_tag = np.eye(4, dtype=np.float64)
    camera_t_tag[:3, :3] = np.asarray(detection.pose_R, dtype=np.float64).reshape(3, 3)
    camera_t_tag[:3, 3] = np.asarray(detection.pose_t, dtype=np.float64).reshape(3)
    return np.asarray(world_t_tag, dtype=np.float64) @ np.linalg.inv(camera_t_tag)


def single_seg_pose_from_opencv_pose(world_t_camera_cv: np.ndarray) -> np.ndarray:
    return np.asarray(world_t_camera_cv, dtype=np.float64) @ CV_TO_GL


def extrinsics_to_matrix(extr: Any) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = np.asarray(extr.rotation, dtype=np.float64).reshape(3, 3).T
    mat[:3, 3] = np.asarray(extr.translation, dtype=np.float64)
    return mat


def intrinsics_matrix_from_rs(intrinsics: Any) -> np.ndarray:
    return np.array(
        [[intrinsics.fx, 0.0, intrinsics.ppx], [0.0, intrinsics.fy, intrinsics.ppy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


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


def build_rectified_depth_geometry(
    left_intr: Any,
    right_intr: Any,
    left_to_right: np.ndarray,
    left_to_color: np.ndarray,
    *,
    image_size: tuple[int, int],
    alpha: float,
    mode: str,
) -> tuple[dict[str, float], np.ndarray, float]:
    if str(mode) == "passthrough":
        rectified_to_left = np.eye(4, dtype=np.float64)
        rectified_k = intrinsics_matrix_from_rs(left_intr)
        baseline_m = abs(float(left_to_right[0, 3]))
        if baseline_m <= 0.0:
            baseline_m = float(np.linalg.norm(left_to_right[:3, 3]))
    else:
        k1 = intrinsics_matrix_from_rs(left_intr)
        k2 = intrinsics_matrix_from_rs(right_intr)
        d1 = np.asarray(left_intr.coeffs[:5], dtype=np.float64)
        d2 = np.asarray(right_intr.coeffs[:5], dtype=np.float64)
        r1, _, p1, _, _, _, _ = cv2.stereoRectify(
            k1,
            d1,
            k2,
            d2,
            image_size,
            left_to_right[:3, :3],
            left_to_right[:3, 3:4],
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=float(alpha),
        )
        rectified_to_left = np.eye(4, dtype=np.float64)
        rectified_to_left[:3, :3] = r1.T
        rectified_k = p1[:3, :3].astype(np.float64)
        baseline_m = float(np.linalg.norm(left_to_right[:3, 3]))
    rectified_to_color = np.asarray(left_to_color, dtype=np.float64) @ rectified_to_left
    depth_intrinsics = intrinsics_payload_from_k(rectified_k, width=image_size[0], height=image_size[1])
    return depth_intrinsics, rectified_to_color, baseline_m


def intrinsics_payload_from_rs(intrinsics: Any) -> dict[str, float]:
    return {
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "cx": float(intrinsics.ppx),
        "cy": float(intrinsics.ppy),
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
    }


def parse_intrinsics(raw: str) -> dict[str, float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(values) not in {4, 6}:
        raise ValueError("--intrinsics must be fx,fy,cx,cy or fx,fy,cx,cy,width,height")
    payload = {"fx": values[0], "fy": values[1], "cx": values[2], "cy": values[3]}
    if len(values) == 6:
        payload["width"] = int(values[4])
        payload["height"] = int(values[5])
    return payload


def camera_params_from_intrinsics(intrinsics: dict[str, float]) -> list[float]:
    return [
        float(intrinsics["fx"]),
        float(intrinsics["fy"]),
        float(intrinsics["cx"]),
        float(intrinsics["cy"]),
    ]


class AprilTagPoseDetector:
    def __init__(
        self,
        *,
        tag_size_m: float,
        families: str,
        nthreads: int,
        quad_decimate: float,
        max_hamming: int,
        min_decision_margin: float | None,
    ) -> None:
        try:
            from dt_apriltags import Detector
        except ImportError as exc:
            raise RuntimeError("需要安装 dt-apriltags 才能检测 AprilTag") from exc
        self.detector = Detector(
            families=families,
            nthreads=int(nthreads),
            quad_decimate=float(quad_decimate),
            debug=0,
        )
        self.tag_size_m = float(tag_size_m)
        self.max_hamming = int(max_hamming)
        self.min_decision_margin = min_decision_margin

    def detect(self, image: np.ndarray, intrinsics: dict[str, float]) -> list[Any]:
        image = np.asarray(image)
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        detections = self.detector.detect(
            gray,
            True,
            camera_params_from_intrinsics(intrinsics),
            self.tag_size_m,
        )
        filtered = []
        for detection in detections:
            if int(getattr(detection, "hamming", 0)) > self.max_hamming:
                continue
            if self.min_decision_margin is not None:
                margin = float(getattr(detection, "decision_margin", 0.0))
                if margin < float(self.min_decision_margin):
                    continue
            filtered.append(detection)
        return filtered


class RealSenseColorCamera:
    def __init__(
        self,
        *,
        camera_id: str,
        serial_number: str,
        width: int,
        height: int,
        stereo_width: int,
        stereo_height: int,
        stereo_alpha: float,
        stereo_rectification_mode: str,
        fps: int,
        wait_timeout_ms: int,
        disable_auto_exposure: bool,
    ) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("需要安装 pyrealsense2 才能从 RealSense 采集图像") from exc
        self.rs = rs
        self.camera_id = str(camera_id)
        self.serial_number = str(serial_number)
        self.width = int(width)
        self.height = int(height)
        self.stereo_width = int(stereo_width)
        self.stereo_height = int(stereo_height)
        self.stereo_alpha = float(stereo_alpha)
        self.stereo_rectification_mode = str(stereo_rectification_mode)
        self.fps = int(fps)
        self.wait_timeout_ms = int(wait_timeout_ms)
        self.disable_auto_exposure = bool(disable_auto_exposure)
        self.pipeline = rs.pipeline()
        self.profile = None
        self.intrinsics: dict[str, float] | None = None
        self.depth_intrinsics: dict[str, float] | None = None
        self.depth_to_color_4x4: np.ndarray | None = None
        self.depth_frame = "rectified_depth"
        self.baseline_m: float | None = None

    def start(self) -> None:
        config = self.rs.config()
        config.enable_device(self.serial_number)
        config.enable_stream(self.rs.stream.color, self.width, self.height, self.rs.format.bgr8, self.fps)
        config.enable_stream(
            self.rs.stream.infrared,
            1,
            self.stereo_width,
            self.stereo_height,
            self.rs.format.y8,
            self.fps,
        )
        config.enable_stream(
            self.rs.stream.infrared,
            2,
            self.stereo_width,
            self.stereo_height,
            self.rs.format.y8,
            self.fps,
        )
        self.profile = self.pipeline.start(config)
        color_profile = self.profile.get_stream(self.rs.stream.color).as_video_stream_profile()
        self.intrinsics = intrinsics_payload_from_rs(color_profile.get_intrinsics())
        left_profile = self.profile.get_stream(self.rs.stream.infrared, 1).as_video_stream_profile()
        right_profile = self.profile.get_stream(self.rs.stream.infrared, 2).as_video_stream_profile()
        depth_intrinsics, depth_to_color, baseline_m = build_rectified_depth_geometry(
            left_profile.get_intrinsics(),
            right_profile.get_intrinsics(),
            extrinsics_to_matrix(left_profile.get_extrinsics_to(right_profile)),
            extrinsics_to_matrix(left_profile.get_extrinsics_to(color_profile)),
            image_size=(self.stereo_width, self.stereo_height),
            alpha=self.stereo_alpha,
            mode=self.stereo_rectification_mode,
        )
        self.depth_intrinsics = depth_intrinsics
        self.depth_to_color_4x4 = depth_to_color
        self.baseline_m = baseline_m
        if self.disable_auto_exposure:
            sensor = self.profile.get_device().first_color_sensor()
            sensor.set_option(self.rs.option.enable_auto_exposure, False)

    def warmup(self, num_frames: int) -> None:
        for _ in range(max(int(num_frames), 0)):
            self.capture()

    def capture(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames(timeout_ms=self.wait_timeout_ms)
        while True:
            ok, newer = self.pipeline.try_wait_for_frames(timeout_ms=1)
            if not ok:
                break
            frames = newer
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError(f"{self.camera_id} did not return a color frame")
        return np.asanyarray(color_frame.get_data()).copy()

    def stop(self) -> None:
        self.pipeline.stop()


def list_realsense_serials() -> list[str]:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError("需要安装 pyrealsense2 才能列出 RealSense 相机") from exc
    devices = rs.context().query_devices()
    serials: list[str] = []
    for device in devices:
        serial = device.get_info(rs.camera_info.serial_number)
        name = device.get_info(rs.camera_info.name)
        serials.append(serial)
        print(f"{len(serials) - 1}: {name} serial={serial}")
    return serials


def resolve_serials(serials_arg: str | None, camera_count: int) -> list[str]:
    if serials_arg:
        return [part.strip() for part in serials_arg.split(",") if part.strip()]
    serials = list_realsense_serials()
    if len(serials) < int(camera_count):
        raise RuntimeError(f"requested {camera_count} cameras but only found {len(serials)}: {serials}")
    return serials[: int(camera_count)]


def draw_debug_detections(image: np.ndarray, detections: list[Any], layout: TagLayout) -> np.ndarray:
    output = image.copy()
    for detection in detections:
        corners = np.asarray(detection.corners, dtype=np.int32).reshape(-1, 2)
        color = (0, 220, 0) if int(detection.tag_id) in layout.world_t_tag_by_id else (0, 0, 220)
        cv2.polylines(output, [corners], True, color, 2)
        center = tuple(np.asarray(detection.center, dtype=np.int32).reshape(2))
        cv2.putText(
            output,
            str(int(detection.tag_id)),
            center,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    return output


def calibrate_camera_from_frames(
    *,
    camera_id: str,
    serial_number: str,
    frames: list[np.ndarray],
    intrinsics: dict[str, float],
    layout: TagLayout,
    detector: AprilTagPoseDetector,
    min_tags_per_frame: int,
    debug_dir: Path | None,
    depth_intrinsics: dict[str, float] | None = None,
    depth_to_color_4x4: np.ndarray | None = None,
    depth_frame: str | None = None,
    baseline_m: float | None = None,
) -> CameraCalibration:
    pose_observations: list[np.ndarray] = []
    pose_errors: list[float] = []
    margins: list[float] = []
    tag_counts = {tag_id: 0 for tag_id in layout.world_t_tag_by_id}
    detected_tag_counts: dict[int, int] = {}
    configured_seen_counts: dict[int, int] = {}
    frames_used = 0
    last_debug: np.ndarray | None = None
    debug_path: Path | None = None
    for frame in frames:
        detections = detector.detect(frame, intrinsics)
        for detection in detections:
            tag_id = int(detection.tag_id)
            detected_tag_counts[tag_id] = detected_tag_counts.get(tag_id, 0) + 1
        matched = [detection for detection in detections if int(detection.tag_id) in layout.world_t_tag_by_id]
        for detection in matched:
            tag_id = int(detection.tag_id)
            configured_seen_counts[tag_id] = configured_seen_counts.get(tag_id, 0) + 1
        if len(matched) < int(min_tags_per_frame):
            last_debug = draw_debug_detections(frame, detections, layout)
            continue
        frames_used += 1
        last_debug = draw_debug_detections(frame, detections, layout)
        for detection in matched:
            tag_id = int(detection.tag_id)
            tag_counts[tag_id] = tag_counts.get(tag_id, 0) + 1
            pose_observations.append(
                camera_to_world_from_detection(detection, layout.world_t_tag_by_id[tag_id])
            )
            if hasattr(detection, "pose_err"):
                pose_errors.append(float(detection.pose_err))
            if hasattr(detection, "decision_margin"):
                margins.append(float(detection.decision_margin))
    if not pose_observations:
        if debug_dir is not None and last_debug is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_path = debug_dir / f"{camera_id}_detections.png"
            cv2.imwrite(str(debug_path), last_debug)
        configured_ids = sorted(int(tag_id) for tag_id in layout.world_t_tag_by_id)
        detected_ids = sorted(int(tag_id) for tag_id in detected_tag_counts)
        configured_seen_ids = sorted(int(tag_id) for tag_id in configured_seen_counts)
        if not detected_ids:
            reason = "没有检测到任何 AprilTag"
        elif not configured_seen_ids:
            reason = "检测到了 AprilTag，但 tag id 不在当前 tag layout 里"
        else:
            reason = f"检测到了配置内 AprilTag，但没有任何一帧满足 min_tags_per_frame={int(min_tags_per_frame)}"
        debug_hint = f"; debug_image={debug_path}" if debug_path is not None else ""
        raise RuntimeError(
            f"{camera_id} serial={serial_number} 无法通过 AprilTag 标定外参：{reason}. "
            f"frames={len(frames)}, valid_frames={frames_used}, "
            f"configured_tag_ids={configured_ids}, detected_tag_ids={detected_ids}, "
            f"configured_detected_counts={configured_seen_counts}, all_detected_counts={detected_tag_counts}"
            f"{debug_hint}"
        )
    world_t_camera_cv = average_transforms(pose_observations)
    depth_to_color = None if depth_to_color_4x4 is None else np.asarray(depth_to_color_4x4, dtype=np.float64)
    world_t_depth_cv = None if depth_to_color is None else world_t_camera_cv @ depth_to_color
    if debug_dir is not None and last_debug is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / f"{camera_id}_detections.png"), last_debug)
    return CameraCalibration(
        camera_id=camera_id,
        serial_number=serial_number,
        intrinsics=dict(intrinsics),
        world_t_camera_cv=world_t_camera_cv,
        world_t_camera_single_seg=single_seg_pose_from_opencv_pose(world_t_camera_cv),
        num_frames_used=frames_used,
        num_tag_observations=len(pose_observations),
        tag_counts={int(key): int(value) for key, value in tag_counts.items() if int(value) > 0},
        mean_pose_error=float(np.mean(pose_errors)) if pose_errors else None,
        mean_decision_margin=float(np.mean(margins)) if margins else None,
        depth_intrinsics=None if depth_intrinsics is None else dict(depth_intrinsics),
        depth_to_color_4x4=depth_to_color,
        world_t_depth_cv=world_t_depth_cv,
        world_t_depth_single_seg=None if world_t_depth_cv is None else single_seg_pose_from_opencv_pose(world_t_depth_cv),
        depth_frame=depth_frame,
        baseline_m=None if baseline_m is None else float(baseline_m),
    )


def calibrate_live_cameras(args: argparse.Namespace, layout: TagLayout, detector: AprilTagPoseDetector) -> list[CameraCalibration]:
    serials = resolve_serials(args.serials, int(args.camera_count))
    cameras = [
        RealSenseColorCamera(
            camera_id=f"cam_{index:02d}",
            serial_number=serial,
            width=int(args.width),
            height=int(args.height),
            stereo_width=int(args.stereo_width),
            stereo_height=int(args.stereo_height),
            stereo_alpha=float(args.stereo_alpha),
            stereo_rectification_mode=str(args.stereo_rectification_mode),
            fps=int(args.fps),
            wait_timeout_ms=int(args.wait_timeout_ms),
            disable_auto_exposure=bool(args.disable_auto_exposure),
        )
        for index, serial in enumerate(serials)
    ]
    try:
        for camera in cameras:
            camera.start()
            print(f"started {camera.camera_id} serial={camera.serial_number}")
        for camera in cameras:
            camera.warmup(int(args.warmup_frames))
        frames_by_camera: dict[str, list[np.ndarray]] = {camera.camera_id: [] for camera in cameras}
        for frame_index in range(int(args.num_frames)):
            for camera in cameras:
                frames_by_camera[camera.camera_id].append(camera.capture())
            if float(args.sample_delay_sec) > 0.0:
                time.sleep(float(args.sample_delay_sec))
            print(f"captured sample {frame_index + 1}/{int(args.num_frames)}")
        calibrations: list[CameraCalibration] = []
        for camera in cameras:
            assert camera.intrinsics is not None
            calibrations.append(
                calibrate_camera_from_frames(
                    camera_id=camera.camera_id,
                    serial_number=camera.serial_number,
                    frames=frames_by_camera[camera.camera_id],
                    intrinsics=camera.intrinsics,
                    layout=layout,
                    detector=detector,
                    min_tags_per_frame=int(args.min_tags_per_frame),
                    debug_dir=resolve_path(args.debug_dir) if args.debug_dir is not None else None,
                    depth_intrinsics=camera.depth_intrinsics,
                    depth_to_color_4x4=camera.depth_to_color_4x4,
                    depth_frame=camera.depth_frame,
                    baseline_m=camera.baseline_m,
                )
            )
        return calibrations
    finally:
        for camera in cameras:
            camera.stop()


def calibrate_images(args: argparse.Namespace, layout: TagLayout, detector: AprilTagPoseDetector) -> list[CameraCalibration]:
    if args.images is None:
        raise ValueError("--images is required in image mode")
    intrinsics_by_camera = parse_image_intrinsics(args.image_intrinsics)
    calibrations: list[CameraCalibration] = []
    for index, spec in enumerate(args.images):
        camera_id, serial_number, image_paths = parse_image_spec(spec, index)
        frames = [cv2.imread(str(resolve_path(path)), cv2.IMREAD_COLOR) for path in image_paths]
        missing = [str(path) for path, image in zip(image_paths, frames) if image is None]
        if missing:
            raise FileNotFoundError(f"failed to read images: {missing}")
        intrinsics = intrinsics_by_camera.get(camera_id) or intrinsics_by_camera.get(serial_number)
        if intrinsics is None:
            intrinsics = intrinsics_by_camera.get("*")
        if intrinsics is None:
            raise ValueError(f"missing intrinsics for {camera_id}; use --image-intrinsics")
        calibrations.append(
            calibrate_camera_from_frames(
                camera_id=camera_id,
                serial_number=serial_number,
                frames=frames,
                intrinsics=intrinsics,
                layout=layout,
                detector=detector,
                min_tags_per_frame=int(args.min_tags_per_frame),
                debug_dir=resolve_path(args.debug_dir) if args.debug_dir is not None else None,
            )
        )
    return calibrations


def parse_image_spec(spec: str, index: int) -> tuple[str, str, list[Path]]:
    if "=" in spec:
        name, paths_raw = spec.split("=", 1)
        camera_id = name.strip()
    else:
        camera_id = f"cam_{index:02d}"
        paths_raw = spec
    paths = [Path(part.strip()) for part in paths_raw.split(",") if part.strip()]
    if not paths:
        raise ValueError(f"image spec has no paths: {spec}")
    return camera_id, camera_id, paths


def parse_image_intrinsics(raw_items: list[str] | None) -> dict[str, dict[str, float]]:
    parsed: dict[str, dict[str, float]] = {}
    if raw_items is None:
        return parsed
    for item in raw_items:
        if "=" not in item:
            parsed["*"] = parse_intrinsics(item)
            continue
        name, payload = item.split("=", 1)
        parsed[name.strip()] = parse_intrinsics(payload)
    return parsed


def calibration_to_payload(calibration: CameraCalibration, output_convention: str) -> dict[str, Any]:
    if output_convention == "opencv_cv":
        cam2world = calibration.world_t_camera_cv
    elif output_convention == "single_seg_gl":
        cam2world = calibration.world_t_camera_single_seg
    else:
        raise ValueError(f"unsupported output convention: {output_convention}")
    payload: dict[str, Any] = {
        "camera_id": calibration.camera_id,
        "serial_number": calibration.serial_number,
        "cam2world_4x4": cam2world.tolist(),
        "world2cam_4x4": np.linalg.inv(cam2world).tolist(),
        "opencv_cv_cam2world_4x4": calibration.world_t_camera_cv.tolist(),
        "single_seg_gl_cam2world_4x4": calibration.world_t_camera_single_seg.tolist(),
        "color_intrinsics": calibration.intrinsics,
        "num_frames_used": int(calibration.num_frames_used),
        "num_tag_observations": int(calibration.num_tag_observations),
        "tag_counts": {str(key): int(value) for key, value in calibration.tag_counts.items()},
        "mean_pose_error": calibration.mean_pose_error,
        "mean_decision_margin": calibration.mean_decision_margin,
    }
    if calibration.world_t_depth_single_seg is not None and calibration.world_t_depth_cv is not None:
        depth_cam2world = (
            calibration.world_t_depth_cv
            if output_convention == "opencv_cv"
            else calibration.world_t_depth_single_seg
        )
        payload.update(
            {
                "depth_frame": calibration.depth_frame or "rectified_depth",
                "depth_cam2world_4x4": depth_cam2world.tolist(),
                "rectified_depth_cam2world_4x4": depth_cam2world.tolist(),
                "world2depth_4x4": np.linalg.inv(depth_cam2world).tolist(),
                "opencv_cv_depth_cam2world_4x4": calibration.world_t_depth_cv.tolist(),
                "single_seg_gl_depth_cam2world_4x4": calibration.world_t_depth_single_seg.tolist(),
                "depth_to_color_4x4": calibration.depth_to_color_4x4.tolist()
                if calibration.depth_to_color_4x4 is not None
                else None,
                "rectified_to_color": calibration.depth_to_color_4x4.tolist()
                if calibration.depth_to_color_4x4 is not None
                else None,
                "depth_intrinsics": calibration.depth_intrinsics or {},
                "baseline_m": calibration.baseline_m,
            }
        )
    return payload


def write_output(
    *,
    output_path: Path,
    calibrations: list[CameraCalibration],
    layout: TagLayout,
    args: argparse.Namespace,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "world_frame": "apriltag_layout",
        "output_convention": str(args.output_convention),
        "tag_layout": {
            "name": layout.name,
            "tag_size_m": float(layout.tag_size_m),
            "tag_ids": sorted(int(tag_id) for tag_id in layout.world_t_tag_by_id),
        },
        "cameras": [
            calibration_to_payload(calibration, str(args.output_convention))
            for calibration in calibrations
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用 AprilTag 标定单台或多台 RealSense 外参，并生成 single_seg live 可读取的 camera_poses_json。"
    )
    parser.add_argument("--list-cameras", action="store_true", help="列出可用 RealSense 相机后退出")
    parser.add_argument("--mode", choices=("live", "images"), default="live", help="live 从 RealSense 采集；images 从图片读取")
    parser.add_argument("--serials", default=None, help="逗号分隔的 RealSense 序列号；不传则按枚举顺序取前 camera-count 台")
    parser.add_argument("--camera-count", type=int, default=1, help="未指定 serials 时使用的相机数量")
    parser.add_argument("--width", type=int, default=1280, help="RealSense color 宽度")
    parser.add_argument("--height", type=int, default=720, help="RealSense color 高度")
    parser.add_argument("--stereo-width", type=int, default=1280, help="RealSense IR/fast depth 宽度")
    parser.add_argument("--stereo-height", type=int, default=720, help="RealSense IR/fast depth 高度")
    parser.add_argument("--stereo-alpha", type=float, default=0.0, help="OpenCV stereoRectify alpha")
    parser.add_argument("--stereo-rectification-mode", choices=("opencv", "passthrough"), default="opencv")
    parser.add_argument("--fps", type=int, default=30, help="RealSense color 帧率")
    parser.add_argument("--warmup-frames", type=int, default=10, help="正式采样前预热帧数")
    parser.add_argument("--num-frames", type=int, default=30, help="每台相机用于平均的采样帧数")
    parser.add_argument("--sample-delay-sec", type=float, default=0.02, help="采样帧之间的等待时间")
    parser.add_argument("--wait-timeout-ms", type=int, default=6000, help="RealSense 等帧超时")
    parser.add_argument("--disable-auto-exposure", action="store_true", help="关闭 color 自动曝光")
    parser.add_argument("--tag-layout", type=Path, default=None, help="自定义 AprilTag 世界布局 YAML/JSON；不传则使用 furniture-bench base tags")
    parser.add_argument("--families", default="tag36h11", help="AprilTag family")
    parser.add_argument("--quad-decimate", type=float, default=1.0, help="dt-apriltags quad_decimate")
    parser.add_argument("--detector-threads", type=int, default=4, help="AprilTag detector 线程数")
    parser.add_argument("--max-hamming", type=int, default=1, help="允许的最大 hamming；furniture-bench 使用 <2")
    parser.add_argument("--min-decision-margin", type=float, default=None, help="过滤低 decision_margin 检测")
    parser.add_argument("--min-tags-per-frame", type=int, default=1, help="一帧至少看到几个配置内 tag 才用于标定")
    parser.add_argument("--output-convention", choices=("single_seg_gl", "opencv_cv"), default="single_seg_gl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="输出 camera_poses_json 路径")
    parser.add_argument("--debug-dir", type=Path, default=None, help="保存每台相机最后一帧 AprilTag 检测可视化")
    parser.add_argument("--images", action="append", default=None, help="图片模式输入，例如 cam_00=img1.png,img2.png；可重复传多台相机")
    parser.add_argument("--image-intrinsics", action="append", default=None, help="图片模式内参，例如 cam_00=fx,fy,cx,cy,width,height 或 fx,fy,cx,cy")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.list_cameras:
        list_realsense_serials()
        return
    layout = load_tag_layout(resolve_path(args.tag_layout) if args.tag_layout is not None else None)
    detector = AprilTagPoseDetector(
        tag_size_m=layout.tag_size_m,
        families=str(args.families),
        nthreads=int(args.detector_threads),
        quad_decimate=float(args.quad_decimate),
        max_hamming=int(args.max_hamming),
        min_decision_margin=args.min_decision_margin,
    )
    if args.mode == "live":
        calibrations = calibrate_live_cameras(args, layout, detector)
    else:
        calibrations = calibrate_images(args, layout, detector)
    for calibration in calibrations:
        print(
            f"{calibration.camera_id} serial={calibration.serial_number} "
            f"frames={calibration.num_frames_used} observations={calibration.num_tag_observations} "
            f"tags={calibration.tag_counts}"
        )
    output_path = resolve_path(args.output)
    write_output(output_path=output_path, calibrations=calibrations, layout=layout, args=args)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
