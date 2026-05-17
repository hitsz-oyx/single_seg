#!/usr/bin/env python3
"""带有正/负提示框的单物体在线 RGBD 点云分割。"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any

import yaml

import numpy as np
import open3d as o3d
from PIL import Image
from PIL import ImageDraw
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
# 默认的剧集（episode）目录
DEFAULT_EPISODE_DIR = REPO_ROOT / "examples" / "data" / "libero_spatial" / "task_00_demo" / "episode_0001"
# 默认的提示图像根目录
DEFAULT_PROMPT_IMAGE_ROOT = REPO_ROOT / "assets" / "prompts" / "libero_spatial" / "semantic_split_parts"
# 默认的提示任务信息文件
DEFAULT_PROMPT_TASK_INFO = DEFAULT_PROMPT_IMAGE_ROOT / "task_info.json"
# 默认的输出目录
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tests" / "outputs" / "demo_spatial_single_object"
DEFAULT_PROMPT_MAX_MASKS = 4
DEFAULT_PROMPT_REF_CELL = 160
DEFAULT_PROMPT_MAX_COLS = 10
DEFAULT_PROMPT_CANVAS_GAP = 24
DEFAULT_SEED_MIN_PIXELS = 200
DEFAULT_SEED_MAX_AREA_RATIO = 0.35
DEFAULT_SEED_BOX_MARGIN = 12
DEFAULT_VIDEO_OBJECT_MIN_SCORE = 0.0
DEFAULT_TRACKER_IMAGE_SIZE = 896
DEFAULT_SYNC_TIMING = False
DEFAULT_TORCH_CLUSTER_EXPANSION_STEPS = 32


def resolve_default_checkpoint() -> Path:
    """解析并返回默认的 SAM3 模型权重路径。"""
    env_path = os.environ.get("SAM3_CHECKPOINT")
    candidates = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(
        [
            REPO_ROOT / "checkpoints" / "sam3.pt",
            Path.home() / ".cache" / "modelscope" / "hub" / "facebook" / "sam3" / "sam3.pt",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_CHECKPOINT = resolve_default_checkpoint()

# 将 SAM3 相关的路径加入系统搜索路径
for sam3_root in (
    Path(os.environ["SAM3_REPO_ROOT"]).expanduser() if os.environ.get("SAM3_REPO_ROOT") else None,
    REPO_ROOT / "third_party" / "sam3",
):
    if sam3_root is not None and sam3_root.exists() and str(sam3_root) not in sys.path:
        sys.path.insert(0, str(sam3_root))


@dataclass(frozen=True)
class PromptEntry:
    """提示条目，包含语义名称、图像名、源路径和包围框。"""
    semantic_name: str
    image_name: str
    source_path: Path
    box_xyxy: list[int]


@dataclass(frozen=True)
class CameraFrame:
    """相机帧数据，包含 RGB、深度、内参、位姿记录和垂直视角。"""
    camera_id: str
    rgb: np.ndarray
    depth_m: np.ndarray | torch.Tensor
    intrinsics: dict[str, float] | None
    pose_record: dict[str, object]
    fovy_deg: float | None


def resolve_repo_path(path_like: str | os.PathLike[str] | Path, *, base_dir: Path = REPO_ROOT) -> Path:
    """解析相对于仓库根目录的路径。"""
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


@dataclass(frozen=True)
class SingleSegConfig:
    """SingleObjectPointCloudSegmenter 的可序列化初始化配置。"""

    target_name: str = "plate"  # 目标物体名称
    prompt_task_info: Path = DEFAULT_PROMPT_TASK_INFO  # 提示任务信息路径
    prompt_image_root: Path = DEFAULT_PROMPT_IMAGE_ROOT  # 提示图像根目录
    checkpoint_path: Path = DEFAULT_CHECKPOINT  # 模型权重路径
    output_dir: Path = DEFAULT_OUTPUT_DIR  # 输出目录
    overwrite_output: bool = False  # 是否覆盖输出
    confidence: float = 0.25  # 置信度阈值
    mask_threshold: float = 0.6  # 掩码阈值
    prompt_keep_score_threshold: float = 0.2  # 提示保留评分阈值
    video_mask_prob_threshold: float = 0.95  # 视频掩码概率阈值
    depth_scale: float = 1000.0  # 深度缩放比例（将单位转换为米）
    depth_min: float = 0.1  # 最小有效深度（米）
    depth_max: float = 3.0  # 最大有效深度（米）
    stride: int = 2  # 处理帧的步长
    frame_voxel_size: float = 0.003  # 帧体素大小（用于下采样点云）
    target_cluster_filter_enabled: bool = True  # 是否启用目标点 3D 聚类去散点
    target_cluster_radius_m: float = 0.025  # 目标点聚类邻域半径（米）
    target_cluster_min_points: int = 35  # 形成有效目标簇所需的最少点数
    target_cluster_keep_largest: bool = True  # 是否只保留最大目标簇
    target_plane_filter_enabled: bool = False  # 是否启用目标点主平面剔除（常用于去掉桌面点）
    target_plane_filter_distance_m: float = 0.004  # 点到主平面的最大距离（米）
    target_plane_filter_min_points: int = 80  # 主平面至少需要的内点数量
    target_plane_filter_min_inlier_ratio: float = 0.25  # 主平面内点占目标点的最低比例
    target_plane_filter_max_inlier_ratio: float = 0.85  # 主平面内点占目标点的最高比例，过高时跳过以免误删目标
    target_plane_filter_max_planes: int = 1  # 单帧单相机最多剔除几个主平面
    target_plane_filter_ransac_iterations: int = 256  # 主平面 RANSAC 迭代次数
    target_depth_band_filter_enabled: bool = True  # 是否按目标核心深度带过滤 3D 取点 mask
    target_depth_band_filter_range_m: float = 0.08  # 保留距离目标深度中位数多少米内的像素
    target_depth_band_filter_min_valid_pixels: int = 50  # 估计目标深度中位数所需的最少有效像素
    target_depth_band_filter_min_keep_pixels: int = 20  # 过滤后至少保留的像素数，过少时跳过过滤
    target_3d_mask_erode_kernel: int = 0  # 反投影前仅用于 3D 取点的目标 mask 腐蚀核大小（像素）
    save_ply: bool = True  # 是否保存 .ply 点云文件
    save_normal: bool = False  # 是否在保存的 PLY 中写入估计法线
    save_debug_2d: bool = False  # 是否保存 2D 调试图
    tracker_image_size: int | None = DEFAULT_TRACKER_IMAGE_SIZE  # 追踪器输入图像尺寸
    target_vis_color: tuple[int, int, int] | None = None  # 目标点可视化颜色 R,G,B，默认红色 (255,70,70)

    @classmethod
    def from_mapping(
        cls,
        payload: dict[str, Any],
        *,
        base_dir: Path = REPO_ROOT,
    ) -> "SingleSegConfig":
        """从字典映射创建配置对象。"""
        raw = dict(payload.get("segmenter", payload))
        defaults = cls()
        path_fields = {"prompt_task_info", "prompt_image_root", "checkpoint_path", "output_dir"}
        values: dict[str, Any] = {}
        for field_name in cls.__dataclass_fields__:
            default_value = getattr(defaults, field_name)
            raw_value = raw.get(field_name, default_value)
            if field_name in path_fields:
                values[field_name] = resolve_repo_path(raw_value, base_dir=base_dir)
            else:
                values[field_name] = raw_value
        return cls(**values)

    @classmethod
    def from_yaml(cls, config_path: Path | str) -> "SingleSegConfig":
        """从 YAML 文件加载配置。"""
        config_path = resolve_repo_path(config_path)
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return cls.from_mapping(payload, base_dir=REPO_ROOT)

    @classmethod
    def from_file(cls, config_path: Path | str) -> "SingleSegConfig":
        """从文件加载配置（支持 .json 和 .yaml 格式）。"""
        return cls.from_yaml(config_path)

    def with_overrides(self, **overrides: Any) -> "SingleSegConfig":
        """应用覆盖选项，返回新的配置对象。"""
        merged = {field_name: getattr(self, field_name) for field_name in self.__dataclass_fields__}
        for key, value in overrides.items():
            if value is None or key not in merged:
                continue
            merged[key] = value
        return SingleSegConfig(**merged)

    def to_segmenter_kwargs(self) -> dict[str, Any]:
        """将配置转换为传给分层器的关键字参数。"""
        return {field_name: getattr(self, field_name) for field_name in self.__dataclass_fields__}


def load_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def semantic_name_from_asset(asset_name: str) -> str:
    """从资产名称中提取语义名称（去除末尾的数字索引）。"""
    return re.sub(r"_\d+$", "", str(asset_name))


def collect_common_frame_names(episode_dir: Path, camera_ids: list[str]) -> list[str]:
    """收集多个相机共有的帧名称列表。"""
    common_names: set[str] | None = None
    for camera_id in camera_ids:
        rgb_names = {path.name for path in (episode_dir / camera_id / "rgb").glob("frame_*.png")}
        depth_names = {path.name for path in (episode_dir / camera_id / "depth").glob("frame_*.png")}
        camera_names = rgb_names & depth_names
        common_names = camera_names if common_names is None else (common_names & camera_names)
    if not common_names:
        raise RuntimeError(f"No common RGBD frame names found under {episode_dir}")
    return sorted(common_names)


def load_frame_camera_extrinsics(episode_dir: Path, frame_name: str) -> dict[str, dict[str, object]]:
    """加载特定帧的相机外参。"""
    frames_path = episode_dir / "camera_extrinsics_frames.jsonl"
    if not frames_path.exists():
        return {}
    with frames_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("frame_name") != frame_name:
                continue
            cameras = record.get("cameras", [])
            if not isinstance(cameras, list):
                return {}
            return {
                str(camera["camera_id"]): camera
                for camera in cameras
                if isinstance(camera, dict) and isinstance(camera.get("camera_id"), str)
            }
    return {}


def load_episode_camera_records(episode_dir: Path) -> list[dict[str, object]]:
    """加载剧集的相机基准记录（外参及内参）。"""
    payload = load_json(episode_dir / "camera_extrinsics.json")
    cameras = payload.get("cameras", [])
    if not isinstance(cameras, list) or not cameras:
        raise ValueError(f"camera_extrinsics.json does not contain cameras: {episode_dir}")
    return cameras


def normalize_intrinsics_payload(intrinsics: object | None) -> dict[str, float] | None:
    """标准化相机内参格式（支持字典或 3x3 矩阵）。"""
    if intrinsics is None:
        return None
    if isinstance(intrinsics, dict):
        return {
            "fx": float(intrinsics["fx"]),
            "fy": float(intrinsics["fy"]),
            "cx": float(intrinsics["cx"]),
            "cy": float(intrinsics["cy"]),
        }
    matrix = np.asarray(intrinsics, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("intrinsics must be a dict or 3x3 matrix")
    return {
        "fx": float(matrix[0, 0]),
        "fy": float(matrix[1, 1]),
        "cx": float(matrix[0, 2]),
        "cy": float(matrix[1, 2]),
    }


def normalize_pose_record(camera_id: str, payload: dict[str, object]) -> dict[str, object]:
    """标准化相机位姿记录，确保包含 cam2world_4x4 和 world2cam_4x4。"""
    pose_record = payload.get("pose_record")
    if isinstance(pose_record, dict) and pose_record.get("cam2world_4x4") is not None:
        cam2world = np.asarray(pose_record["cam2world_4x4"], dtype=np.float64)
        world2cam = pose_record.get("world2cam_4x4")
        if world2cam is None:
            world2cam = np.linalg.inv(cam2world)
        else:
            world2cam = np.asarray(world2cam, dtype=np.float64)
        return {
            "camera_id": camera_id,
            "cam2world_4x4": cam2world.tolist(),
            "world2cam_4x4": world2cam.tolist(),
        }
    cam2world = payload.get("cam2world_4x4")
    if cam2world is None and isinstance(payload.get("extrinsics"), dict):
        cam2world = payload["extrinsics"].get("cam2world_4x4")
    if cam2world is None:
        raise KeyError(f"camera {camera_id} is missing cam2world_4x4")
    cam2world_np = np.asarray(cam2world, dtype=np.float64)
    world2cam = payload.get("world2cam_4x4")
    if world2cam is None and isinstance(payload.get("extrinsics"), dict):
        world2cam = payload["extrinsics"].get("world2cam_4x4")
    world2cam_np = np.asarray(world2cam, dtype=np.float64) if world2cam is not None else np.linalg.inv(cam2world_np)
    return {
        "camera_id": camera_id,
        "cam2world_4x4": cam2world_np.tolist(),
        "world2cam_4x4": world2cam_np.tolist(),
    }


def load_rgb_depth(rgb_path: Path, depth_path: Path, depth_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """读取并处理 RGB 和深度图。"""
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    depth_m = np.asarray(Image.open(depth_path), dtype=np.float32) / float(depth_scale)
    return rgb, depth_m


def load_episode_frame_inputs(
    episode_dir: Path,
    frame_name: str,
    camera_records: list[dict[str, object]],
    depth_scale: float,
) -> dict[str, dict[str, object]]:
    """加载指定剧集和帧名称的所有相机的输入数据。"""
    frame_extrinsics = load_frame_camera_extrinsics(episode_dir, frame_name)
    camera_inputs: dict[str, dict[str, object]] = {}
    for camera_record in camera_records:
        camera_id = str(camera_record["camera_id"])
        rgb, depth_m = load_rgb_depth(
            rgb_path=episode_dir / camera_id / "rgb" / frame_name,
            depth_path=episode_dir / camera_id / "depth" / frame_name,
            depth_scale=depth_scale,
        )
        pose_record = frame_extrinsics.get(camera_id, camera_record)
        camera_inputs[camera_id] = {
            "rgb": rgb,
            "depth_m": depth_m,
            "intrinsics": camera_record.get("intrinsics"),
            "fovy_deg": camera_record.get("fovy_deg"),
            "pose_record": pose_record,
        }
    return camera_inputs


def build_prompt_grid_layout(prompt_ids: list[str], ref_cell: int, max_cols: int) -> dict[str, tuple[int, int]]:
    """构建提示网格布局，计算每个提示在画布上的位置。"""
    cols = min(max(max_cols, 1), max(1, math.ceil(math.sqrt(len(prompt_ids)))))
    layout: dict[str, tuple[int, int]] = {}
    for idx, prompt_id in enumerate(prompt_ids):
        col = idx % cols
        row = idx // cols
        layout[prompt_id] = (col * ref_cell, row * ref_cell)
    return layout


def scale_bbox_to_layout(box_xyxy: list[int], source_size: list[int], pasted_size: list[int]) -> list[float]:
    """将包围框从源图像坐标缩放到目标画布坐标。"""
    src_w = max(float(source_size[0]), 1.0)
    src_h = max(float(source_size[1]), 1.0)
    dst_w = float(pasted_size[0])
    dst_h = float(pasted_size[1])
    x0, y0, x1, y1 = [float(value) for value in box_xyxy]
    return [
        x0 * dst_w / src_w,
        y0 * dst_h / src_h,
        x1 * dst_w / src_w,
        y1 * dst_h / src_h,
    ]


def xywh_to_normalized_cxcywh(box_xywh: list[float], image_size: tuple[int, int]) -> list[float]:
    """将中心点坐标和宽高格式的包围框转换为归一化的中心坐标和宽高格式。"""
    x, y, w, h = [float(value) for value in box_xywh]
    image_w = max(float(image_size[0]), 1.0)
    image_h = max(float(image_size[1]), 1.0)
    return [
        (x + 0.5 * w) / image_w,
        (y + 0.5 * h) / image_h,
        w / image_w,
        h / image_h,
    ]


def filter_predictions_to_camera(
    boxes: np.ndarray,
    scores: np.ndarray,
    masks: np.ndarray,
    camera_layout: dict[str, object],
    keep_score_threshold: float,
    max_keep: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """根据相机视角过滤并选择预测的目标框、得分和掩码。"""
    cam_x, cam_y = camera_layout["paste_xy"]
    cam_w, cam_h = camera_layout["image_size"]
    kept: list[tuple[int, float]] = []
    for idx, (box, score) in enumerate(zip(boxes, scores)):
        cx = float((box[0] + box[2]) / 2.0)
        cy = float((box[1] + box[3]) / 2.0)
        if cam_x <= cx <= cam_x + cam_w and cam_y <= cy <= cam_y + cam_h:
            kept.append((idx, float(score)))
    kept.sort(key=lambda item: item[1], reverse=True)
    kept = [item for item in kept if item[1] >= float(keep_score_threshold)]
    if max_keep > 0:
        kept = kept[:max_keep]
    if not kept:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, cam_h, cam_w), dtype=bool),
        )
    keep_idx = np.array([idx for idx, _ in kept], dtype=np.int64)
    local_boxes = boxes[keep_idx].copy()
    local_scores = scores[keep_idx].copy()
    local_masks = masks[keep_idx][:, cam_y : cam_y + cam_h, cam_x : cam_x + cam_w]
    local_boxes[:, [0, 2]] -= cam_x
    local_boxes[:, [1, 3]] -= cam_y
    return local_boxes, local_scores, local_masks


def load_prompt_entries(task_info_path: Path, prompt_image_root: Path) -> list[PromptEntry]:
    """加载提示条目，包含正负提示的图像路径和包围框信息。"""
    payload = load_json(task_info_path)
    assets = payload.get("assets", [])
    if not isinstance(assets, list) or not assets:
        raise ValueError(f"task_info.json does not contain assets: {task_info_path}")
    entries: list[PromptEntry] = []
    for asset in assets:
        semantic_name = semantic_name_from_asset(str(asset["asset_name"]))
        image_records = [{"image_path": asset["image_path"], "bbox_xyxy": asset["bbox_xyxy"]}]
        image_records.extend(
            {
                "image_path": extra["image_path"],
                "bbox_xyxy": extra["bbox_xyxy"],
            }
            for extra in asset.get("extra_views", [])
        )
        for image_record in image_records:
            image_name = str(image_record["image_path"])
            box_xyxy = image_record.get("bbox_xyxy")
            if box_xyxy is None:
                continue
            source_path = (prompt_image_root / image_name).resolve()
            if not source_path.exists():
                raise FileNotFoundError(f"Prompt image not found: {source_path}")
            entries.append(
                PromptEntry(
                    semantic_name=semantic_name,
                    image_name=image_name,
                    source_path=source_path,
                    box_xyxy=[int(value) for value in box_xyxy],
                )
            )
    if not entries:
        raise RuntimeError(f"No usable prompt entries found in {task_info_path}")
    return entries


def split_prompt_entries(entries: list[PromptEntry], target_name: str) -> tuple[list[PromptEntry], list[PromptEntry]]:
    """将提示条目分为正例和负例。"""
    positive = [entry for entry in entries if entry.semantic_name == target_name]
    negative = [entry for entry in entries if entry.semantic_name != target_name]
    if not positive:
        raise ValueError(f"Target semantic {target_name!r} not found in prompt entries")
    return positive, negative


def build_prompt_canvas(
    camera_image: Image.Image,
    camera_source_path: Path,
    prompt_entries: list[PromptEntry],
    ref_cell: int,
    max_cols: int,
    canvas_gap: int,
) -> tuple[Image.Image, dict[str, dict[str, object]]]:
    """构建提示画布，将提示图像拼接到相机图像旁边。"""
    prompt_ids = [entry.image_name for entry in prompt_entries]
    ref_positions = build_prompt_grid_layout(prompt_ids, ref_cell=ref_cell, max_cols=max_cols)
    cols = min(max(max_cols, 1), max(1, int(np.ceil(np.sqrt(len(prompt_entries))))))
    rows = int(np.ceil(len(prompt_entries) / max(cols, 1)))
    ref_panel_w = cols * ref_cell
    ref_panel_h = rows * ref_cell
    canvas_h = max(ref_panel_h, camera_image.height)
    canvas_w = ref_panel_w + max(canvas_gap, 0) + camera_image.width
    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))
    layout: dict[str, dict[str, object]] = {}
    ref_y_offset = (canvas_h - ref_panel_h) // 2
    camera_paste_xy = (ref_panel_w + max(canvas_gap, 0), (canvas_h - camera_image.height) // 2)
    for entry in prompt_entries:
        ref_img = Image.open(entry.source_path).convert("RGB")
        source_size = [ref_img.width, ref_img.height]
        if ref_img.size != (ref_cell, ref_cell):
            ref_img = ref_img.resize((ref_cell, ref_cell))
        x, y = ref_positions[entry.image_name]
        paste_xy = (x, y + ref_y_offset)
        canvas.paste(ref_img, paste_xy)
        layout[entry.image_name] = {
            "kind": "reference",
            "paste_xy": [paste_xy[0], paste_xy[1]],
            "image_size": [ref_img.width, ref_img.height],
            "source_size": source_size,
            "source_path": str(entry.source_path),
            "image_name": entry.image_name,
            "box_xyxy": list(entry.box_xyxy),
        }
    canvas.paste(camera_image, camera_paste_xy)
    layout["__camera__"] = {
        "kind": "camera",
        "paste_xy": [camera_paste_xy[0], camera_paste_xy[1]],
        "image_size": [camera_image.width, camera_image.height],
        "source_path": str(camera_source_path),
    }
    return canvas, layout


def create_canvas_prompts(prompt_entries: list[PromptEntry], layout: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    """根据提示条目和布局信息创建画布提示。"""
    prompts: list[dict[str, object]] = []
    for entry in prompt_entries:
        image_layout = layout[entry.image_name]
        x0, y0, x1, y1 = scale_bbox_to_layout(
            entry.box_xyxy,
            source_size=image_layout["source_size"],
            pasted_size=image_layout["image_size"],
        )
        px, py = image_layout["paste_xy"]
        prompts.append(
            {
                "prompt_image": entry.image_name,
                "box_xywh_canvas": [x0 + px, y0 + py, x1 - x0, y1 - y0],
            }
        )
    return prompts


def autocast_context():
    """自动混合精度上下文管理器（根据可用的 CUDA 设备选择）。"""
    if torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def no_autocast_context(device: torch.device | None = None):
    """禁用自动混合精度的上下文管理器。"""
    if device is not None and device.type == "cuda":
        return torch.autocast("cuda", enabled=False)
    return contextlib.nullcontext()


def maybe_cuda_synchronize(device: torch.device | None, enabled: bool) -> None:
    """在指定设备上执行 CUDA 同步（如果启用）。"""
    if enabled and device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def load_sam3_image_processor(
    checkpoint_path: Path,
    confidence: float,
    mask_threshold: float,
):
    """加载 SAM3 图像处理器。"""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint_path}")
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint_path),
        load_from_HF=False,
        device=resolved_device,
        eval_mode=True,
    )
    try:
        return Sam3Processor(
            model,
            device=resolved_device,
            confidence_threshold=float(confidence),
            mask_threshold=float(mask_threshold),
        )
    except TypeError:
        return Sam3Processor(
            model,
            device=resolved_device,
            confidence_threshold=float(confidence),
        )


def load_video_predictor(
    checkpoint_path: Path,
    *,
    tracker_image_size: int | None = None,
):
    """加载固定的 stitched tracker 视频预测器。"""
    try:
        from single_seg.tracker_only_backend import TrackerOnlyVideoPredictor
    except ImportError:
        from tracker_only_backend import TrackerOnlyVideoPredictor

    return TrackerOnlyVideoPredictor(
        checkpoint_path=checkpoint_path,
        tracker_image_size=tracker_image_size,
    )


def run_single_object_prompt_query(
    *,
    image: Image.Image,
    camera_source_path: Path,
    positive_entries: list[PromptEntry],
    negative_entries: list[PromptEntry],
    keep_score_threshold: float,
    max_masks: int,
    ref_cell: int,
    max_cols: int,
    canvas_gap: int,
    processor,
    debug_canvas_path: Path | None = None,
    debug_prompt_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """运行单次对象提示查询，返回过滤后的目标框、得分和掩码。"""
    selected_entries = list(positive_entries) + list(negative_entries)
    canvas, layout = build_prompt_canvas(
        camera_image=image,
        camera_source_path=camera_source_path,
        prompt_entries=selected_entries,
        ref_cell=max(int(ref_cell), 8),
        max_cols=max(int(max_cols), 1),
        canvas_gap=max(int(canvas_gap), 0),
    )
    positive_prompts = create_canvas_prompts(positive_entries, layout)
    negative_prompts = create_canvas_prompts(negative_entries, layout)
    if debug_canvas_path is not None:
        debug_canvas_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(debug_canvas_path)
    if debug_prompt_path is not None:
        prompt_vis = canvas.copy()
        draw = ImageDraw.Draw(prompt_vis)
        for prompt in positive_prompts:
            x, y, w, h = prompt["box_xywh_canvas"]
            draw.rectangle((x, y, x + w, y + h), outline=(40, 220, 40), width=3)
        for prompt in negative_prompts:
            x, y, w, h = prompt["box_xywh_canvas"]
            draw.rectangle((x, y, x + w, y + h), outline=(220, 40, 40), width=2)
        debug_prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_vis.save(debug_prompt_path)

    state = processor.set_image(canvas)
    processor.reset_all_prompts(state)
    for prompt in positive_prompts:
        state = processor.add_geometric_prompt(
            xywh_to_normalized_cxcywh(prompt["box_xywh_canvas"], canvas.size),
            True,
            state,
        )
    for prompt in negative_prompts:
        state = processor.add_geometric_prompt(
            xywh_to_normalized_cxcywh(prompt["box_xywh_canvas"], canvas.size),
            False,
            state,
        )
    if "masks" not in state or state["masks"] is None:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, image.size[1], image.size[0]), dtype=bool),
        )
    boxes = state["boxes"].float().detach().cpu().numpy()
    scores = state["scores"].float().detach().cpu().numpy()
    masks = state["masks"].detach().cpu().numpy().squeeze(1)
    return filter_predictions_to_camera(
        boxes=boxes,
        scores=scores,
        masks=masks,
        camera_layout=layout["__camera__"],
        keep_score_threshold=float(keep_score_threshold),
        max_keep=int(max_masks),
    )


def select_best_seed_mask(
    boxes: np.ndarray,
    scores: np.ndarray,
    masks: np.ndarray,
    min_pixels: int,
) -> tuple[np.ndarray, float, list[int]] | None:
    """从多个种子掩码中选择最佳者，依据是掩码的像素数和得分。"""
    if boxes.shape[0] == 0:
        return None
    order = sorted(
        range(boxes.shape[0]),
        key=lambda idx: (float(scores[idx]), int(np.count_nonzero(masks[idx]))),
        reverse=True,
    )
    for idx in order:
        mask = np.asarray(masks[idx], dtype=bool)
        if int(np.count_nonzero(mask)) < int(min_pixels):
            continue
        return mask, float(scores[idx]), [int(value) for value in boxes[idx].tolist()]
    return None


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """提取掩码的最大连通组件。"""
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return mask_bool
    try:
        from scipy import ndimage
    except ImportError:
        return mask_bool
    labeled, num = ndimage.label(mask_bool)
    if int(num) <= 1:
        return mask_bool
    sizes = np.bincount(labeled.ravel())[1:]
    if sizes.size == 0:
        return mask_bool
    keep_label = int(np.argmax(sizes)) + 1
    return labeled == keep_label


def refine_seed_mask(
    mask: np.ndarray,
    box_xyxy: list[int],
    *,
    image_shape: tuple[int, int],
    max_area_ratio: float,
    box_margin: int,
    min_pixels: int,
) -> tuple[np.ndarray, str]:
    """精炼种子掩码，确保其在包围框内并符合面积要求。"""
    mask_bool = largest_connected_component(np.asarray(mask, dtype=bool))
    image_area = int(image_shape[0] * image_shape[1])
    mask_area = int(np.count_nonzero(mask_bool))
    if image_area <= 0:
        return mask_bool, "raw"
    if mask_area <= int(image_area * float(max_area_ratio)):
        return mask_bool, "raw_lcc"
    x0, y0, x1, y1 = [int(value) for value in box_xyxy]
    x0 = max(0, x0 - int(box_margin))
    y0 = max(0, y0 - int(box_margin))
    x1 = min(int(image_shape[1]) - 1, x1 + int(box_margin))
    y1 = min(int(image_shape[0]) - 1, y1 + int(box_margin))
    roi_mask = np.zeros_like(mask_bool, dtype=bool)
    roi_mask[y0 : y1 + 1, x0 : x1 + 1] = True
    refined = mask_bool & roi_mask
    refined_cc = largest_connected_component(refined)
    if int(np.count_nonzero(refined_cc)) >= int(min_pixels):
        return refined_cc, "box_refined_lcc"
    if int(np.count_nonzero(refined)) >= int(min_pixels):
        return refined, "box_refined"
    if int(np.count_nonzero(mask_bool)) >= int(min_pixels):
        return mask_bool, "raw_lcc"
    if int(np.count_nonzero(roi_mask)) >= int(min_pixels):
        return roi_mask, "box_fallback"
    return mask_bool, "raw"


def as_numpy(array_like) -> np.ndarray:
    """将数组类对象转换为 NumPy 数组。"""
    if isinstance(array_like, np.ndarray):
        return array_like
    if torch.is_tensor(array_like):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def as_torch(
    array_like,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """将数组类对象转换为 PyTorch 张量。"""
    if torch.is_tensor(array_like):
        tensor = array_like.to(device=device, non_blocking=True)
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        return tensor
    return torch.as_tensor(array_like, dtype=dtype, device=device)


def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid 函数实现。"""
    x = np.asarray(x, dtype=np.float32)
    positive = x >= 0
    out = np.empty_like(x, dtype=np.float32)
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def prob_threshold_to_logit(threshold: float) -> float:
    """将概率阈值转换为对数几率（logit）值。"""
    threshold = float(threshold)
    if threshold <= 0.0:
        return float("-inf")
    if threshold >= 1.0:
        return float("inf")
    return float(math.log(threshold / (1.0 - threshold)))


def build_score_label_map(
    out_obj_ids: np.ndarray,
    out_binary_masks: np.ndarray,
    out_probs: np.ndarray,
    out_tracker_probs: np.ndarray,
    image_shape: tuple[int, int],
    min_object_score: float = 0.0,
    out_mask_logits: np.ndarray | None = None,
    mask_prob_threshold: float = 0.5,
) -> tuple[np.ndarray, dict[int, dict[str, float]]]:
    """根据模型输出构建分数-标签映射，供后续处理使用。"""
    height, width = image_shape
    label_map = np.full((height, width), -1, dtype=np.int32)
    score_map = np.full((height, width), -np.inf, dtype=np.float32)
    object_stats: dict[int, dict[str, float]] = {}
    if out_obj_ids.size == 0:
        return label_map, object_stats
    order = sorted(
        range(out_obj_ids.shape[0]),
        key=lambda idx: (
            float(out_tracker_probs[idx]),
            float(out_probs[idx]),
            int(np.count_nonzero(out_binary_masks[idx])),
        ),
        reverse=True,
    )
    for idx in order:
        obj_id = int(out_obj_ids[idx])
        if out_mask_logits is not None:
            mask_logits = np.asarray(out_mask_logits[idx], dtype=np.float32)
            mask_probs = stable_sigmoid(mask_logits)
            mask = mask_probs >= float(mask_prob_threshold)
        else:
            mask = out_binary_masks[idx].astype(bool)
        if not np.any(mask):
            continue
        score = float(max(out_tracker_probs[idx], out_probs[idx]))
        if score < float(min_object_score):
            continue
        replace = mask & (score > score_map)
        if not np.any(replace):
            continue
        label_map[replace] = obj_id
        score_map[replace] = score
        object_stats[obj_id] = {
            "score": score,
            "seed_score": float(out_probs[idx]),
            "tracker_score": float(out_tracker_probs[idx]),
            "area_pixels": int(np.count_nonzero(mask)),
            "assigned_pixels": int(np.count_nonzero(replace)),
        }
    return label_map, object_stats


def _score_label_map_from_output(
    output: dict[str, object],
    image_shape: tuple[int, int],
    min_object_score: float,
    mask_prob_threshold: float,
) -> dict[str, object]:
    """从模型输出中提取分数-标签映射，供后续处理使用。"""
    out_obj_ids = as_numpy(output["out_obj_ids"])
    out_masks = as_numpy(output["out_binary_masks"])
    out_probs = as_numpy(output["out_probs"])
    out_tracker_probs = as_numpy(output.get("out_tracker_probs", output["out_probs"]))
    out_mask_logits = None
    if output.get("out_mask_logits") is not None:
        out_mask_logits = as_numpy(output["out_mask_logits"])
    label_map, object_stats = build_score_label_map(
        out_obj_ids=out_obj_ids,
        out_binary_masks=out_masks,
        out_probs=out_probs,
        out_tracker_probs=out_tracker_probs,
        image_shape=image_shape,
        min_object_score=min_object_score,
        out_mask_logits=out_mask_logits,
        mask_prob_threshold=mask_prob_threshold,
    )
    return {"label_map": label_map, "object_stats": object_stats}


def _extract_target_mask_from_output_torch(
    output: dict[str, object],
    image_shape: tuple[int, int],
    min_object_score: float,
    mask_prob_threshold: float,
    *,
    target_obj_id: int = 1,
    device: torch.device,
) -> dict[str, object]:
    """从模型输出中提取目标掩码，供后续处理使用（PyTorch 实现）。"""
    height, width = (int(image_shape[0]), int(image_shape[1]))
    empty_mask = torch.zeros((height, width), dtype=torch.bool, device=device)
    out_obj_ids = as_torch(output["out_obj_ids"], device=device, dtype=torch.int64).reshape(-1)
    if out_obj_ids.numel() == 0:
        return {"mask": empty_mask, "score": None, "object_stats": {}}
    out_probs = as_torch(output["out_probs"], device=device, dtype=torch.float32).reshape(-1)
    out_tracker_probs = as_torch(output.get("out_tracker_probs", output["out_probs"]), device=device, dtype=torch.float32).reshape(-1)
    target_matches = torch.nonzero(out_obj_ids == int(target_obj_id), as_tuple=False).flatten()
    if target_matches.numel() == 0:
        return {"mask": empty_mask, "score": None, "object_stats": {}}
    combined_scores = torch.maximum(out_tracker_probs, out_probs)
    if target_matches.numel() > 1:
        best_idx = target_matches[torch.argmax(combined_scores[target_matches])]
    else:
        best_idx = target_matches[0]
    best_idx_i = int(best_idx.item())
    score = float(combined_scores[best_idx_i].item())
    if score < float(min_object_score):
        return {"mask": empty_mask, "score": None, "object_stats": {}}
    if output.get("out_mask_logits") is not None:
        out_mask_logits = as_torch(output["out_mask_logits"], device=device, dtype=torch.float32)
        if out_mask_logits.ndim == 4 and out_mask_logits.shape[1] == 1:
            out_mask_logits = out_mask_logits.squeeze(1)
        mask = out_mask_logits[best_idx_i] >= prob_threshold_to_logit(mask_prob_threshold)
    else:
        out_binary_masks = as_torch(output["out_binary_masks"], device=device, dtype=torch.bool)
        if out_binary_masks.ndim == 4 and out_binary_masks.shape[1] == 1:
            out_binary_masks = out_binary_masks.squeeze(1)
        mask = out_binary_masks[best_idx_i].to(torch.bool)
    if tuple(mask.shape) != (height, width):
        raise ValueError(f"target mask shape mismatch: {tuple(mask.shape)} vs {(height, width)}")
    area_pixels = int(torch.count_nonzero(mask).item())
    if area_pixels == 0:
        return {"mask": empty_mask, "score": None, "object_stats": {}}
    object_stats = {
        int(target_obj_id): {
            "score": score,
            "seed_score": float(out_probs[best_idx_i].item()),
            "tracker_score": float(out_tracker_probs[best_idx_i].item()),
            "area_pixels": area_pixels,
            "assigned_pixels": area_pixels,
        }
    }
    return {"mask": mask, "score": score, "object_stats": object_stats}


def extract_frame_output(
    video_predictor,
    session_id: str,
    frame_idx: int,
    image_shape: tuple[int, int],
    min_object_score: float = 0.0,
    mask_prob_threshold: float = 0.5,
) -> dict[str, object] | None:
    """提取单帧的目标掩码输出，供后续处理使用。"""
    if hasattr(video_predictor, "infer_frame"):
        payload = video_predictor.infer_frame(session_id=session_id, frame_idx=int(frame_idx), reverse=False)
        if int(payload["frame_index"]) == int(frame_idx) and payload["outputs"] is not None:
            return _score_label_map_from_output(
                payload["outputs"],
                image_shape,
                min_object_score,
                mask_prob_threshold,
            )
    start_frame_idx = int(frame_idx)
    max_frame_num_to_track = 0
    if frame_idx > 0:
        start_frame_idx = int(frame_idx) - 1
        max_frame_num_to_track = 1
    stream = video_predictor.propagate_in_video(
        session_id=session_id,
        propagation_direction="forward",
        start_frame_idx=start_frame_idx,
        max_frame_num_to_track=max_frame_num_to_track,
    )
    for payload in stream:
        if int(payload["frame_index"]) == int(frame_idx) and payload["outputs"] is not None:
            return _score_label_map_from_output(
                payload["outputs"],
                image_shape,
                min_object_score,
                mask_prob_threshold,
            )
    return None


def extract_target_mask_output(
    video_predictor,
    session_id: str,
    frame_idx: int,
    image_shape: tuple[int, int],
    min_object_score: float = 0.0,
    mask_prob_threshold: float = 0.5,
    *,
    target_obj_id: int = 1,
    device: torch.device,
) -> dict[str, object] | None:
    """提取单帧的目标掩码输出，供后续处理使用（PyTorch 实现）。"""
    if hasattr(video_predictor, "infer_frame"):
        payload = video_predictor.infer_frame(session_id=session_id, frame_idx=int(frame_idx), reverse=False)
        if int(payload["frame_index"]) == int(frame_idx) and payload["outputs"] is not None:
            return _extract_target_mask_from_output_torch(
                payload["outputs"],
                image_shape,
                min_object_score,
                mask_prob_threshold,
                target_obj_id=target_obj_id,
                device=device,
            )
    start_frame_idx = int(frame_idx)
    max_frame_num_to_track = 0
    if frame_idx > 0:
        start_frame_idx = int(frame_idx) - 1
        max_frame_num_to_track = 1
    stream = video_predictor.propagate_in_video(
        session_id=session_id,
        propagation_direction="forward",
        start_frame_idx=start_frame_idx,
        max_frame_num_to_track=max_frame_num_to_track,
    )
    for payload in stream:
        if int(payload["frame_index"]) == int(frame_idx) and payload["outputs"] is not None:
            return _extract_target_mask_from_output_torch(
                payload["outputs"],
                image_shape,
                min_object_score,
                mask_prob_threshold,
                target_obj_id=target_obj_id,
                device=device,
            )
    return None


def extract_frame_outputs_batch(
    video_predictor,
    requests: list[dict[str, object]],
    min_object_score: float = 0.0,
    mask_prob_threshold: float = 0.5,
) -> dict[str, dict[str, object] | None]:
    """批量提取帧的目标掩码输出，供后续处理使用。"""
    results: dict[str, dict[str, object] | None] = {}
    if hasattr(video_predictor, "infer_frames_batch"):
        payloads = video_predictor.infer_frames_batch(
            [
                {
                    "session_id": request["session_id"],
                    "frame_index": int(request["frame_idx"]),
                    "reverse": False,
                }
                for request in requests
            ]
        )
        for request, payload in zip(requests, payloads):
            camera_id = str(request["camera_id"])
            image_shape = request["image_shape"]
            frame_idx = int(request["frame_idx"])
            if int(payload["frame_index"]) != frame_idx or payload["outputs"] is None:
                results[camera_id] = None
                continue
            results[camera_id] = _score_label_map_from_output(
                payload["outputs"],
                image_shape,
                min_object_score,
                mask_prob_threshold,
            )
        return results
    for request in requests:
        camera_id = str(request["camera_id"])
        results[camera_id] = extract_frame_output(
            video_predictor=video_predictor,
            session_id=str(request["session_id"]),
            frame_idx=int(request["frame_idx"]),
            image_shape=request["image_shape"],
            min_object_score=float(min_object_score),
            mask_prob_threshold=float(mask_prob_threshold),
        )
    return results


def extract_target_mask_outputs_batch(
    video_predictor,
    requests: list[dict[str, object]],
    min_object_score: float = 0.0,
    mask_prob_threshold: float = 0.5,
    *,
    target_obj_id: int = 1,
    device: torch.device,
) -> dict[str, dict[str, object] | None]:
    """批量提取帧的目标掩码输出，供后续处理使用（PyTorch 实现）。"""
    results: dict[str, dict[str, object] | None] = {}
    if hasattr(video_predictor, "infer_frames_batch"):
        payloads = video_predictor.infer_frames_batch(
            [
                {
                    "session_id": request["session_id"],
                    "frame_index": int(request["frame_idx"]),
                    "reverse": False,
                }
                for request in requests
            ]
        )
        for request, payload in zip(requests, payloads):
            camera_id = str(request["camera_id"])
            image_shape = request["image_shape"]
            frame_idx = int(request["frame_idx"])
            if int(payload["frame_index"]) != frame_idx or payload["outputs"] is None:
                results[camera_id] = None
                continue
            results[camera_id] = _extract_target_mask_from_output_torch(
                payload["outputs"],
                image_shape,
                min_object_score,
                mask_prob_threshold,
                target_obj_id=target_obj_id,
                device=device,
            )
        return results
    for request in requests:
        camera_id = str(request["camera_id"])
        results[camera_id] = extract_target_mask_output(
            video_predictor=video_predictor,
            session_id=str(request["session_id"]),
            frame_idx=int(request["frame_idx"]),
            image_shape=request["image_shape"],
            min_object_score=float(min_object_score),
            mask_prob_threshold=float(mask_prob_threshold),
            target_obj_id=int(target_obj_id),
            device=device,
        )
    return results


def backproject_scene_points_with_labels(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    mask: np.ndarray,
    cam2world_gl: np.ndarray,
    intrinsics: dict[str, float] | None,
    fovy_deg: float | None,
    depth_min: float,
    depth_max: float,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """根据 RGB 图像、深度图和掩码反投影场景点，并返回 3D 坐标和颜色。"""
    height, width = depth_m.shape
    if intrinsics is not None:
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
    else:
        if fovy_deg is None:
            raise ValueError("Either intrinsics or fovy_deg must be provided")
        fy = 0.5 * height / np.tan(np.deg2rad(float(fovy_deg)) * 0.5)
        fx = fy
        cx = (width - 1) * 0.5
        cy = (height - 1) * 0.5
    sampled_depth = depth_m[::stride, ::stride]
    sampled_mask = mask[::stride, ::stride]
    valid = np.isfinite(sampled_depth) & (sampled_depth > depth_min) & (sampled_depth < depth_max)
    if not np.any(valid):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            np.empty((0,), dtype=np.int32),
        )
    v = np.arange(0, int(height), int(stride), dtype=np.float32)
    u = np.arange(0, int(width), int(stride), dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    depth = sampled_depth[valid]
    x_cv = ((uu - float(cx)) / float(fx))[valid] * depth
    y_cv = ((vv - float(cy)) / float(fy))[valid] * depth
    z_cv = depth
    pts_cv = np.stack([x_cv, y_cv, z_cv], axis=1)
    pts_gl = pts_cv * np.array([1.0, -1.0, -1.0], dtype=np.float32)[None, :]
    pts_gl_h = np.concatenate([pts_gl, np.ones((pts_gl.shape[0], 1), dtype=np.float32)], axis=1)
    pts_world = (cam2world_gl.astype(np.float32) @ pts_gl_h.T).T[:, :3]
    colors = rgb[::stride, ::stride][valid]
    labels = sampled_mask[valid].astype(np.int32, copy=False)
    return pts_world.astype(np.float32), colors.astype(np.uint8), labels


def backproject_scene_points_with_labels_torch(
    sampled_rgb: np.ndarray | torch.Tensor,
    sampled_depth_m: np.ndarray | torch.Tensor,
    sampled_mask: np.ndarray | torch.Tensor,
    cam2world_gl: np.ndarray,
    x_scale: torch.Tensor,
    y_scale: torch.Tensor,
    depth_min: float,
    depth_max: float,
    *,
    device: torch.device,
    target_id: int = 1,
    camera_score: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """根据 RGB 图像、深度图和掩码反投影场景点（PyTorch 实现）。返回 points, colors, labels, scores。"""
    with no_autocast_context(device):
        if torch.is_tensor(sampled_depth_m):
            depth = sampled_depth_m.to(device=device, dtype=torch.float32, non_blocking=True)
        else:
            depth = torch.as_tensor(np.ascontiguousarray(sampled_depth_m), dtype=torch.float32, device=device)
        if torch.is_tensor(sampled_mask):
            mask = sampled_mask.to(device=device, dtype=torch.bool, non_blocking=True)
        else:
            mask = torch.as_tensor(np.ascontiguousarray(sampled_mask), dtype=torch.bool, device=device)
        valid = torch.isfinite(depth) & (depth > float(depth_min)) & (depth < float(depth_max))
        if not bool(valid.any().item()):
            return (
                torch.empty((0, 3), dtype=torch.float32, device=device),
                torch.empty((0, 3), dtype=torch.uint8, device=device),
                torch.empty((0,), dtype=torch.int32, device=device),
                torch.empty((0,), dtype=torch.float32, device=device),
            )
        depth_valid = depth[valid]
        x_cv = x_scale.to(torch.float32)[valid] * depth_valid
        y_cv = y_scale.to(torch.float32)[valid] * depth_valid
        z_cv = depth_valid
        pts_cv = torch.stack([x_cv, y_cv, z_cv], dim=1)
        pts_gl = pts_cv * torch.tensor([1.0, -1.0, -1.0], dtype=torch.float32, device=device)[None, :]
        pts_gl_h = torch.cat(
            [
                pts_gl,
                torch.ones((pts_gl.shape[0], 1), dtype=torch.float32, device=device),
            ],
            dim=1,
        )
        cam2world = torch.as_tensor(np.asarray(cam2world_gl, dtype=np.float32), dtype=torch.float32, device=device)
        pts_world = (cam2world @ pts_gl_h.T).T[:, :3]
        if torch.is_tensor(sampled_rgb):
            colors_src = sampled_rgb.to(device=device, dtype=torch.uint8, non_blocking=True)
        else:
            colors_src = torch.as_tensor(np.ascontiguousarray(sampled_rgb), dtype=torch.uint8, device=device)
        colors = colors_src[valid]
        labels = mask[valid].to(torch.int32) * target_id
        scores = torch.full((labels.shape[0],), camera_score, dtype=torch.float32, device=device)
        return pts_world, colors, labels, scores


def erode_binary_mask_torch(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Erode a 2D boolean mask with an ellipse-like kernel on the mask device."""
    kernel = int(kernel_size)
    if kernel <= 1:
        return mask.to(dtype=torch.bool)
    if kernel % 2 == 0:
        kernel += 1
    mask_bool = mask.to(dtype=torch.bool)
    if mask_bool.ndim != 2:
        raise ValueError(f"mask erosion expects a 2D mask, got shape={tuple(mask_bool.shape)}")
    radius = kernel // 2
    yy, xx = torch.meshgrid(
        torch.arange(kernel, dtype=torch.float32, device=mask_bool.device),
        torch.arange(kernel, dtype=torch.float32, device=mask_bool.device),
        indexing="ij",
    )
    center = float(radius)
    if radius <= 0:
        return mask_bool
    footprint = (((xx - center) / float(radius)) ** 2 + ((yy - center) / float(radius)) ** 2) <= 1.0
    footprint_f = footprint.to(dtype=torch.float32)
    weight = footprint_f.reshape(1, 1, kernel, kernel)
    mask_f = mask_bool.to(dtype=torch.float32).reshape(1, 1, *mask_bool.shape)
    counts = torch.nn.functional.conv2d(mask_f, weight, padding=radius).reshape_as(mask_bool)
    return counts >= footprint_f.sum()


def filter_target_mask_by_depth_band(
    mask: np.ndarray,
    depth_m: np.ndarray,
    *,
    enabled: bool,
    range_m: float,
    min_valid_pixels: int,
    min_keep_pixels: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """按目标 mask 内有效深度的中位数做深度带过滤，优先减少背景误点。"""
    mask_bool = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = mask_bool & np.isfinite(depth) & (depth > 0.0)
    valid_pixels = int(np.count_nonzero(valid))
    summary: dict[str, object] = {
        "enabled": bool(enabled),
        "backend": "numpy",
        "range_m": float(range_m),
        "min_valid_pixels": int(min_valid_pixels),
        "min_keep_pixels": int(min_keep_pixels),
        "target_pixels_before": int(np.count_nonzero(mask_bool)),
        "valid_depth_pixels": valid_pixels,
        "target_pixels_after": int(np.count_nonzero(mask_bool)),
        "removed_target_pixels": 0,
        "center_depth_m": None,
        "applied": False,
        "skipped_reason": None,
    }
    if not enabled:
        return mask_bool, summary
    if valid_pixels < max(int(min_valid_pixels), 1):
        summary["skipped_reason"] = "not_enough_valid_depth_pixels"
        return mask_bool, summary
    depth_center = float(np.median(depth[valid]))
    keep = valid & (np.abs(depth - depth_center) <= max(float(range_m), 0.0))
    keep_pixels = int(np.count_nonzero(keep))
    summary["center_depth_m"] = depth_center
    if keep_pixels < max(int(min_keep_pixels), 1):
        summary["skipped_reason"] = "not_enough_kept_pixels"
        return mask_bool, summary
    summary["target_pixels_after"] = keep_pixels
    summary["removed_target_pixels"] = int(max(summary["target_pixels_before"] - keep_pixels, 0))
    summary["applied"] = True
    return keep, summary


def filter_target_mask_by_depth_band_torch(
    mask: torch.Tensor,
    depth_m: torch.Tensor,
    *,
    enabled: bool,
    range_m: float,
    min_valid_pixels: int,
    min_keep_pixels: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    """按目标 mask 内有效深度的中位数做深度带过滤（torch 实现）。"""
    mask_bool = mask.to(dtype=torch.bool)
    depth = depth_m.to(dtype=torch.float32)
    valid = mask_bool & torch.isfinite(depth) & (depth > 0.0)
    valid_pixels = int(torch.count_nonzero(valid).item())
    target_pixels_before = int(torch.count_nonzero(mask_bool).item())
    summary: dict[str, object] = {
        "enabled": bool(enabled),
        "backend": "torch",
        "range_m": float(range_m),
        "min_valid_pixels": int(min_valid_pixels),
        "min_keep_pixels": int(min_keep_pixels),
        "target_pixels_before": target_pixels_before,
        "valid_depth_pixels": valid_pixels,
        "target_pixels_after": target_pixels_before,
        "removed_target_pixels": 0,
        "center_depth_m": None,
        "applied": False,
        "skipped_reason": None,
    }
    if not enabled:
        return mask_bool, summary
    if valid_pixels < max(int(min_valid_pixels), 1):
        summary["skipped_reason"] = "not_enough_valid_depth_pixels"
        return mask_bool, summary
    with no_autocast_context(depth.device):
        center = torch.median(depth[valid])
        keep = valid & (torch.abs(depth - center) <= max(float(range_m), 0.0))
        keep_pixels = int(torch.count_nonzero(keep).item())
    summary["center_depth_m"] = float(center.item())
    if keep_pixels < max(int(min_keep_pixels), 1):
        summary["skipped_reason"] = "not_enough_kept_pixels"
        return mask_bool, summary
    summary["target_pixels_after"] = keep_pixels
    summary["removed_target_pixels"] = int(max(target_pixels_before - keep_pixels, 0))
    summary["applied"] = True
    return keep, summary


def fuse_scene_geometry(
    point_chunks: list[np.ndarray],
    color_chunks: list[np.ndarray],
    label_chunks: list[np.ndarray],
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """融合场景几何信息，生成下采样后的点云、颜色和标签。"""
    points = np.concatenate(point_chunks, axis=0)
    colors = np.concatenate(color_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0).astype(np.int32, copy=False)
    if points.shape[0] == 0 or float(voxel_size) <= 0.0:
        return points.astype(np.float32), colors.astype(np.uint8), labels

    voxel_keys = np.floor(points / float(voxel_size)).astype(np.int64, copy=False)
    voxel_keys -= voxel_keys.min(axis=0, keepdims=True)
    spans = voxel_keys.max(axis=0).astype(np.int64, copy=False) + 1
    hashed = voxel_keys[:, 0].astype(np.int64, copy=False)
    hashed += spans[0] * voxel_keys[:, 1].astype(np.int64, copy=False)
    hashed += spans[0] * spans[1] * voxel_keys[:, 2].astype(np.int64, copy=False)

    order = np.argsort(hashed, kind="mergesort")
    hashed_sorted = hashed[order]
    group_starts = np.concatenate(
        [
            np.array([0], dtype=np.int64),
            np.flatnonzero(np.diff(hashed_sorted)) + 1,
        ]
    )
    counts = np.diff(
        np.concatenate(
            [
                group_starts,
                np.array([hashed_sorted.shape[0]], dtype=np.int64),
            ]
        )
    ).astype(np.int64, copy=False)

    points_sorted = points[order].astype(np.float64, copy=False)
    colors_sorted = colors[order].astype(np.float64, copy=False)
    labels_sorted = labels[order].astype(np.int32, copy=False)
    point_sum = np.add.reduceat(points_sorted, group_starts, axis=0)
    color_sum = np.add.reduceat(colors_sorted, group_starts, axis=0)
    label_max = np.maximum.reduceat(labels_sorted, group_starts)

    counts_f = counts[:, None].astype(np.float64, copy=False)
    down_points = (point_sum / counts_f).astype(np.float32, copy=False)
    down_colors = np.clip(np.rint(color_sum / counts_f), 0.0, 255.0).astype(np.uint8, copy=False)
    return down_points, down_colors, label_max.astype(np.int32, copy=False)


def fuse_scene_geometry_torch(
    point_chunks: list[torch.Tensor],
    color_chunks: list[torch.Tensor],
    label_chunks: list[torch.Tensor],
    voxel_size: float,
    *,
    device: torch.device,
    score_chunks: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """融合场景几何信息（PyTorch 实现），生成下采样后的点云、颜色、标签和置信度（求和）。"""
    points = torch.cat(point_chunks, dim=0)
    colors = torch.cat(color_chunks, dim=0)
    labels = torch.cat(label_chunks, dim=0).to(torch.int32)
    has_scores = score_chunks is not None and len(score_chunks) > 0
    scores: torch.Tensor | None = torch.cat(score_chunks, dim=0).to(torch.float32) if has_scores else None
    if points.numel() == 0 or float(voxel_size) <= 0.0:
        return points.to(torch.float32), colors.to(torch.uint8), labels, scores

    with no_autocast_context(device):
        voxel_keys = torch.floor(points.to(torch.float64) / float(voxel_size)).to(torch.int64)
        voxel_keys = voxel_keys - voxel_keys.min(dim=0, keepdim=True).values
        unique_keys, inverse = torch.unique(voxel_keys, dim=0, return_inverse=True)
        num_groups = int(unique_keys.shape[0])
        counts = torch.bincount(inverse, minlength=num_groups).to(torch.float32)
        point_sum = torch.zeros((num_groups, 3), dtype=torch.float32, device=device)
        color_sum = torch.zeros((num_groups, 3), dtype=torch.float32, device=device)
        point_sum.scatter_add_(0, inverse[:, None].expand(-1, 3), points.to(torch.float32))
        color_sum.scatter_add_(0, inverse[:, None].expand(-1, 3), colors.to(torch.float32))
        label_max = torch.full(
            (num_groups,),
            torch.iinfo(torch.int32).min,
            dtype=torch.int32,
            device=device,
        )
        label_max.scatter_reduce_(0, inverse, labels, reduce="amax", include_self=True)
        down_points = point_sum / counts[:, None]
        down_colors = torch.clamp(torch.round(color_sum / counts[:, None]), 0, 255).to(torch.uint8)
        down_scores: torch.Tensor | None = None
        if scores is not None:
            score_sum = torch.zeros((num_groups,), dtype=torch.float32, device=device)
            score_sum.scatter_add_(0, inverse, scores)
            down_scores = score_sum
        return down_points, down_colors, label_max, down_scores


def dbscan_labels_from_points(points: np.ndarray, radius_m: float, min_points: int) -> np.ndarray:
    """用网格邻域查询对 3D 点执行轻量 DBSCAN，返回每个点的簇 ID，噪声为 -1。"""
    points = np.asarray(points, dtype=np.float32)
    num_points = int(points.shape[0])
    labels = np.full((num_points,), -2, dtype=np.int32)
    if num_points == 0:
        return labels
    radius = float(radius_m)
    if radius <= 0.0:
        return np.full((num_points,), -1, dtype=np.int32)
    min_points = max(int(min_points), 1)
    radius_sq = radius * radius
    voxel_keys = np.floor(points / radius).astype(np.int64, copy=False)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(voxel_keys):
        buckets.setdefault((int(key[0]), int(key[1]), int(key[2])), []).append(index)

    neighbor_offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ]

    def region_query(point_index: int) -> np.ndarray:
        key = voxel_keys[point_index]
        candidates: list[int] = []
        for dx, dy, dz in neighbor_offsets:
            candidates.extend(
                buckets.get(
                    (int(key[0]) + dx, int(key[1]) + dy, int(key[2]) + dz),
                    [],
                )
            )
        candidate_ids = np.asarray(candidates, dtype=np.int64)
        if candidate_ids.size == 0:
            return candidate_ids
        delta = points[candidate_ids] - points[point_index]
        return candidate_ids[np.einsum("ij,ij->i", delta, delta) <= radius_sq]

    cluster_id = 0
    for point_index in range(num_points):
        if labels[point_index] != -2:
            continue
        neighbors = region_query(point_index)
        if int(neighbors.size) < min_points:
            labels[point_index] = -1
            continue
        labels[point_index] = cluster_id
        queued = np.zeros((num_points,), dtype=bool)
        seeds = []
        for value in neighbors:
            seed_value = int(value)
            if seed_value == point_index:
                continue
            seeds.append(seed_value)
            queued[seed_value] = True
        seed_cursor = 0
        while seed_cursor < len(seeds):
            seed_index = seeds[seed_cursor]
            seed_cursor += 1
            if labels[seed_index] == -1:
                labels[seed_index] = cluster_id
            if labels[seed_index] != -2:
                continue
            labels[seed_index] = cluster_id
            seed_neighbors = region_query(seed_index)
            if int(seed_neighbors.size) >= min_points:
                for neighbor_index in seed_neighbors:
                    neighbor_int = int(neighbor_index)
                    if labels[neighbor_int] in {-2, -1} and not bool(queued[neighbor_int]):
                        seeds.append(neighbor_int)
                        queued[neighbor_int] = True
        cluster_id += 1
    labels[labels == -2] = -1
    return labels


def filter_target_labels_by_3d_clusters(
    points_xyz: np.ndarray,
    labels: np.ndarray,
    *,
    enabled: bool,
    radius_m: float,
    min_points: int,
    keep_largest: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    """按 3D 聚类过滤目标标签，散点会被改为背景标签 0。"""
    labels_out = np.asarray(labels, dtype=np.int32).copy()
    target_mask = labels_out > 0
    target_count = int(np.count_nonzero(target_mask))
    summary: dict[str, object] = {
        "enabled": bool(enabled),
        "backend": "numpy",
        "radius_m": float(radius_m),
        "min_points": int(min_points),
        "keep_largest": bool(keep_largest),
        "target_points_before": target_count,
        "target_points_after": target_count,
        "removed_target_points": 0,
        "num_clusters": 0,
        "cluster_sizes": [],
    }
    if not enabled or target_count == 0:
        return labels_out, summary

    target_points = np.asarray(points_xyz, dtype=np.float32)[target_mask]
    cluster_labels = dbscan_labels_from_points(
        target_points,
        radius_m=float(radius_m),
        min_points=int(min_points),
    )
    valid_cluster_ids, cluster_sizes = np.unique(cluster_labels[cluster_labels >= 0], return_counts=True)
    summary["num_clusters"] = int(valid_cluster_ids.shape[0])
    summary["cluster_sizes"] = [int(value) for value in cluster_sizes.tolist()]
    keep_target = np.zeros((target_count,), dtype=bool)
    if valid_cluster_ids.size > 0:
        if keep_largest:
            largest_cluster_id = int(valid_cluster_ids[int(np.argmax(cluster_sizes))])
            keep_target = cluster_labels == largest_cluster_id
        else:
            keep_target = cluster_labels >= 0
    target_indices = np.flatnonzero(target_mask)
    labels_out[target_indices[~keep_target]] = 0
    target_after = int(np.count_nonzero(labels_out > 0))
    summary["target_points_after"] = target_after
    summary["removed_target_points"] = target_count - target_after
    return labels_out, summary


def fit_dominant_plane_ransac(
    points: np.ndarray,
    *,
    distance_m: float,
    min_points: int,
    num_iterations: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """用确定性 RANSAC 拟合主平面，返回 plane=[nx,ny,nz,d] 和内点 mask。"""
    points_f = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    num_points = int(points_f.shape[0])
    if num_points < max(int(min_points), 3):
        return None
    threshold = max(float(distance_m), 0.0)
    if threshold <= 0.0:
        return None

    rng = np.random.default_rng(int(seed))
    best_plane: np.ndarray | None = None
    best_inliers: np.ndarray | None = None
    best_count = -1
    best_mean_distance = float("inf")
    iterations = max(int(num_iterations), 1)
    for _ in range(iterations):
        ids = rng.choice(num_points, size=3, replace=False)
        p0, p1, p2 = points_f[ids]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-8:
            continue
        normal = normal / norm
        d = -float(np.dot(normal, p0))
        distances = np.abs(points_f @ normal + d)
        inliers = distances <= threshold
        count = int(np.count_nonzero(inliers))
        if count <= 0:
            continue
        mean_distance = float(np.mean(distances[inliers]))
        if count > best_count or (count == best_count and mean_distance < best_mean_distance):
            best_plane = np.asarray([normal[0], normal[1], normal[2], d], dtype=np.float32)
            best_inliers = inliers
            best_count = count
            best_mean_distance = mean_distance

    if best_plane is None or best_inliers is None or best_count < max(int(min_points), 3):
        return None

    inlier_points = points_f[best_inliers]
    centroid = inlier_points.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_points - centroid, full_matrices=False)
    normal = vh[-1].astype(np.float32, copy=False)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-8:
        return best_plane, best_inliers
    normal = normal / norm
    d = -float(np.dot(normal, centroid))
    refined_plane = np.asarray([normal[0], normal[1], normal[2], d], dtype=np.float32)
    refined_inliers = np.abs(points_f @ normal + d) <= threshold
    if int(np.count_nonzero(refined_inliers)) >= max(int(min_points), 3):
        return refined_plane, refined_inliers
    return best_plane, best_inliers


def filter_target_labels_by_dominant_plane(
    points_xyz: np.ndarray,
    labels: np.ndarray,
    *,
    enabled: bool,
    distance_m: float,
    min_points: int,
    min_inlier_ratio: float,
    max_inlier_ratio: float,
    max_planes: int,
    ransac_iterations: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """按目标点中的主平面过滤标签，常用于去掉被 2D mask 扫进来的桌面点。"""
    labels_out = np.asarray(labels, dtype=np.int32).copy()
    target_mask = labels_out > 0
    target_indices = np.flatnonzero(target_mask)
    target_count = int(target_indices.shape[0])
    summary: dict[str, object] = {
        "enabled": bool(enabled),
        "backend": "numpy_ransac",
        "distance_m": float(distance_m),
        "min_points": int(min_points),
        "min_inlier_ratio": float(min_inlier_ratio),
        "max_inlier_ratio": float(max_inlier_ratio),
        "max_planes": int(max_planes),
        "ransac_iterations": int(ransac_iterations),
        "target_points_before": target_count,
        "target_points_after": target_count,
        "removed_target_points": 0,
        "plane_found": False,
        "plane_applied": False,
        "plane": None,
        "plane_inliers": 0,
        "plane_inlier_ratio": 0.0,
        "planes": [],
        "skipped_reason": None,
    }
    if not enabled or target_count == 0:
        return labels_out, summary
    min_points_i = max(int(min_points), 3)
    if target_count < min_points_i:
        summary["skipped_reason"] = "not_enough_target_points"
        return labels_out, summary
    points_all = np.asarray(points_xyz, dtype=np.float32)
    remaining_indices = target_indices.copy()
    planes: list[dict[str, object]] = []
    max_planes_i = max(int(max_planes), 1)
    for plane_index in range(max_planes_i):
        remaining_count = int(remaining_indices.shape[0])
        if remaining_count < min_points_i:
            if not planes:
                summary["skipped_reason"] = "not_enough_target_points"
            break
        target_points = points_all[remaining_indices]
        fitted = fit_dominant_plane_ransac(
            target_points,
            distance_m=float(distance_m),
            min_points=min_points_i,
            num_iterations=int(ransac_iterations),
            seed=int(plane_index),
        )
        if fitted is None:
            if not planes:
                summary["skipped_reason"] = "no_plane"
            break
        plane, inlier_mask = fitted
        inlier_count = int(np.count_nonzero(inlier_mask))
        inlier_ratio = float(inlier_count / max(remaining_count, 1))
        plane_record = {
            "index": int(plane_index),
            "plane": [float(value) for value in plane.tolist()],
            "inliers": inlier_count,
            "inlier_ratio": inlier_ratio,
            "applied": False,
            "skipped_reason": None,
        }
        summary["plane_found"] = True
        if summary["plane"] is None:
            summary["plane"] = plane_record["plane"]
            summary["plane_inliers"] = inlier_count
            summary["plane_inlier_ratio"] = inlier_ratio
        if inlier_count < min_points_i:
            plane_record["skipped_reason"] = "not_enough_plane_inliers"
            if not planes:
                summary["skipped_reason"] = plane_record["skipped_reason"]
            break
        if inlier_ratio < float(min_inlier_ratio):
            plane_record["skipped_reason"] = "inlier_ratio_too_low"
            if not planes:
                summary["skipped_reason"] = plane_record["skipped_reason"]
            break
        if inlier_ratio > float(max_inlier_ratio):
            plane_record["skipped_reason"] = "inlier_ratio_too_high"
            if not planes:
                summary["skipped_reason"] = plane_record["skipped_reason"]
            break

        labels_out[remaining_indices[inlier_mask]] = 0
        plane_record["applied"] = True
        planes.append(plane_record)
        remaining_indices = remaining_indices[~inlier_mask]

    target_after = int(np.count_nonzero(labels_out > 0))
    summary["target_points_after"] = target_after
    summary["removed_target_points"] = int(target_count - target_after)
    summary["plane_applied"] = bool(planes)
    summary["planes"] = planes
    return labels_out, summary


def dbscan_labels_from_points_torch(points: torch.Tensor, radius_m: float, min_points: int) -> torch.Tensor:
    """用 torch 半径图对 3D 点执行 DBSCAN，返回每个点的簇 ID，噪声为 -1。"""
    points = points.reshape(-1, 3)
    num_points = int(points.shape[0])
    device = points.device
    if num_points == 0:
        return torch.empty((0,), dtype=torch.int32, device=device)
    radius = float(radius_m)
    if radius <= 0.0:
        return torch.full((num_points,), -1, dtype=torch.int32, device=device)
    min_points = max(int(min_points), 1)
    with no_autocast_context(device):
        points_f = points.to(dtype=torch.float32)
        point_norm = torch.sum(points_f * points_f, dim=1, keepdim=True)
        dist_sq = point_norm + point_norm.T - 2.0 * (points_f @ points_f.T)
        adjacency = dist_sq <= (radius * radius)
        adjacency.fill_diagonal_(True)
        core_mask = torch.sum(adjacency, dim=1) >= int(min_points)
        core_indices = torch.nonzero(core_mask, as_tuple=False).flatten()
        num_core = int(core_indices.shape[0])
        labels = torch.full((num_points,), -1, dtype=torch.int32, device=device)
        if num_core == 0:
            return labels

        core_adjacency = adjacency[core_indices][:, core_indices]
        core_labels = torch.full((num_core,), -1, dtype=torch.int32, device=device)
        unvisited = torch.ones((num_core,), dtype=torch.bool, device=device)
        cluster_id = 0
        while bool(torch.any(unvisited).item()):
            seed = int(torch.nonzero(unvisited, as_tuple=False)[0].item())
            component = torch.zeros((num_core,), dtype=torch.bool, device=device)
            component[seed] = True
            while True:
                expanded = torch.any(core_adjacency[component], dim=0) | component
                if bool(torch.equal(expanded, component)):
                    break
                component = expanded
            core_labels[component] = int(cluster_id)
            unvisited[component] = False
            cluster_id += 1

        labels[core_indices] = core_labels
        border_mask = ~core_mask
        if bool(torch.any(border_mask).item()):
            border_indices = torch.nonzero(border_mask, as_tuple=False).flatten()
            border_to_core = adjacency[border_indices][:, core_indices]
            has_core_neighbor = torch.any(border_to_core, dim=1)
            if bool(torch.any(has_core_neighbor).item()):
                large_label = torch.full(
                    (1,),
                    int(cluster_id),
                    dtype=torch.int32,
                    device=device,
                )
                neighbor_labels = torch.where(
                    border_to_core,
                    core_labels[None, :],
                    large_label,
                )
                border_labels = torch.min(neighbor_labels, dim=1).values
                labels[border_indices[has_core_neighbor]] = border_labels[has_core_neighbor]
        return labels


def filter_target_labels_by_3d_clusters_torch(
    points_xyz: torch.Tensor,
    labels: torch.Tensor,
    *,
    enabled: bool,
    radius_m: float,
    min_points: int,
    keep_largest: bool,
) -> tuple[torch.Tensor, dict[str, object]]:
    """按 3D 聚类过滤目标标签（torch 实现），散点会被改为背景标签 0。"""
    labels_out = labels.to(dtype=torch.int32).clone()
    target_mask = labels_out > 0
    target_count = int(torch.count_nonzero(target_mask).item())
    summary: dict[str, object] = {
        "enabled": bool(enabled),
        "backend": "torch",
        "radius_m": float(radius_m),
        "min_points": int(min_points),
        "keep_largest": bool(keep_largest),
        "target_points_before": target_count,
        "target_points_after": target_count,
        "removed_target_points": 0,
        "num_clusters": 0,
        "cluster_sizes": [],
    }
    if not enabled or target_count == 0:
        return labels_out, summary

    target_indices = torch.nonzero(target_mask, as_tuple=False).flatten()
    target_points = points_xyz[target_indices]
    if keep_largest:
        radius = float(radius_m)
        if radius <= 0.0:
            keep_target = torch.zeros((target_count,), dtype=torch.bool, device=labels_out.device)
            target_after = 0
        else:
            with no_autocast_context(points_xyz.device):
                target_points_f = target_points.to(dtype=torch.float32)
                point_norm = torch.sum(target_points_f * target_points_f, dim=1, keepdim=True)
                adjacency = point_norm + point_norm.T - 2.0 * (target_points_f @ target_points_f.T)
                adjacency = adjacency <= (radius * radius)
                adjacency.fill_diagonal_(True)
                neighbor_counts = torch.sum(adjacency, dim=1)
                core_mask = neighbor_counts >= max(int(min_points), 1)
                core_indices = torch.nonzero(core_mask, as_tuple=False).flatten()
                if int(core_indices.shape[0]) == 0:
                    keep_target = torch.zeros((target_count,), dtype=torch.bool, device=labels_out.device)
                    target_after = 0
                else:
                    core_adjacency = adjacency[core_indices][:, core_indices].to(dtype=torch.float32)
                    seed = int(torch.argmax(neighbor_counts[core_indices]).item())
                    component_score = torch.zeros(
                        (int(core_indices.shape[0]),),
                        dtype=torch.float32,
                        device=labels_out.device,
                    )
                    component_score[seed] = 1.0
                    for _ in range(DEFAULT_TORCH_CLUSTER_EXPANSION_STEPS):
                        component_score = torch.clamp(core_adjacency @ component_score, max=1.0)
                    component = component_score > 0.0
                    keep_target = torch.zeros((target_count,), dtype=torch.bool, device=labels_out.device)
                    kept_core_indices = core_indices[component]
                    keep_target[kept_core_indices] = True
                    border_indices = torch.nonzero(~core_mask, as_tuple=False).flatten()
                    if int(border_indices.shape[0]) > 0 and int(kept_core_indices.shape[0]) > 0:
                        keep_target[border_indices] = torch.any(
                            adjacency[border_indices][:, kept_core_indices],
                            dim=1,
                        )
                    target_after = int(torch.count_nonzero(keep_target).item())
        summary["num_clusters"] = 1 if target_after > 0 else 0
        summary["cluster_sizes"] = [target_after] if target_after > 0 else []
        summary["largest_only_fast_path"] = True
        summary["expansion_steps"] = DEFAULT_TORCH_CLUSTER_EXPANSION_STEPS
        labels_out[target_indices[~keep_target]] = 0
        summary["target_points_after"] = target_after
        summary["removed_target_points"] = target_count - target_after
        return labels_out, summary

    cluster_labels = dbscan_labels_from_points_torch(
        target_points,
        radius_m=float(radius_m),
        min_points=int(min_points),
    )
    valid = cluster_labels >= 0
    if bool(torch.any(valid).item()):
        cluster_sizes_t = torch.bincount(cluster_labels[valid].to(torch.int64))
        keep_target = valid.clone()
        if keep_largest:
            keep_cluster = int(torch.argmax(cluster_sizes_t).item())
            keep_target = cluster_labels == keep_cluster
        target_after = int(torch.count_nonzero(keep_target).item())
        summary["num_clusters"] = int(cluster_sizes_t.shape[0])
        summary["cluster_sizes"] = [int(value) for value in cluster_sizes_t.detach().cpu().tolist()]
    else:
        keep_target = torch.zeros((target_count,), dtype=torch.bool, device=labels_out.device)
        target_after = 0
    labels_out[target_indices[~keep_target]] = 0
    summary["target_points_after"] = target_after
    summary["removed_target_points"] = target_count - target_after
    return labels_out, summary


def filter_target_labels_by_dominant_plane_torch(
    points_xyz: torch.Tensor,
    labels: torch.Tensor,
    *,
    enabled: bool,
    distance_m: float,
    min_points: int,
    min_inlier_ratio: float,
    max_inlier_ratio: float,
    max_planes: int,
    ransac_iterations: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    """按目标点中的主平面过滤标签（torch 输入，CPU RANSAC 拟合）。"""
    labels_out = labels.to(dtype=torch.int32).clone()
    target_mask = labels_out > 0
    target_indices = torch.nonzero(target_mask, as_tuple=False).flatten()
    target_count = int(target_indices.shape[0])
    summary: dict[str, object] = {
        "enabled": bool(enabled),
        "backend": "torch_cpu_ransac",
        "distance_m": float(distance_m),
        "min_points": int(min_points),
        "min_inlier_ratio": float(min_inlier_ratio),
        "max_inlier_ratio": float(max_inlier_ratio),
        "max_planes": int(max_planes),
        "ransac_iterations": int(ransac_iterations),
        "target_points_before": target_count,
        "target_points_after": target_count,
        "removed_target_points": 0,
        "plane_found": False,
        "plane_applied": False,
        "plane": None,
        "plane_inliers": 0,
        "plane_inlier_ratio": 0.0,
        "planes": [],
        "skipped_reason": None,
    }
    if not enabled or target_count == 0:
        return labels_out, summary

    target_points_np = points_xyz[target_indices].detach().cpu().numpy().astype(np.float32, copy=False)
    target_labels_np = np.ones((target_count,), dtype=np.int32)
    filtered_target_labels, np_summary = filter_target_labels_by_dominant_plane(
        target_points_np,
        target_labels_np,
        enabled=True,
        distance_m=float(distance_m),
        min_points=int(min_points),
        min_inlier_ratio=float(min_inlier_ratio),
        max_inlier_ratio=float(max_inlier_ratio),
        max_planes=int(max_planes),
        ransac_iterations=int(ransac_iterations),
    )
    summary.update(np_summary)
    summary["backend"] = "torch_cpu_ransac"
    keep_target = torch.as_tensor(filtered_target_labels > 0, dtype=torch.bool, device=labels_out.device)
    labels_out[target_indices[~keep_target]] = 0
    return labels_out, summary


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """将点云数据写入 .ply 文件。"""
    if points.shape[0] != colors.shape[0]:
        raise ValueError("points and colors must have the same length")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    verts = np.empty(
        points.shape[0],
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    verts["x"], verts["y"], verts["z"] = points[:, 0], points[:, 1], points[:, 2]
    verts["red"], verts["green"], verts["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(verts.tobytes())


def write_ply_with_normals(path: Path, points: np.ndarray, colors: np.ndarray, normals: np.ndarray) -> None:
    """将带法线和颜色的点云数据写入 .ply 文件。"""
    if points.shape[0] != colors.shape[0] or points.shape[0] != normals.shape[0]:
        raise ValueError("points, colors, and normals must have the same length")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    verts = np.empty(
        points.shape[0],
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("nx", "<f4"),
            ("ny", "<f4"),
            ("nz", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    verts["x"], verts["y"], verts["z"] = points[:, 0], points[:, 1], points[:, 2]
    verts["nx"], verts["ny"], verts["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    verts["red"], verts["green"], verts["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(verts.tobytes())


def write_label_ply(path: Path, points: np.ndarray, labels: np.ndarray) -> None:
    """将带标签的点云数据写入 .ply 文件。"""
    if points.shape[0] != labels.shape[0]:
        raise ValueError("points and labels must have the same length")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property int label\n"
        "end_header\n"
    ).encode("ascii")
    verts = np.empty(points.shape[0], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("label", "<i4")])
    verts["x"], verts["y"], verts["z"] = points[:, 0], points[:, 1], points[:, 2]
    verts["label"] = labels.astype(np.int32, copy=False)
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(verts.tobytes())


def write_label_ply_with_normals(path: Path, points: np.ndarray, labels: np.ndarray, normals: np.ndarray) -> None:
    """将带法线和标签的点云数据写入 .ply 文件。"""
    if points.shape[0] != labels.shape[0] or points.shape[0] != normals.shape[0]:
        raise ValueError("points, labels, and normals must have the same length")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property int label\n"
        "end_header\n"
    ).encode("ascii")
    verts = np.empty(
        points.shape[0],
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("nx", "<f4"),
            ("ny", "<f4"),
            ("nz", "<f4"),
            ("label", "<i4"),
        ],
    )
    verts["x"], verts["y"], verts["z"] = points[:, 0], points[:, 1], points[:, 2]
    verts["nx"], verts["ny"], verts["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    verts["label"] = labels.astype(np.int32, copy=False)
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(verts.tobytes())


def estimate_normals_towards_cameras(
    points: np.ndarray,
    *,
    camera_centers: list[np.ndarray],
    voxel_size: float,
    normals_radius: float | None = None,
    normals_max_nn: int = 30,
) -> np.ndarray:
    """估计点云法线，并将每个法线翻转到朝向最近相机。"""
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)
    import open3d as o3d  # noqa: PLC0415

    radius = float(normals_radius) if normals_radius is not None else 0.0
    if radius <= 0.0:
        radius = max(float(voxel_size) * 4.0, 0.02)
    max_nn = max(int(normals_max_nn), 8)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=max_nn,
        )
    )
    normals = np.asarray(cloud.normals, dtype=np.float32)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(
        normals,
        np.maximum(norms, 1e-8),
        out=np.zeros_like(normals),
        where=norms > 0.0,
    )
    centers = (
        np.asarray(camera_centers, dtype=np.float32).reshape(-1, 3)
        if camera_centers
        else np.empty((0, 3), dtype=np.float32)
    )
    if centers.shape[0] > 0:
        deltas = centers[None, :, :] - points[:, None, :]
        nearest = np.argmin(np.einsum("nck,nck->nc", deltas, deltas), axis=1)
        to_camera = deltas[np.arange(points.shape[0]), nearest]
        flip = np.einsum("ij,ij->i", normals, to_camera) < 0.0
        normals[flip] *= -1.0
    return normals.astype(np.float32, copy=False)


def camera_centers_from_inputs(camera_inputs: dict[str, dict[str, object]]) -> list[np.ndarray]:
    """从当前帧相机输入中提取 world 坐标下的相机中心。"""
    centers: list[np.ndarray] = []
    for payload in camera_inputs.values():
        pose_record = payload.get("pose_record")
        if not isinstance(pose_record, dict) or pose_record.get("cam2world_4x4") is None:
            continue
        cam2world = np.asarray(pose_record["cam2world_4x4"], dtype=np.float64)
        if cam2world.shape == (4, 4):
            centers.append(cam2world[:3, 3].astype(np.float32, copy=False))
    return centers


def camera_center_from_pose_record(pose_record: dict[str, object]) -> np.ndarray | None:
    """从单个相机位姿记录中提取 world 坐标下的相机中心。"""
    cam2world_payload = pose_record.get("cam2world_4x4")
    if cam2world_payload is None:
        return None
    cam2world = np.asarray(cam2world_payload, dtype=np.float64)
    if cam2world.shape != (4, 4):
        return None
    return cam2world[:3, 3].astype(np.float32, copy=False)


def write_live_debug_target_object_cloud(
    *,
    live_debug_root: Path,
    frame_name: str,
    camera_id: str,
    target_name: str,
    points: np.ndarray,
    colors: np.ndarray,
    save_normal: bool,
    camera_center: np.ndarray | None,
    voxel_size: float,
    score: float | None = None,
    target_pixels: int | None = None,
) -> dict[str, object]:
    """保存 live debug 单相机目标物体点云，不包含背景点。"""
    frame_stem = frame_name.replace(".png", "")
    camera_dir = live_debug_root / frame_stem / camera_id
    camera_dir.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    ply_name = "target_object_rgb.ply"
    normals: np.ndarray | None = None
    if save_normal:
        camera_centers = [] if camera_center is None else [np.asarray(camera_center, dtype=np.float32)]
        normals = estimate_normals_towards_cameras(
            points,
            camera_centers=camera_centers,
            voxel_size=float(voxel_size),
        )
        write_ply_with_normals(camera_dir / ply_name, points, colors, normals)
    else:
        write_ply(camera_dir / ply_name, points, colors)
    summary = {
        "frame_name": frame_name,
        "camera_id": camera_id,
        "target_name": target_name,
        "ply_file": ply_name,
        "num_points": int(points.shape[0]),
        "has_normals": bool(normals is not None),
        "score": None if score is None else float(score),
        "target_pixels": None if target_pixels is None else int(target_pixels),
    }
    (camera_dir / "target_object_pointcloud.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def save_binary_mask_debug(
    output_dir: Path,
    frame_name: str,
    camera_id: str,
    rgb: np.ndarray,
    mask: np.ndarray,
    score: float | None,
) -> None:
    """保存二进制掩码调试信息，包括 RGB 图像、掩码叠加图和掩码本身。"""
    frame_stem = frame_name.replace(".png", "")
    camera_dir = output_dir / "masks_2d" / frame_stem / camera_id
    camera_dir.mkdir(parents=True, exist_ok=True)
    overlay = rgb.astype(np.float32).copy()
    valid = mask.astype(bool)
    overlay[valid] = 0.45 * overlay[valid] + 0.55 * np.array([255.0, 70.0, 70.0], dtype=np.float32)
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
    overlay[boundary] = np.array([255.0, 255.0, 255.0], dtype=np.float32)
    Image.fromarray(rgb).save(camera_dir / "rgb.png")
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(camera_dir / "semantic_overlay.png")
    mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    mask_rgb[valid] = np.array([255, 70, 70], dtype=np.uint8)
    Image.fromarray(mask_rgb).save(camera_dir / "semantic_label.png")
    summary = {
        "frame_name": frame_name,
        "camera_id": camera_id,
        "target_pixels": int(np.count_nonzero(valid)),
        "score": None if score is None else float(score),
    }
    (camera_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


class SingleObjectPointCloudSegmenter:
    """快速单物体在线分割器，使用一个正类提示和所有其他类的负框。"""

    @classmethod
    def from_config(cls, config: SingleSegConfig, **overrides: Any) -> "SingleObjectPointCloudSegmenter":
        resolved = config.with_overrides(**overrides)
        return cls(**resolved.to_segmenter_kwargs())

    @classmethod
    def from_config_file(
        cls,
        config_path: Path | str,
        **overrides: Any,
    ) -> "SingleObjectPointCloudSegmenter":
        return cls.from_config(SingleSegConfig.from_file(config_path), **overrides)

    def __init__(
        self,
        *,
        target_name: str,
        prompt_task_info: Path = DEFAULT_PROMPT_TASK_INFO,
        prompt_image_root: Path = DEFAULT_PROMPT_IMAGE_ROOT,
        checkpoint_path: Path = DEFAULT_CHECKPOINT,
        output_dir: Path | None = None,
        overwrite_output: bool = False,
        confidence: float = 0.25,
        mask_threshold: float = 0.6,
        prompt_keep_score_threshold: float = 0.2,
        video_mask_prob_threshold: float = 0.95,
        depth_scale: float = 1000.0,
        depth_min: float = 0.1,
        depth_max: float = 3.0,
        stride: int = 2,
        frame_voxel_size: float = 0.003,
        target_cluster_filter_enabled: bool = True,
        target_cluster_radius_m: float = 0.025,
        target_cluster_min_points: int = 35,
        target_cluster_keep_largest: bool = True,
        target_plane_filter_enabled: bool = False,
        target_plane_filter_distance_m: float = 0.004,
        target_plane_filter_min_points: int = 80,
        target_plane_filter_min_inlier_ratio: float = 0.25,
        target_plane_filter_max_inlier_ratio: float = 0.85,
        target_plane_filter_max_planes: int = 1,
        target_plane_filter_ransac_iterations: int = 256,
        target_depth_band_filter_enabled: bool = True,
        target_depth_band_filter_range_m: float = 0.08,
        target_depth_band_filter_min_valid_pixels: int = 50,
        target_depth_band_filter_min_keep_pixels: int = 20,
        target_3d_mask_erode_kernel: int = 0,
        save_ply: bool = True,
        save_normal: bool = False,
        save_debug_2d: bool = False,
        tracker_image_size: int | None = DEFAULT_TRACKER_IMAGE_SIZE,
        target_vis_color: tuple[int, int, int] | None = None,
        target_id: int = 1,
    ) -> None:
        self.target_name = str(target_name)
        self.target_id = int(target_id)
        self.prompt_task_info = Path(prompt_task_info).resolve()
        self.prompt_image_root = Path(prompt_image_root).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.output_dir = Path(output_dir).resolve() if output_dir is not None else DEFAULT_OUTPUT_DIR.resolve()
        self.confidence = float(confidence)
        self.mask_threshold = float(mask_threshold)
        self.prompt_keep_score_threshold = float(prompt_keep_score_threshold)
        self.video_mask_prob_threshold = float(video_mask_prob_threshold)
        self.depth_scale = float(depth_scale)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.stride = int(stride)
        self.frame_voxel_size = float(frame_voxel_size)
        self.target_cluster_filter_enabled = bool(target_cluster_filter_enabled)
        self.target_cluster_radius_m = float(target_cluster_radius_m)
        self.target_cluster_min_points = int(target_cluster_min_points)
        self.target_cluster_keep_largest = bool(target_cluster_keep_largest)
        self.target_plane_filter_enabled = bool(target_plane_filter_enabled)
        self.target_plane_filter_distance_m = float(target_plane_filter_distance_m)
        self.target_plane_filter_min_points = int(target_plane_filter_min_points)
        self.target_plane_filter_min_inlier_ratio = float(target_plane_filter_min_inlier_ratio)
        self.target_plane_filter_max_inlier_ratio = float(target_plane_filter_max_inlier_ratio)
        self.target_plane_filter_max_planes = int(target_plane_filter_max_planes)
        self.target_plane_filter_ransac_iterations = int(target_plane_filter_ransac_iterations)
        self.target_depth_band_filter_enabled = bool(target_depth_band_filter_enabled)
        self.target_depth_band_filter_range_m = float(target_depth_band_filter_range_m)
        self.target_depth_band_filter_min_valid_pixels = int(target_depth_band_filter_min_valid_pixels)
        self.target_depth_band_filter_min_keep_pixels = int(target_depth_band_filter_min_keep_pixels)
        target_erode_kernel = max(int(target_3d_mask_erode_kernel), 0)
        if target_erode_kernel > 1 and target_erode_kernel % 2 == 0:
            target_erode_kernel += 1
        self.target_3d_mask_erode_kernel = target_erode_kernel
        self.save_ply = bool(save_ply)
        self.save_normal = bool(save_normal)
        self.save_debug_2d = bool(save_debug_2d)
        self.tracker_image_size = None if tracker_image_size is None else int(tracker_image_size)
        self.target_vis_color = (
            tuple(int(c) for c in target_vis_color) if target_vis_color is not None else (255, 70, 70)
        )
        self.prompt_max_masks = DEFAULT_PROMPT_MAX_MASKS
        self.prompt_ref_cell = DEFAULT_PROMPT_REF_CELL
        self.prompt_max_cols = DEFAULT_PROMPT_MAX_COLS
        self.prompt_canvas_gap = DEFAULT_PROMPT_CANVAS_GAP
        self.seed_min_pixels = DEFAULT_SEED_MIN_PIXELS
        self.seed_max_area_ratio = DEFAULT_SEED_MAX_AREA_RATIO
        self.seed_box_margin = DEFAULT_SEED_BOX_MARGIN
        self.video_object_min_score = DEFAULT_VIDEO_OBJECT_MIN_SCORE
        self.sync_timing = DEFAULT_SYNC_TIMING

        if overwrite_output and self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_output_dir = self.output_dir / "frame_outputs"
        self.frame_output_dir.mkdir(parents=True, exist_ok=True)
        if not self.prompt_task_info.is_file():
            raise FileNotFoundError(f"prompt_task_info not found: {self.prompt_task_info}")
        if not self.prompt_image_root.is_dir():
            raise FileNotFoundError(f"prompt_image_root not found: {self.prompt_image_root}")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {self.checkpoint_path}")

        all_entries = load_prompt_entries(self.prompt_task_info, self.prompt_image_root)
        self.positive_entries, self.negative_entries = split_prompt_entries(all_entries, self.target_name)
        image_processor_t0 = time.perf_counter()
        self.image_processor = load_sam3_image_processor(
            checkpoint_path=self.checkpoint_path,
            confidence=self.confidence,
            mask_threshold=self.mask_threshold,
        )
        self.image_processor_load_time_sec = time.perf_counter() - image_processor_t0
        video_predictor_t0 = time.perf_counter()
        self.video_predictor = load_video_predictor(
            checkpoint_path=self.checkpoint_path,
            tracker_image_size=self.tracker_image_size,
        )
        self.video_predictor_load_time_sec = time.perf_counter() - video_predictor_t0

        self.session_ids: dict[str, str] = {}
        self.stitched_layout: Any | None = None
        self.active_camera_ids: list[str] = []
        self.seed_info_by_camera: dict[str, dict[str, object]] = {}
        self.tensor_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._torch_backproject_scale_cache: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor]] = {}
        self.frame_index = 0
        self.initialized = False
        self.closed = False
        self.pipeline_t0 = time.perf_counter()
        self.startup_time_before_streaming: float | None = None
        self.first_frame_ready_time: float | None = None
        self.timeline: list[dict[str, object]] = []
        self._last_initialize_timing: dict[str, object] = {}

    def _build_frame_resources(self, camera_inputs: dict[str, dict[str, object]]) -> dict[str, list[Image.Image | torch.Tensor]]:
        resources: dict[str, list[Image.Image | torch.Tensor]] = {}
        for camera_id, payload in camera_inputs.items():
            rgb = payload["rgb"]
            if torch.is_tensor(rgb):
                resources[camera_id] = [rgb]
            else:
                resources[camera_id] = [Image.fromarray(np.asarray(rgb, dtype=np.uint8))]
        return resources

    def _get_torch_backproject_scales(
        self,
        *,
        height: int,
        width: int,
        intrinsics: dict[str, float] | None,
        fovy_deg: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """获取 PyTorch 中的反投影缩放因子（X 和 Y 方向）。"""
        if intrinsics is not None:
            fx = float(intrinsics["fx"])
            fy = float(intrinsics["fy"])
            cx = float(intrinsics["cx"])
            cy = float(intrinsics["cy"])
        else:
            if fovy_deg is None:
                raise ValueError("Either intrinsics or fovy_deg must be provided")
            fy = 0.5 * int(height) / np.tan(np.deg2rad(float(fovy_deg)) * 0.5)
            fx = fy
            cx = (int(width) - 1) * 0.5
            cy = (int(height) - 1) * 0.5
        sample_h = (int(height) + self.stride - 1) // self.stride
        sample_w = (int(width) + self.stride - 1) // self.stride
        key = (
            int(sample_h),
            int(sample_w),
            int(self.stride),
            round(fx, 6),
            round(fy, 6),
            round(cx, 6),
            round(cy, 6),
            self.tensor_device.type,
            self.tensor_device.index,
        )
        cached = self._torch_backproject_scale_cache.get(key)
        if cached is not None:
            return cached
        with no_autocast_context(self.tensor_device):
            v = torch.arange(0, int(height), int(self.stride), dtype=torch.float32, device=self.tensor_device)
            u = torch.arange(0, int(width), int(self.stride), dtype=torch.float32, device=self.tensor_device)
            vv, uu = torch.meshgrid(v, u, indexing="ij")
            x_scale = ((uu - float(cx)) / float(fx)).to(torch.float32)
            y_scale = ((vv - float(cy)) / float(fy)).to(torch.float32)
        cached = (x_scale, y_scale)
        self._torch_backproject_scale_cache[key] = cached
        return cached

    def _initialize_sessions(
        self,
        frame_name: str,
        camera_inputs: dict[str, dict[str, object]],
        frame_resources: dict[str, list[Image.Image]],
    ) -> None:
        """初始化每个相机的追踪会话，并根据正负提示框设置种子掩码。"""
        init_t0 = time.perf_counter()
        seed_query_time = 0.0
        compose_time = 0.0
        start_session_time = 0.0
        add_prompt_time = 0.0
        per_camera_timing: list[dict[str, object]] = []
        seed_masks_by_camera: dict[str, np.ndarray] = {}
        active_camera_ids: list[str] = []
        for camera_id, payload in camera_inputs.items():
            image_or_tensor = frame_resources[camera_id][0]
            if torch.is_tensor(image_or_tensor):
                image = Image.fromarray(image_or_tensor.detach().cpu().numpy())
            else:
                image = image_or_tensor
            debug_dir = self.output_dir / "prompt_debug" / camera_id if self.save_debug_2d else None
            maybe_cuda_synchronize(self.tensor_device, self.sync_timing)
            prompt_t0 = time.perf_counter()
            boxes, scores, masks = run_single_object_prompt_query(
                image=image,
                camera_source_path=Path(f"{camera_id}/{frame_name}"),
                positive_entries=self.positive_entries,
                negative_entries=self.negative_entries,
                keep_score_threshold=self.prompt_keep_score_threshold,
                max_masks=self.prompt_max_masks,
                ref_cell=self.prompt_ref_cell,
                max_cols=self.prompt_max_cols,
                canvas_gap=self.prompt_canvas_gap,
                processor=self.image_processor,
                debug_canvas_path=(debug_dir / "concat_canvas.png") if debug_dir is not None else None,
                debug_prompt_path=(debug_dir / "prompt_boxes.png") if debug_dir is not None else None,
            )
            maybe_cuda_synchronize(self.tensor_device, self.sync_timing)
            selection = select_best_seed_mask(
                boxes=boxes,
                scores=scores,
                masks=masks,
                min_pixels=self.seed_min_pixels,
            )
            seed_source = "pos+neg"
            if selection is None:
                boxes, scores, masks = run_single_object_prompt_query(
                    image=image,
                    camera_source_path=Path(f"{camera_id}/{frame_name}"),
                    positive_entries=self.positive_entries,
                    negative_entries=[],
                    keep_score_threshold=0.0,
                    max_masks=max(self.prompt_max_masks, 8),
                    ref_cell=self.prompt_ref_cell,
                    max_cols=self.prompt_max_cols,
                    canvas_gap=self.prompt_canvas_gap,
                    processor=self.image_processor,
                    debug_canvas_path=(debug_dir / "concat_canvas_pos_only.png") if debug_dir is not None else None,
                    debug_prompt_path=(debug_dir / "prompt_boxes_pos_only.png") if debug_dir is not None else None,
                )
                maybe_cuda_synchronize(self.tensor_device, self.sync_timing)
                selection = select_best_seed_mask(
                    boxes=boxes,
                    scores=scores,
                    masks=masks,
                    min_pixels=self.seed_min_pixels,
                )
                seed_source = "pos_only_fallback"
            prompt_time = time.perf_counter() - prompt_t0
            seed_query_time += prompt_time
            per_camera_item: dict[str, object] = {
                "camera_id": camera_id,
                "seed_prompt_time_sec": prompt_time,
                "seed_source": seed_source,
                "usable_seed": selection is not None,
            }
            if selection is None:
                per_camera_timing.append(per_camera_item)
                continue
            seed_mask, seed_score, seed_box = selection
            seed_mask, seed_shape_mode = refine_seed_mask(
                seed_mask,
                seed_box,
                image_shape=tuple(payload["rgb"].shape[:2]) if torch.is_tensor(payload["rgb"]) else np.asarray(payload["rgb"], dtype=np.uint8).shape[:2],
                max_area_ratio=self.seed_max_area_ratio,
                box_margin=self.seed_box_margin,
                min_pixels=self.seed_min_pixels,
            )
            active_camera_ids.append(camera_id)
            seed_masks_by_camera[camera_id] = np.asarray(seed_mask, dtype=bool)
            self.seed_info_by_camera[camera_id] = {
                "seed_score": float(seed_score),
                "seed_pixels": int(np.count_nonzero(seed_mask)),
                "seed_box_xyxy": [int(value) for value in seed_box],
                "seed_source": seed_source,
                "seed_shape_mode": seed_shape_mode,
            }
            per_camera_item.update(
                {
                    "seed_score": float(seed_score),
                    "seed_pixels": int(np.count_nonzero(seed_mask)),
                    "seed_shape_mode": seed_shape_mode,
                }
            )
            per_camera_timing.append(per_camera_item)
            if self.save_debug_2d:
                rgb_data = payload["rgb"]
                if torch.is_tensor(rgb_data):
                    rgb_np = rgb_data.detach().cpu().numpy()
                else:
                    rgb_np = np.asarray(rgb_data, dtype=np.uint8)
                save_binary_mask_debug(
                    output_dir=self.output_dir,
                    frame_name=frame_name,
                    camera_id=camera_id,
                    rgb=rgb_np,
                    mask=np.asarray(seed_mask, dtype=bool),
                    score=seed_score,
                )
        self.active_camera_ids = active_camera_ids
        if not self.active_camera_ids:
            raise RuntimeError(f"No camera produced a usable seed for target {self.target_name!r}")
        try:
            from single_seg.tracker_only_backend import compose_camera_rgb_frame_resources, compose_camera_rgb_frame_resources_torch, stitch_camera_binary_masks
        except ImportError:
            from tracker_only_backend import compose_camera_rgb_frame_resources, compose_camera_rgb_frame_resources_torch, stitch_camera_binary_masks

        compose_t0 = time.perf_counter()
        
        first_rgb = camera_inputs[self.active_camera_ids[0]]["rgb"]
        if torch.is_tensor(first_rgb):
            canvas_tensor, self.stitched_layout = compose_camera_rgb_frame_resources_torch(
                rgb_by_camera={
                    camera_id: camera_inputs[camera_id]["rgb"]
                    for camera_id in self.active_camera_ids
                },
                camera_order=self.active_camera_ids,
                device=self.tensor_device,
            )
            composite_resources = [canvas_tensor]
        else:
            composite_resources, self.stitched_layout = compose_camera_rgb_frame_resources(
                rgb_by_camera={
                    camera_id: np.asarray(camera_inputs[camera_id]["rgb"], dtype=np.uint8)
                    for camera_id in self.active_camera_ids
                },
                camera_order=self.active_camera_ids,
            )
        composite_mask = stitch_camera_binary_masks(seed_masks_by_camera, self.stitched_layout)
        compose_time = time.perf_counter() - compose_t0
        with autocast_context():
            maybe_cuda_synchronize(self.tensor_device, self.sync_timing)
            start_t0 = time.perf_counter()
            session_id = self.video_predictor.start_session(composite_resources)["session_id"]
            maybe_cuda_synchronize(self.tensor_device, self.sync_timing)
            start_session_time = time.perf_counter() - start_t0
            add_prompt_t0 = time.perf_counter()
            self.video_predictor.add_prompt(
                session_id=session_id,
                frame_idx=0,
                mask=np.asarray(composite_mask, dtype=np.uint8),
                obj_id=1,
            )
            maybe_cuda_synchronize(self.tensor_device, self.sync_timing)
            add_prompt_time = time.perf_counter() - add_prompt_t0
            self.session_ids["__stitched__"] = session_id
        self.startup_time_before_streaming = time.perf_counter() - self.pipeline_t0
        self._last_initialize_timing = {
            "total_time_sec": time.perf_counter() - init_t0,
            "seed_query_time_sec": seed_query_time,
            "compose_seed_time_sec": compose_time,
            "tracker_start_session_time_sec": start_session_time,
            "tracker_add_prompt_time_sec": add_prompt_time,
            "per_camera": per_camera_timing,
        }
        self.initialized = True

    def process_frame(
        self,
        *,
        frame_name: str,
        camera_inputs: dict[str, dict[str, object]],
        live_debug_root: Path | None = None,
    ) -> dict[str, object]:
        """处理一帧多相机 RGBD 输入，并返回标记的点云。"""
        if self.closed:
            raise RuntimeError("segmenter is already closed")
        if not camera_inputs:
            raise ValueError("camera_inputs must not be empty")
        frame_t0 = time.perf_counter()
        frame_resource_build_time = 0.0
        frame_resource_t0 = time.perf_counter()
        frame_resources = self._build_frame_resources(camera_inputs)
        frame_resource_build_time = time.perf_counter() - frame_resource_t0
        initialize_sessions_time = 0.0
        initialize_sessions_breakdown: dict[str, object] | None = None
        if not self.initialized:
            initialize_t0 = time.perf_counter()
            self._initialize_sessions(frame_name, camera_inputs, frame_resources)
            initialize_sessions_time = time.perf_counter() - initialize_t0
            initialize_sessions_breakdown = dict(self._last_initialize_timing)

        append_frame_time = 0.0
        backproject_time = 0.0
        fuse_time = 0.0
        compose_inputs_time = 0.0
        mask_postprocess_time = 0.0
        target_mask_erode_time = 0.0
        target_depth_band_filter_time = 0.0
        camera_prepare_time = 0.0
        camera_rgb_copy_time = 0.0
        camera_mask_convert_time = 0.0
        camera_normalize_time = 0.0
        camera_depth_convert_time = 0.0
        camera_scale_compute_time = 0.0
        camera_bookkeeping_time = 0.0
        camera_save_debug_time = 0.0
        camera_mask_split_time = 0.0
        camera_target_summary_time = 0.0
        target_plane_filter_time = 0.0
        target_cluster_filter_time = 0.0
        live_debug_object_ply_time = 0.0
        cpu_transfer_time = 0.0
        colorize_time = 0.0
        normal_time = 0.0
        stereo_time = 0.0
        masks_by_camera: dict[str, torch.Tensor] = {}
        scores_by_camera: dict[str, float | None] = {}
        try:
            from single_seg.tracker_only_backend import compose_camera_rgb_frame_resources, compose_camera_rgb_frame_resources_torch, split_stitched_binary_mask_torch
        except ImportError:
            from tracker_only_backend import compose_camera_rgb_frame_resources, compose_camera_rgb_frame_resources_torch, split_stitched_binary_mask_torch

        compose_t0 = time.perf_counter()
        
        first_rgb = camera_inputs[self.active_camera_ids[0]]["rgb"]
        if torch.is_tensor(first_rgb):
            canvas_tensor, current_layout = compose_camera_rgb_frame_resources_torch(
                rgb_by_camera={
                    camera_id: camera_inputs[camera_id]["rgb"]
                    for camera_id in self.active_camera_ids
                },
                camera_order=self.active_camera_ids,
                layout=self.stitched_layout,
                device=self.tensor_device,
            )
            composite_resources = [canvas_tensor]
        else:
            composite_resources, current_layout = compose_camera_rgb_frame_resources(
                rgb_by_camera={
                    camera_id: np.asarray(camera_inputs[camera_id]["rgb"], dtype=np.uint8)
                    for camera_id in self.active_camera_ids
                },
                camera_order=self.active_camera_ids,
                layout=self.stitched_layout,
            )
        compose_inputs_time += time.perf_counter() - compose_t0
        if self.frame_index > 0:
            maybe_cuda_synchronize(self.tensor_device, self.sync_timing)
            append_t0 = time.perf_counter()
            self.video_predictor.append_frame(
                session_id=self.session_ids["__stitched__"],
                resource_path=composite_resources,
            )
            maybe_cuda_synchronize(self.tensor_device, self.sync_timing)
            append_frame_time += time.perf_counter() - append_t0
        maybe_cuda_synchronize(self.tensor_device, self.sync_timing)
        propagate_t0 = time.perf_counter()
        stitched_payload = self.video_predictor.infer_frame(
            session_id=self.session_ids["__stitched__"],
            frame_idx=int(self.frame_index),
            reverse=False,
        )
        maybe_cuda_synchronize(self.tensor_device, self.sync_timing)
        propagate_time = time.perf_counter() - propagate_t0
        stitched_score: float | None = None
        maybe_cuda_synchronize(self.tensor_device, self.sync_timing)
        mask_post_t0 = time.perf_counter()
        stitched_output_t = {
            "mask": torch.zeros(
                (int(current_layout.canvas_height), int(current_layout.canvas_width)),
                dtype=torch.bool,
                device=self.tensor_device,
            ),
            "score": None,
            "object_stats": {},
        }
        if stitched_payload.get("outputs") is not None:
            stitched_output_t = _extract_target_mask_from_output_torch(
                stitched_payload["outputs"],
                image_shape=(int(current_layout.canvas_height), int(current_layout.canvas_width)),
                min_object_score=self.video_object_min_score,
                mask_prob_threshold=self.video_mask_prob_threshold,
                target_obj_id=1,
                device=self.tensor_device,
            )
        stitched_score = None if stitched_output_t["score"] is None else float(stitched_output_t["score"])
        mask_split_t0 = time.perf_counter()
        masks_by_camera = split_stitched_binary_mask_torch(stitched_output_t["mask"], current_layout)
        camera_mask_split_time += time.perf_counter() - mask_split_t0
        maybe_cuda_synchronize(self.tensor_device, self.sync_timing)
        mask_postprocess_time += time.perf_counter() - mask_post_t0
        scores_by_camera = {camera_id: stitched_score for camera_id in self.active_camera_ids}

        point_chunks: list[torch.Tensor] = []
        color_chunks: list[torch.Tensor] = []
        label_chunks: list[torch.Tensor] = []
        score_chunks: list[torch.Tensor] = []
        camera_summaries: list[dict[str, object]] = []
        for camera_id in self.active_camera_ids:
            camera_prepare_t0 = time.perf_counter()
            payload = camera_inputs[camera_id]
            stereo_time += float(payload.get("stereo_time_sec", 0.0))

            rgb_copy_t0 = time.perf_counter()
            rgb_input = payload["rgb"]
            if torch.is_tensor(rgb_input):
                rgb = rgb_input
            else:
                rgb = np.asarray(rgb_input, dtype=np.uint8)
            camera_rgb_copy_time += time.perf_counter() - rgb_copy_t0

            mask_convert_t0 = time.perf_counter()
            mask_value = masks_by_camera.get(camera_id)
            if torch.is_tensor(mask_value):
                mask_t = mask_value.to(device=self.tensor_device, dtype=torch.bool, non_blocking=True)
            else:
                mask_t = torch.as_tensor(
                    np.asarray(
                        mask_value if mask_value is not None else np.zeros(rgb.shape[:2], dtype=bool),
                        dtype=bool,
                    ),
                    dtype=torch.bool,
                    device=self.tensor_device,
                )
            camera_mask_convert_time += time.perf_counter() - mask_convert_t0

            score = scores_by_camera.get(camera_id)
            save_debug_t0 = time.perf_counter()
            if self.save_debug_2d:
                rgb_for_debug = rgb.detach().cpu().numpy() if torch.is_tensor(rgb) else rgb
                save_binary_mask_debug(
                    output_dir=self.output_dir,
                    frame_name=frame_name,
                    camera_id=camera_id,
                    rgb=rgb_for_debug,
                    mask=mask_t.detach().cpu().numpy(),
                    score=score,
                )
            camera_save_debug_time += time.perf_counter() - save_debug_t0

            normalize_t0 = time.perf_counter()
            intrinsics = normalize_intrinsics_payload(payload.get("intrinsics"))
            pose_record = normalize_pose_record(camera_id, payload)
            camera_normalize_time += time.perf_counter() - normalize_t0

            depth_convert_t0 = time.perf_counter()
            depth_value = payload["depth_m"]
            if torch.is_tensor(depth_value):
                depth_m = depth_value.to(device=self.tensor_device, dtype=torch.float32, non_blocking=True)
            else:
                depth_m = np.asarray(depth_value, dtype=np.float32)
            camera_depth_convert_time += time.perf_counter() - depth_convert_t0

            fovy_deg = float(payload["fovy_deg"]) if payload.get("fovy_deg") is not None else None
            scale_compute_t0 = time.perf_counter()
            x_scale, y_scale = self._get_torch_backproject_scales(
                height=rgb.shape[0],
                width=rgb.shape[1],
                intrinsics=intrinsics,
                fovy_deg=fovy_deg,
            )
            camera_scale_compute_time += time.perf_counter() - scale_compute_t0

            camera_prepare_time += time.perf_counter() - camera_prepare_t0
            target_summary_t0 = time.perf_counter()
            mask_3d_t = mask_t
            target_pixels_before_3d = int(torch.count_nonzero(mask_t).item())
            target_mask_erode_summary: dict[str, object] = {
                "enabled": bool(self.target_3d_mask_erode_kernel > 1),
                "kernel_size": int(self.target_3d_mask_erode_kernel),
                "target_pixels_before": target_pixels_before_3d,
                "target_pixels_after": target_pixels_before_3d,
                "removed_target_pixels": 0,
            }
            if self.target_3d_mask_erode_kernel > 1:
                maybe_cuda_synchronize(self.tensor_device, self.sync_timing)
                target_mask_erode_t0 = time.perf_counter()
                mask_3d_t = erode_binary_mask_torch(mask_t, self.target_3d_mask_erode_kernel)
                maybe_cuda_synchronize(self.tensor_device, self.sync_timing)
                target_mask_erode_time += time.perf_counter() - target_mask_erode_t0
                after_pixels = int(torch.count_nonzero(mask_3d_t).item())
                target_mask_erode_summary["target_pixels_after"] = after_pixels
                target_mask_erode_summary["removed_target_pixels"] = int(
                    max(target_pixels_before_3d - after_pixels, 0)
                )
            target_depth_band_summary: dict[str, object] = {
                "enabled": bool(self.target_depth_band_filter_enabled),
                "backend": "torch",
                "range_m": float(self.target_depth_band_filter_range_m),
                "min_valid_pixels": int(self.target_depth_band_filter_min_valid_pixels),
                "min_keep_pixels": int(self.target_depth_band_filter_min_keep_pixels),
                "target_pixels_before": int(torch.count_nonzero(mask_3d_t).item()),
                "valid_depth_pixels": 0,
                "target_pixels_after": int(torch.count_nonzero(mask_3d_t).item()),
                "removed_target_pixels": 0,
                "center_depth_m": None,
                "applied": False,
                "skipped_reason": None,
            }
            if self.target_depth_band_filter_enabled:
                maybe_cuda_synchronize(self.tensor_device, self.sync_timing)
                target_depth_band_t0 = time.perf_counter()
                depth_for_filter = (
                    depth_m
                    if torch.is_tensor(depth_m)
                    else torch.as_tensor(np.ascontiguousarray(depth_m), dtype=torch.float32, device=self.tensor_device)
                )
                mask_3d_t, target_depth_band_summary = filter_target_mask_by_depth_band_torch(
                    mask_3d_t,
                    depth_for_filter,
                    enabled=True,
                    range_m=self.target_depth_band_filter_range_m,
                    min_valid_pixels=self.target_depth_band_filter_min_valid_pixels,
                    min_keep_pixels=self.target_depth_band_filter_min_keep_pixels,
                )
                maybe_cuda_synchronize(self.tensor_device, self.sync_timing)
                target_depth_band_filter_time += time.perf_counter() - target_depth_band_t0
            maybe_cuda_synchronize(self.tensor_device, self.sync_timing)
            camera_backproject_t0 = time.perf_counter()
            sampled_depth = (
                depth_m[:: self.stride, :: self.stride]
                if torch.is_tensor(depth_m)
                else np.ascontiguousarray(depth_m[:: self.stride, :: self.stride])
            )
            sampled_rgb = rgb[:: self.stride, :: self.stride]
            if torch.is_tensor(sampled_rgb):
                sampled_rgb = sampled_rgb.contiguous()
            else:
                sampled_rgb = np.ascontiguousarray(sampled_rgb)
            points, colors, point_labels, point_scores = backproject_scene_points_with_labels_torch(
                sampled_rgb=sampled_rgb,
                sampled_depth_m=sampled_depth,
                sampled_mask=mask_3d_t[:: self.stride, :: self.stride],
                cam2world_gl=np.asarray(pose_record["cam2world_4x4"], dtype=np.float64),
                x_scale=x_scale,
                y_scale=y_scale,
                depth_min=self.depth_min,
                depth_max=self.depth_max,
                device=self.tensor_device,
                target_id=self.target_id,
                camera_score=score,
            )
            maybe_cuda_synchronize(self.tensor_device, self.sync_timing)
            camera_backproject_time = time.perf_counter() - camera_backproject_t0
            backproject_time += camera_backproject_time
            target_plane_filter_summary: dict[str, object] = {
                "enabled": bool(self.target_plane_filter_enabled),
                "backend": "torch_cpu_ransac",
                "distance_m": float(self.target_plane_filter_distance_m),
                "min_points": int(self.target_plane_filter_min_points),
                "min_inlier_ratio": float(self.target_plane_filter_min_inlier_ratio),
                "max_inlier_ratio": float(self.target_plane_filter_max_inlier_ratio),
                "max_planes": int(self.target_plane_filter_max_planes),
                "ransac_iterations": int(self.target_plane_filter_ransac_iterations),
                "target_points_before": int(torch.count_nonzero(point_labels > 0).item()),
                "target_points_after": int(torch.count_nonzero(point_labels > 0).item()),
                "removed_target_points": 0,
                "plane_found": False,
                "plane_applied": False,
                "plane": None,
                "plane_inliers": 0,
                "plane_inlier_ratio": 0.0,
                "planes": [],
                "skipped_reason": None,
            }
            if self.target_plane_filter_enabled and int(points.shape[0]) > 0:
                maybe_cuda_synchronize(self.tensor_device, self.sync_timing)
                target_plane_t0 = time.perf_counter()
                point_labels, target_plane_filter_summary = filter_target_labels_by_dominant_plane_torch(
                    points,
                    point_labels,
                    enabled=True,
                    distance_m=self.target_plane_filter_distance_m,
                    min_points=self.target_plane_filter_min_points,
                    min_inlier_ratio=self.target_plane_filter_min_inlier_ratio,
                    max_inlier_ratio=self.target_plane_filter_max_inlier_ratio,
                    max_planes=self.target_plane_filter_max_planes,
                    ransac_iterations=self.target_plane_filter_ransac_iterations,
                )
                maybe_cuda_synchronize(self.tensor_device, self.sync_timing)
                target_plane_filter_time += time.perf_counter() - target_plane_t0
            camera_bookkeeping_t0 = time.perf_counter()
            target_pixels = int(torch.count_nonzero(mask_t).item())
            target_pixels_for_3d = int(torch.count_nonzero(mask_3d_t).item())
            target_points_count = int(torch.count_nonzero(point_labels).item())
            target_object_summary: dict[str, object] | None = None
            if live_debug_root is not None:
                live_debug_object_t0 = time.perf_counter()
                target_point_mask = point_labels > 0
                target_points = points[target_point_mask].detach().cpu().numpy().astype(np.float32, copy=False)
                target_colors = colors[target_point_mask].detach().cpu().numpy().astype(np.uint8, copy=False)
                target_object_summary = write_live_debug_target_object_cloud(
                    live_debug_root=Path(live_debug_root),
                    frame_name=frame_name,
                    camera_id=camera_id,
                    target_name=self.target_name,
                    points=target_points,
                    colors=target_colors,
                    save_normal=self.save_normal,
                    camera_center=camera_center_from_pose_record(pose_record),
                    voxel_size=self.frame_voxel_size,
                    score=score,
                    target_pixels=target_pixels,
                )
                live_debug_object_ply_time += time.perf_counter() - live_debug_object_t0
            if int(points.shape[0]) > 0:
                point_chunks.append(points)
                color_chunks.append(colors)
                label_chunks.append(point_labels)
                score_chunks.append(point_scores)
            camera_target_summary_time += time.perf_counter() - target_summary_t0
            camera_summary = {
                "camera_id": camera_id,
                "target_pixels": target_pixels,
                "target_pixels_for_3d": target_pixels_for_3d,
                "num_points_backprojected": int(points.shape[0]),
                "num_target_points_backprojected": target_points_count,
                "backproject_time_sec": camera_backproject_time,
                "stereo_time_sec": float(payload.get("stereo_time_sec", 0.0)),
                "target_3d_mask_erode": target_mask_erode_summary,
                "target_depth_band_filter": target_depth_band_summary,
                "target_plane_filter": target_plane_filter_summary,
            }
            if target_object_summary is not None:
                camera_summary["target_object_pointcloud"] = target_object_summary
            camera_summaries.append(camera_summary)
            camera_bookkeeping_time += time.perf_counter() - camera_bookkeeping_t0

        maybe_cuda_synchronize(self.tensor_device, self.sync_timing)
        fuse_t0 = time.perf_counter()
        scores_t: torch.Tensor | None = None
        if point_chunks:
            fuse_result = fuse_scene_geometry_torch(
                point_chunks=point_chunks,
                color_chunks=color_chunks,
                label_chunks=label_chunks,
                voxel_size=self.frame_voxel_size,
                device=self.tensor_device,
                score_chunks=score_chunks,
            )
            points_xyz_t, raw_colors_t, labels_t, scores_t = fuse_result
        else:
            points_xyz_t = torch.empty((0, 3), dtype=torch.float32, device=self.tensor_device)
            raw_colors_t = torch.empty((0, 3), dtype=torch.uint8, device=self.tensor_device)
            labels_t = torch.empty((0,), dtype=torch.int32, device=self.tensor_device)
            scores_t = torch.empty((0,), dtype=torch.float32, device=self.tensor_device)
        maybe_cuda_synchronize(self.tensor_device, self.sync_timing)
        fuse_time = time.perf_counter() - fuse_t0
        target_cluster_summary: dict[str, object] = {
            "enabled": bool(self.target_cluster_filter_enabled),
            "backend": "torch",
            "radius_m": float(self.target_cluster_radius_m),
            "min_points": int(self.target_cluster_min_points),
            "keep_largest": bool(self.target_cluster_keep_largest),
            "target_points_before": 0,
            "target_points_after": 0,
            "removed_target_points": 0,
            "num_clusters": 0,
            "cluster_sizes": [],
        }
        if self.target_cluster_filter_enabled and int(labels_t.numel()) > 0:
            maybe_cuda_synchronize(self.tensor_device, self.sync_timing)
            target_cluster_t0 = time.perf_counter()
            labels_t, target_cluster_summary = filter_target_labels_by_3d_clusters_torch(
                points_xyz_t,
                labels_t,
                enabled=True,
                radius_m=self.target_cluster_radius_m,
                min_points=self.target_cluster_min_points,
                keep_largest=self.target_cluster_keep_largest,
            )
            maybe_cuda_synchronize(self.tensor_device, self.sync_timing)
            target_cluster_filter_time += time.perf_counter() - target_cluster_t0
        need_cpu_output = self.save_ply or live_debug_root is not None
        points_xyz: np.ndarray | None = None
        raw_colors: np.ndarray | None = None
        labels: np.ndarray | None = None
        vis_colors: np.ndarray | None = None
        vis_colors_t: torch.Tensor | None = None
        if need_cpu_output:
            cpu_transfer_t0 = time.perf_counter()
            points_xyz = points_xyz_t.detach().cpu().numpy().astype(np.float32, copy=False)
            raw_colors = raw_colors_t.detach().cpu().numpy().astype(np.uint8, copy=False)
            labels = labels_t.detach().cpu().numpy().astype(np.int32, copy=False)
            cpu_transfer_time += time.perf_counter() - cpu_transfer_t0
            colorize_t0 = time.perf_counter()
            vis_colors = raw_colors.copy()
            if vis_colors.shape[0] > 0:
                vis_colors[labels > 0] = np.array(self.target_vis_color, dtype=np.uint8)
            colorize_time += time.perf_counter() - colorize_t0
        colorize_t0 = time.perf_counter()
        vis_colors_t = raw_colors_t.clone()
        if vis_colors_t.shape[0] > 0:
            vis_colors_t[labels_t > 0] = torch.tensor(
                self.target_vis_color, dtype=torch.uint8, device=self.tensor_device
            )
        colorize_time += time.perf_counter() - colorize_t0

        save_t0 = time.perf_counter()
        if self.save_ply:
            assert points_xyz is not None and raw_colors is not None and labels is not None and vis_colors is not None
            frame_stem = frame_name.replace(".png", "")
            normals: np.ndarray | None = None
            if self.save_normal:
                normal_t0 = time.perf_counter()
                normals = estimate_normals_towards_cameras(
                    points_xyz,
                    camera_centers=camera_centers_from_inputs(camera_inputs),
                    voxel_size=self.frame_voxel_size,
                )
                normal_time += time.perf_counter() - normal_t0
            if normals is not None:
                write_ply_with_normals(
                    self.frame_output_dir / f"{frame_stem}_scene_rgb.ply",
                    points_xyz,
                    raw_colors,
                    normals,
                )
                write_ply_with_normals(
                    self.frame_output_dir / f"{frame_stem}_instance_rgb.ply",
                    points_xyz,
                    vis_colors,
                    normals,
                )
                write_label_ply_with_normals(
                    self.frame_output_dir / f"{frame_stem}_instance_label.ply",
                    points_xyz,
                    labels,
                    normals,
                )
                target_mask = labels > 0
                if np.any(target_mask):
                    write_ply_with_normals(
                        self.frame_output_dir / f"{frame_stem}_target_only.ply",
                        points_xyz[target_mask],
                        vis_colors[target_mask],
                        normals[target_mask],
                    )
            else:
                write_ply(self.frame_output_dir / f"{frame_stem}_scene_rgb.ply", points_xyz, raw_colors)
                write_ply(self.frame_output_dir / f"{frame_stem}_instance_rgb.ply", points_xyz, vis_colors)
                write_label_ply(self.frame_output_dir / f"{frame_stem}_instance_label.ply", points_xyz, labels)
                target_mask = labels > 0
                if np.any(target_mask):
                    write_ply(
                        self.frame_output_dir / f"{frame_stem}_target_only.ply",
                        points_xyz[target_mask],
                        vis_colors[target_mask],
                    )
            if scores_t is not None and scores_t.numel() > 0:
                confidence_path = self.frame_output_dir / f"{frame_stem}_instance_confidence.npy"
                np.save(str(confidence_path), scores_t.detach().cpu().numpy().astype(np.float32, copy=False))
            meta = {
                "frame_name": frame_name,
                "target_name": self.target_name,
                "num_points": int(points_xyz.shape[0]),
                "num_labeled_points": int(np.count_nonzero(labels)),
                "has_normals": bool(normals is not None),
                "target_cluster_filter": target_cluster_summary,
                "camera_summaries": camera_summaries,
                "seed_info_by_camera": self.seed_info_by_camera,
            }
            (self.frame_output_dir / f"{frame_stem}_instance_meta.json").write_text(
                json.dumps(meta, indent=2),
                encoding="utf-8",
            )
        save_time = time.perf_counter() - save_t0

        frame_runtime = time.perf_counter() - frame_t0
        if self.first_frame_ready_time is None:
            self.first_frame_ready_time = time.perf_counter() - self.pipeline_t0
        self.timeline.append(
            {
                "frame_index": int(self.frame_index),
                "frame_name": frame_name,
                "num_points": int(points_xyz_t.shape[0]),
                "num_labeled_points": int(torch.count_nonzero(labels_t > 0).item()),
                "target_cluster_filter": target_cluster_summary,
                "camera_summaries": camera_summaries,
                "total_frame_time_sec": frame_runtime,
                "append_frame_time_sec": append_frame_time,
                "initialize_sessions_time_sec": initialize_sessions_time,
                "propagate_time_sec": propagate_time,
                "backproject_time_sec": backproject_time,
                "fuse_time_sec": fuse_time,
                "save_time_sec": save_time,
                "frame_runtime_sec": frame_runtime,
                "residual_breakdown_sec": {
                    "frame_resource_build_time_sec": frame_resource_build_time,
                    "initialize_sessions_time_sec": initialize_sessions_time,
                    "compose_inputs_time_sec": compose_inputs_time,
                    "mask_postprocess_time_sec": mask_postprocess_time,
                    "camera_mask_split_time_sec": camera_mask_split_time,
                    "target_mask_erode_time_sec": target_mask_erode_time,
                    "target_depth_band_filter_time_sec": target_depth_band_filter_time,
                    "target_plane_filter_time_sec": target_plane_filter_time,
                    "camera_prepare_time_sec": camera_prepare_time,
                    "camera_rgb_copy_time_sec": camera_rgb_copy_time,
                    "camera_mask_convert_time_sec": camera_mask_convert_time,
                    "camera_normalize_time_sec": camera_normalize_time,
                    "camera_depth_convert_time_sec": camera_depth_convert_time,
                    "camera_scale_compute_time_sec": camera_scale_compute_time,
                    "camera_save_debug_time_sec": camera_save_debug_time,
                    "camera_target_summary_time_sec": camera_target_summary_time,
                    "camera_bookkeeping_time_sec": camera_bookkeeping_time,
                    "target_cluster_filter_time_sec": target_cluster_filter_time,
                    "live_debug_object_ply_time_sec": live_debug_object_ply_time,
                    "cpu_transfer_time_sec": cpu_transfer_time,
                    "colorize_time_sec": colorize_time,
                    "normal_time_sec": normal_time,
                    "stereo_time_sec": stereo_time,
                },
                "initialize_sessions_breakdown_sec": initialize_sessions_breakdown,
            }
        )
        result = {
            "frame_index": int(self.frame_index),
            "frame_name": frame_name,
            "points_xyz": points_xyz_t,
            "instance_labels": labels_t,
            "instance_colors": vis_colors_t,
            "raw_colors": raw_colors_t,
            "semantic_labels": labels_t,
            "semantic_colors": vis_colors_t,
            "label_names": [self.target_name],
            "label_values": [0, self.target_id],
            "palette": torch.tensor([self.target_vis_color], dtype=torch.uint8, device=self.tensor_device),
            "camera_summaries": camera_summaries,
            "target_cluster_filter": target_cluster_summary,
            "meta": {
                "frame_name": frame_name,
                "target_name": self.target_name,
                "num_points": int(points_xyz_t.shape[0]),
                "num_labeled_points": int(torch.count_nonzero(labels_t).item()),
                "camera_summaries": camera_summaries,
                "target_cluster_filter": target_cluster_summary,
                "has_normals": bool(self.save_ply and self.save_normal),
                "output_format": "torch",
            },
        }
        self.frame_index += 1
        return result

    def write_summary(self) -> None:
        """将处理摘要信息写入文件。"""
        later = self.timeline[1:] if len(self.timeline) > 1 else []
        later_mean = (
            float(sum(item.get("total_frame_time_sec", item.get("frame_runtime_sec", 0.0)) for item in later) / len(later))
            if later
            else None
        )

        def optional_later_mean(key: str) -> float | None:
            values = [float(item[key]) for item in later if key in item and item[key] is not None]
            if not values:
                return None
            return float(sum(values) / len(values))

        later_loop_mean = optional_later_mean("loop_runtime_sec")
        later_capture_mean = optional_later_mean("capture_time_sec")
        later_rgbd_mean = optional_later_mean("build_camera_inputs_time_sec")
        later_process_wall_mean = optional_later_mean("process_frame_time_sec")
        summary = {
            "target_name": self.target_name,
            "prompt_task_info": str(self.prompt_task_info),
            "prompt_image_root": str(self.prompt_image_root),
            "checkpoint_path": str(self.checkpoint_path),
            "tracker_image_size": self.tracker_image_size,
            "confidence": float(self.confidence),
            "mask_threshold": float(self.mask_threshold),
            "video_mask_prob_threshold": float(self.video_mask_prob_threshold),
            "target_cluster_filter_enabled": bool(self.target_cluster_filter_enabled),
            "target_cluster_radius_m": float(self.target_cluster_radius_m),
            "target_cluster_min_points": int(self.target_cluster_min_points),
            "target_cluster_keep_largest": bool(self.target_cluster_keep_largest),
            "target_plane_filter_enabled": bool(self.target_plane_filter_enabled),
            "target_plane_filter_distance_m": float(self.target_plane_filter_distance_m),
            "target_plane_filter_min_points": int(self.target_plane_filter_min_points),
            "target_plane_filter_min_inlier_ratio": float(self.target_plane_filter_min_inlier_ratio),
            "target_plane_filter_max_inlier_ratio": float(self.target_plane_filter_max_inlier_ratio),
            "target_plane_filter_max_planes": int(self.target_plane_filter_max_planes),
            "target_plane_filter_ransac_iterations": int(self.target_plane_filter_ransac_iterations),
            "target_depth_band_filter_enabled": bool(self.target_depth_band_filter_enabled),
            "target_depth_band_filter_range_m": float(self.target_depth_band_filter_range_m),
            "target_depth_band_filter_min_valid_pixels": int(self.target_depth_band_filter_min_valid_pixels),
            "target_depth_band_filter_min_keep_pixels": int(self.target_depth_band_filter_min_keep_pixels),
            "target_3d_mask_erode_kernel": int(self.target_3d_mask_erode_kernel),
            "save_normal": bool(self.save_normal),
            "image_processor_load_time_sec": self.image_processor_load_time_sec,
            "video_predictor_load_time_sec": self.video_predictor_load_time_sec,
            "active_camera_ids": list(self.active_camera_ids),
            "seed_info_by_camera": self.seed_info_by_camera,
            "startup_time_before_streaming_sec": self.startup_time_before_streaming,
            "first_frame_ready_sec": self.first_frame_ready_time,
            "later_frame_runtime_sec_mean": later_mean,
            "later_frame_fps": None if later_mean in {None, 0.0} else float(1.0 / later_mean),
            "later_loop_runtime_sec_mean": later_loop_mean,
            "later_loop_fps": None if later_loop_mean in {None, 0.0} else float(1.0 / later_loop_mean),
            "later_capture_time_sec_mean": later_capture_mean,
            "later_build_camera_inputs_time_sec_mean": later_rgbd_mean,
            "later_process_frame_time_sec_mean": later_process_wall_mean,
            "timeline": self.timeline,
        }
        (self.output_dir.parent / "single_object_timeline.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def close(self) -> None:
        """关闭分割器，释放资源并保存摘要信息。"""
        if self.closed:
            return
        if hasattr(self.video_predictor, "close_session"):
            for session_id in list(self.session_ids.values()):
                with contextlib.suppress(Exception):
                    self.video_predictor.close_session(session_id=session_id)
        self.write_summary()
        self.closed = True

    def __enter__(self) -> "SingleObjectPointCloudSegmenter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def update_frame_metadata(self, metadata: dict[str, object]) -> None:
        self.timeline[-1].update(metadata)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="为 LIBERO 空间任务的 RGBD 剧集运行快速单物体在线分割。")
    parser.add_argument("--target-name", default="akita_black_bowl", help="目标物体名称")
    parser.add_argument("--episode-dir", type=Path, default=DEFAULT_EPISODE_DIR, help="剧集数据目录")
    parser.add_argument("--prompt-task-info", type=Path, default=DEFAULT_PROMPT_TASK_INFO, help="提示任务信息路径")
    parser.add_argument("--prompt-image-root", type=Path, default=DEFAULT_PROMPT_IMAGE_ROOT, help="提示图像根目录")
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT, help="模型权重路径")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="结果输出目录")
    parser.add_argument("--max-frames", type=int, default=5, help="最大处理帧数")
    parser.add_argument("--stride", type=int, default=2, help="帧采样步长")
    parser.add_argument("--frame-voxel-size", type=float, default=0.003, help="点云下采样的体素大小")
    parser.add_argument("--target-cluster-filter-enabled", type=int, default=0, help="是否启用目标点 3D 聚类去散点")
    parser.add_argument("--target-cluster-radius-m", type=float, default=0.03, help="目标点 3D 聚类邻域半径（米）")
    parser.add_argument("--target-cluster-min-points", type=int, default=30, help="形成有效目标簇所需的最少点数")
    parser.add_argument("--target-cluster-keep-largest", type=int, default=1, help="是否只保留最大目标簇")
    parser.add_argument("--target-plane-filter-enabled", type=int, default=0, help="是否启用目标点主平面剔除")
    parser.add_argument("--target-plane-filter-distance-m", type=float, default=0.004, help="目标主平面内点距离阈值（米）")
    parser.add_argument("--target-plane-filter-min-points", type=int, default=80, help="目标主平面最少内点数")
    parser.add_argument("--target-plane-filter-min-inlier-ratio", type=float, default=0.25, help="目标主平面最低内点比例")
    parser.add_argument("--target-plane-filter-max-inlier-ratio", type=float, default=0.85, help="目标主平面最高内点比例")
    parser.add_argument("--target-plane-filter-max-planes", type=int, default=1, help="单帧单相机最多剔除几个目标主平面")
    parser.add_argument("--target-plane-filter-ransac-iterations", type=int, default=256, help="目标主平面 RANSAC 迭代次数")
    parser.add_argument("--target-depth-band-filter-enabled", type=int, default=0, help="是否按目标核心深度带过滤 3D 取点 mask")
    parser.add_argument("--target-depth-band-filter-range-m", type=float, default=0.015, help="目标核心深度带半径（米）")
    parser.add_argument("--target-depth-band-filter-min-valid-pixels", type=int, default=50, help="估计目标核心深度所需最少有效像素")
    parser.add_argument("--target-depth-band-filter-min-keep-pixels", type=int, default=20, help="深度带过滤后至少保留的像素数")
    parser.add_argument("--target-3d-mask-erode-kernel", type=int, default=0, help="反投影前仅用于 3D 取点的目标 mask 腐蚀核大小，0/1 表示关闭")
    parser.add_argument("--prompt-keep-score-threshold", type=float, default=0.2, help="保留提示掩码的评分阈值")
    parser.add_argument("--depth-scale", type=float, default=1000.0, help="深度图缩放比例")
    parser.add_argument("--depth-min", type=float, default=0.1, help="最小有效深度")
    parser.add_argument("--depth-max", type=float, default=3.0, help="最大有效深度")
    parser.add_argument("--confidence", type=float, default=0.25, help="检测置信度")
    parser.add_argument("--mask-threshold", type=float, default=0.6, help="分割掩码阈值")
    parser.add_argument("--video-mask-prob-threshold", type=float, default=0.95, help="视频掩码概率阈值")
    parser.add_argument("--tracker-image-size", type=int, default=DEFAULT_TRACKER_IMAGE_SIZE, help="追踪器图像尺寸")
    parser.add_argument("--save-ply", action="store_true", default=False, help="是否保存 PLY 点云文件")
    parser.add_argument(
        "--save-normal",
        "--save-normals",
        dest="save_normal",
        action="store_true",
        default=False,
        help="保存 PLY 时是否写入估计法线",
    )
    parser.add_argument("--save-debug-2d", action="store_true", default=False, help="是否保存 2D 调试图")
    parser.add_argument("--overwrite-output", action="store_true", help="是否覆盖已有的输出目录")
    return parser.parse_args()


def run_demo(args: argparse.Namespace) -> None:
    """运行演示程序。"""
    episode_dir = Path(args.episode_dir).resolve()
    camera_records = load_episode_camera_records(episode_dir)
    camera_ids = [str(record["camera_id"]) for record in camera_records]
    frame_names = collect_common_frame_names(episode_dir, camera_ids)
    if int(args.max_frames) > 0:
        frame_names = frame_names[: int(args.max_frames)]
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
        depth_scale=float(args.depth_scale),
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
        tracker_image_size=args.tracker_image_size,
    ) as segmenter:
        t0 = time.perf_counter()
        for frame_name in frame_names:
            camera_inputs = load_episode_frame_inputs(
                episode_dir=episode_dir,
                frame_name=frame_name,
                camera_records=camera_records,
                depth_scale=float(args.depth_scale),
            )
            result = segmenter.process_frame(
                frame_name=frame_name,
                camera_inputs=camera_inputs,
            )
            print(
                f"[frame {result['frame_index']:03d}] {frame_name} points={result['points_xyz'].shape[0]} "
                f"runtime={segmenter.timeline[-1]['frame_runtime_sec']:.3f}s"
            )
        elapsed = time.perf_counter() - t0
        print(f"Processed {len(frame_names)} frames in {elapsed:.2f}s")
        print(f"Output dir: {segmenter.output_dir}")


def main() -> None:
    run_demo(parse_args())


if __name__ == "__main__":
    main()
