#!/usr/bin/env python3
"""Single-object online RGBD point-cloud segmentation with positive/negative prompt boxes."""

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
DEFAULT_EPISODE_DIR = REPO_ROOT / "examples" / "data" / "libero_spatial" / "task_00_demo" / "episode_0001"
DEFAULT_PROMPT_IMAGE_ROOT = REPO_ROOT / "assets" / "prompts" / "libero_spatial" / "semantic_split_parts"
DEFAULT_PROMPT_TASK_INFO = DEFAULT_PROMPT_IMAGE_ROOT / "task_info.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tests" / "outputs" / "demo_spatial_single_object"


def resolve_default_checkpoint() -> Path:
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

for sam3_root in (
    Path(os.environ["SAM3_REPO_ROOT"]).expanduser() if os.environ.get("SAM3_REPO_ROOT") else None,
    REPO_ROOT / "third_party" / "sam3",
):
    if sam3_root is not None and sam3_root.exists() and str(sam3_root) not in sys.path:
        sys.path.insert(0, str(sam3_root))


@dataclass(frozen=True)
class PromptEntry:
    semantic_name: str
    image_name: str
    source_path: Path
    box_xyxy: list[int]


@dataclass(frozen=True)
class CameraFrame:
    camera_id: str
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: dict[str, float] | None
    pose_record: dict[str, object]
    fovy_deg: float | None


def resolve_repo_path(path_like: str | os.PathLike[str] | Path, *, base_dir: Path = REPO_ROOT) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


@dataclass(frozen=True)
class SingleSegConfig:
    """Serializable initialization config for SingleObjectPointCloudSegmenter."""

    target_name: str = "plate"
    prompt_task_info: Path = DEFAULT_PROMPT_TASK_INFO
    prompt_image_root: Path = DEFAULT_PROMPT_IMAGE_ROOT
    checkpoint_path: Path = DEFAULT_CHECKPOINT
    output_dir: Path = DEFAULT_OUTPUT_DIR
    overwrite_output: bool = False
    confidence: float = 0.25
    mask_threshold: float = 0.6
    prompt_keep_score_threshold: float = 0.2
    prompt_max_masks: int = 4
    prompt_ref_cell: int = 160
    prompt_max_cols: int = 10
    prompt_canvas_gap: int = 24
    seed_min_pixels: int = 200
    seed_max_area_ratio: float = 0.35
    seed_box_margin: int = 12
    video_object_min_score: float = 0.0
    video_mask_prob_threshold: float = 0.95
    depth_scale: float = 1000.0
    depth_min: float = 0.1
    depth_max: float = 3.0
    stride: int = 2
    frame_voxel_size: float = 0.003
    save_ply: bool = True
    save_debug_2d: bool = False
    sam3_image_device: str | None = None
    video_backend: str = "tracker_only_stitched"
    compile_video_predictor: bool = False
    tracker_profile: str = "default"
    tracker_image_size: int | None = 896
    stitched_roi_tracking: bool = False
    stitched_roi_margin_scale: float = 2.0
    stitched_roi_min_size_ratio: float = 0.35
    sync_timing: bool | None = None

    @classmethod
    def from_mapping(
        cls,
        payload: dict[str, Any],
        *,
        base_dir: Path = REPO_ROOT,
    ) -> "SingleSegConfig":
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
        config_path = resolve_repo_path(config_path)
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return cls.from_mapping(payload, base_dir=REPO_ROOT)

    @classmethod
    def from_file(cls, config_path: Path | str) -> "SingleSegConfig":
        return cls.from_yaml(config_path)

    def with_overrides(self, **overrides: Any) -> "SingleSegConfig":
        merged = {field_name: getattr(self, field_name) for field_name in self.__dataclass_fields__}
        for key, value in overrides.items():
            if value is None or key not in merged:
                continue
            merged[key] = value
        return SingleSegConfig(**merged)

    def to_segmenter_kwargs(self) -> dict[str, Any]:
        return {field_name: getattr(self, field_name) for field_name in self.__dataclass_fields__}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def semantic_name_from_asset(asset_name: str) -> str:
    return re.sub(r"_\d+$", "", str(asset_name))


def collect_common_frame_names(episode_dir: Path, camera_ids: list[str]) -> list[str]:
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
    payload = load_json(episode_dir / "camera_extrinsics.json")
    cameras = payload.get("cameras", [])
    if not isinstance(cameras, list) or not cameras:
        raise ValueError(f"camera_extrinsics.json does not contain cameras: {episode_dir}")
    return cameras


def normalize_intrinsics_payload(intrinsics: object | None) -> dict[str, float] | None:
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
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    depth_m = np.asarray(Image.open(depth_path), dtype=np.float32) / float(depth_scale)
    return rgb, depth_m


def load_episode_frame_inputs(
    episode_dir: Path,
    frame_name: str,
    camera_records: list[dict[str, object]],
    depth_scale: float,
) -> dict[str, dict[str, object]]:
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
    cols = min(max(max_cols, 1), max(1, math.ceil(math.sqrt(len(prompt_ids)))))
    layout: dict[str, tuple[int, int]] = {}
    for idx, prompt_id in enumerate(prompt_ids):
        col = idx % cols
        row = idx // cols
        layout[prompt_id] = (col * ref_cell, row * ref_cell)
    return layout


def scale_bbox_to_layout(box_xyxy: list[int], source_size: list[int], pasted_size: list[int]) -> list[float]:
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
    if torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def no_autocast_context(device: torch.device | None = None):
    if device is not None and device.type == "cuda":
        return torch.autocast("cuda", enabled=False)
    return contextlib.nullcontext()


def maybe_cuda_synchronize(device: torch.device | None, enabled: bool) -> None:
    if enabled and device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def load_sam3_image_processor(
    checkpoint_path: Path,
    confidence: float,
    mask_threshold: float,
    device: str | None = None,
):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint_path}")
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    resolved_device = str(device).strip() if device is not None else ""
    if not resolved_device:
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
            confidence_threshold=float(confidence),
            mask_threshold=float(mask_threshold),
        )
    except TypeError:
        return Sam3Processor(
            model,
            confidence_threshold=float(confidence),
        )


def load_sam3_video_predictor(
    checkpoint_path: Path,
    *,
    compile_video_predictor: bool = False,
):
    if not torch.cuda.is_available():
        raise RuntimeError("SAM3 video predictor requires CUDA")
    from sam3.model_builder import build_sam3_video_predictor

    predictor = build_sam3_video_predictor(
        checkpoint_path=str(checkpoint_path),
        gpus_to_use=[torch.cuda.current_device()],
        async_loading_frames=True,
        compile=bool(compile_video_predictor),
    )
    return predictor


def load_video_backend_predictor(
    checkpoint_path: Path,
    *,
    video_backend: str,
    compile_video_predictor: bool = False,
    tracker_profile: str = "default",
    tracker_image_size: int | None = None,
):
    backend = str(video_backend).strip().lower()
    if backend == "video_model":
        return load_sam3_video_predictor(
            checkpoint_path=checkpoint_path,
            compile_video_predictor=bool(compile_video_predictor),
        )
    if backend == "tracker_only":
        try:
            from single_seg.tracker_only_backend import TrackerOnlyVideoPredictor
        except ImportError:
            from tracker_only_backend import TrackerOnlyVideoPredictor

        return TrackerOnlyVideoPredictor(
            checkpoint_path=checkpoint_path,
            compile_model=bool(compile_video_predictor),
            tracker_profile=str(tracker_profile),
            tracker_image_size=tracker_image_size,
        )
    if backend == "tracker_only_stitched":
        try:
            from single_seg.tracker_only_backend import TrackerOnlyVideoPredictor
        except ImportError:
            from tracker_only_backend import TrackerOnlyVideoPredictor

        return TrackerOnlyVideoPredictor(
            checkpoint_path=checkpoint_path,
            compile_model=bool(compile_video_predictor),
            tracker_profile=str(tracker_profile),
            tracker_image_size=tracker_image_size,
        )
    raise ValueError(f"Unsupported video_backend: {video_backend}")


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
    if torch.is_tensor(array_like):
        tensor = array_like.to(device=device, non_blocking=True)
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        return tensor
    return torch.as_tensor(array_like, dtype=dtype, device=device)


def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    positive = x >= 0
    out = np.empty_like(x, dtype=np.float32)
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def prob_threshold_to_logit(threshold: float) -> float:
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        labels = mask[valid].to(torch.int32)
        return pts_world, colors, labels


def fuse_scene_geometry(
    point_chunks: list[np.ndarray],
    color_chunks: list[np.ndarray],
    label_chunks: list[np.ndarray],
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    points = torch.cat(point_chunks, dim=0)
    colors = torch.cat(color_chunks, dim=0)
    labels = torch.cat(label_chunks, dim=0).to(torch.int32)
    if points.numel() == 0 or float(voxel_size) <= 0.0:
        return points.to(torch.float32), colors.to(torch.uint8), labels

    with no_autocast_context(device):
        # Match the historical NumPy path, which promotes float32 points to float64
        # when dividing by the Python float voxel size before applying floor().
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
        return down_points, down_colors, label_max


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
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


def write_label_ply(path: Path, points: np.ndarray, labels: np.ndarray) -> None:
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


def save_binary_mask_debug(
    output_dir: Path,
    frame_name: str,
    camera_id: str,
    rgb: np.ndarray,
    mask: np.ndarray,
    score: float | None,
) -> None:
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
    """Fast single-object online segmenter using one positive class and negative boxes for all others."""

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
        prompt_max_masks: int = 4,
        prompt_ref_cell: int = 160,
        prompt_max_cols: int = 10,
        prompt_canvas_gap: int = 24,
        seed_min_pixels: int = 200,
        seed_max_area_ratio: float = 0.35,
        seed_box_margin: int = 12,
        video_object_min_score: float = 0.0,
        video_mask_prob_threshold: float = 0.95,
        depth_scale: float = 1000.0,
        depth_min: float = 0.1,
        depth_max: float = 3.0,
        stride: int = 2,
        frame_voxel_size: float = 0.003,
        save_ply: bool = True,
        save_debug_2d: bool = False,
        sam3_image_device: str | None = None,
        video_backend: str = "tracker_only_stitched",
        compile_video_predictor: bool = False,
        tracker_profile: str = "default",
        tracker_image_size: int | None = 896,
        stitched_roi_tracking: bool = False,
        stitched_roi_margin_scale: float = 2.0,
        stitched_roi_min_size_ratio: float = 0.35,
        sync_timing: bool | None = None,
    ) -> None:
        self.target_name = str(target_name)
        self.prompt_task_info = Path(prompt_task_info).resolve()
        self.prompt_image_root = Path(prompt_image_root).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.output_dir = Path(output_dir).resolve() if output_dir is not None else DEFAULT_OUTPUT_DIR.resolve()
        self.confidence = float(confidence)
        self.mask_threshold = float(mask_threshold)
        self.prompt_keep_score_threshold = float(prompt_keep_score_threshold)
        self.prompt_max_masks = int(prompt_max_masks)
        self.prompt_ref_cell = int(prompt_ref_cell)
        self.prompt_max_cols = int(prompt_max_cols)
        self.prompt_canvas_gap = int(prompt_canvas_gap)
        self.seed_min_pixels = int(seed_min_pixels)
        self.seed_max_area_ratio = float(seed_max_area_ratio)
        self.seed_box_margin = int(seed_box_margin)
        self.video_object_min_score = float(video_object_min_score)
        self.video_mask_prob_threshold = float(video_mask_prob_threshold)
        self.depth_scale = float(depth_scale)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.stride = int(stride)
        self.frame_voxel_size = float(frame_voxel_size)
        self.save_ply = bool(save_ply)
        self.save_debug_2d = bool(save_debug_2d)
        self.sam3_image_device = sam3_image_device
        self.video_backend = str(video_backend).strip().lower()
        self.compile_video_predictor = bool(compile_video_predictor)
        self.tracker_profile = str(tracker_profile).strip().lower()
        self.tracker_image_size = None if tracker_image_size is None else int(tracker_image_size)
        self.stitched_roi_tracking = bool(stitched_roi_tracking)
        self.stitched_roi_margin_scale = float(stitched_roi_margin_scale)
        self.stitched_roi_min_size_ratio = float(stitched_roi_min_size_ratio)
        if sync_timing is None:
            sync_timing = os.environ.get("SINGLE_SEG_SYNC_TIMING", "0") not in {"", "0", "false", "False"}
        self.sync_timing = bool(sync_timing)

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
            device=self.sam3_image_device,
        )
        self.image_processor_load_time_sec = time.perf_counter() - image_processor_t0
        video_predictor_t0 = time.perf_counter()
        self.video_predictor = load_video_backend_predictor(
            checkpoint_path=self.checkpoint_path,
            video_backend=self.video_backend,
            compile_video_predictor=self.compile_video_predictor,
            tracker_profile=self.tracker_profile,
            tracker_image_size=self.tracker_image_size,
        )
        self.video_predictor_load_time_sec = time.perf_counter() - video_predictor_t0

        self.session_ids: dict[str, str] = {}
        self.stitched_layout: Any | None = None
        self.stitched_crop_windows: dict[str, object] = {}
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

    def _build_frame_resources(self, camera_inputs: dict[str, dict[str, object]]) -> dict[str, list[Image.Image]]:
        resources: dict[str, list[Image.Image]] = {}
        for camera_id, payload in camera_inputs.items():
            resources[camera_id] = [Image.fromarray(np.asarray(payload["rgb"], dtype=np.uint8))]
        return resources

    def _get_torch_backproject_scales(
        self,
        *,
        height: int,
        width: int,
        intrinsics: dict[str, float] | None,
        fovy_deg: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def _uses_stitched_tracking(self) -> bool:
        return self.video_backend == "tracker_only_stitched"

    def _uses_stitched_roi_tracking(self) -> bool:
        return self._uses_stitched_tracking() and bool(self.stitched_roi_tracking)

    def _derive_stitched_crop_windows(
        self,
        frame_resources: dict[str, list[Image.Image]],
        masks_by_camera: dict[str, np.ndarray],
        previous_windows: dict[str, object] | None = None,
    ) -> dict[str, object]:
        try:
            from single_seg.tracker_only_backend import crop_window_from_mask, full_frame_crop_window
        except ImportError:
            from tracker_only_backend import crop_window_from_mask, full_frame_crop_window

        windows: dict[str, object] = {}
        previous_windows = previous_windows or {}
        for camera_id in self.active_camera_ids:
            image = frame_resources[camera_id][0]
            image_size = (int(image.width), int(image.height))
            mask = np.asarray(masks_by_camera.get(camera_id), dtype=bool) if camera_id in masks_by_camera else None
            if mask is not None and mask.shape == (image.height, image.width) and np.any(mask):
                windows[camera_id] = crop_window_from_mask(
                    mask,
                    image_size=image_size,
                    margin_scale=self.stitched_roi_margin_scale,
                    min_size_ratio=self.stitched_roi_min_size_ratio,
                )
            elif camera_id in previous_windows:
                windows[camera_id] = previous_windows[camera_id]
            else:
                windows[camera_id] = full_frame_crop_window(image_size)
        return windows

    def _build_roi_frame_resources(
        self,
        frame_resources: dict[str, list[Image.Image]],
        crop_windows: dict[str, object],
    ) -> dict[str, list[Image.Image]]:
        try:
            from single_seg.tracker_only_backend import crop_and_resize_frame
        except ImportError:
            from tracker_only_backend import crop_and_resize_frame

        roi_resources: dict[str, list[Image.Image]] = {}
        for camera_id in self.active_camera_ids:
            image = frame_resources[camera_id][0]
            crop_window = crop_windows[camera_id]
            roi_resources[camera_id] = [
                crop_and_resize_frame(
                    image,
                    crop_window,
                    output_size=(int(image.width), int(image.height)),
                )
            ]
        return roi_resources

    def _crop_masks_to_roi_view(
        self,
        masks_by_camera: dict[str, np.ndarray],
        frame_resources: dict[str, list[Image.Image]],
        crop_windows: dict[str, object],
    ) -> dict[str, np.ndarray]:
        try:
            from single_seg.tracker_only_backend import crop_mask_to_tracker_view
        except ImportError:
            from tracker_only_backend import crop_mask_to_tracker_view

        roi_masks: dict[str, np.ndarray] = {}
        for camera_id in self.active_camera_ids:
            if camera_id not in masks_by_camera:
                continue
            image = frame_resources[camera_id][0]
            roi_masks[camera_id] = crop_mask_to_tracker_view(
                np.asarray(masks_by_camera[camera_id], dtype=bool),
                crop_windows[camera_id],
                output_size=(int(image.width), int(image.height)),
            )
        return roi_masks

    def _project_roi_masks_to_full(
        self,
        roi_masks_by_camera: dict[str, np.ndarray],
        frame_resources: dict[str, list[Image.Image]],
        crop_windows: dict[str, object],
    ) -> dict[str, np.ndarray]:
        try:
            from single_seg.tracker_only_backend import project_tracker_mask_to_full_image
        except ImportError:
            from tracker_only_backend import project_tracker_mask_to_full_image

        full_masks: dict[str, np.ndarray] = {}
        for camera_id in self.active_camera_ids:
            image = frame_resources[camera_id][0]
            roi_mask = np.asarray(
                roi_masks_by_camera.get(camera_id, np.zeros((image.height, image.width), dtype=bool)),
                dtype=bool,
            )
            full_masks[camera_id] = project_tracker_mask_to_full_image(
                roi_mask,
                crop_windows[camera_id],
                full_size=(int(image.width), int(image.height)),
            )
        return full_masks

    def _initialize_sessions(
        self,
        frame_name: str,
        camera_inputs: dict[str, dict[str, object]],
        frame_resources: dict[str, list[Image.Image]] | None,
    ) -> None:
        seed_masks_by_camera: dict[str, np.ndarray] = {}
        active_camera_ids: list[str] = []
        for camera_id, payload in camera_inputs.items():
            image = Image.fromarray(np.asarray(payload["rgb"], dtype=np.uint8))
            debug_dir = self.output_dir / "prompt_debug" / camera_id if self.save_debug_2d else None
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
                selection = select_best_seed_mask(
                    boxes=boxes,
                    scores=scores,
                    masks=masks,
                    min_pixels=self.seed_min_pixels,
                )
                seed_source = "pos_only_fallback"
            if selection is None:
                continue
            seed_mask, seed_score, seed_box = selection
            seed_mask, seed_shape_mode = refine_seed_mask(
                seed_mask,
                seed_box,
                image_shape=np.asarray(payload["rgb"], dtype=np.uint8).shape[:2],
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
            if self.save_debug_2d:
                save_binary_mask_debug(
                    output_dir=self.output_dir,
                    frame_name=frame_name,
                    camera_id=camera_id,
                    rgb=np.asarray(payload["rgb"], dtype=np.uint8),
                    mask=np.asarray(seed_mask, dtype=bool),
                    score=seed_score,
                )
            if self._uses_stitched_tracking():
                continue
            if frame_resources is None:
                raise RuntimeError("frame_resources are required for per-camera tracking initialization")
            with autocast_context():
                session_id = self.video_predictor.start_session(frame_resources[camera_id])["session_id"]
                self.video_predictor.add_prompt(
                    session_id=session_id,
                    frame_idx=0,
                    mask=np.asarray(seed_mask, dtype=np.uint8),
                    obj_id=1,
                )
                self.session_ids[camera_id] = session_id
        self.active_camera_ids = active_camera_ids
        if not self.active_camera_ids:
            raise RuntimeError(f"No camera produced a usable seed for target {self.target_name!r}")
        if self._uses_stitched_tracking():
            try:
                from single_seg.tracker_only_backend import (
                    compose_camera_frame_resources,
                    compose_camera_rgb_frame_resources,
                    stitch_camera_binary_masks,
                )
            except ImportError:
                from tracker_only_backend import (
                    compose_camera_frame_resources,
                    compose_camera_rgb_frame_resources,
                    stitch_camera_binary_masks,
                )

            stitched_frame_resources = frame_resources
            stitched_seed_masks = seed_masks_by_camera
            self.stitched_crop_windows = {}
            if self._uses_stitched_roi_tracking():
                if frame_resources is None:
                    raise RuntimeError("frame_resources are required for stitched ROI tracking")
                self.stitched_crop_windows = self._derive_stitched_crop_windows(
                    frame_resources=frame_resources,
                    masks_by_camera=seed_masks_by_camera,
                )
                stitched_frame_resources = self._build_roi_frame_resources(
                    frame_resources=frame_resources,
                    crop_windows=self.stitched_crop_windows,
                )
                stitched_seed_masks = self._crop_masks_to_roi_view(
                    masks_by_camera=seed_masks_by_camera,
                    frame_resources=frame_resources,
                    crop_windows=self.stitched_crop_windows,
                )
                composite_resources, self.stitched_layout = compose_camera_frame_resources(
                    frame_resources=stitched_frame_resources,
                    camera_order=self.active_camera_ids,
                )
            else:
                composite_resources, self.stitched_layout = compose_camera_rgb_frame_resources(
                    rgb_by_camera={
                        camera_id: np.asarray(camera_inputs[camera_id]["rgb"], dtype=np.uint8)
                        for camera_id in self.active_camera_ids
                    },
                    camera_order=self.active_camera_ids,
                )
            composite_mask = stitch_camera_binary_masks(stitched_seed_masks, self.stitched_layout)
            with autocast_context():
                session_id = self.video_predictor.start_session(composite_resources)["session_id"]
                self.video_predictor.add_prompt(
                    session_id=session_id,
                    frame_idx=0,
                    mask=np.asarray(composite_mask, dtype=np.uint8),
                    obj_id=1,
                )
                self.session_ids["__stitched__"] = session_id
        self.startup_time_before_streaming = time.perf_counter() - self.pipeline_t0
        self.initialized = True

    def process_frame(
        self,
        *,
        frame_name: str,
        camera_inputs: dict[str, dict[str, object]],
        output_format: str = "numpy",
    ) -> dict[str, object]:
        """Process one timestep of multi-camera RGBD inputs and return a labeled point cloud."""
        if self.closed:
            raise RuntimeError("segmenter is already closed")
        if not camera_inputs:
            raise ValueError("camera_inputs must not be empty")
        output_format = str(output_format).strip().lower()
        if output_format not in {"numpy", "torch"}:
            raise ValueError(f"Unsupported output_format: {output_format!r}")
        frame_t0 = time.perf_counter()
        frame_resources: dict[str, list[Image.Image]] | None = None
        frame_resource_build_time = 0.0
        if (not self._uses_stitched_tracking()) or self._uses_stitched_roi_tracking():
            frame_resource_t0 = time.perf_counter()
            frame_resources = self._build_frame_resources(camera_inputs)
            frame_resource_build_time = time.perf_counter() - frame_resource_t0
        if not self.initialized:
            self._initialize_sessions(frame_name, camera_inputs, frame_resources)

        append_frame_time = 0.0
        backproject_time = 0.0
        fuse_time = 0.0
        compose_inputs_time = 0.0
        mask_postprocess_time = 0.0
        roi_project_time = 0.0
        roi_update_time = 0.0
        camera_prepare_time = 0.0
        camera_bookkeeping_time = 0.0
        cpu_transfer_time = 0.0
        colorize_time = 0.0
        masks_by_camera: dict[str, np.ndarray | torch.Tensor] = {}
        scores_by_camera: dict[str, float | None] = {}
        requests: list[dict[str, object]] = []

        if self._uses_stitched_tracking():
            try:
                from single_seg.tracker_only_backend import (
                    compose_camera_frame_resources,
                    compose_camera_rgb_frame_resources,
                    split_stitched_binary_mask,
                    split_stitched_binary_mask_torch,
                )
            except ImportError:
                from tracker_only_backend import (
                    compose_camera_frame_resources,
                    compose_camera_rgb_frame_resources,
                    split_stitched_binary_mask,
                    split_stitched_binary_mask_torch,
                )

            stitched_frame_resources = frame_resources
            current_crop_windows = dict(self.stitched_crop_windows)
            if self._uses_stitched_roi_tracking():
                if frame_resources is None:
                    raise RuntimeError("frame_resources are required for stitched ROI tracking")
                stitched_frame_resources = self._build_roi_frame_resources(
                    frame_resources=frame_resources,
                    crop_windows=current_crop_windows,
                )
                compose_t0 = time.perf_counter()
                composite_resources, current_layout = compose_camera_frame_resources(
                    frame_resources=stitched_frame_resources,
                    camera_order=self.active_camera_ids,
                    layout=self.stitched_layout,
                )
                compose_inputs_time += time.perf_counter() - compose_t0
            else:
                compose_t0 = time.perf_counter()
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
            if self._uses_stitched_roi_tracking():
                stitched_mask: np.ndarray | torch.Tensor = np.zeros(
                    (int(current_layout.canvas_height), int(current_layout.canvas_width)),
                    dtype=bool,
                )
                if stitched_payload.get("outputs") is not None:
                    stitched_output = _score_label_map_from_output(
                        stitched_payload["outputs"],
                        image_shape=(int(current_layout.canvas_height), int(current_layout.canvas_width)),
                        min_object_score=self.video_object_min_score,
                        mask_prob_threshold=self.video_mask_prob_threshold,
                    )
                    stitched_mask = np.asarray(stitched_output["label_map"] == 1, dtype=bool)
                    if isinstance(stitched_output.get("object_stats"), dict) and 1 in stitched_output["object_stats"]:
                        stitched_score = float(stitched_output["object_stats"][1]["score"])
                roi_masks_by_camera = split_stitched_binary_mask(stitched_mask, current_layout)
            else:
                stitched_output_t = {"mask": torch.zeros((int(current_layout.canvas_height), int(current_layout.canvas_width)), dtype=torch.bool, device=self.tensor_device), "score": None, "object_stats": {}}
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
                roi_masks_by_camera = split_stitched_binary_mask_torch(stitched_output_t["mask"], current_layout)
            maybe_cuda_synchronize(self.tensor_device, self.sync_timing)
            mask_postprocess_time += time.perf_counter() - mask_post_t0
            if self._uses_stitched_roi_tracking():
                roi_project_t0 = time.perf_counter()
                masks_by_camera = self._project_roi_masks_to_full(
                    roi_masks_by_camera=roi_masks_by_camera,
                    frame_resources=frame_resources,
                    crop_windows=current_crop_windows,
                )
                roi_project_time += time.perf_counter() - roi_project_t0
            else:
                masks_by_camera = roi_masks_by_camera
            scores_by_camera = {camera_id: stitched_score for camera_id in self.active_camera_ids}
        else:
            if frame_resources is None:
                raise RuntimeError("frame_resources are required for per-camera tracking")
            for camera_id in self.active_camera_ids:
                payload = camera_inputs[camera_id]
                if self.frame_index > 0:
                    maybe_cuda_synchronize(self.tensor_device, self.sync_timing)
                    append_t0 = time.perf_counter()
                    self.video_predictor.append_frame(
                        session_id=self.session_ids[camera_id],
                        resource_path=frame_resources[camera_id],
                    )
                    maybe_cuda_synchronize(self.tensor_device, self.sync_timing)
                    append_frame_time += time.perf_counter() - append_t0
                rgb = np.asarray(payload["rgb"], dtype=np.uint8)
                requests.append(
                    {
                        "camera_id": camera_id,
                        "session_id": self.session_ids[camera_id],
                        "frame_idx": int(self.frame_index),
                        "image_shape": (rgb.shape[0], rgb.shape[1]),
                    }
                )

            maybe_cuda_synchronize(self.tensor_device, self.sync_timing)
            propagate_t0 = time.perf_counter()
            batch_outputs = extract_target_mask_outputs_batch(
                video_predictor=self.video_predictor,
                requests=requests,
                min_object_score=self.video_object_min_score,
                mask_prob_threshold=self.video_mask_prob_threshold,
                target_obj_id=1,
                device=self.tensor_device,
            )
            maybe_cuda_synchronize(self.tensor_device, self.sync_timing)
            propagate_time = time.perf_counter() - propagate_t0
            for request in requests:
                camera_id = str(request["camera_id"])
                frame_output = batch_outputs.get(camera_id)
                if frame_output is None:
                    continue
                masks_by_camera[camera_id] = frame_output["mask"]
                scores_by_camera[camera_id] = None if frame_output.get("score") is None else float(frame_output["score"])

        if self._uses_stitched_roi_tracking():
            roi_update_t0 = time.perf_counter()
            self.stitched_crop_windows = self._derive_stitched_crop_windows(
                frame_resources=frame_resources,
                masks_by_camera=masks_by_camera,
                previous_windows=current_crop_windows,
            )
            roi_update_time += time.perf_counter() - roi_update_t0

        point_chunks: list[torch.Tensor] = []
        color_chunks: list[torch.Tensor] = []
        label_chunks: list[torch.Tensor] = []
        camera_summaries: list[dict[str, object]] = []
        for camera_id in self.active_camera_ids:
            camera_prepare_t0 = time.perf_counter()
            payload = camera_inputs[camera_id]
            rgb = np.asarray(payload["rgb"], dtype=np.uint8)
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
            score = scores_by_camera.get(camera_id)
            if self.save_debug_2d:
                save_binary_mask_debug(
                    output_dir=self.output_dir,
                    frame_name=frame_name,
                    camera_id=camera_id,
                    rgb=rgb,
                    mask=mask_t.detach().cpu().numpy(),
                    score=score,
                )
            intrinsics = normalize_intrinsics_payload(payload.get("intrinsics"))
            pose_record = normalize_pose_record(camera_id, payload)
            depth_m = np.asarray(payload["depth_m"], dtype=np.float32)
            fovy_deg = float(payload["fovy_deg"]) if payload.get("fovy_deg") is not None else None
            x_scale, y_scale = self._get_torch_backproject_scales(
                height=rgb.shape[0],
                width=rgb.shape[1],
                intrinsics=intrinsics,
                fovy_deg=fovy_deg,
            )
            camera_prepare_time += time.perf_counter() - camera_prepare_t0
            maybe_cuda_synchronize(self.tensor_device, self.sync_timing)
            camera_backproject_t0 = time.perf_counter()
            points, colors, point_labels = backproject_scene_points_with_labels_torch(
                sampled_rgb=np.ascontiguousarray(rgb[:: self.stride, :: self.stride]),
                sampled_depth_m=np.ascontiguousarray(depth_m[:: self.stride, :: self.stride]),
                sampled_mask=mask_t[:: self.stride, :: self.stride],
                cam2world_gl=np.asarray(pose_record["cam2world_4x4"], dtype=np.float64),
                x_scale=x_scale,
                y_scale=y_scale,
                depth_min=self.depth_min,
                depth_max=self.depth_max,
                device=self.tensor_device,
            )
            maybe_cuda_synchronize(self.tensor_device, self.sync_timing)
            camera_backproject_time = time.perf_counter() - camera_backproject_t0
            backproject_time += camera_backproject_time
            camera_bookkeeping_t0 = time.perf_counter()
            if int(points.shape[0]) > 0:
                point_chunks.append(points)
                color_chunks.append(colors)
                label_chunks.append(point_labels)
            camera_summaries.append(
                {
                    "camera_id": camera_id,
                    "target_pixels": int(torch.count_nonzero(mask_t).item()),
                    "num_points_backprojected": int(points.shape[0]),
                    "num_target_points_backprojected": int(torch.count_nonzero(point_labels).item()),
                    "backproject_time_sec": camera_backproject_time,
                }
            )
            camera_bookkeeping_time += time.perf_counter() - camera_bookkeeping_t0

        maybe_cuda_synchronize(self.tensor_device, self.sync_timing)
        fuse_t0 = time.perf_counter()
        if point_chunks:
            points_xyz_t, raw_colors_t, labels_t = fuse_scene_geometry_torch(
                point_chunks=point_chunks,
                color_chunks=color_chunks,
                label_chunks=label_chunks,
                voxel_size=self.frame_voxel_size,
                device=self.tensor_device,
            )
        else:
            points_xyz_t = torch.empty((0, 3), dtype=torch.float32, device=self.tensor_device)
            raw_colors_t = torch.empty((0, 3), dtype=torch.uint8, device=self.tensor_device)
            labels_t = torch.empty((0,), dtype=torch.int32, device=self.tensor_device)
        maybe_cuda_synchronize(self.tensor_device, self.sync_timing)
        fuse_time = time.perf_counter() - fuse_t0
        need_cpu_output = self.save_ply or output_format == "numpy"
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
                vis_colors[labels > 0] = np.array([255, 70, 70], dtype=np.uint8)
            colorize_time += time.perf_counter() - colorize_t0
        else:
            colorize_t0 = time.perf_counter()
            vis_colors_t = raw_colors_t.clone()
            if vis_colors_t.shape[0] > 0:
                vis_colors_t[labels_t > 0] = torch.tensor([255, 70, 70], dtype=torch.uint8, device=self.tensor_device)
            colorize_time += time.perf_counter() - colorize_t0

        save_t0 = time.perf_counter()
        if self.save_ply:
            assert points_xyz is not None and raw_colors is not None and labels is not None and vis_colors is not None
            frame_stem = frame_name.replace(".png", "")
            write_ply(self.frame_output_dir / f"{frame_stem}_scene_rgb.ply", points_xyz, raw_colors)
            write_ply(self.frame_output_dir / f"{frame_stem}_instance_rgb.ply", points_xyz, vis_colors)
            write_label_ply(self.frame_output_dir / f"{frame_stem}_instance_label.ply", points_xyz, labels)
            meta = {
                "frame_name": frame_name,
                "target_name": self.target_name,
                "num_points": int(points_xyz.shape[0]),
                "num_labeled_points": int(np.count_nonzero(labels)),
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
                "camera_summaries": camera_summaries,
                "append_frame_time_sec": append_frame_time,
                "propagate_time_sec": propagate_time,
                "backproject_time_sec": backproject_time,
                "fuse_time_sec": fuse_time,
                "save_time_sec": save_time,
                "frame_runtime_sec": frame_runtime,
                "residual_breakdown_sec": {
                    "frame_resource_build_time_sec": frame_resource_build_time,
                    "compose_inputs_time_sec": compose_inputs_time,
                    "mask_postprocess_time_sec": mask_postprocess_time,
                    "roi_project_time_sec": roi_project_time,
                    "roi_update_time_sec": roi_update_time,
                    "camera_prepare_time_sec": camera_prepare_time,
                    "camera_bookkeeping_time_sec": camera_bookkeeping_time,
                    "cpu_transfer_time_sec": cpu_transfer_time,
                    "colorize_time_sec": colorize_time,
                },
            }
        )
        if output_format == "torch":
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
                "label_values": [0, 1],
                "palette": torch.tensor([[255, 70, 70]], dtype=torch.uint8, device=self.tensor_device),
                "camera_summaries": camera_summaries,
                "meta": {
                    "frame_name": frame_name,
                    "target_name": self.target_name,
                    "num_points": int(points_xyz_t.shape[0]),
                    "num_labeled_points": int(torch.count_nonzero(labels_t).item()),
                    "camera_summaries": camera_summaries,
                    "output_format": "torch",
                },
            }
        else:
            assert points_xyz is not None and raw_colors is not None and labels is not None and vis_colors is not None
            result = {
                "frame_index": int(self.frame_index),
                "frame_name": frame_name,
                "points_xyz": points_xyz,
                "instance_labels": labels,
                "instance_colors": vis_colors,
                "raw_colors": raw_colors,
                "semantic_labels": labels.copy(),
                "semantic_colors": vis_colors,
                "label_names": [self.target_name],
                "label_values": [0, 1],
                "palette": np.asarray([[255, 70, 70]], dtype=np.uint8),
                "camera_summaries": camera_summaries,
                "meta": {
                    "frame_name": frame_name,
                    "target_name": self.target_name,
                    "num_points": int(points_xyz.shape[0]),
                    "num_labeled_points": int(np.count_nonzero(labels)),
                    "camera_summaries": camera_summaries,
                    "output_format": "numpy",
                },
            }
        self.frame_index += 1
        return result

    def write_summary(self) -> None:
        later = self.timeline[1:] if len(self.timeline) > 1 else []
        later_mean = (
            float(sum(item["frame_runtime_sec"] for item in later) / len(later))
            if later
            else None
        )
        summary = {
            "target_name": self.target_name,
            "prompt_task_info": str(self.prompt_task_info),
            "prompt_image_root": str(self.prompt_image_root),
            "checkpoint_path": str(self.checkpoint_path),
            "video_backend": self.video_backend,
            "tracker_profile": self.tracker_profile,
            "tracker_image_size": self.tracker_image_size,
            "stitched_roi_tracking": bool(self.stitched_roi_tracking),
            "stitched_roi_margin_scale": float(self.stitched_roi_margin_scale),
            "stitched_roi_min_size_ratio": float(self.stitched_roi_min_size_ratio),
            "compile_video_predictor": bool(self.compile_video_predictor),
            "confidence": float(self.confidence),
            "mask_threshold": float(self.mask_threshold),
            "video_object_min_score": float(self.video_object_min_score),
            "video_mask_prob_threshold": float(self.video_mask_prob_threshold),
            "image_processor_load_time_sec": self.image_processor_load_time_sec,
            "video_predictor_load_time_sec": self.video_predictor_load_time_sec,
            "active_camera_ids": list(self.active_camera_ids),
            "seed_info_by_camera": self.seed_info_by_camera,
            "startup_time_before_streaming_sec": self.startup_time_before_streaming,
            "first_frame_ready_sec": self.first_frame_ready_time,
            "later_frame_runtime_sec_mean": later_mean,
            "later_frame_fps": None if later_mean in {None, 0.0} else float(1.0 / later_mean),
            "timeline": self.timeline,
        }
        (self.output_dir / "single_object_timeline.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def close(self) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fast single-object online segmentation for LIBERO spatial RGBD episodes.")
    parser.add_argument("--target-name", default="akita_black_bowl")
    parser.add_argument("--episode-dir", type=Path, default=DEFAULT_EPISODE_DIR)
    parser.add_argument("--prompt-task-info", type=Path, default=DEFAULT_PROMPT_TASK_INFO)
    parser.add_argument("--prompt-image-root", type=Path, default=DEFAULT_PROMPT_IMAGE_ROOT)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-frames", type=int, default=5)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--frame-voxel-size", type=float, default=0.003)
    parser.add_argument("--prompt-ref-cell", type=int, default=160)
    parser.add_argument("--prompt-max-cols", type=int, default=10)
    parser.add_argument("--prompt-canvas-gap", type=int, default=24)
    parser.add_argument("--prompt-keep-score-threshold", type=float, default=0.2)
    parser.add_argument("--prompt-max-masks", type=int, default=4)
    parser.add_argument("--seed-min-pixels", type=int, default=200)
    parser.add_argument("--seed-max-area-ratio", type=float, default=0.35)
    parser.add_argument("--seed-box-margin", type=int, default=12)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--depth-min", type=float, default=0.1)
    parser.add_argument("--depth-max", type=float, default=3.0)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--mask-threshold", type=float, default=0.6)
    parser.add_argument("--video-object-min-score", type=float, default=0.0)
    parser.add_argument("--video-mask-prob-threshold", type=float, default=0.95)
    parser.add_argument("--sam3-image-device", default=None)
    parser.add_argument("--video-backend", choices=("video_model", "tracker_only", "tracker_only_stitched"), default="tracker_only_stitched")
    parser.add_argument("--tracker-profile", choices=("default", "lite"), default="default")
    parser.add_argument("--tracker-image-size", type=int, default=896)
    parser.add_argument("--stitched-roi-tracking", action="store_true")
    parser.add_argument("--stitched-roi-margin-scale", type=float, default=2.0)
    parser.add_argument("--stitched-roi-min-size-ratio", type=float, default=0.35)
    parser.add_argument("--compile-video-predictor", action="store_true")
    parser.add_argument("--output-format", choices=("numpy", "torch"), default="numpy")
    parser.add_argument("--save-ply", action="store_true", default=False)
    parser.add_argument("--save-debug-2d", action="store_true", default=False)
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def run_demo(args: argparse.Namespace) -> None:
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
        prompt_max_masks=int(args.prompt_max_masks),
        prompt_ref_cell=int(args.prompt_ref_cell),
        prompt_max_cols=int(args.prompt_max_cols),
        prompt_canvas_gap=int(args.prompt_canvas_gap),
        seed_min_pixels=int(args.seed_min_pixels),
        seed_max_area_ratio=float(args.seed_max_area_ratio),
        seed_box_margin=int(args.seed_box_margin),
        video_object_min_score=float(args.video_object_min_score),
        video_mask_prob_threshold=float(args.video_mask_prob_threshold),
        depth_scale=float(args.depth_scale),
        depth_min=float(args.depth_min),
        depth_max=float(args.depth_max),
        stride=int(args.stride),
        frame_voxel_size=float(args.frame_voxel_size),
        save_ply=bool(args.save_ply),
        save_debug_2d=bool(args.save_debug_2d),
        sam3_image_device=args.sam3_image_device,
        video_backend=str(args.video_backend),
        compile_video_predictor=bool(args.compile_video_predictor),
        tracker_profile=str(args.tracker_profile),
        tracker_image_size=args.tracker_image_size,
        stitched_roi_tracking=bool(args.stitched_roi_tracking),
        stitched_roi_margin_scale=float(args.stitched_roi_margin_scale),
        stitched_roi_min_size_ratio=float(args.stitched_roi_min_size_ratio),
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
                output_format=str(args.output_format),
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
