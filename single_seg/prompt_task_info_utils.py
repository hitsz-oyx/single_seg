from __future__ import annotations

import json
import numpy as np
from pathlib import Path
import re
from typing import Any


def semantic_name_from_asset(asset_name: str) -> str:
    return re.sub(r"_\d+$", "", str(asset_name))


def asset_index_from_name(asset_name: str, semantic_name: str) -> int | None:
    match = re.fullmatch(rf"{re.escape(semantic_name)}_(\d+)", str(asset_name))
    if match is None:
        return None
    return int(match.group(1))


def next_asset_index(assets: list[dict[str, Any]], semantic_name: str, *, start_index: int | None = None) -> int:
    next_index = 0 if start_index is None else int(start_index)
    for asset in assets:
        idx = asset_index_from_name(str(asset.get("asset_name", "")), semantic_name)
        if idx is None:
            continue
        next_index = max(next_index, idx + 1)
    return next_index


def relative_prompt_image_path(image_path: Path, prompt_image_root: Path) -> str:
    resolved_image = Path(image_path).resolve()
    resolved_root = Path(prompt_image_root).resolve()
    try:
        relative = resolved_image.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"image path {resolved_image} is not under prompt root {resolved_root}") from exc
    return relative.as_posix()


def default_annotated_dir(task_info_path: Path) -> Path:
    return Path(task_info_path).resolve().parent / "annotated"


def xywh_to_xyxy_inclusive(
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    image_width: int,
    image_height: int,
) -> list[int] | None:
    if int(w) <= 0 or int(h) <= 0:
        return None
    x0 = max(int(x), 0)
    y0 = max(int(y), 0)
    x1 = min(int(x) + int(w) - 1, int(image_width) - 1)
    y1 = min(int(y) + int(h) - 1, int(image_height) - 1)
    if x1 < x0 or y1 < y0:
        return None
    return [x0, y0, x1, y1]


def xyxy_inclusive_to_normalized_cxcywh(
    bbox_xyxy: list[int],
    *,
    image_width: int,
    image_height: int,
) -> list[float]:
    x0, y0, x1, y1 = [float(value) for value in bbox_xyxy]
    width = max(x1 - x0 + 1.0, 1.0)
    height = max(y1 - y0 + 1.0, 1.0)
    image_w = max(float(image_width), 1.0)
    image_h = max(float(image_height), 1.0)
    return [
        (x0 + 0.5 * width) / image_w,
        (y0 + 0.5 * height) / image_h,
        width / image_w,
        height / image_h,
    ]


def bbox_xyxy_from_mask(mask: np.ndarray) -> list[int] | None:
    mask_np = np.asarray(mask, dtype=bool)
    if mask_np.ndim == 3 and mask_np.shape[0] == 1:
        mask_np = mask_np[0]
    if mask_np.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask_np.shape}")
    ys, xs = np.nonzero(mask_np)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def expand_bbox_xyxy(
    bbox_xyxy: list[int] | None,
    *,
    image_shape: tuple[int, int],
    pad_ratio: float = 0.0,
    min_pad: int = 0,
) -> list[int] | None:
    if bbox_xyxy is None:
        return None
    x0, y0, x1, y1 = [int(value) for value in bbox_xyxy]
    width = max(x1 - x0 + 1, 1)
    height = max(y1 - y0 + 1, 1)
    pad_x = max(int(round(width * float(pad_ratio))), int(min_pad))
    pad_y = max(int(round(height * float(pad_ratio))), int(min_pad))
    image_h, image_w = [int(value) for value in image_shape]
    return [
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(image_w - 1, x1 + pad_x),
        min(image_h - 1, y1 + pad_y),
    ]


def select_best_mask_by_score(
    masks: np.ndarray,
    scores: np.ndarray,
    *,
    min_mask_pixels: int = 1,
) -> tuple[np.ndarray, float] | None:
    masks_np = np.asarray(masks, dtype=bool)
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np[:, 0]
    if masks_np.ndim != 3:
        raise ValueError(f"masks must have shape (N, H, W), got {masks_np.shape}")
    scores_np = np.asarray(scores, dtype=np.float32).reshape(-1)
    if masks_np.shape[0] != scores_np.shape[0]:
        raise ValueError("masks and scores must have the same first dimension")
    order = sorted(
        range(masks_np.shape[0]),
        key=lambda idx: (float(scores_np[idx]), int(np.count_nonzero(masks_np[idx]))),
        reverse=True,
    )
    for idx in order:
        mask = masks_np[idx]
        if int(np.count_nonzero(mask)) < int(min_mask_pixels):
            continue
        return mask, float(scores_np[idx])
    return None


def load_task_info(task_info_path: Path) -> dict[str, Any]:
    path = Path(task_info_path)
    if not path.exists():
        return {"assets": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    assets = payload.get("assets")
    if assets is None:
        payload["assets"] = []
        return payload
    if not isinstance(assets, list):
        raise ValueError(f"task_info assets must be a list: {path}")
    return payload


def write_task_info(task_info_path: Path, payload: dict[str, Any]) -> None:
    Path(task_info_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def upsert_prompt_assets(
    payload: dict[str, Any],
    *,
    prompt_image_root: Path,
    semantic_name: str,
    annotations: list[dict[str, Any]],
    start_index: int | None = None,
) -> dict[str, list[str]]:
    assets = payload.setdefault("assets", [])
    if not isinstance(assets, list):
        raise ValueError("task_info payload must contain an assets list")
    image_to_asset: dict[str, dict[str, Any]] = {}
    for asset in assets:
        image_path = asset.get("image_path")
        if isinstance(image_path, str):
            image_to_asset[image_path] = asset
    next_index = next_asset_index(assets, semantic_name, start_index=start_index)
    created: list[str] = []
    updated: list[str] = []
    for annotation in annotations:
        image_path = Path(annotation["image_path"]).resolve()
        bbox_xyxy = [int(value) for value in annotation["bbox_xyxy"]]
        relative_path = relative_prompt_image_path(image_path, prompt_image_root)
        existing_asset = image_to_asset.get(relative_path)
        if existing_asset is not None:
            existing_semantic = semantic_name_from_asset(str(existing_asset.get("asset_name", "")))
            if existing_semantic != semantic_name:
                raise ValueError(
                    f"image {relative_path} already belongs to semantic {existing_semantic!r}, not {semantic_name!r}"
                )
            existing_asset["bbox_xyxy"] = bbox_xyxy
            updated.append(str(existing_asset.get("asset_name", relative_path)))
            continue
        asset_name = f"{semantic_name}_{next_index}"
        next_index += 1
        asset = {
            "asset_name": asset_name,
            "image_path": relative_path,
            "bbox_xyxy": bbox_xyxy,
        }
        assets.append(asset)
        image_to_asset[relative_path] = asset
        created.append(asset_name)
    return {"created": created, "updated": updated}
