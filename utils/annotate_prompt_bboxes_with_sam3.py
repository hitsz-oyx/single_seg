#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
import sys
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from single_seg.prompt_task_info_utils import (
    bbox_xyxy_from_mask,
    default_annotated_dir,
    expand_bbox_xyxy,
    load_task_info,
    relative_prompt_image_path,
    select_best_mask_by_score,
    upsert_prompt_assets,
    write_task_info,
)


IMAGE_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.webp")


def default_checkpoint_path() -> Path:
    env_path = os.environ.get("SAM3_CHECKPOINT")
    candidates: list[Path] = []
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM3 with a text prompt on prompt images and derive bbox_xyxy from the best segmentation mask."
    )
    parser.add_argument("--prompt-image-root", type=Path, required=True, help="Root directory containing prompt images")
    parser.add_argument("--semantic-name", type=str, required=True, help="Semantic name, e.g. plate")
    parser.add_argument(
        "--text-prompt",
        type=str,
        default=None,
        help="SAM3 text prompt; defaults to semantic-name with underscores replaced by spaces",
    )
    parser.add_argument(
        "--task-info",
        type=Path,
        default=None,
        help="task_info.json path; defaults to <prompt-image-root>/task_info.json",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=default_checkpoint_path(),
        help="SAM3 checkpoint path",
    )
    parser.add_argument("--confidence", type=float, default=0.25, help="SAM3 confidence threshold")
    parser.add_argument("--mask-threshold", type=float, default=0.6, help="Reserved for SAM3 variants that accept it")
    parser.add_argument(
        "--images",
        type=Path,
        nargs="*",
        default=None,
        help="Optional explicit image paths to annotate; defaults to scanning the prompt root",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan images under prompt-image-root when --images is omitted",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images that already have task_info.json entries",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Optional starting index for new asset_name suffixes",
    )
    parser.add_argument(
        "--annotated-dir",
        "--preview-dir",
        type=Path,
        default=None,
        help="Directory to save SAM3 overlay images; defaults to <task-info-dir>/annotated",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned updates without writing task_info.json")
    parser.add_argument("--min-mask-pixels", type=int, default=64, help="Reject masks smaller than this area")
    parser.add_argument("--bbox-pad-ratio", type=float, default=0.0, help="Optional bbox expansion ratio")
    parser.add_argument("--bbox-min-pad", type=int, default=0, help="Optional minimum bbox expansion in pixels")
    return parser.parse_args()


def discover_images(prompt_image_root: Path, *, recursive: bool) -> list[Path]:
    image_paths: list[Path] = []
    for pattern in IMAGE_PATTERNS:
        matches: Iterable[Path]
        matches = prompt_image_root.rglob(pattern) if recursive else prompt_image_root.glob(pattern)
        image_paths.extend(Path(path).resolve() for path in matches)
    return sorted(dict.fromkeys(image_paths))


def ensure_sam3_import_path() -> None:
    sam3_root = REPO_ROOT / "third_party" / "sam3"
    if sam3_root.exists() and str(sam3_root) not in sys.path:
        sys.path.insert(0, str(sam3_root))


def load_sam3_processor(*, checkpoint_path: Path, confidence: float, mask_threshold: float):
    ensure_sam3_import_path()
    import torch
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint_path),
        load_from_HF=False,
        device=device,
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


def infer_refined_bbox(
    *,
    processor,
    image_bgr: np.ndarray,
    text_prompt: str,
    min_mask_pixels: int,
    bbox_pad_ratio: float,
    bbox_min_pad: int,
) -> tuple[list[int] | None, np.ndarray | None, float | None]:
    import torch

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    autocast_context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available()
        else contextlib.nullcontext()
    )
    with autocast_context:
        state = processor.set_image(image_pil)
        state = processor.set_text_prompt(state=state, prompt=text_prompt)
    if "masks" not in state or state["masks"] is None or "scores" not in state:
        return None, None, None
    selected = select_best_mask_by_score(
        state["masks"].detach().cpu().numpy(),
        state["scores"].float().detach().cpu().numpy(),
        min_mask_pixels=int(min_mask_pixels),
    )
    if selected is None:
        return None, None, None
    mask, score = selected
    bbox_xyxy = bbox_xyxy_from_mask(mask)
    bbox_xyxy = expand_bbox_xyxy(
        bbox_xyxy,
        image_shape=image_bgr.shape[:2],
        pad_ratio=float(bbox_pad_ratio),
        min_pad=int(bbox_min_pad),
    )
    return bbox_xyxy, mask, score


def build_preview_image(
    *,
    image_bgr: np.ndarray,
    refined_bbox_xyxy: list[int] | None,
    mask: np.ndarray | None,
    title: str,
) -> np.ndarray:
    preview = image_bgr.copy()
    if mask is not None:
        mask_bool = np.asarray(mask, dtype=bool)
        overlay = preview.copy()
        overlay[mask_bool] = np.array([60, 220, 60], dtype=np.uint8)
        preview = cv2.addWeighted(preview, 0.65, overlay, 0.35, 0.0)
    if refined_bbox_xyxy is not None:
        x0, y0, x1, y1 = [int(value) for value in refined_bbox_xyxy]
        cv2.rectangle(preview, (x0, y0), (x1, y1), (40, 40, 255), 2)
    cv2.rectangle(preview, (0, 0), (min(preview.shape[1] - 1, 780), 34), (20, 20, 20), thickness=-1)
    cv2.putText(preview, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return preview


def save_preview_image(
    *,
    preview_dir: Path,
    prompt_image_root: Path,
    image_path: Path,
    preview_bgr: np.ndarray,
) -> None:
    relative_path = Path(relative_prompt_image_path(image_path, prompt_image_root))
    preview_path = preview_dir / relative_path
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), preview_bgr)


def main() -> None:
    args = parse_args()

    prompt_image_root = Path(args.prompt_image_root).resolve()
    if not prompt_image_root.is_dir():
        raise FileNotFoundError(f"prompt image root not found: {prompt_image_root}")
    task_info_path = (Path(args.task_info) if args.task_info is not None else prompt_image_root / "task_info.json").resolve()
    annotated_dir = (
        Path(args.annotated_dir).resolve()
        if args.annotated_dir is not None
        else default_annotated_dir(task_info_path)
    )
    text_prompt = str(args.text_prompt or str(args.semantic_name).replace("_", " "))
    processor = load_sam3_processor(
        checkpoint_path=Path(args.checkpoint_path).resolve(),
        confidence=float(args.confidence),
        mask_threshold=float(args.mask_threshold),
    )

    payload = load_task_info(task_info_path)
    existing_images = {
        str(asset["image_path"])
        for asset in payload.get("assets", [])
        if isinstance(asset, dict) and isinstance(asset.get("image_path"), str)
    }
    image_paths = [Path(path).resolve() for path in args.images] if args.images else discover_images(
        prompt_image_root,
        recursive=bool(args.recursive),
    )
    if not image_paths:
        raise RuntimeError(f"no prompt images found under {prompt_image_root}")

    pending_paths: list[Path] = []
    for image_path in image_paths:
        relative_path = relative_prompt_image_path(image_path, prompt_image_root)
        if bool(args.skip_existing) and relative_path in existing_images:
            continue
        pending_paths.append(image_path)
    if not pending_paths:
        print("No images require annotation.")
        return

    annotations: list[dict[str, object]] = []
    for image_path in pending_paths:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"failed to read image: {image_path}")
        refined_bbox_xyxy, mask, score = infer_refined_bbox(
            processor=processor,
            image_bgr=image_bgr,
            text_prompt=text_prompt,
            min_mask_pixels=int(args.min_mask_pixels),
            bbox_pad_ratio=float(args.bbox_pad_ratio),
            bbox_min_pad=int(args.bbox_min_pad),
        )
        if refined_bbox_xyxy is None or mask is None or score is None:
            print(f"Skipped {image_path.name}: SAM3 produced no usable mask for prompt {text_prompt!r}")
            continue
        annotations.append({"image_path": image_path, "bbox_xyxy": refined_bbox_xyxy})
        print(f"{image_path.name}: bbox_xyxy={refined_bbox_xyxy} score={float(score):.3f}")
        preview = build_preview_image(
            image_bgr=image_bgr,
            refined_bbox_xyxy=refined_bbox_xyxy,
            mask=mask,
            title=f"text_prompt={text_prompt!r} score={float(score):.3f}",
        )
        save_preview_image(
            preview_dir=annotated_dir,
            prompt_image_root=prompt_image_root,
            image_path=image_path,
            preview_bgr=preview,
        )

    if not annotations:
        print("No bbox was produced; task_info.json was not changed.")
        return

    summary = upsert_prompt_assets(
        payload,
        prompt_image_root=prompt_image_root,
        semantic_name=str(args.semantic_name),
        annotations=annotations,
        start_index=args.start_index,
    )
    if bool(args.dry_run):
        print(f"[dry-run] would write {task_info_path}")
    else:
        write_task_info(task_info_path, payload)
        write_task_info(annotated_dir / "task_info.json", payload)
        print(f"Wrote {task_info_path}")
        print(f"Wrote {annotated_dir / 'task_info.json'}")
    print(f"Created: {summary['created']}")
    print(f"Updated: {summary['updated']}")
    print(f"Text prompt: {text_prompt!r}")
    print(f"Annotated images dir: {annotated_dir}")
    print("bbox_xyxy comes from the best SAM3 mask, using image coordinates [x0, y0, x1, y1].")


if __name__ == "__main__":
    main()
