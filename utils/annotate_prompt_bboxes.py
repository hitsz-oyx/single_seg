#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Iterable

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from single_seg.prompt_task_info_utils import (
    default_annotated_dir,
    load_task_info,
    relative_prompt_image_path,
    upsert_prompt_assets,
    write_task_info,
    xywh_to_xyxy_inclusive,
)


IMAGE_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively draw bbox_xyxy annotations for prompt images and write/update task_info.json."
    )
    parser.add_argument("--prompt-image-root", type=Path, required=True, help="Root directory containing prompt images")
    parser.add_argument("--semantic-name", type=str, required=True, help="Semantic name, e.g. plate")
    parser.add_argument(
        "--task-info",
        type=Path,
        default=None,
        help="task_info.json path; defaults to <prompt-image-root>/task_info.json",
    )
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
        help="Directory to save annotated preview images; defaults to <task-info-dir>/annotated",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned updates without writing task_info.json",
    )
    return parser.parse_args()


def discover_images(prompt_image_root: Path, *, recursive: bool) -> list[Path]:
    image_paths: list[Path] = []
    for pattern in IMAGE_PATTERNS:
        matches: Iterable[Path]
        matches = prompt_image_root.rglob(pattern) if recursive else prompt_image_root.glob(pattern)
        image_paths.extend(Path(path).resolve() for path in matches)
    return sorted(dict.fromkeys(image_paths))


def draw_preview(image_path: Path, bbox_xyxy: list[int], preview_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image for preview: {image_path}")
    x0, y0, x1, y1 = [int(value) for value in bbox_xyxy]
    cv2.rectangle(image, (x0, y0), (x1, y1), (60, 220, 60), 2)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), image)


def select_bbox(image_path: Path) -> list[int] | None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    window_name = f"{image_path.name} | drag bbox, ENTER/SPACE confirm, c cancel/skip"
    x, y, w, h = cv2.selectROI(window_name, image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)
    return xywh_to_xyxy_inclusive(
        int(x),
        int(y),
        int(w),
        int(h),
        image_width=int(image.shape[1]),
        image_height=int(image.shape[0]),
    )


def main() -> None:
    args = parse_args()
    if os.name != "nt" and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        raise RuntimeError("No GUI display detected. Run this script on a machine/session with desktop display access.")

    prompt_image_root = Path(args.prompt_image_root).resolve()
    if not prompt_image_root.is_dir():
        raise FileNotFoundError(f"prompt image root not found: {prompt_image_root}")
    task_info_path = (Path(args.task_info) if args.task_info is not None else prompt_image_root / "task_info.json").resolve()
    annotated_dir = (
        Path(args.annotated_dir).resolve()
        if args.annotated_dir is not None
        else default_annotated_dir(task_info_path)
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
        bbox_xyxy = select_bbox(image_path)
        if bbox_xyxy is None:
            print(f"Skipped {image_path.name}")
            continue
        annotations.append({"image_path": image_path, "bbox_xyxy": bbox_xyxy})
        print(f"{image_path.name}: bbox_xyxy={bbox_xyxy}")
        preview_path = annotated_dir / relative_prompt_image_path(image_path, prompt_image_root)
        draw_preview(image_path, bbox_xyxy, preview_path)

    if not annotations:
        print("No bbox was confirmed; task_info.json was not changed.")
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
    print(f"Annotated images dir: {annotated_dir}")
    print("bbox_xyxy uses image coordinates [x0, y0, x1, y1] with top-left origin and inclusive x1/y1.")


if __name__ == "__main__":
    main()
