from __future__ import annotations

from pathlib import Path

import numpy as np

from single_seg.prompt_task_info_utils import (
    bbox_xyxy_from_mask,
    default_annotated_dir,
    expand_bbox_xyxy,
    next_asset_index,
    relative_prompt_image_path,
    select_best_mask_by_score,
    upsert_prompt_assets,
    xyxy_inclusive_to_normalized_cxcywh,
    xywh_to_xyxy_inclusive,
)


def test_xywh_to_xyxy_inclusive_uses_top_left_and_inclusive_max_corner() -> None:
    bbox = xywh_to_xyxy_inclusive(10, 20, 30, 40, image_width=100, image_height=80)
    assert bbox == [10, 20, 39, 59]


def test_xyxy_inclusive_to_normalized_cxcywh_matches_inclusive_box_definition() -> None:
    normalized = xyxy_inclusive_to_normalized_cxcywh([10, 20, 39, 59], image_width=100, image_height=80)
    assert normalized == [0.25, 0.5, 0.3, 0.5]


def test_bbox_xyxy_from_mask_uses_top_left_origin() -> None:
    mask = np.zeros((8, 10), dtype=bool)
    mask[2:6, 3:8] = True
    assert bbox_xyxy_from_mask(mask) == [3, 2, 7, 5]


def test_expand_bbox_xyxy_respects_inclusive_bounds() -> None:
    expanded = expand_bbox_xyxy([3, 2, 7, 5], image_shape=(8, 10), pad_ratio=0.2, min_pad=1)
    assert expanded == [2, 1, 8, 6]


def test_select_best_mask_by_score_prefers_high_score_then_area() -> None:
    masks = np.zeros((2, 6, 6), dtype=bool)
    masks[0, 1:3, 1:3] = True
    masks[1, 1:5, 1:5] = True
    selected = select_best_mask_by_score(masks, np.array([0.8, 0.7], dtype=np.float32), min_mask_pixels=1)
    assert selected is not None
    mask, score = selected
    assert abs(float(score) - 0.8) < 1e-6
    assert int(mask.sum()) == 4


def test_relative_prompt_image_path_requires_image_under_prompt_root(tmp_path: Path) -> None:
    prompt_root = tmp_path / "prompts"
    prompt_root.mkdir()
    image_path = prompt_root / "plate_0.png"
    image_path.write_bytes(b"")
    assert relative_prompt_image_path(image_path, prompt_root) == "plate_0.png"


def test_default_annotated_dir_uses_task_info_parent(tmp_path: Path) -> None:
    task_info_path = tmp_path / "semantic_split_parts" / "task_info.json"
    assert default_annotated_dir(task_info_path) == tmp_path / "semantic_split_parts" / "annotated"


def test_upsert_prompt_assets_assigns_current_style_asset_names(tmp_path: Path) -> None:
    prompt_root = tmp_path / "prompts"
    prompt_root.mkdir()
    image_a = prompt_root / "plate_0.png"
    image_b = prompt_root / "plate_1.png"
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")
    payload = {"assets": []}
    summary = upsert_prompt_assets(
        payload,
        prompt_image_root=prompt_root,
        semantic_name="plate",
        annotations=[
            {"image_path": image_a, "bbox_xyxy": [1, 2, 10, 11]},
            {"image_path": image_b, "bbox_xyxy": [3, 4, 12, 13]},
        ],
    )
    assert summary == {"created": ["plate_0", "plate_1"], "updated": []}
    assert payload["assets"] == [
        {"asset_name": "plate_0", "image_path": "plate_0.png", "bbox_xyxy": [1, 2, 10, 11]},
        {"asset_name": "plate_1", "image_path": "plate_1.png", "bbox_xyxy": [3, 4, 12, 13]},
    ]


def test_upsert_prompt_assets_updates_existing_image_without_new_index(tmp_path: Path) -> None:
    prompt_root = tmp_path / "prompts"
    prompt_root.mkdir()
    image_path = prompt_root / "plate_0.png"
    image_path.write_bytes(b"a")
    payload = {
        "assets": [
            {"asset_name": "plate_0", "image_path": "plate_0.png", "bbox_xyxy": [1, 2, 10, 11]},
        ]
    }
    summary = upsert_prompt_assets(
        payload,
        prompt_image_root=prompt_root,
        semantic_name="plate",
        annotations=[{"image_path": image_path, "bbox_xyxy": [5, 6, 20, 21]}],
    )
    assert summary == {"created": [], "updated": ["plate_0"]}
    assert payload["assets"][0]["bbox_xyxy"] == [5, 6, 20, 21]


def test_next_asset_index_skips_other_semantics() -> None:
    assets = [
        {"asset_name": "plate_0"},
        {"asset_name": "plate_4"},
        {"asset_name": "bowl_9"},
    ]
    assert next_asset_index(assets, "plate") == 5
