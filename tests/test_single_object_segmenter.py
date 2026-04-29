from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import pytest

from single_seg.tracker_only_backend import (
    CropWindow,
    crop_mask_to_tracker_view,
    crop_window_from_mask,
    full_frame_crop_window,
    project_tracker_mask_to_full_image,
    TrackerBuildConfig,
    adapt_tracker_state_dict_for_config,
    build_stitched_layout,
    resolve_tracker_build_config,
    split_stitched_binary_mask,
    split_stitched_binary_mask_torch,
    stitch_camera_binary_masks,
)
from single_seg.single_object_segmenter import (
    DEFAULT_EPISODE_DIR,
    DEFAULT_PROMPT_IMAGE_ROOT,
    DEFAULT_PROMPT_TASK_INFO,
    REPO_ROOT,
    SingleSegConfig,
    _extract_target_mask_from_output_torch,
    backproject_scene_points_with_labels,
    backproject_scene_points_with_labels_torch,
    build_score_label_map,
    collect_common_frame_names,
    fuse_scene_geometry,
    fuse_scene_geometry_torch,
    largest_connected_component,
    load_prompt_entries,
    refine_seed_mask,
    select_best_seed_mask,
    semantic_name_from_asset,
    split_prompt_entries,
)
from single_seg.realsense_rgbd_segmenter import (
    align_rectified_depth_to_color,
    load_live_arg_defaults,
    project_points_to_depth_image,
)


def test_repo_default_resources_exist() -> None:
    assert DEFAULT_PROMPT_TASK_INFO.exists()
    assert DEFAULT_PROMPT_IMAGE_ROOT.exists()
    assert DEFAULT_EPISODE_DIR.exists()
    frame_names = collect_common_frame_names(DEFAULT_EPISODE_DIR, ["cam_00", "cam_01", "cam_02"])
    assert frame_names == ["frame_00000.png", "frame_00001.png", "frame_00002.png"]


def test_single_seg_config_from_yaml() -> None:
    config = SingleSegConfig.from_yaml(REPO_ROOT / "configs" / "fast_plate_demo.yaml")
    assert config.target_name == "plate"
    assert config.tracker_image_size == 896
    assert config.prompt_task_info.exists()
    assert config.prompt_image_root.exists()


def test_realsense_live_config_defaults_from_yaml() -> None:
    defaults = load_live_arg_defaults(REPO_ROOT / "configs" / "realsense_d435_live.yaml")
    assert defaults["target_name"] == "plate"
    assert defaults["camera_count"] == 1
    assert defaults["low_bandwidth_mode"] == 1
    assert defaults["save_live_debug"] == 1
    assert defaults["prompt_task_info"].exists()
    assert defaults["fast_model_path"] == (
        REPO_ROOT / "third_party" / "fastfoundationstereo" / "weights" / "23-36-37" / "model_best_bp2_serialize.pth"
    )


def test_realsense_live_config_normalizes_serial_lists(tmp_path: Path) -> None:
    config_path = tmp_path / "realsense.yaml"
    config_path.write_text(
        "\n".join(
            [
                "segmenter:",
                "  target_name: plate",
                "  prompt_task_info: assets/prompts/libero_spatial/semantic_split_parts/task_info.json",
                "  prompt_image_root: assets/prompts/libero_spatial/semantic_split_parts",
                "realsense:",
                "  camera_count: 2",
                "  camera_serials:",
                "    - 123",
                "    - 456",
                "  camera_poses_json: tests/outputs/camera_poses.json",
                "fast_stereo:",
                "  model_path: third_party/fastfoundationstereo/weights/23-36-37/model_best_bp2_serialize.pth",
                "",
            ]
        ),
        encoding="utf-8",
    )
    defaults = load_live_arg_defaults(config_path)
    assert defaults["camera_count"] == 2
    assert defaults["camera_serials"] == "123,456"
    assert defaults["camera_poses_json"] == REPO_ROOT / "tests" / "outputs" / "camera_poses.json"


def test_segmenter_from_config_uses_paths() -> None:
    config = SingleSegConfig.from_yaml(REPO_ROOT / "configs" / "default.yaml")
    overridden = config.with_overrides(target_name="akita_black_bowl")
    kwargs = overridden.to_segmenter_kwargs()
    assert kwargs["target_name"] == "akita_black_bowl"
    assert kwargs["prompt_task_info"] == config.prompt_task_info
    assert kwargs["prompt_image_root"] == config.prompt_image_root


def test_semantic_name_from_asset() -> None:
    assert semantic_name_from_asset("akita_black_bowl_0") == "akita_black_bowl"
    assert semantic_name_from_asset("robot_arm_10") == "robot_arm"
    assert semantic_name_from_asset("plate") == "plate"


def test_load_prompt_entries_and_split(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    for name in ("akita_black_bowl_0.png", "plate_0.png"):
        Image.fromarray(np.full((32, 32, 3), 127, dtype=np.uint8)).save(image_root / name)
    task_info = {
        "assets": [
            {
                "asset_name": "akita_black_bowl_0",
                "image_path": "akita_black_bowl_0.png",
                "bbox_xyxy": [4, 5, 20, 24],
            },
            {
                "asset_name": "plate_0",
                "image_path": "plate_0.png",
                "bbox_xyxy": [2, 3, 29, 30],
            },
        ]
    }
    task_info_path = tmp_path / "task_info.json"
    task_info_path.write_text(json.dumps(task_info), encoding="utf-8")
    entries = load_prompt_entries(task_info_path, image_root)
    positive, negative = split_prompt_entries(entries, "akita_black_bowl")
    assert len(entries) == 2
    assert len(positive) == 1
    assert len(negative) == 1
    assert positive[0].semantic_name == "akita_black_bowl"
    assert negative[0].semantic_name == "plate"


def test_select_best_seed_mask_prefers_high_score() -> None:
    boxes = np.asarray([[0, 0, 5, 5], [10, 10, 20, 20]], dtype=np.float32)
    scores = np.asarray([0.4, 0.9], dtype=np.float32)
    masks = np.zeros((2, 32, 32), dtype=bool)
    masks[0, :6, :6] = True
    masks[1, 10:21, 10:21] = True
    selected = select_best_seed_mask(boxes, scores, masks, min_pixels=16)
    assert selected is not None
    mask, score, box = selected
    assert mask.shape == (32, 32)
    assert abs(score - 0.9) < 1e-6
    assert box == [10, 10, 20, 20]


def test_collect_common_frame_names(tmp_path: Path) -> None:
    episode_dir = tmp_path / "episode"
    for camera_id in ("cam_00", "cam_01"):
        (episode_dir / camera_id / "rgb").mkdir(parents=True)
        (episode_dir / camera_id / "depth").mkdir(parents=True)
    for frame_name in ("frame_00000.png", "frame_00001.png"):
        for camera_id in ("cam_00", "cam_01"):
            (episode_dir / camera_id / "rgb" / frame_name).write_bytes(b"rgb")
            (episode_dir / camera_id / "depth" / frame_name).write_bytes(b"depth")
    names = collect_common_frame_names(episode_dir, ["cam_00", "cam_01"])
    assert names == ["frame_00000.png", "frame_00001.png"]


def test_refine_seed_mask_uses_box_when_mask_is_too_large() -> None:
    mask = np.ones((20, 30), dtype=bool)
    refined, mode = refine_seed_mask(
        mask,
        [10, 5, 14, 9],
        image_shape=(20, 30),
        max_area_ratio=0.2,
        box_margin=1,
        min_pixels=4,
    )
    assert mode in {"box_refined_lcc", "box_refined"}
    assert int(np.count_nonzero(refined)) < int(np.count_nonzero(mask))


def test_largest_connected_component_keeps_biggest_blob() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:4, 1:4] = True
    mask[6:8, 6:8] = True
    kept = largest_connected_component(mask)
    assert kept.sum() == 9
    assert kept[1:4, 1:4].all()
    assert not kept[6:8, 6:8].any()


def test_build_score_label_map_uses_mask_prob_threshold() -> None:
    logits = np.full((1, 6, 6), -10.0, dtype=np.float32)
    logits[0, 1:5, 1:5] = 2.0
    logits[0, 2:4, 2:4] = 5.0
    label_map, stats = build_score_label_map(
        out_obj_ids=np.asarray([1], dtype=np.int32),
        out_binary_masks=(logits > 0),
        out_probs=np.asarray([0.9], dtype=np.float32),
        out_tracker_probs=np.asarray([0.9], dtype=np.float32),
        image_shape=(6, 6),
        min_object_score=0.0,
        out_mask_logits=logits,
        mask_prob_threshold=0.97,
    )
    assert 1 in stats
    assert int(np.count_nonzero(label_map == 1)) == 4
    assert (label_map[2:4, 2:4] == 1).all()


def test_build_stitched_layout_for_three_cameras() -> None:
    layout = build_stitched_layout(
        frame_sizes={
            "cam_00": (1280, 720),
            "cam_01": (1280, 720),
            "cam_02": (1280, 720),
        },
        camera_order=["cam_00", "cam_01", "cam_02"],
    )
    assert layout.canvas_width == 2560
    assert layout.canvas_height == 1440
    assert layout.tiles["cam_00"].x == 0
    assert layout.tiles["cam_01"].x == 1280
    assert layout.tiles["cam_02"].y == 720


def test_stitch_and_split_binary_masks_roundtrip() -> None:
    layout = build_stitched_layout(
        frame_sizes={
            "cam_00": (4, 3),
            "cam_01": (4, 3),
            "cam_02": (4, 3),
        },
        camera_order=["cam_00", "cam_01", "cam_02"],
    )
    masks = {
        "cam_00": np.array([[0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1]], dtype=bool),
        "cam_01": np.array([[1, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=bool),
        "cam_02": np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 1, 0]], dtype=bool),
    }
    stitched = stitch_camera_binary_masks(masks, layout)
    recovered = split_stitched_binary_mask(stitched, layout)
    for camera_id, mask in masks.items():
        assert np.array_equal(mask, recovered[camera_id])


def test_split_stitched_binary_masks_torch_roundtrip() -> None:
    layout = build_stitched_layout(
        frame_sizes={
            "cam_00": (4, 3),
            "cam_01": (4, 3),
            "cam_02": (4, 3),
        },
        camera_order=["cam_00", "cam_01", "cam_02"],
    )
    masks = {
        "cam_00": np.array([[0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1]], dtype=bool),
        "cam_01": np.array([[1, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=bool),
        "cam_02": np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 1, 0]], dtype=bool),
    }
    stitched = torch.as_tensor(stitch_camera_binary_masks(masks, layout), dtype=torch.bool)
    recovered = split_stitched_binary_mask_torch(stitched, layout)
    for camera_id, mask in masks.items():
        assert np.array_equal(mask, recovered[camera_id].cpu().numpy())


def test_extract_target_mask_from_output_torch_uses_logit_threshold() -> None:
    logits = torch.full((1, 6, 6), -10.0, dtype=torch.float32)
    logits[0, 1:5, 1:5] = 2.0
    logits[0, 2:4, 2:4] = 5.0
    output = {
        "out_obj_ids": np.asarray([1], dtype=np.int32),
        "out_binary_masks": logits > 0,
        "out_mask_logits": logits,
        "out_probs": torch.tensor([0.9], dtype=torch.float32),
        "out_tracker_probs": torch.tensor([0.9], dtype=torch.float32),
    }
    result = _extract_target_mask_from_output_torch(
        output,
        image_shape=(6, 6),
        min_object_score=0.0,
        mask_prob_threshold=0.97,
        target_obj_id=1,
        device=torch.device("cpu"),
    )
    mask = result["mask"].cpu().numpy()
    assert int(mask.sum()) == 4
    assert mask[2:4, 2:4].all()


def test_resolve_tracker_build_config_with_image_size_override() -> None:
    config = resolve_tracker_build_config(image_size_override=840)
    assert config.profile_name == "default"
    assert config.image_size == 840


def test_resolve_tracker_build_config_rejects_non_multiple_of_14() -> None:
    with pytest.raises(ValueError):
        resolve_tracker_build_config(image_size_override=768)


def test_adapt_tracker_state_dict_for_smaller_maskmem() -> None:
    state_dict = {
        "maskmem_tpos_enc": torch.zeros((7, 1, 1, 256), dtype=torch.float32),
        "other_key": torch.ones((2, 2), dtype=torch.float32),
    }
    config = TrackerBuildConfig(num_maskmem=4)
    adapted = adapt_tracker_state_dict_for_config(state_dict, build_config=config)
    assert adapted["maskmem_tpos_enc"].shape[0] == 4
    assert adapted["other_key"].shape == (2, 2)


def test_crop_window_from_mask_returns_local_window() -> None:
    mask = np.zeros((10, 12), dtype=bool)
    mask[3:5, 7:9] = True
    window = crop_window_from_mask(
        mask,
        image_size=(12, 10),
        margin_scale=2.0,
        min_size_ratio=0.25,
    )
    assert window.width < 12
    assert window.height < 10
    assert window.x0 <= 7 < window.x1
    assert window.y0 <= 3 < window.y1


def test_roi_mask_projection_fills_crop_window() -> None:
    crop_window = CropWindow(x0=2, y0=1, x1=6, y1=5)
    tracker_mask = np.ones((6, 8), dtype=bool)
    projected = project_tracker_mask_to_full_image(
        tracker_mask,
        crop_window,
        full_size=(8, 6),
    )
    assert projected.shape == (6, 8)
    assert projected[1:5, 2:6].all()
    assert not projected[:1, :].any()
    assert not projected[:, :2].any()


def test_crop_mask_to_tracker_view_preserves_nonempty_region() -> None:
    mask = np.zeros((6, 8), dtype=bool)
    mask[2:4, 3:5] = True
    crop_window = CropWindow(x0=2, y0=1, x1=6, y1=5)
    tracker_view = crop_mask_to_tracker_view(
        mask,
        crop_window,
        output_size=(8, 6),
    )
    assert tracker_view.shape == (6, 8)
    assert tracker_view.any()


def test_backproject_scene_points_with_labels_torch_matches_numpy() -> None:
    rgb = np.array(
        [
            [[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]],
            [[15, 25, 35], [45, 55, 65], [75, 85, 95], [105, 115, 125]],
            [[20, 30, 40], [50, 60, 70], [80, 90, 100], [110, 120, 130]],
            [[25, 35, 45], [55, 65, 75], [85, 95, 105], [115, 125, 135]],
        ],
        dtype=np.uint8,
    )
    depth = np.array(
        [
            [1.0, 1.1, 1.2, 1.3],
            [1.4, 1.5, 1.6, 1.7],
            [1.8, 1.9, 2.0, 2.1],
            [2.2, 2.3, 2.4, 2.5],
        ],
        dtype=np.float32,
    )
    mask = np.array(
        [
            [0, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=bool,
    )
    intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 1.5, "cy": 1.5}
    cam2world = np.eye(4, dtype=np.float32)
    np_points, np_colors, np_labels = backproject_scene_points_with_labels(
        rgb=rgb,
        depth_m=depth,
        mask=mask,
        cam2world_gl=cam2world,
        intrinsics=intrinsics,
        fovy_deg=None,
        depth_min=0.1,
        depth_max=3.0,
        stride=2,
    )
    sampled_rgb = np.ascontiguousarray(rgb[::2, ::2])
    sampled_depth = np.ascontiguousarray(depth[::2, ::2])
    sampled_mask = np.ascontiguousarray(mask[::2, ::2])
    device = torch.device("cpu")
    v = torch.arange(0, 4, 2, dtype=torch.float32, device=device)
    u = torch.arange(0, 4, 2, dtype=torch.float32, device=device)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    x_scale = (uu - 1.5) / 100.0
    y_scale = (vv - 1.5) / 100.0
    t_points, t_colors, t_labels = backproject_scene_points_with_labels_torch(
        sampled_rgb=sampled_rgb,
        sampled_depth_m=sampled_depth,
        sampled_mask=sampled_mask,
        cam2world_gl=cam2world,
        x_scale=x_scale,
        y_scale=y_scale,
        depth_min=0.1,
        depth_max=3.0,
        device=device,
    )
    assert np.allclose(np_points, t_points.numpy(), atol=1e-6)
    assert np.array_equal(np_colors, t_colors.numpy())
    assert np.array_equal(np_labels, t_labels.numpy())


def test_fuse_scene_geometry_torch_matches_numpy() -> None:
    point_chunks = [
        np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 0.001, 0.0], [0.101, 0.0, 0.0]], dtype=np.float32),
    ]
    color_chunks = [
        np.array([[10, 20, 30], [20, 30, 40], [100, 110, 120]], dtype=np.uint8),
        np.array([[30, 40, 50], [110, 120, 130]], dtype=np.uint8),
    ]
    label_chunks = [
        np.array([0, 1, 0], dtype=np.int32),
        np.array([1, 1], dtype=np.int32),
    ]
    np_points, np_colors, np_labels = fuse_scene_geometry(point_chunks, color_chunks, label_chunks, voxel_size=0.01)
    t_points, t_colors, t_labels = fuse_scene_geometry_torch(
        [torch.from_numpy(chunk) for chunk in point_chunks],
        [torch.from_numpy(chunk) for chunk in color_chunks],
        [torch.from_numpy(chunk) for chunk in label_chunks],
        voxel_size=0.01,
        device=torch.device("cpu"),
    )
    assert np.allclose(np_points, t_points.numpy(), atol=1e-6)
    assert np.array_equal(np_colors, t_colors.numpy())
    assert np.array_equal(np_labels, t_labels.numpy())


def test_project_points_to_depth_image_keeps_nearest_depth() -> None:
    points_src = np.array(
        [
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    transform = np.eye(4, dtype=np.float32)
    intrinsics = {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0}
    depth = project_points_to_depth_image(
        points_src,
        transform,
        intrinsics,
        (1, 1),
    )
    assert depth.shape == (1, 1)
    assert abs(float(depth[0, 0]) - 1.0) < 1e-6


def test_align_rectified_depth_to_color_identity_projection() -> None:
    depth_rect = np.array(
        [
            [1.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=np.float32,
    )
    intrinsics = {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0}
    aligned = align_rectified_depth_to_color(
        depth_rect,
        rectified_intrinsics=intrinsics,
        rectified_to_color=np.eye(4, dtype=np.float32),
        color_intrinsics=intrinsics,
        color_shape=(2, 2),
    )
    assert np.allclose(aligned, depth_rect, atol=1e-6)
