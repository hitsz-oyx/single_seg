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
    erode_binary_mask_torch,
    filter_target_mask_by_depth_band,
    filter_target_mask_by_depth_band_torch,
    filter_target_labels_by_3d_clusters,
    filter_target_labels_by_3d_clusters_torch,
    filter_target_labels_by_dominant_plane,
    filter_target_labels_by_dominant_plane_torch,
    fuse_scene_geometry,
    fuse_scene_geometry_torch,
    largest_connected_component,
    load_prompt_entries,
    refine_seed_mask,
    select_best_seed_mask,
    semantic_name_from_asset,
    split_prompt_entries,
    write_live_debug_target_object_cloud,
)
from single_seg.realsense_rgbd_segmenter import (
    align_rectified_depth_to_color,
    align_rectified_depth_to_color_torch,
    build_arg_parser,
    build_camera_inputs_from_live_frames,
    build_effective_live_config,
    filter_depth_edges_torch,
    LibrealsenseSoftwareAligner,
    load_live_arg_defaults,
    project_points_to_depth_image,
)
from utils.calibrate_realsense_apriltag_extrinsics import (
    CV_TO_GL,
    average_transforms,
    calibrate_camera_from_frames,
    camera_to_world_from_detection,
    furniture_base_tag_layout,
    make_transform,
    single_seg_pose_from_opencv_pose,
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
    assert defaults["depth_source"] == "fast"
    assert defaults["low_bandwidth_mode"] == 0
    assert defaults["save_live_debug"] == 1
    assert defaults["target_cluster_filter_enabled"] is True
    assert defaults["target_cluster_radius_m"] == 0.013
    assert defaults["target_cluster_min_points"] == 45
    assert defaults["target_cluster_keep_largest"] is True
    assert defaults["target_plane_filter_enabled"] is False
    assert defaults["target_plane_filter_distance_m"] == 0.004
    assert defaults["target_plane_filter_min_points"] == 80
    assert defaults["target_plane_filter_min_inlier_ratio"] == 0.25
    assert defaults["target_plane_filter_max_inlier_ratio"] == 0.85
    assert defaults["target_plane_filter_max_planes"] == 1
    assert defaults["target_plane_filter_ransac_iterations"] == 256
    assert defaults["target_depth_band_filter_enabled"] is False
    assert defaults["target_depth_band_filter_range_m"] == 0.015
    assert defaults["target_depth_band_filter_min_valid_pixels"] == 50
    assert defaults["target_depth_band_filter_min_keep_pixels"] == 20
    assert defaults["target_3d_mask_erode_kernel"] == 0
    assert defaults["fast_depth_edge_filter_enabled"] == 0
    assert defaults["fast_depth_edge_filter_threshold_m"] == 0.5
    assert defaults["fast_depth_edge_filter_stage"] == "rectified"
    assert defaults["camera_poses_json"] == REPO_ROOT / "tests" / "outputs" / "camera_poses_apriltag.json"
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


def test_effective_live_config_reflects_cli_overrides(tmp_path: Path) -> None:
    config_path = REPO_ROOT / "configs" / "realsense_d435_live.yaml"
    output_dir = tmp_path / "live_out"
    parser = build_arg_parser(load_live_arg_defaults(config_path))
    args = parser.parse_args(
        [
            "--config",
            str(config_path),
            "--target-name",
            "bowl",
            "--camera-count",
            "3",
            "--camera-serials",
            "111,222,333",
            "--output-dir",
            str(output_dir),
            "--fast-scale",
            "1.0",
            "--fast-depth-edge-filter-enabled",
            "1",
            "--fast-depth-edge-filter-threshold-m",
            "0.25",
            "--fast-depth-edge-filter-stage",
            "aligned",
            "--target-plane-filter-enabled",
            "1",
            "--target-plane-filter-distance-m",
            "0.006",
            "--target-plane-filter-min-points",
            "20",
            "--target-plane-filter-min-inlier-ratio",
            "0.2",
            "--target-plane-filter-max-inlier-ratio",
            "0.9",
            "--target-plane-filter-max-planes",
            "2",
            "--target-plane-filter-ransac-iterations",
            "64",
            "--target-depth-band-filter-enabled",
            "1",
            "--target-depth-band-filter-range-m",
            "0.012",
            "--target-depth-band-filter-min-valid-pixels",
            "30",
            "--target-depth-band-filter-min-keep-pixels",
            "10",
            "--target-3d-mask-erode-kernel",
            "5",
            "--stereo-rectification-mode",
            "passthrough",
            "--emitter-enabled",
            "0",
            "--save-ply",
            "--save-normal",
            "1",
        ]
    )
    effective = build_effective_live_config(args, serials=["111", "222", "333"])

    assert effective["segmenter"]["target_name"] == "bowl"
    assert effective["segmenter"]["output_dir"] == str(output_dir.resolve())
    assert effective["segmenter"]["save_ply"] is True
    assert effective["segmenter"]["save_normal"] is True
    assert effective["segmenter"]["target_plane_filter_enabled"] is True
    assert effective["segmenter"]["target_plane_filter_distance_m"] == 0.006
    assert effective["segmenter"]["target_plane_filter_min_points"] == 20
    assert effective["segmenter"]["target_plane_filter_min_inlier_ratio"] == 0.2
    assert effective["segmenter"]["target_plane_filter_max_inlier_ratio"] == 0.9
    assert effective["segmenter"]["target_plane_filter_max_planes"] == 2
    assert effective["segmenter"]["target_plane_filter_ransac_iterations"] == 64
    assert effective["segmenter"]["target_depth_band_filter_enabled"] is True
    assert effective["segmenter"]["target_depth_band_filter_range_m"] == 0.012
    assert effective["segmenter"]["target_depth_band_filter_min_valid_pixels"] == 30
    assert effective["segmenter"]["target_depth_band_filter_min_keep_pixels"] == 10
    assert effective["segmenter"]["target_3d_mask_erode_kernel"] == 5
    assert effective["realsense"]["camera_count"] == 3
    assert effective["realsense"]["camera_serials"] == "111,222,333"
    assert effective["realsense"]["stereo_rectification_mode"] == "passthrough"
    assert effective["realsense"]["emitter_enabled"] == 0
    assert effective["fast_stereo"]["scale"] == 1.0
    assert effective["fast_stereo"]["depth_edge_filter_enabled"] is True
    assert effective["fast_stereo"]["depth_edge_filter_threshold_m"] == 0.25
    assert effective["fast_stereo"]["depth_edge_filter_stage"] == "aligned"


def read_ply_header(path: Path) -> str:
    with path.open("rb") as handle:
        chunks: list[bytes] = []
        while True:
            line = handle.readline()
            if not line:
                break
            chunks.append(line)
            if line == b"end_header\n":
                break
    return b"".join(chunks).decode("ascii")


def test_live_debug_target_object_cloud_writes_object_only_ply(tmp_path: Path) -> None:
    summary = write_live_debug_target_object_cloud(
        live_debug_root=tmp_path / "live_rgbd_debug",
        frame_name="frame_00000.png",
        camera_id="cam_00",
        target_name="bowl",
        points=np.asarray([[0.0, 0.0, 0.5], [0.1, 0.0, 0.5]], dtype=np.float32),
        colors=np.asarray([[10, 20, 30], [40, 50, 60]], dtype=np.uint8),
        save_normal=False,
        camera_center=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        voxel_size=0.003,
        score=0.75,
        target_pixels=8,
    )

    ply_path = tmp_path / "live_rgbd_debug" / "frame_00000" / "cam_00" / "target_object_rgb.ply"
    meta_path = tmp_path / "live_rgbd_debug" / "frame_00000" / "cam_00" / "target_object_pointcloud.json"
    header = read_ply_header(ply_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    assert summary["num_points"] == 2
    assert summary["has_normals"] is False
    assert meta["target_name"] == "bowl"
    assert meta["target_pixels"] == 8
    assert "element vertex 2" in header
    assert "property float nx" not in header


def test_live_debug_target_object_cloud_respects_save_normal(tmp_path: Path) -> None:
    write_live_debug_target_object_cloud(
        live_debug_root=tmp_path / "live_rgbd_debug",
        frame_name="frame_00001.png",
        camera_id="cam_01",
        target_name="bowl",
        points=np.asarray(
            [
                [0.0, 0.0, 0.5],
                [0.1, 0.0, 0.5],
                [0.0, 0.1, 0.5],
                [0.1, 0.1, 0.5],
            ],
            dtype=np.float32,
        ),
        colors=np.full((4, 3), 128, dtype=np.uint8),
        save_normal=True,
        camera_center=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        voxel_size=0.003,
    )

    ply_path = tmp_path / "live_rgbd_debug" / "frame_00001" / "cam_01" / "target_object_rgb.ply"
    meta_path = tmp_path / "live_rgbd_debug" / "frame_00001" / "cam_01" / "target_object_pointcloud.json"
    header = read_ply_header(ply_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    assert meta["num_points"] == 4
    assert meta["has_normals"] is True
    assert "property float nx" in header
    assert "property float ny" in header
    assert "property float nz" in header


def test_furniture_base_apriltag_layout_matches_expected_offsets() -> None:
    layout = furniture_base_tag_layout()
    assert layout.tag_size_m == 0.048
    assert sorted(layout.world_t_tag_by_id) == [0, 1, 2, 3]
    assert np.allclose(layout.world_t_tag_by_id[0][:3, 3], [-0.03, -0.03, 0.0])
    assert np.allclose(layout.world_t_tag_by_id[3][:3, 3], [0.03, 0.03, 0.0])


def test_apriltag_camera_pose_conversion_uses_world_tag_pose() -> None:
    class Detection:
        pass

    detection = Detection()
    detection.pose_R = np.eye(3, dtype=np.float64)
    detection.pose_t = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    world_t_tag = make_transform(np.asarray([0.2, -0.1, 0.4], dtype=np.float64))
    world_t_camera_cv = camera_to_world_from_detection(detection, world_t_tag)
    expected = make_transform(np.asarray([0.2, -0.1, -0.6], dtype=np.float64))
    assert np.allclose(world_t_camera_cv, expected)
    assert np.allclose(single_seg_pose_from_opencv_pose(world_t_camera_cv), expected @ CV_TO_GL)


def test_average_transforms_averages_positions_and_keeps_rotation() -> None:
    transforms = [
        make_transform(np.asarray([0.0, 0.0, 0.0], dtype=np.float64)),
        make_transform(np.asarray([0.2, 0.0, 0.0], dtype=np.float64)),
    ]
    averaged = average_transforms(transforms)
    assert np.allclose(averaged[:3, 3], [0.1, 0.0, 0.0])
    assert np.allclose(averaged[:3, :3], np.eye(3), atol=1e-6)


def test_apriltag_calibration_errors_when_camera_sees_no_tags(tmp_path: Path) -> None:
    class EmptyDetector:
        def detect(self, image, intrinsics):
            return []

    debug_dir = tmp_path / "debug"
    with pytest.raises(RuntimeError, match="没有检测到任何 AprilTag"):
        calibrate_camera_from_frames(
            camera_id="cam_00",
            serial_number="serial_00",
            frames=[np.zeros((24, 32, 3), dtype=np.uint8)],
            intrinsics={"fx": 20.0, "fy": 20.0, "cx": 16.0, "cy": 12.0},
            layout=furniture_base_tag_layout(),
            detector=EmptyDetector(),
            min_tags_per_frame=1,
            debug_dir=debug_dir,
        )
    assert (debug_dir / "cam_00_detections.png").exists()


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


def test_filter_target_labels_by_3d_clusters_removes_outliers() -> None:
    points = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [0.00, 0.00, 0.00],
            [0.01, 0.00, 0.00],
            [0.00, 0.01, 0.00],
            [0.00, 0.00, 0.01],
            [0.50, 0.50, 0.50],
            [0.54, 0.50, 0.50],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 1, 1, 1, 1, 1, 1], dtype=np.int32)

    filtered, summary = filter_target_labels_by_3d_clusters(
        points,
        labels,
        enabled=True,
        radius_m=0.03,
        min_points=3,
        keep_largest=True,
    )

    assert filtered.tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert summary["target_points_before"] == 6
    assert summary["target_points_after"] == 4
    assert summary["removed_target_points"] == 2


def test_filter_target_labels_by_3d_clusters_disabled_is_noop() -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    labels = np.asarray([1, 1], dtype=np.int32)
    filtered, summary = filter_target_labels_by_3d_clusters(
        points,
        labels,
        enabled=False,
        radius_m=0.03,
        min_points=3,
        keep_largest=True,
    )
    assert np.array_equal(filtered, labels)
    assert summary["removed_target_points"] == 0


def test_filter_target_labels_by_3d_clusters_torch_matches_numpy() -> None:
    points = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [0.00, 0.00, 0.00],
            [0.01, 0.00, 0.00],
            [0.00, 0.01, 0.00],
            [0.00, 0.00, 0.01],
            [0.50, 0.50, 0.50],
            [0.54, 0.50, 0.50],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 1, 1, 1, 1, 1, 1], dtype=np.int32)
    expected, expected_summary = filter_target_labels_by_3d_clusters(
        points,
        labels,
        enabled=True,
        radius_m=0.03,
        min_points=3,
        keep_largest=True,
    )

    filtered_t, summary = filter_target_labels_by_3d_clusters_torch(
        torch.as_tensor(points),
        torch.as_tensor(labels),
        enabled=True,
        radius_m=0.03,
        min_points=3,
        keep_largest=True,
    )

    assert np.array_equal(filtered_t.cpu().numpy(), expected)
    assert summary["backend"] == "torch"
    assert summary["target_points_before"] == expected_summary["target_points_before"]
    assert summary["target_points_after"] == expected_summary["target_points_after"]
    assert summary["removed_target_points"] == expected_summary["removed_target_points"]


def test_filter_target_labels_by_dominant_plane_removes_planar_target_points() -> None:
    plane_points = np.asarray(
        [[x * 0.01, y * 0.01, 0.0] for y in range(4) for x in range(6)],
        dtype=np.float32,
    )
    object_points = np.asarray(
        [
            [0.02, 0.01, 0.05],
            [0.03, 0.01, 0.06],
            [0.02, 0.02, 0.07],
            [0.03, 0.02, 0.08],
        ],
        dtype=np.float32,
    )
    points = np.concatenate([np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32), plane_points, object_points], axis=0)
    labels = np.concatenate(
        [
            np.asarray([0], dtype=np.int32),
            np.ones((plane_points.shape[0] + object_points.shape[0],), dtype=np.int32),
        ],
        axis=0,
    )

    filtered, summary = filter_target_labels_by_dominant_plane(
        points,
        labels,
        enabled=True,
        distance_m=0.001,
        min_points=12,
        min_inlier_ratio=0.4,
        max_inlier_ratio=0.9,
        max_planes=1,
        ransac_iterations=64,
    )

    assert summary["plane_applied"] is True
    assert summary["removed_target_points"] == plane_points.shape[0]
    assert filtered[0] == 0
    assert int(np.count_nonzero(filtered[1 : 1 + plane_points.shape[0]])) == 0
    assert np.all(filtered[-object_points.shape[0] :] == 1)


def test_filter_target_labels_by_dominant_plane_torch_matches_numpy() -> None:
    plane_points = np.asarray(
        [[x * 0.01, y * 0.01, 0.0] for y in range(4) for x in range(6)],
        dtype=np.float32,
    )
    object_points = np.asarray([[0.02, 0.01, 0.05], [0.03, 0.02, 0.07]], dtype=np.float32)
    points = np.concatenate([plane_points, object_points], axis=0)
    labels = np.ones((points.shape[0],), dtype=np.int32)
    expected, expected_summary = filter_target_labels_by_dominant_plane(
        points,
        labels,
        enabled=True,
        distance_m=0.001,
        min_points=12,
        min_inlier_ratio=0.4,
        max_inlier_ratio=0.95,
        max_planes=1,
        ransac_iterations=64,
    )

    filtered_t, summary = filter_target_labels_by_dominant_plane_torch(
        torch.as_tensor(points),
        torch.as_tensor(labels),
        enabled=True,
        distance_m=0.001,
        min_points=12,
        min_inlier_ratio=0.4,
        max_inlier_ratio=0.95,
        max_planes=1,
        ransac_iterations=64,
    )

    assert np.array_equal(filtered_t.cpu().numpy(), expected)
    assert summary["backend"] == "torch_cpu_ransac"
    assert summary["plane_applied"] == expected_summary["plane_applied"]
    assert summary["removed_target_points"] == expected_summary["removed_target_points"]


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


def test_erode_binary_mask_torch_shrinks_boundary_only() -> None:
    mask = torch.zeros((7, 7), dtype=torch.bool)
    mask[1:6, 1:6] = True
    eroded = erode_binary_mask_torch(mask, 3)
    expected = torch.zeros((7, 7), dtype=torch.bool)
    expected[2:5, 2:5] = True
    assert torch.equal(eroded, expected)


def test_filter_target_mask_by_depth_band_removes_depth_outliers() -> None:
    mask = np.ones((3, 4), dtype=bool)
    depth = np.asarray(
        [
            [1.00, 1.01, 1.02, 1.30],
            [1.00, 1.01, 1.02, 1.35],
            [1.00, 1.01, 1.02, 0.00],
        ],
        dtype=np.float32,
    )
    filtered, summary = filter_target_mask_by_depth_band(
        mask,
        depth,
        enabled=True,
        range_m=0.03,
        min_valid_pixels=3,
        min_keep_pixels=3,
    )

    assert summary["applied"] is True
    assert summary["center_depth_m"] == pytest.approx(1.01, abs=1e-6)
    assert int(np.count_nonzero(filtered)) == 9
    assert not filtered[0, 3]
    assert not filtered[1, 3]
    assert not filtered[2, 3]


def test_filter_target_mask_by_depth_band_torch_matches_numpy() -> None:
    mask = np.ones((3, 4), dtype=bool)
    depth = np.asarray(
        [
            [1.00, 1.01, 1.02, 1.30],
            [1.00, 1.01, 1.02, 1.35],
            [1.00, 1.01, 1.02, 0.00],
        ],
        dtype=np.float32,
    )
    expected, expected_summary = filter_target_mask_by_depth_band(
        mask,
        depth,
        enabled=True,
        range_m=0.03,
        min_valid_pixels=3,
        min_keep_pixels=3,
    )
    filtered_t, summary = filter_target_mask_by_depth_band_torch(
        torch.as_tensor(mask),
        torch.as_tensor(depth),
        enabled=True,
        range_m=0.03,
        min_valid_pixels=3,
        min_keep_pixels=3,
    )

    assert np.array_equal(filtered_t.cpu().numpy(), expected)
    assert summary["backend"] == "torch"
    assert summary["target_pixels_after"] == expected_summary["target_pixels_after"]
    assert summary["removed_target_pixels"] == expected_summary["removed_target_pixels"]


def test_filter_depth_edges_torch_removes_depth_jumps() -> None:
    depth = torch.ones((5, 5), dtype=torch.float32)
    depth[:, 3:] = 2.0
    filtered = filter_depth_edges_torch(depth, threshold_m=0.5)
    assert int(torch.count_nonzero(filtered > 0).item()) < int(depth.numel())
    assert torch.allclose(filtered[:, 0], torch.ones(5))


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


def test_align_rectified_depth_to_color_torch_matches_identity_projection() -> None:
    depth_rect = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=torch.float32,
    )
    intrinsics = {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0}
    aligned = align_rectified_depth_to_color_torch(
        depth_rect,
        rectified_intrinsics=intrinsics,
        rectified_to_color=torch.eye(4, dtype=torch.float32),
        color_intrinsics=intrinsics,
        color_shape=(2, 2),
    )
    assert torch.allclose(aligned, depth_rect, atol=1e-6)


def test_librealsense_software_aligner_matches_identity_projection() -> None:
    depth_rect = np.zeros((16, 16), dtype=np.float32)
    depth_rect[2:10, 3:12] = 1.25
    intrinsics = {"fx": 20.0, "fy": 20.0, "cx": 8.0, "cy": 8.0}
    aligner = LibrealsenseSoftwareAligner(
        rectified_intrinsics=intrinsics,
        rectified_to_color=np.eye(4, dtype=np.float32),
        color_intrinsics=intrinsics,
        depth_shape=depth_rect.shape,
        color_shape=depth_rect.shape,
    )
    try:
        aligned = aligner.align(depth_rect, np.zeros((16, 16, 3), dtype=np.uint8))
    finally:
        aligner.close()
    assert aligned.shape == depth_rect.shape
    assert np.allclose(aligned, depth_rect, atol=1e-4)


def test_librealsense_software_aligner_uses_same_rotation_convention_as_torch() -> None:
    depth_rect = np.zeros((48, 48), dtype=np.float32)
    depth_rect[16:32, 16:32] = 1.0
    intrinsics = {"fx": 70.0, "fy": 70.0, "cx": 24.0, "cy": 24.0}
    angle = np.deg2rad(3.0)
    rectified_to_color = np.eye(4, dtype=np.float32)
    rectified_to_color[:3, :3] = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=np.float32,
    )
    aligner = LibrealsenseSoftwareAligner(
        rectified_intrinsics=intrinsics,
        rectified_to_color=rectified_to_color,
        color_intrinsics=intrinsics,
        depth_shape=depth_rect.shape,
        color_shape=depth_rect.shape,
    )
    try:
        aligned = aligner.align(depth_rect, np.zeros((48, 48, 3), dtype=np.uint8))
    finally:
        aligner.close()
    torch_aligned = align_rectified_depth_to_color_torch(
        torch.from_numpy(depth_rect),
        rectified_intrinsics=intrinsics,
        rectified_to_color=rectified_to_color,
        color_intrinsics=intrinsics,
        color_shape=depth_rect.shape,
    ).numpy()
    assert np.array_equal(aligned > 0.0, torch_aligned > 0.0)
    assert np.allclose(aligned[aligned > 0.0], torch_aligned[torch_aligned > 0.0], atol=1e-2)


def test_build_camera_inputs_can_use_native_realsense_depth_without_fast(tmp_path: Path) -> None:
    rgb = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    depth_m = np.array(
        [
            [0.05, 0.2],
            [4.0, np.inf],
        ],
        dtype=np.float32,
    )
    intrinsics = {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0, "width": 2, "height": 2}
    camera_inputs = build_camera_inputs_from_live_frames(
        captured_frames=[
            {
                "camera_id": "cam_00",
                "depth_source": "native",
                "rgb": rgb,
                "depth_m": depth_m,
                "color_intrinsics": intrinsics,
                "pose_record": {"camera_id": "cam_00"},
            }
        ],
        stereo_runner=None,
        depth_min=0.1,
        depth_max=3.0,
        output_dir=tmp_path,
        frame_index=0,
        write_debug_images=False,
    )
    depth_out = camera_inputs["cam_00"]["depth_m"]
    assert np.array_equal(camera_inputs["cam_00"]["rgb"], rgb)
    assert np.allclose(depth_out, np.array([[0.0, 0.2], [0.0, 0.0]], dtype=np.float32))


def test_build_camera_inputs_keeps_fast_depth_on_torch_path(tmp_path: Path) -> None:
    class FakeStereoRunner:
        def infer_depth(self, **kwargs):
            assert kwargs["return_torch"] is True
            return {
                "depth_m": torch.tensor([[0.05, 0.2], [4.0, 0.3]], dtype=torch.float32),
                "rectified_intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
            }

    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    intrinsics = {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0, "width": 2, "height": 2}
    camera_inputs = build_camera_inputs_from_live_frames(
        captured_frames=[
            {
                "camera_id": "cam_00",
                "depth_source": "fast",
                "rgb": rgb,
                "ir_left_rect": np.zeros((2, 2), dtype=np.uint8),
                "ir_right_rect": np.zeros((2, 2), dtype=np.uint8),
                "rectified_k": np.eye(3, dtype=np.float32),
                "rectified_to_color": np.eye(4, dtype=np.float32),
                "baseline_m": 0.05,
                "color_intrinsics": intrinsics,
                "pose_record": {"camera_id": "cam_00"},
            }
        ],
        stereo_runner=FakeStereoRunner(),
        depth_min=0.1,
        depth_max=3.0,
        output_dir=tmp_path,
        frame_index=0,
        write_debug_images=True,
    )
    depth_out = camera_inputs["cam_00"]["depth_m"]
    payload_path = tmp_path / "live_rgbd_debug" / "frame_00000" / "cam_00" / "camera_payload.json"
    debug_payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert torch.is_tensor(depth_out)
    assert torch.allclose(depth_out.cpu(), torch.tensor([[0.0, 0.2], [0.0, 0.3]], dtype=torch.float32))
    assert debug_payload["depth_source"] == "fast"
    assert debug_payload["ir_left_rect_file"] == "ir_left_rect.png"
    assert debug_payload["ir_right_rect_file"] == "ir_right_rect.png"
    assert debug_payload["baseline_m"] == 0.05
    assert debug_payload["rectified_k"] == np.eye(3, dtype=np.float32).tolist()
    assert debug_payload["rectified_to_color"] == np.eye(4, dtype=np.float32).tolist()


def test_build_camera_inputs_can_use_librealsense_fast_align_backend(tmp_path: Path) -> None:
    class FakeStereoRunner:
        def infer_depth(self, **kwargs):
            assert kwargs["return_torch"] is True
            depth = torch.zeros((16, 16), dtype=torch.float32)
            depth[4:12, 5:13] = 1.25
            return {
                "depth_m": depth,
                "rectified_intrinsics": {"fx": 20.0, "fy": 20.0, "cx": 8.0, "cy": 8.0},
            }

    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    intrinsics = {"fx": 20.0, "fy": 20.0, "cx": 8.0, "cy": 8.0, "width": 16, "height": 16}
    timing: dict[str, object] = {}
    aligners: dict[str, LibrealsenseSoftwareAligner] = {}
    try:
        camera_inputs = build_camera_inputs_from_live_frames(
            captured_frames=[
                {
                    "camera_id": "cam_00",
                    "depth_source": "fast",
                    "rgb": rgb,
                    "ir_left_rect": np.zeros((16, 16), dtype=np.uint8),
                    "ir_right_rect": np.zeros((16, 16), dtype=np.uint8),
                    "rectified_k": np.eye(3, dtype=np.float32),
                    "rectified_to_color": np.eye(4, dtype=np.float32),
                    "baseline_m": 0.05,
                    "color_intrinsics": intrinsics,
                    "pose_record": {"camera_id": "cam_00"},
                }
            ],
            stereo_runner=FakeStereoRunner(),
            depth_min=0.1,
            depth_max=3.0,
            output_dir=tmp_path,
            frame_index=0,
            write_debug_images=False,
            timing=timing,
            fast_align_backend="librealsense",
            fast_aligners=aligners,
        )
    finally:
        for aligner in aligners.values():
            aligner.close()
    depth_out = camera_inputs["cam_00"]["depth_m"]
    assert isinstance(depth_out, np.ndarray)
    assert np.allclose(depth_out[4:12, 5:13], 1.25, atol=1e-4)
    assert camera_inputs["cam_00"]["fast_align_backend"] == "librealsense"
    assert timing["fast_align_backend"] == "librealsense"
    assert timing["depth_to_cpu_time_sec"] >= 0.0
    assert timing["librealsense_align_time_sec"] >= 0.0
