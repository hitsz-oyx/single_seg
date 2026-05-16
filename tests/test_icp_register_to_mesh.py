from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("open3d")

from icp.register_to_mesh import (
    color_cam2world_from_depth_pose,
    load_extrinsics_with_info,
    save_results,
)
from icp.visualize_mesh_registration import load_cam2worlds, load_payload_extrinsics


def test_load_extrinsics_with_info_prefers_depth_pose_and_preserves_camera_info(tmp_path: Path) -> None:
    color_pose = np.eye(4, dtype=np.float64)
    color_pose[:3, 3] = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    depth_pose = np.eye(4, dtype=np.float64)
    depth_pose[:3, 3] = np.array([-0.1, 0.05, 0.7], dtype=np.float64)
    depth_to_color = np.eye(4, dtype=np.float64)
    depth_to_color[0, 3] = 0.02
    payload = {
        "cameras": [
            {
                "camera_id": "cam_00",
                "serial_number": "123",
                "cam2world_4x4": color_pose.tolist(),
                "depth_cam2world_4x4": depth_pose.tolist(),
                "depth_to_color_4x4": depth_to_color.tolist(),
                "color_intrinsics": {"fx": 1.0},
                "depth_intrinsics": {"fx": 2.0},
                "pointcloud_frame": "rectified_depth",
            }
        ]
    }
    path = tmp_path / "extrinsics.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    extrinsics, camera_info = load_extrinsics_with_info(path)

    assert np.allclose(extrinsics["cam_00"], depth_pose)
    assert camera_info["cam_00"]["serial_number"] == "123"
    assert camera_info["cam_00"]["depth_to_color_4x4"] == depth_to_color.tolist()
    assert camera_info["cam_00"]["color_intrinsics"] == {"fx": 1.0}
    assert camera_info["cam_00"]["depth_intrinsics"] == {"fx": 2.0}
    assert camera_info["cam_00"]["pointcloud_frame"] == "rectified_depth"


def test_save_results_outputs_color_and_depth_poses_from_depth_primary_input(tmp_path: Path) -> None:
    depth_pose = np.eye(4, dtype=np.float64)
    depth_pose[:3, 3] = np.array([0.3, -0.2, 0.8], dtype=np.float64)
    depth_to_color = np.eye(4, dtype=np.float64)
    depth_to_color[0, 3] = 0.04
    output_path = tmp_path / "refined_extrinsics.json"

    save_results(
        output_path=output_path,
        master_camera="cam_00",
        camera_ids=["cam_00"],
        extrinsics={"cam_00": depth_pose},
        adj_results={},
        camera_info={
            "cam_00": {
                "serial_number": "123",
                "depth_to_color_4x4": depth_to_color.tolist(),
                "color_intrinsics": {"fx": 1.0},
                "depth_intrinsics": {"fx": 2.0},
                "pointcloud_frame": "rectified_depth",
            }
        },
        t_elapsed=0.0,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    camera = payload["cameras"][0]
    expected_color_pose = color_cam2world_from_depth_pose(depth_pose, depth_to_color)

    assert np.allclose(np.asarray(camera["depth_cam2world_4x4"], dtype=np.float64), depth_pose)
    assert np.allclose(np.asarray(camera["cam2world_4x4"], dtype=np.float64), expected_color_pose)
    assert camera["pointcloud_frame"] == "rectified_depth"
    assert camera["depth_intrinsics"] == {"fx": 2.0}


def test_load_cam2worlds_prefers_depth_pose_for_depth_pointcloud_json(tmp_path: Path) -> None:
    color_pose = np.eye(4, dtype=np.float64)
    color_pose[:3, 3] = np.array([0.0, 0.0, 0.1], dtype=np.float64)
    depth_pose = np.eye(4, dtype=np.float64)
    depth_pose[:3, 3] = np.array([0.2, 0.0, 0.0], dtype=np.float64)
    path = tmp_path / "refined_extrinsics.json"
    path.write_text(
        json.dumps(
            {
                "cameras": [
                    {
                        "camera_id": "cam_00",
                        "cam2world_4x4": color_pose.tolist(),
                        "depth_cam2world_4x4": depth_pose.tolist(),
                        "pointcloud_frame": "rectified_depth",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    loaded = load_cam2worlds(path, prefer="refined")

    assert np.allclose(loaded["cam_00"], depth_pose)


def test_load_payload_extrinsics_uses_depth_pose_for_depth_pointclouds(tmp_path: Path) -> None:
    live_debug = tmp_path / "live_rgbd_debug" / "frame_00000" / "cam_00"
    live_debug.mkdir(parents=True)
    color_pose = np.eye(4, dtype=np.float64)
    color_pose[:3, 3] = np.array([0.0, 0.0, 0.1], dtype=np.float64)
    depth_pose = np.eye(4, dtype=np.float64)
    depth_pose[:3, 3] = np.array([0.0, 0.3, 0.0], dtype=np.float64)
    payload = {
        "pointcloud_frame": "rectified_depth",
        "pose_record": {"cam2world_4x4": color_pose.tolist()},
        "color_pose_record": {"cam2world_4x4": color_pose.tolist()},
        "depth_pose_record": {"cam2world_4x4": depth_pose.tolist()},
    }
    (live_debug / "camera_payload.json").write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_payload_extrinsics(tmp_path / "live_rgbd_debug", ["cam_00"])

    assert np.allclose(loaded["cam_00"], depth_pose)
