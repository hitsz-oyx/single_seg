#!/usr/bin/env python3
"""根据外参和相机RGBD数据重建点云，支持多视角融合。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_camera_extrinsics(extrinsics_path: Path) -> dict[str, Any]:
    """加载相机外参文件。"""
    with open(extrinsics_path) as f:
        return json.load(f)


def load_rgbd_data(frame_dir: Path, camera_id: str) -> tuple[np.ndarray, np.ndarray]:
    """加载单个相机的RGB和深度数据。"""
    rgb_path = frame_dir / "predicted" / camera_id / "rgb.npy"
    depth_path = frame_dir / "predicted" / camera_id / "depth_m.npy"
    
    rgb = np.load(rgb_path)
    depth_m = np.load(depth_path)
    
    return rgb, depth_m


def backproject_rgbd_to_pointcloud(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: dict[str, float],
    cam2world_gl: np.ndarray,
    depth_min: float = 0.1,
    depth_max: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """将RGBD图像反投影到3D点云（无下采样版本）。"""
    height, width = depth_m.shape
    
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    
    valid = np.isfinite(depth_m) & (depth_m > depth_min) & (depth_m < depth_max)
    
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    
    v_coords, u_coords = np.where(valid)
    depths = depth_m[valid]
    
    x_cv = ((u_coords - cx) / fx) * depths
    y_cv = ((v_coords - cy) / fy) * depths
    z_cv = depths
    pts_cv = np.stack([x_cv, y_cv, z_cv], axis=1)
    
    pts_gl = pts_cv * np.array([1.0, -1.0, -1.0], dtype=np.float32)[None, :]
    pts_gl_h = np.concatenate([pts_gl, np.ones((pts_gl.shape[0], 1), dtype=np.float32)], axis=1)
    pts_world = (cam2world_gl.astype(np.float32) @ pts_gl_h.T).T[:, :3]
    
    colors = rgb[v_coords, u_coords]
    
    return pts_world.astype(np.float32), colors.astype(np.uint8)


def fuse_pointclouds(point_chunks: list[np.ndarray], color_chunks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """融合多个点云（无下采样版本）。"""
    if not point_chunks:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    
    points = np.concatenate(point_chunks, axis=0)
    colors = np.concatenate(color_chunks, axis=0)
    
    return points.astype(np.float32), colors.astype(np.uint8)


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """将点云数据写入 .ply 文件。"""
    if points.shape[0] != colors.shape[0]:
        raise ValueError("points and colors must have the same length")
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
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


def process_frame(
    frame_dir: Path,
    extrinsics_data: dict[str, Any],
    output_dir: Path,
    save_individual_cameras: bool = True,
    save_fused: bool = True,
) -> None:
    """处理单个帧，生成点云文件。"""
    frame_name = frame_dir.name
    cameras = extrinsics_data.get("cameras", [])
    
    all_points = []
    all_colors = []
    
    for camera in cameras:
        camera_id = camera["camera_id"]
        intrinsics = camera.get("intrinsics", {})
        cam2world_gl = np.array(camera.get("cam2world_4x4"))
        
        if cam2world_gl.shape != (4, 4):
            print(f"  跳过 {camera_id}: 无效的外参矩阵")
            continue
        
        try:
            rgb, depth_m = load_rgbd_data(frame_dir, camera_id)
        except FileNotFoundError:
            print(f"  跳过 {camera_id}: RGBD数据文件不存在")
            continue
        
        points, colors = backproject_rgbd_to_pointcloud(rgb, depth_m, intrinsics, cam2world_gl)
        
        if points.shape[0] == 0:
            print(f"  相机 {camera_id}: 无有效点云")
            continue
        
        all_points.append(points)
        all_colors.append(colors)
        
        if save_individual_cameras:
            cam_output_dir = output_dir / frame_name / camera_id
            cam_ply_path = cam_output_dir / f"{frame_name}_{camera_id}_scene_rgb.ply"
            write_ply(cam_ply_path, points, colors)
            print(f"  {camera_id}: {points.shape[0]} 点 -> {cam_ply_path}")
    
    if save_fused and all_points:
        fused_points, fused_colors = fuse_pointclouds(all_points, all_colors)
        fused_ply_path = output_dir / frame_name / f"{frame_name}_fused.ply"
        write_ply(fused_ply_path, fused_points, fused_colors)
        print(f"  融合后: {fused_points.shape[0]} 点 -> {fused_ply_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="根据外参和相机RGBD数据重建点云"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="rollout_results目录路径（包含render子目录）",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="输出目录路径",
    )
    parser.add_argument(
        "--save-individual",
        action="store_true",
        default=True,
        help="保存每个相机的独立点云文件（默认启用）",
    )
    parser.add_argument(
        "--no-individual",
        action="store_true",
        help="不保存每个相机的独立点云文件",
    )
    parser.add_argument(
        "--save-fused",
        action="store_true",
        default=True,
        help="保存融合后的点云文件（默认启用）",
    )
    parser.add_argument(
        "--no-fused",
        action="store_true",
        help="不保存融合后的点云文件",
    )
    parser.add_argument(
        "--depth-min",
        type=float,
        default=0.1,
        help="最小有效深度（米），默认0.1",
    )
    parser.add_argument(
        "--depth-max",
        type=float,
        default=10.0,
        help="最大有效深度（米），默认10.0",
    )
    
    args = parser.parse_args()
    
    input_path = args.input_dir.expanduser().resolve()
    output_path = args.output_dir.expanduser().resolve()
    
    render_dir = input_path / "render"
    if not render_dir.exists():
        print(f"错误: render目录不存在: {render_dir}")
        return
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    frame_dirs = sorted(render_dir.glob("depth_frame_*"))
    
    if not frame_dirs:
        print(f"错误: 在 {render_dir} 中未找到 depth_frame_* 目录")
        return
    
    print(f"找到 {len(frame_dirs)} 个帧目录")
    print(f"输入目录: {input_path}")
    print(f"输出目录: {output_path}")
    print(f"深度范围: [{args.depth_min}, {args.depth_max}] 米")
    print()
    
    save_individual = args.save_individual and not args.no_individual
    save_fused = args.save_fused and not args.no_fused
    
    total_frames = len(frame_dirs)
    for idx, frame_dir in enumerate(frame_dirs, 1):
        print(f"处理帧 {idx}/{total_frames}: {frame_dir.name}")
        
        extrinsics_path = frame_dir / "predicted" / "camera_extrinsics.json"
        if not extrinsics_path.exists():
            print(f"  警告: 外参文件不存在，跳过此帧")
            continue
        
        try:
            extrinsics_data = load_camera_extrinsics(extrinsics_path)
        except Exception as e:
            print(f"  错误: 读取外参失败: {e}")
            continue
        
        try:
            process_frame(
                frame_dir,
                extrinsics_data,
                output_path,
                save_individual_cameras=save_individual,
                save_fused=save_fused,
            )
        except Exception as e:
            print(f"  错误: 处理帧失败: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    print("完成！")


if __name__ == "__main__":
    main()
