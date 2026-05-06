#!/usr/bin/env python3
"""PLY 帧序列的交互式 Open3D 查看器。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="为 PLY 帧输出开启一个交互式 Open3D 窗口。")
    parser.add_argument("--input-dir", type=Path, required=True, help="输入目录路径")
    parser.add_argument("--pattern", default="frame_*_instance_rgb.ply", help="匹配 PLY 文件的模式")
    parser.add_argument("--max-frames", type=int, default=0, help="处理的最大帧数（0 表示不限制）")
    parser.add_argument("--start-index", type=int, default=0, help="开始帧索引")
    parser.add_argument("--point-size", type=float, default=2.0, help="点的大小")
    parser.add_argument("--bg", choices=["black", "white"], default="black", help="背景颜色")
    parser.add_argument("--width", type=int, default=1280, help="窗口宽度")
    parser.add_argument("--height", type=int, default=800, help="窗口高度")
    parser.add_argument(
        "--color-mode",
        choices=["ply", "target-height"],
        default="ply",
        help="点云着色模式：ply 使用文件原颜色；target-height 将目标点按高度渐变着色",
    )
    parser.add_argument("--target-color", default="255,70,70", help="目标实例在 PLY 中的 RGB 颜色，用逗号分隔")
    parser.add_argument("--target-color-tolerance", type=float, default=0.18, help="目标颜色匹配容差，范围 0-1")
    return parser.parse_args()


def collect_ply_paths(input_dir: Path, pattern: str, max_frames: int) -> list[Path]:
    """收集输入目录下所有匹配模式的 PLY 文件路径。"""
    paths = sorted(input_dir.glob(pattern))
    if int(max_frames) > 0:
        paths = paths[: int(max_frames)]
    if not paths:
        raise FileNotFoundError(f"在 {input_dir} 下没有找到匹配 {pattern!r} 的 PLY 文件")
    return paths


def load_cloud(path: Path) -> o3d.geometry.PointCloud:
    """加载点云文件，如果缺色则涂上默认颜色。"""
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise RuntimeError(f"Open3D 加载了一个空点云: {path}")
    if not cloud.has_colors():
        cloud.paint_uniform_color([0.86, 0.86, 0.86])
    return cloud


def parse_rgb_color(text: str) -> np.ndarray:
    """解析 0-255 或 0-1 RGB 字符串为 0-1 float 颜色。"""
    parts = [float(part.strip()) for part in str(text).split(",")]
    if len(parts) != 3:
        raise ValueError(f"颜色必须是 R,G,B 三个数值，得到: {text!r}")
    color = np.asarray(parts, dtype=np.float64)
    if np.any(color > 1.0):
        color = color / 255.0
    return np.clip(color, 0.0, 1.0)


def height_colormap(values: np.ndarray) -> np.ndarray:
    """将 0-1 高度值映射为蓝-青-绿-黄-红渐变。"""
    stops = np.asarray(
        [
            [0.10, 0.20, 0.90],
            [0.00, 0.75, 1.00],
            [0.05, 0.80, 0.25],
            [1.00, 0.88, 0.10],
            [1.00, 0.15, 0.05],
        ],
        dtype=np.float64,
    )
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    scaled = values * float(len(stops) - 1)
    lower = np.floor(scaled).astype(np.int64)
    upper = np.clip(lower + 1, 0, len(stops) - 1)
    lower = np.clip(lower, 0, len(stops) - 1)
    blend = (scaled - lower)[:, None]
    return stops[lower] * (1.0 - blend) + stops[upper] * blend


def apply_target_height_colors(
    cloud: o3d.geometry.PointCloud,
    *,
    target_color: np.ndarray,
    tolerance: float,
) -> o3d.geometry.PointCloud:
    """将目标实例点按高度渐变着色，其它点保留原始颜色。"""
    points = np.asarray(cloud.points)
    if points.size == 0:
        return cloud
    colors = np.asarray(cloud.colors) if cloud.has_colors() else np.full((points.shape[0], 3), 0.86, dtype=np.float64)
    target = np.linalg.norm(colors - target_color[None, :], axis=1) <= float(tolerance)
    if not np.any(target):
        return cloud
    z = points[target, 2]
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    if z_max <= z_min:
        normalized = np.zeros_like(z, dtype=np.float64)
    else:
        normalized = (z - z_min) / (z_max - z_min)
    new_colors = colors.copy()
    new_colors[target] = height_colormap(normalized)
    cloud.colors = o3d.utility.Vector3dVector(new_colors)
    return cloud


def cleanup_open3d_camera_artifacts(search_root: Path) -> None:
    """清理 Open3D 自动生成的相机参数等临时文件。"""
    for pattern in ("DepthCamera_*.json", "DepthCamera_*.png", "ScreenCamera_*.json", "ScreenCamera_*.png"):
        for artifact_path in search_root.glob(pattern):
            if artifact_path.is_file():
                artifact_path.unlink()


def copy_cloud(dst: o3d.geometry.PointCloud, src: o3d.geometry.PointCloud) -> None:
    """将源点云的数据复制到目标点云对象中。"""
    dst.points = src.points
    dst.colors = src.colors
    if src.has_normals():
        dst.normals = src.normals


def configure_view(vis: o3d.visualization.VisualizerWithKeyCallback, cloud: o3d.geometry.PointCloud, bg: str) -> None:
    """配置 Open3D 可视化窗口的视角和渲染选项。"""
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([1.0, 1.0, 1.0]) if bg == "white" else np.array([0.0, 0.0, 0.0])
    bbox = cloud.get_axis_aligned_bounding_box()
    ctr = vis.get_view_control()
    ctr.set_lookat(bbox.get_center().tolist())
    ctr.set_up([0.0, 0.0, 1.0])
    ctr.set_front([1.0, -0.9, -0.35])
    ctr.set_zoom(0.72)


class PlySequenceViewer:
    """PLY 序列查看器类，负责管理帧切换和缓存。"""
    def __init__(self, paths: list[Path], bg: str, color_mode: str, target_color: np.ndarray, target_color_tolerance: float) -> None:
        self.paths = paths
        self.bg = bg
        self.color_mode = color_mode
        self.target_color = target_color
        self.target_color_tolerance = float(target_color_tolerance)
        self.cache: dict[int, o3d.geometry.PointCloud] = {}
        self.current_idx = 0
        self.pcd = o3d.geometry.PointCloud()

    def load(self, idx: int) -> o3d.geometry.PointCloud:
        """带缓存加载指定索引的点云帧。"""
        if idx not in self.cache:
            cloud = load_cloud(self.paths[idx])
            if self.color_mode == "target-height":
                cloud = apply_target_height_colors(
                    cloud,
                    target_color=self.target_color,
                    tolerance=self.target_color_tolerance,
                )
            self.cache[idx] = cloud
        return self.cache[idx]

    def set_frame(self, idx: int, vis: o3d.visualization.VisualizerWithKeyCallback, reset_view: bool = False) -> None:
        """将当前显示帧切换为指定索引的帧。"""
        idx = max(0, min(int(idx), len(self.paths) - 1))
        self.current_idx = idx
        # 先保存当前视角，以免切换帧后视角重置
        camera_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
        copy_cloud(self.pcd, self.load(idx))
        vis.update_geometry(self.pcd)
        if reset_view:
            configure_view(vis, self.pcd, self.bg)
        else:
            vis.get_view_control().convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)
        cleanup_open3d_camera_artifacts(Path.cwd())
        print(f"[{self.current_idx + 1}/{len(self.paths)}] {self.paths[self.current_idx].name}")

    def next_frame(self, vis: o3d.visualization.VisualizerWithKeyCallback) -> bool:
        """切换到下一帧。"""
        if self.current_idx + 1 < len(self.paths):
            self.set_frame(self.current_idx + 1, vis)
        return False

    def previous_frame(self, vis: o3d.visualization.VisualizerWithKeyCallback) -> bool:
        """切换到前一帧。"""
        if self.current_idx > 0:
            self.set_frame(self.current_idx - 1, vis)
        return False

    def reset_view(self, vis: o3d.visualization.VisualizerWithKeyCallback) -> bool:
        configure_view(vis, self.pcd, self.bg)
        return False

    def close(self, vis: o3d.visualization.VisualizerWithKeyCallback) -> bool:
        vis.close()
        return False


def main() -> None:
    args = parse_args()
    cleanup_open3d_camera_artifacts(Path.cwd())
    input_dir = args.input_dir.resolve()
    paths = collect_ply_paths(input_dir, args.pattern, int(args.max_frames))
    start_index = max(0, min(int(args.start_index), len(paths) - 1))
    viewer = PlySequenceViewer(
        paths=paths,
        bg=str(args.bg),
        color_mode=str(args.color_mode),
        target_color=parse_rgb_color(str(args.target_color)),
        target_color_tolerance=float(args.target_color_tolerance),
    )
    viewer.current_idx = start_index
    copy_cloud(viewer.pcd, viewer.load(start_index))
    vis = o3d.visualization.VisualizerWithKeyCallback()
    if not vis.create_window(
        window_name=f"PLY sequence: {input_dir.name}",
        width=int(args.width),
        height=int(args.height),
    ):
        raise RuntimeError("Failed to create Open3D window. Check DISPLAY, e.g. DISPLAY=localhost:10.0")
    vis.add_geometry(viewer.pcd)
    vis.get_render_option().point_size = max(float(args.point_size), 1.0)
    configure_view(vis, viewer.pcd, str(args.bg))
    print("Controls: D next, A previous, R reset view, Q close")
    viewer.set_frame(start_index, vis, reset_view=True)
    vis.register_key_callback(ord("D"), viewer.next_frame)
    vis.register_key_callback(ord("A"), viewer.previous_frame)
    vis.register_key_callback(ord("R"), viewer.reset_view)
    vis.register_key_callback(ord("Q"), viewer.close)
    vis.run()
    cleanup_open3d_camera_artifacts(Path.cwd())
    vis.destroy_window()


if __name__ == "__main__":
    main()
