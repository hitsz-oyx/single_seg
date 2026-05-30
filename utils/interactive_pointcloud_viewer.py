#!/usr/bin/env python3
"""交互式点云查看器，终端输入命令控制。"""

from __future__ import annotations

import argparse
import math
import queue
import sys
import threading
import numpy as np
import open3d as o3d
from pathlib import Path


def rotation_matrix_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    if sy > 1e-6:
        rx = math.atan2(R[2, 1], R[2, 2])
        ry = math.atan2(-R[2, 0], sy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(-R[2, 0], sy)
        rz = 0.0
    return (math.degrees(rx), math.degrees(ry), math.degrees(rz))


def format_camera_params(params: dict) -> str:
    return (
        f"  --camera-position {params['position'][0]:.3f} {params['position'][1]:.3f} {params['position'][2]:.3f} \\\n"
        f"  --camera-rotation {params['rotation'][0]:.1f} {params['rotation'][1]:.1f} {params['rotation'][2]:.1f} \\\n"
        f"  --rotation-center {params['center'][0]:.3f} {params['center'][1]:.3f} {params['center'][2]:.3f}"
    )


def compute_euler_for_zup(forward: np.ndarray) -> tuple[float, float, float]:
    """根据当前forward方向，计算让相机Z轴朝上（平行桌面）的旋转角度。

    render_ply_pointcloud.py 内部：
      1. up_hint = R_euler @ [0, 1, 0]   (R_euler = Rx@Ry@Rz)
      2. right  = normalize(cross(forward, up_hint))
      3. up     = normalize(cross(right, forward))

    当 forward 与 world_z=[0,0,1] 不平行时，令 up_hint = world_z
    即可算出正确的 Z-up 旋转角度。
    """
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    fwd_dot_up = np.dot(forward, world_up)
    if abs(abs(fwd_dot_up) - 1.0) < 1e-6:
        return (0.0, 0.0, 0.0)

    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    R_new = np.array([right, up, -forward], dtype=np.float64)
    rx, ry, rz = rotation_matrix_to_euler(R_new)
    return (rx, ry, rz)


def get_current_camera_params(vis) -> dict:
    """从Open3D提取当前相机参数。"""
    ctr = vis.get_view_control()
    cam_params = ctr.convert_to_pinhole_camera_parameters()
    extrinsic = cam_params.extrinsic
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    camera_pos = -R.T @ t
    forward = R.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward = forward / np.linalg.norm(forward)
    lookat = camera_pos + forward * 1.0
    rx, ry, rz = rotation_matrix_to_euler(R.T)
    return {'position': camera_pos, 'rotation': (rx, ry, rz), 'center': lookat,
            'forward': forward}


def apply_zup_correction(vis, scene_center: np.ndarray):
    """实时将Open3D视角修正为Z轴朝上（平行桌面）。"""
    ctr = vis.get_view_control()
    cam_params = ctr.convert_to_pinhole_camera_parameters()
    extrinsic = cam_params.extrinsic
    intrinsic = cam_params.intrinsic

    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    camera_pos = -R.T @ t
    forward = R.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward = forward / np.linalg.norm(forward)

    current_up = extrinsic[1, :3]

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    angle_to_z = math.degrees(math.acos(np.clip(np.dot(current_up, world_up), -1.0, 1.0)))
    if angle_to_z < 1.0:
        print(f"当前已接近标准朝上方向（夹角 {angle_to_z:.1f}°），无需修正")
        return

    new_forward = scene_center - camera_pos
    new_forward = new_forward / np.linalg.norm(new_forward)

    current_right = extrinsic[0, :3]
    right_base = np.cross(new_forward, world_up)
    right_norm = np.linalg.norm(right_base)
    if right_norm < 1e-6:
        print("警告: forward与标准朝上方向平行，无法修正")
        return
    right_base = right_base / right_norm

    if np.dot(right_base, current_right) < 0:
        right = -right_base
    else:
        right = right_base

    new_up = np.cross(right, new_forward)
    new_up = new_up / np.linalg.norm(new_up)

    new_R = np.array([right, new_up, -new_forward], dtype=np.float64)
    new_t = -new_R @ camera_pos

    new_extrinsic = np.eye(4, dtype=np.float64)
    new_extrinsic[:3, :3] = new_R
    new_extrinsic[:3, 3] = new_t

    cam_params.extrinsic = new_extrinsic
    ctr.convert_from_pinhole_camera_parameters(cam_params)

    rx, ry, rz = compute_euler_for_zup(new_forward)
    dist = np.linalg.norm(camera_pos - scene_center)

    print("\n" + "=" * 60)
    print("已修正为Z轴朝上（平行桌面）！")
    print("=" * 60)
    print(f"  --camera-position {camera_pos[0]:.3f} {camera_pos[1]:.3f} {camera_pos[2]:.3f} \\")
    print(f"  --camera-rotation {rx:.1f} {ry:.1f} {rz:.1f} \\")
    print(f"  --rotation-center {scene_center[0]:.3f} {scene_center[1]:.3f} {scene_center[2]:.3f}")
    print("=" * 60)
    print(f"  距离场景中心: {dist:.3f}m")
    sys.stdout.flush()


def input_listener(cmd_queue: queue.Queue):
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            cmd_queue.put(line.strip().lower())
        except (EOFError, KeyboardInterrupt):
            break


def main():
    parser = argparse.ArgumentParser(description="交互式点云查看器")
    parser.add_argument("--ply", type=str, required=True, help="PLY点云文件路径")
    parser.add_argument("--point-size", type=float, default=2.0, help="点大小")
    args = parser.parse_args()

    ply_path = Path(args.ply)
    if not ply_path.exists():
        print(f"错误: 文件不存在: {ply_path}")
        return

    print("=" * 60)
    print("交互式点云查看器")
    print("=" * 60)
    print("\n鼠标操作:")
    print("  左键拖拽: 旋转 | 滚轮: 缩放 | 右键拖拽: 平移")
    print("\n终端命令（输入后回车）:")
    print("  c → 打印当前相机参数")
    print("  u → 实时修正视角为Z轴朝上（平行桌面）")
    print("  q → 退出")
    print("=" * 60)

    print(f"\n加载点云: {ply_path}")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    print(f"点数: {len(pcd.points)}")

    scene_center = np.asarray(pcd.get_center())
    print(f"点云中心: [{scene_center[0]:.3f}, {scene_center[1]:.3f}, {scene_center[2]:.3f}]")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='Point Cloud Viewer', width=1280, height=720, visible=True)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    ro = vis.get_render_option()
    if ro is not None:
        ro.point_size = args.point_size
        ro.background_color = np.array([0, 0, 0])
        ro.show_coordinate_frame = False

    ctr = vis.get_view_control()
    ctr.set_lookat(scene_center)
    ctr.set_up(np.array([0.0, 0.0, 1.0]))
    ctr.set_front(np.array([0.0, -1.0, 0.0]))
    ctr.set_zoom(1.0)

    vis.poll_events()
    vis.update_renderer()

    params = get_current_camera_params(vis)
    print("\n初始相机参数:")
    print(format_camera_params(params))

    cmd_queue = queue.Queue()
    listener = threading.Thread(target=input_listener, args=(cmd_queue,), daemon=True)
    listener.start()

    print("\n开始交互！在终端输入命令后回车:")

    running = True
    while running:
        if vis.poll_events():
            vis.update_renderer()
        else:
            break

        try:
            cmd = cmd_queue.get_nowait()
        except queue.Empty:
            cmd = None

        if cmd is not None:
            if cmd == 'c':
                params = get_current_camera_params(vis)
                print("\n" + "=" * 60)
                print("当前相机参数（可用于 render_ply_pointcloud.py）：")
                print("=" * 60)
                print(format_camera_params(params))
                print("=" * 60)
                dist = np.linalg.norm(params['position'] - params['center'])
                print(f"距离: {dist:.3f}m")
                sys.stdout.flush()
            elif cmd == 'u':
                apply_zup_correction(vis, scene_center)
            elif cmd == 'q':
                print("退出中...")
                running = False

    vis.destroy_window()
    print("已退出")


if __name__ == "__main__":
    main()
