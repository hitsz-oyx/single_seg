#!/usr/bin/env python3
"""多相机外参微调：以主相机定义 mesh 位姿，其他相机配准到同一个 mesh。

算法（用户设计）：
  1. 主相机世界点云 → ICP → mesh → T_MW (world→mesh)
  2. 主相机的 T_MW 定义 mesh 在世界坐标系中的位姿
  3. 对每个非主相机：
     a. predicted_T_MC = T_MW @ original_cam2world (camera→mesh 空间的初始值)
     b. 相机坐标系点云 → mesh
        - Open3D ICP：以 predicted_T_MC 为初值做局部微调
        - Go-ICP：全局匹配，不使用 predicted_T_MC 作为初值
     c. 得到 refined_T_MC
     d. new_cam2world = inv(T_MW) @ refined_T_MC

主相机和非主相机都支持两种后端：
  - Open3D ICP（默认）：以质心对齐为初值，局部迭代，调整量小
  - Go-ICP（--use-goicp / --refine-use-goicp）：全局搜索最优解

两种数据模式：

  模式1 - 自动检测（从数据目录自动寻找点云和外参）：
    python icp/register_to_mesh.py \
      --data-dir tests/outputs/realsense_live_register_hand_three_cam_fast \
      --mesh icp/Register.STL \
      --master-camera cam_00

  模式2 - 显式传入（手动指定外参JSON和各相机点云PLY）：
    python icp/register_to_mesh.py \
      --mesh icp/Register.STL \
      --extrinsics path/to/extrinsics.json \
      --point-cloud cam_00=/path/to/cam00.ply cam_01=/path/to/cam01.ply \
      --master-camera cam_00 \
      --output path/to/refined_extrinsics.json

外参JSON格式（--extrinsics）支持：
  {"cam_00": [[4x4]], "cam_01": [[4x4]], ...}
  或  {"extrinsics": {"cam_00": [[4x4]], ...}}
  或  {"cam_00": {"cam2world_4x4": [[4x4]], ...}, ...}

"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Go-ICP 后端（可选）
try:
    from icp.goicp import GoICPConfig, register_point_clouds as goicp_register
    _HAS_GOICP = True
except (ImportError, RuntimeError):
    _HAS_GOICP = False
    GoICPConfig = None
    goicp_register = None


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取二进制PLY中的 XYZ 和 RGB。"""
    with open(path, "rb") as f:
        header_bytes = b""
        while True:
            line = f.readline()
            header_bytes += line
            if line.startswith(b"end_header"):
                break
        header = header_bytes.decode("ascii")
        has_normal = any(
            line.startswith("property float nx")
            for line in header.strip().split("\n")
        )
        if has_normal:
            dtype = np.dtype([
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                ("r", "u1"), ("g", "u1"), ("b", "u1"),
            ])
        else:
            dtype = np.dtype([
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("r", "u1"), ("g", "u1"), ("b", "u1"),
            ])
        data = np.frombuffer(f.read(), dtype=dtype)
        xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
        rgb = np.stack([data["r"], data["g"], data["b"]], axis=1).astype(np.uint8)
    return xyz, rgb


def load_mesh_pcd(mesh_path: Path, num_points: int = 100000) -> o3d.geometry.PointCloud:
    """加载STL并均匀采样为点云。"""
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise RuntimeError(f"无法加载 mesh: {mesh_path}")
    mesh.compute_vertex_normals()
    bounds = np.asarray(mesh.vertices)
    print(f"  mesh 顶点: {len(np.asarray(mesh.vertices))}, 三角面: {len(np.asarray(mesh.triangles))}")
    print(f"  mesh 范围: X[{bounds[:,0].min():.4f},{bounds[:,0].max():.4f}] "
          f"Y[{bounds[:,1].min():.4f},{bounds[:,1].max():.4f}] "
          f"Z[{bounds[:,2].min():.4f},{bounds[:,2].max():.4f}]")
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    print(f"  mesh 采样点: {len(np.asarray(pcd.points))}")
    return pcd


def load_extrinsics_json(path: Path) -> dict[str, np.ndarray]:
    """加载外参JSON，支持多种格式。"""
    with open(path) as f:
        data = json.load(f)

    candidates = []

    # 格式1: {"cam_00": [[4x4]], "cam_01": [[4x4]], ...}
    if isinstance(data, dict):
        flat = {}
        for k, v in data.items():
            arr = np.array(v, dtype=np.float64)
            if arr.shape == (4, 4) and k.startswith("cam_"):
                flat[k] = arr
        if len(flat) >= 1:
            candidates.append(("直接 cam_xx 格式", flat))

        # 格式2: {"extrinsics": {"cam_00": [[4x4]], ...}}
        if "extrinsics" in data and isinstance(data["extrinsics"], dict):
            nested = {}
            for k, v in data["extrinsics"].items():
                arr = np.array(v, dtype=np.float64)
                if arr.shape == (4, 4):
                    nested[k] = arr
            if len(nested) >= 1:
                candidates.append(('extrinsics 嵌套格式', nested))

        # 格式3: {"cam_00": {"cam2world_4x4": [[4x4]], ...}}
        nested2 = {}
        for k, v in data.items():
            if isinstance(v, dict) and "cam2world_4x4" in v:
                arr = np.array(v["cam2world_4x4"], dtype=np.float64)
                if arr.shape == (4, 4):
                    nested2[k] = arr
        if len(nested2) >= 1:
            candidates.append(('cam2world_4x4 嵌套格式', nested2))

        # 格式4: camera_payload.json 格式（单相机）
        if "pose_record" in data and "cam2world_4x4" in data["pose_record"]:
            arr = np.array(data["pose_record"]["cam2world_4x4"], dtype=np.float64)
            if arr.shape == (4, 4):
                candidates.append(("camera_payload 格式", {"cam_00": arr}))

    if not candidates:
        raise ValueError(
            f"无法识别的外参格式: {path}\n"
            "支持: {\"cam_00\": [[4x4]], ...} 或 "
            "{\"extrinsics\": {\"cam_00\": [[4x4]], ...}} 或 "
            "{\"cam_00\": {\"cam2world_4x4\": [[4x4]], ...}}"
        )

    best = candidates[-1]
    print(f"  ✓ 外参格式: {best[0]}")
    return best[1]


def load_extrinsics_from_payload(frame_dir: Path, camera_ids: list[str]) -> dict[str, np.ndarray]:
    """从 camera_payload.json 加载外参（自动检测模式用）。"""
    result = {}
    for cam_id in camera_ids:
        payload_path = frame_dir / cam_id / "camera_payload.json"
        if not payload_path.is_file():
            continue
        with open(payload_path) as f:
            payload = json.load(f)
        cam2world = np.array(payload["pose_record"]["cam2world_4x4"], dtype=np.float64)
        result[cam_id] = cam2world
    return result


def transform_world_to_camera(points_world: np.ndarray, cam2world: np.ndarray) -> np.ndarray:
    world2cam = np.linalg.inv(cam2world)
    return points_world @ world2cam[:3, :3].T + world2cam[:3, 3][None, :]


def gl_to_opencv_cam2world(gl_cam2world: np.ndarray) -> np.ndarray:
    """将 single_seg GL 约定的 cam2world 转为 OpenCV 约定。

    GL: Y-up, 相机看向 -Z
    OpenCV: Y-down, 相机看向 +Z
    转换: 旋转矩阵的第2、3列取反, 平移不变
    """
    cv = gl_cam2world.copy()
    cv[:3, 1] *= -1.0
    cv[:3, 2] *= -1.0
    return cv


def compute_angle(T: np.ndarray) -> float:
    trace = np.trace(T[:3, :3])
    return float(np.rad2deg(np.arccos(np.clip((trace - 1) / 2, -1, 1))))


def merge_point_clouds(
    pts_list: list[np.ndarray],
    cols_list: list[np.ndarray] | None = None,
    voxel_size: float = 0.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """合并多个点云，可选下采样。"""
    if not pts_list:
        return np.empty((0, 3), dtype=np.float64), None
    all_pts = np.concatenate(pts_list, axis=0)
    all_cols = np.concatenate(cols_list, axis=0) if cols_list else None
    if voxel_size > 0 and all_pts.shape[0] > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_pts)
        if all_cols is not None:
            pcd.colors = o3d.utility.Vector3dVector(all_cols.astype(np.float64) / 255.0)
        pcd = pcd.voxel_down_sample(voxel_size)
        all_pts = np.asarray(pcd.points, dtype=np.float64)
        if all_cols is not None:
            cols_np = np.asarray(pcd.colors, dtype=np.float64)
            all_cols = np.clip(np.round(cols_np * 255.0), 0, 255).astype(np.uint8)
    return all_pts, all_cols


# ──────────────────────────────────────────────
# 数据加载：自动检测模式
# ──────────────────────────────────────────────


def auto_discover(data_dir: Path, camera_ids: list[str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    """自动从 data_dir/live_rgbd_debug/frame_*/cam_id/ 加载点云、外参和相机信息。

    返回: (world_clouds, extrinsics, camera_info)
      world_clouds[cam_id] = (N, 3) 世界坐标点云
      extrinsics[cam_id]   = (4, 4) cam2world
      camera_info[cam_id]  = {"serial_number": str, "color_intrinsics": dict}
    """
    live_debug_dir = data_dir / "live_rgbd_debug"
    frame_dirs = sorted(live_debug_dir.glob("frame_*"))
    if not frame_dirs:
        raise FileNotFoundError(f"无 frame_* 目录: {live_debug_dir}")

    world_clouds: dict[str, np.ndarray] = {}
    extrinsics: dict[str, np.ndarray] = {}
    camera_info: dict = {}
    for cam_id in camera_ids:
        pts_list: list[np.ndarray] = []
        frame_count = 0
        for frame_dir in frame_dirs:
            ply_path = frame_dir / cam_id / "target_object_rgb.ply"
            if not ply_path.is_file():
                continue
            pts, _ = read_ply(ply_path)
            if pts.shape[0] == 0:
                continue
            pts_list.append(pts)
            frame_count += 1

        if not pts_list:
            print(f"  ⚠️ {cam_id}: 无有效点云")
            world_clouds[cam_id] = np.empty((0, 3), dtype=np.float64)
            extrinsics[cam_id] = np.eye(4, dtype=np.float64)
            continue

        world_clouds[cam_id] = np.concatenate(pts_list, axis=0)
        print(f"  [{cam_id}] {world_clouds[cam_id].shape[0]} 点, {frame_count} 帧")

    # 从第一帧加载外参和相机信息（所有相机都有数据的第一帧）
    for frame_dir in frame_dirs:
        partial_ext = {}
        partial_info = {}
        for cam_id in camera_ids:
            payload_path = frame_dir / cam_id / "camera_payload.json"
            if not payload_path.is_file():
                continue
            with open(payload_path) as f:
                payload = json.load(f)
            partial_ext[cam_id] = np.array(payload["pose_record"]["cam2world_4x4"], dtype=np.float64)
            partial_info[cam_id] = {
                "serial_number": str(payload.get("serial_number", cam_id)),
                "color_intrinsics": dict(payload.get("color_intrinsics", {})),
            }
        if len(partial_ext) == len(camera_ids):
            extrinsics = partial_ext
            camera_info = partial_info
            for cam_id in camera_ids:
                print(f"  [{cam_id}] 外参+序列号: {frame_dir / cam_id / 'camera_payload.json'}")
            break
    else:
        raise FileNotFoundError(f"无法找到所有相机的外参")

    return world_clouds, extrinsics, camera_info


# ──────────────────────────────────────────────
# 数据加载：显式模式
# ──────────────────────────────────────────────


def load_explicit(
    extrinsics_path: Path,
    point_cloud_specs: list[str],
    camera_ids: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """从显式传入的外参JSON和点云路径加载数据。

    point_cloud_specs: ["cam_00=/path/to/a.ply", "cam_01=/path/to/b.ply", ...]
    返回: (world_clouds, extrinsics)
    """
    extrinsics = load_extrinsics_json(extrinsics_path)

    world_clouds: dict[str, np.ndarray] = {}
    for spec in point_cloud_specs:
        if "=" not in spec:
            raise ValueError(f"点云参数格式错误，应为 cam_id=path: {spec}")
        cam_id, ply_path_str = spec.split("=", 1)
        cam_id = cam_id.strip()
        ply_path = Path(ply_path_str).expanduser().resolve()

        if cam_id not in camera_ids:
            print(f"  ⚠️ {cam_id} 不在 camera_ids 列表中，跳过")
            continue

        if not ply_path.is_file():
            raise FileNotFoundError(f"点云文件不存在: {ply_path}")

        pts, _ = read_ply(ply_path)
        world_clouds[cam_id] = pts
        print(f"  [{cam_id}] {pts.shape[0]} 点 <- {ply_path}")

    # 确保所有需要的相机都有数据
    for cam_id in camera_ids:
        if cam_id not in extrinsics:
            raise ValueError(f"外参中缺少 {cam_id}")
        if cam_id not in world_clouds:
            raise ValueError(f"点云中缺少 {cam_id}")

    return world_clouds, extrinsics


# ──────────────────────────────────────────────
# 核心配准逻辑
# ──────────────────────────────────────────────


def _make_goicp_config(trim_fraction: float | None = None):
    """Build a Go-ICP config aligned with icp_ref.py defaults.

    By default this returns GoICPConfig() without overriding max correspondence,
    min correspondence, or MSE threshold. trim_fraction is the only optional
    override kept for quick experiments.
    """
    cfg = GoICPConfig()
    if trim_fraction is not None:
        cfg.goicp_trim_fraction = float(trim_fraction)
    return cfg


def _format_goicp_config(cfg) -> str:
    return (
        f"voxel_size_ratio={cfg.voxel_size_ratio}, "
        f"max_corr_ratio={cfg.goicp_max_corr_ratio:.4f}, "
        f"min_voxel_size={cfg.min_voxel_size:.4f}, "
        f"min_corr={cfg.min_goicp_corr:.4f}, "
        f"trim_fraction={cfg.goicp_trim_fraction:.3f}, "
        f"mse_thresh={cfg.goicp_mse_thresh:g}"
    )


def run_registration(
    world_clouds: dict[str, np.ndarray],
    extrinsics: dict[str, np.ndarray],
    mesh_pcd: o3d.geometry.PointCloud,
    master_camera: str,
    camera_ids: list[str],
    voxel_size: float = 0.003,
    master_icp_dist: float = 0.05,
    refine_icp_dist: float = 0.003,
    use_goicp: bool = False,
    refine_use_goicp: bool = False,
    goicp_trim_fraction: float | None = None,
    refine_goicp_trim_fraction: float | None = None,
) -> dict:
    """执行配准流程，返回调整结果。

    返回 dict，结构同保存的JSON中的 per-camera 数据。
    """
    other_cameras = [c for c in camera_ids if c != master_camera]
    mesh_pts = np.asarray(mesh_pcd.points, dtype=np.float64)
    if refine_goicp_trim_fraction is None:
        refine_goicp_trim_fraction = goicp_trim_fraction

    # ── 下采样 ──
    print("  下采样...")
    pcd_world_down: dict[str, o3d.geometry.PointCloud] = {}
    for cam_id in camera_ids:
        pts = world_clouds[cam_id]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        if voxel_size > 0 and pts.shape[0] > 0:
            pcd = pcd.voxel_down_sample(voxel_size)
        pcd_world_down[cam_id] = pcd
        print(f"    [{cam_id}] {pts.shape[0]} → {len(np.asarray(pcd.points))} 点")
    print()

    # ── Step 2: 主相机 → mesh ──
    master_backend = "Open3D ICP"
    print("=" * 60)
    print("2. 主相机配准 → mesh 位姿 T_MW")
    print("=" * 60)
    master_pts_down = np.asarray(pcd_world_down[master_camera].points, dtype=np.float64)

    T_MW = None
    if use_goicp:
        if not _HAS_GOICP:
            raise RuntimeError(
                "--use-goicp 已指定但 Go-ICP 未安装。请安装: pip install py_goicp\n"
                "  或用 Open3D ICP（默认，不传 --use-goicp）"
            )
        goicp_cfg = _make_goicp_config(goicp_trim_fraction)
        print(f"  Go-ICP 配准中: {_format_goicp_config(goicp_cfg)}")
        goicp_result, _ = goicp_register(
            moving_points=master_pts_down,
            reference_points=mesh_pts,
            config=goicp_cfg,
        )
        T_MW = goicp_result.transformation
        master_fitness = goicp_result.fitness
        master_rmse = goicp_result.inlier_rmse
        master_backend = "Go-ICP"

    else:
        m_center = np.mean(master_pts_down, axis=0)
        mesh_center = np.mean(mesh_pts, axis=0)
        T_init = np.eye(4)
        T_init[:3, 3] = mesh_center - m_center
        master_pcd = o3d.geometry.PointCloud()
        master_pcd.points = o3d.utility.Vector3dVector(master_pts_down)
        mesh_pcd_temp = o3d.geometry.PointCloud()
        mesh_pcd_temp.points = o3d.utility.Vector3dVector(mesh_pts)
        print(f"  Open3D ICP 配准中: max_correspondence_distance={master_icp_dist}")
        result = o3d.pipelines.registration.registration_icp(
            source=master_pcd, target=mesh_pcd_temp,
            max_correspondence_distance=master_icp_dist,
            init=T_init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=300),
        )
        T_MW = result.transformation
        master_fitness = result.fitness
        master_rmse = result.inlier_rmse

    print(f"  后端: {master_backend}")
    print(f"  主相机配准: fitness={master_fitness:.4f}, rmse={master_rmse:.4f}")
    print(f"  T_MW (world→mesh):")
    print(f"    {np.array2string(T_MW, precision=6, suppress_small=True)}")
    print()

    # ── Step 3: 非主相机微调 ──
    refine_backend = "Go-ICP" if refine_use_goicp else "Open3D ICP"
    print("=" * 60)
    print(f"3. 非主相机：camera→mesh 空间 ICP 微调（{refine_backend}）")
    print("=" * 60)
    adj_results: dict = {}
    _refine_solver_ctx = None

    for cam_id in other_cameras:
        cam2world = extrinsics[cam_id]

        predicted_T_MC = T_MW @ cam2world
        predicted_angle = compute_angle(predicted_T_MC)
        print(f"\n  [{cam_id}]")
        if refine_use_goicp:
            print(f"    predicted_cam_to_mesh: 旋转={predicted_angle:.4f}°（仅记录；Go-ICP 全局分支不使用初值）")
        else:
            print(f"    predicted_cam_to_mesh: 旋转={predicted_angle:.4f}°（Open3D ICP 初值）")

        pts_world = world_clouds[cam_id]
        pts_camera = transform_world_to_camera(pts_world, cam2world)
        pcd_camera = o3d.geometry.PointCloud()
        pcd_camera.points = o3d.utility.Vector3dVector(pts_camera)
        if voxel_size > 0 and pts_camera.shape[0] > 0:
            pcd_camera = pcd_camera.voxel_down_sample(voxel_size)
        pts_camera_down = np.asarray(pcd_camera.points, dtype=np.float64)
        print(f"    相机坐标系下: {pts_world.shape[0]} → {pts_camera_down.shape[0]} 点")

        if refine_use_goicp:
            if not _HAS_GOICP:
                raise RuntimeError(
                    "--refine-use-goicp 已指定但 Go-ICP 未安装。请安装: pip install py_goicp"
                )
            goicp_cfg = _make_goicp_config(refine_goicp_trim_fraction)
            print(f"    Go-ICP 配准中: {_format_goicp_config(goicp_cfg)}")
            goicp_result, _refine_solver_ctx = goicp_register(
                moving_points=pts_camera_down,
                reference_points=mesh_pts,
                config=goicp_cfg,
                solver_ctx=_refine_solver_ctx,
            )
            T_MC_refined = goicp_result.transformation
            refine_fitness = goicp_result.fitness
            refine_rmse = goicp_result.inlier_rmse
        else:
            result = o3d.pipelines.registration.registration_icp(
                source=pcd_camera, target=mesh_pcd,
                max_correspondence_distance=refine_icp_dist,
                init=predicted_T_MC,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
            )
            T_MC_refined = result.transformation
            refine_fitness = float(result.fitness)
            refine_rmse = float(result.inlier_rmse)

        T_MW_inv = np.linalg.inv(T_MW)
        new_cam2world = T_MW_inv @ T_MC_refined
        adj = new_cam2world @ np.linalg.inv(cam2world)
        adj_angle = compute_angle(adj)
        adj_trans = float(np.linalg.norm(adj[:3, 3]))

        adj_results[cam_id] = {
            "predicted_T_MC": predicted_T_MC.tolist(),
            "refined_T_MC": T_MC_refined.tolist(),
            "new_cam2world": new_cam2world.tolist(),
            "adjustment_4x4": adj.tolist(),
            "adjustment_rotation_deg": adj_angle,
            "adjustment_translation_m": adj_trans,
            "icp_fitness": refine_fitness,
            "icp_inlier_rmse": refine_rmse,
        }
        print(f"    ICP ({refine_backend}): fitness={refine_fitness:.4f}, rmse={refine_rmse:.4f}")
        print(f"    外参调整: 旋转={adj_angle:.4f}°, 平移={adj_trans:.4f}m")

    print()

    # ── Step 4: 验证 ──
    print("=" * 60)
    print("4. 验证：新外参下点云一致性")
    print("=" * 60)
    master_pts = np.asarray(pcd_world_down[master_camera].points)
    for cam_id in other_cameras:
        if cam_id not in adj_results:
            continue
        adj = adj_results[cam_id]
        pts_world = world_clouds[cam_id]
        pts_camera = transform_world_to_camera(pts_world, extrinsics[cam_id])
        new_T_WC = np.array(adj["new_cam2world"], dtype=np.float64)
        pts_new_world = pts_camera @ new_T_WC[:3, :3].T + new_T_WC[:3, 3][None, :]

        pcd_new = o3d.geometry.PointCloud()
        pcd_new.points = o3d.utility.Vector3dVector(pts_new_world)
        if voxel_size > 0:
            pcd_new = pcd_new.voxel_down_sample(voxel_size)

        init_center = np.mean(np.asarray(pcd_new.points), axis=0) - np.mean(master_pts, axis=0)
        T_init_v = np.eye(4)
        T_init_v[:3, 3] = -init_center

        rv = o3d.pipelines.registration.registration_icp(
            source=pcd_new, target=pcd_world_down[master_camera],
            max_correspondence_distance=0.01,
            init=T_init_v,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
        )
        adj["verify_vs_master_fitness"] = float(rv.fitness)
        adj["verify_vs_master_rmse"] = float(rv.inlier_rmse)
        print(f"  [{cam_id}] 新外参下与主相机重叠: fitness={rv.fitness:.4f}, rmse={rv.inlier_rmse:.4f}")
    print()

    adj_results["_world_to_mesh_4x4"] = T_MW.tolist()
    return adj_results


# ──────────────────────────────────────────────
# 保存融合场景点云
# ──────────────────────────────────────────────


def save_fused_point_clouds(
    output_root: Path,
    world_clouds: dict[str, np.ndarray],
    extrinsics: dict[str, np.ndarray],
    master_camera: str,
    camera_ids: list[str],
    adj_results: dict,
    voxel_size: float,
) -> None:
    """保存原始和新外参下的融合场景点云，便于对比效果。"""
    fused_dir = output_root / "fused"
    fused_dir.mkdir(parents=True, exist_ok=True)

    orig_all_pts: list[np.ndarray] = []
    orig_all_cols: list[np.ndarray] = []
    ref_all_pts: list[np.ndarray] = []
    ref_all_cols: list[np.ndarray] = []

    for cam_id in camera_ids:
        pts = world_clouds[cam_id]
        if pts.shape[0] == 0:
            continue
        cols = np.zeros((pts.shape[0], 3), dtype=np.uint8)

        orig_all_pts.append(pts)
        orig_all_cols.append(cols)

        if cam_id == master_camera:
            ref_all_pts.append(pts)
            ref_all_cols.append(cols.copy())
        elif cam_id in adj_results:
            cam2world = extrinsics[cam_id]
            world2cam = np.linalg.inv(cam2world)
            pts_camera = pts @ world2cam[:3, :3].T + world2cam[:3, 3][None, :]
            new_T = np.array(adj_results[cam_id]["new_cam2world"], dtype=np.float64)
            pts_new_world = pts_camera @ new_T[:3, :3].T + new_T[:3, 3][None, :]
            ref_all_pts.append(pts_new_world)
            ref_all_cols.append(cols.copy())

    def fuse_and_save(pts_list, cols_list, save_path, label):
        if not pts_list:
            return
        all_pts = np.concatenate(pts_list, axis=0)
        all_cols = np.concatenate(cols_list, axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_pts)
        if voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size)
        pts_down = np.asarray(pcd.points, dtype=np.float32)
        dummy_cols = np.full((pts_down.shape[0], 3), 200, dtype=np.uint8)
        _write_ply_binary(save_path, pts_down, dummy_cols)
        print(f"  {label}: {save_path} ({pts_down.shape[0]} 点)")

    orig_path = fused_dir / "original_fused.ply"
    ref_path = fused_dir / "refined_fused.ply"
    fuse_and_save(orig_all_pts, orig_all_cols, orig_path, "原始外参融合")
    fuse_and_save(ref_all_pts, ref_all_cols, ref_path, "新外参融合")

    # 着色对比（蓝=新外参, 橙=原始外参）
    if orig_all_pts and ref_all_pts:
        ref_all = np.concatenate(ref_all_pts, axis=0)
        orig_all = np.concatenate(orig_all_pts, axis=0)
        ref_pcd = o3d.geometry.PointCloud()
        ref_pcd.points = o3d.utility.Vector3dVector(ref_all)
        orig_pcd = o3d.geometry.PointCloud()
        orig_pcd.points = o3d.utility.Vector3dVector(orig_all)
        if voxel_size > 0:
            ref_pcd = ref_pcd.voxel_down_sample(voxel_size)
            orig_pcd = orig_pcd.voxel_down_sample(voxel_size)
        ref_d = np.asarray(ref_pcd.points, dtype=np.float32)
        orig_d = np.asarray(orig_pcd.points, dtype=np.float32)
        offset_x = float(np.max(ref_d[:, 0]) - np.min(orig_d[:, 0]) + 0.2)
        orig_d[:, 0] += offset_x
        combined_pts = np.concatenate([ref_d, orig_d], axis=0)
        blue = np.full((ref_d.shape[0], 3), [100, 160, 255], dtype=np.uint8)
        orange = np.full((orig_d.shape[0], 3), [255, 160, 80], dtype=np.uint8)
        _write_ply_binary(fused_dir / "comparison_colored.ply", combined_pts, np.concatenate([blue, orange], axis=0))
        print(f"  着色对比（蓝=新外参 橙=原始外参）: {fused_dir / 'comparison_colored.ply'}")

    print()


def _write_ply_binary(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """写二进制 PLY（仅 XYZ+RGB）。"""
    n = xyz.shape[0]
    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {n}\n".encode())
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        f.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(b"end_header\n")
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        buf = np.empty(n, dtype=dtype)
        buf["x"] = xyz[:, 0]
        buf["y"] = xyz[:, 1]
        buf["z"] = xyz[:, 2]
        buf["r"] = rgb[:, 0]
        buf["g"] = rgb[:, 1]
        buf["b"] = rgb[:, 2]
        f.write(buf.tobytes())


# ──────────────────────────────────────────────
# 保存默认输出目录内容
# ──────────────────────────────────────────────


def _camera_pose_entries(
    camera_ids: list[str],
    extrinsics: dict[str, np.ndarray],
    camera_info: dict | None = None,
) -> list[dict]:
    entries = []
    for cam_id in camera_ids:
        cam2world = np.asarray(extrinsics[cam_id], dtype=np.float64)
        world2cam = np.linalg.inv(cam2world)
        opencv_c2w = gl_to_opencv_cam2world(cam2world)
        info = (camera_info or {}).get(cam_id, {})
        entries.append(
            {
                "camera_id": cam_id,
                "serial_number": str(info.get("serial_number", cam_id)),
                "cam2world_4x4": cam2world.tolist(),
                "world2cam_4x4": world2cam.tolist(),
                "opencv_cv_cam2world_4x4": opencv_c2w.tolist(),
                "single_seg_gl_cam2world_4x4": cam2world.tolist(),
                "color_intrinsics": info.get("color_intrinsics", {}),
            }
        )
    return entries


def build_refined_extrinsics(
    master_camera: str,
    camera_ids: list[str],
    extrinsics: dict[str, np.ndarray],
    adj_results: dict,
) -> dict[str, np.ndarray]:
    refined = {}
    for cam_id in camera_ids:
        if cam_id == master_camera or cam_id not in adj_results:
            refined[cam_id] = np.asarray(extrinsics[cam_id], dtype=np.float64).copy()
        else:
            refined[cam_id] = np.asarray(adj_results[cam_id]["new_cam2world"], dtype=np.float64)
    return refined


def transform_clouds_to_refined_world(
    world_clouds: dict[str, np.ndarray],
    original_extrinsics: dict[str, np.ndarray],
    refined_extrinsics: dict[str, np.ndarray],
    camera_ids: list[str],
) -> dict[str, np.ndarray]:
    refined_clouds = {}
    for cam_id in camera_ids:
        pts = world_clouds[cam_id]
        if pts.shape[0] == 0:
            refined_clouds[cam_id] = pts.copy()
            continue
        pts_camera = transform_world_to_camera(pts, original_extrinsics[cam_id])
        new_t = refined_extrinsics[cam_id]
        refined_clouds[cam_id] = pts_camera @ new_t[:3, :3].T + new_t[:3, 3][None, :]
    return refined_clouds


def save_default_output_artifacts(
    output_dir: Path,
    mesh_path: Path,
    world_clouds: dict[str, np.ndarray],
    original_extrinsics: dict[str, np.ndarray],
    refined_extrinsics: dict[str, np.ndarray],
    master_camera: str,
    camera_ids: list[str],
    camera_info: dict | None = None,
    voxel_size: float = 0.003,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_dst = output_dir / mesh_path.name
    if mesh_path.resolve() != mesh_dst.resolve():
        shutil.copy2(mesh_path, mesh_dst)

    original_data = {
        "cameras": _camera_pose_entries(camera_ids, original_extrinsics, camera_info),
    }
    with (output_dir / "original_extrinsics.json").open("w") as f:
        json.dump(original_data, f, indent=2, ensure_ascii=False)

    refined_clouds = transform_clouds_to_refined_world(
        world_clouds=world_clouds,
        original_extrinsics=original_extrinsics,
        refined_extrinsics=refined_extrinsics,
        camera_ids=camera_ids,
    )

    cam_colors = {
        "cam_00": np.array([40, 130, 255], dtype=np.uint8),
        "cam_01": np.array([255, 128, 40], dtype=np.uint8),
        "cam_02": np.array([40, 210, 90], dtype=np.uint8),
    }
    for cam_id in camera_ids:
        pts = refined_clouds[cam_id]
        if pts.shape[0] == 0:
            continue
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        if voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size)
        pts_down = np.asarray(pcd.points, dtype=np.float32)
        color = cam_colors.get(cam_id, np.array([230, 230, 230], dtype=np.uint8))
        colors = np.tile(color[None, :], (pts_down.shape[0], 1))
        _write_ply_binary(output_dir / f"{cam_id}_registered.ply", pts_down, colors)

    print(f"  默认输出目录: {output_dir}")
    print(f"    原始 mesh: {mesh_dst}")
    print(f"    原始外参: {output_dir / 'original_extrinsics.json'}")
    print(f"    调整后外参: {output_dir / 'refined_extrinsics.json'}")
    for cam_id in camera_ids:
        print(f"    [{cam_id}] 配准后点云: {output_dir / f'{cam_id}_registered.ply'}")
    print()


# ──────────────────────────────────────────────
# 保存结果
# ──────────────────────────────────────────────


def save_results(
    output_path: Path,
    master_camera: str,
    camera_ids: list[str],
    extrinsics: dict[str, np.ndarray],
    adj_results: dict,
    camera_info: dict | None = None,
    t_elapsed: float = 0.0,
) -> None:
    """保存为 camera_poses_apriltag.json 格式，与其他脚本兼容。"""
    cameras_list = []
    for cam_id in camera_ids:
        cam2world_orig = extrinsics[cam_id]

        if cam_id == master_camera:
            refined_c2w = cam2world_orig.copy()
        elif cam_id in adj_results:
            refined_c2w = np.array(adj_results[cam_id]["new_cam2world"], dtype=np.float64)
        else:
            refined_c2w = cam2world_orig.copy()

        world2refined = np.linalg.inv(refined_c2w)
        opencv_c2w = gl_to_opencv_cam2world(refined_c2w)

        info = (camera_info or {}).get(cam_id, {})
        cam_entry = {
            "camera_id": cam_id,
            "serial_number": str(info.get("serial_number", cam_id)),
            "cam2world_4x4": refined_c2w.tolist(),
            "world2cam_4x4": world2refined.tolist(),
            "opencv_cv_cam2world_4x4": opencv_c2w.tolist(),
            "single_seg_gl_cam2world_4x4": refined_c2w.tolist(),
            "color_intrinsics": info.get("color_intrinsics", {}),
        }
        cameras_list.append(cam_entry)

    output_data = {"cameras": cameras_list}
    if isinstance(adj_results, dict) and "_world_to_mesh_4x4" in adj_results:
        output_data["world_to_mesh_4x4"] = adj_results["_world_to_mesh_4x4"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"  优化外参: {output_path}")

    print()
    print("=" * 60)
    print("完成")
    print("=" * 60)
    print(f"  主相机: {master_camera}（不变）")
    for cam_id in camera_ids:
        if cam_id != master_camera and cam_id in adj_results:
            a = adj_results[cam_id]
            print(f"  [{cam_id}] 调整: {a['adjustment_rotation_deg']:.4f}°, "
                  f"{a['adjustment_translation_m']:.4f}m")
    print(f"  耗时: {t_elapsed:.1f}s")


# ──────────────────────────────────────────────
# 可视化（可选）
# ──────────────────────────────────────────────


def visualize(
    mesh_pcd: o3d.geometry.PointCloud,
    T_MW: np.ndarray,
    pcd_world: dict[str, o3d.geometry.PointCloud],
    camera_ids: list[str],
) -> None:
    mesh_in_world = copy.deepcopy(mesh_pcd)
    mesh_in_world.transform(T_MW)
    mesh_in_world.paint_uniform_color([0.5, 0.5, 0.5])
    geoms = [mesh_in_world]
    cam_colors = {
        "cam_00": [0.2, 0.6, 1.0],
        "cam_01": [1.0, 0.4, 0.2],
        "cam_02": [0.2, 0.8, 0.2],
    }
    for cam_id in camera_ids:
        if cam_id in pcd_world:
            c = copy.deepcopy(pcd_world[cam_id])
            c.paint_uniform_color(cam_colors.get(cam_id, [0.8, 0.8, 0.8]))
            geoms.append(c)
    o3d.visualization.draw_geometries(geoms, window_name="Refined Extrinsics")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="多相机外参微调：相机点云配准到 mesh，支持 Open3D ICP 和 Go-ICP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
两种模式（二选一）:
  自动检测模式:
    --data-dir <目录>
  显式模式:
    --extrinsics <JSON> --point-cloud cam_00=<ply> cam_01=<ply> ...

外参JSON格式:
  {"cam_00": [[4x4]], "cam_01": [[4x4]], ...}
  或 {"extrinsics": {"cam_00": [[4x4]], ...}}
  或 {"cam_00": {"cam2world_4x4": [[4x4]], ...}}
        """,
    )

    # 核心参数
    parser.add_argument("--mesh", type=Path, required=True, help="标定物体 .stl 文件")
    parser.add_argument("--master-camera", type=str, default="cam_00", help="主相机 ID")
    parser.add_argument("--camera-ids", type=str, nargs="+", default=["cam_00", "cam_01", "cam_02"],
                        help="所有相机 ID 列表")
    parser.add_argument("--output", type=Path, default=None,
                        help="输出 JSON 路径（默认: icp/output/<mesh名>/refined_extrinsics.json）")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="默认输出目录（默认: icp/output/<mesh名>）")

    # 模式1: 自动检测
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="[自动检测] 数据目录，自动寻找点云和外参")

    # 模式2: 显式传入
    parser.add_argument("--extrinsics", type=Path, default=None,
                        help="[显式模式] 原始外参 JSON 文件")
    parser.add_argument("--point-cloud", type=str, nargs="+", default=None,
                        help="[显式模式] 点云文件，格式: cam_00=/path/to/a.ply cam_01=/path/to/b.ply")

    # 配准参数
    parser.add_argument("--use-goicp", action="store_true",
                        help="主相机使用 Go-ICP 全局配准（默认用 Open3D ICP 局部配准）")
    parser.add_argument("--refine-use-goicp", action="store_true",
                        help="从相机微调也使用 Go-ICP 全局配准（默认用 Open3D ICP 局部微调）")
    parser.add_argument("--goicp-trim-fraction", type=float, default=None,
                        help="主相机 Go-ICP TrimFraction（默认使用 GoICPConfig 默认值 0.05）")
    parser.add_argument("--refine-goicp-trim-fraction", type=float, default=None,
                        help="从相机 Go-ICP TrimFraction（默认同 --goicp-trim-fraction；未传则使用 GoICPConfig 默认值 0.05）")
    parser.add_argument("--voxel-size", type=float, default=0.003,
                        help="体素下采样大小")
    parser.add_argument("--master-icp-dist", type=float, default=0.05,
                        help="主相机配准到mesh的最大对应距离")
    parser.add_argument("--refine-icp-dist", type=float, default=0.003,
                        help="非主相机微调的最大对应距离")

    # 其他
    parser.add_argument("--visualize", type=int, default=0,
                        help="是否显示可视化窗口")
    parser.add_argument("--num-mesh-points", type=int, default=100000,
                        help="mesh 采样点数")
    parser.add_argument("--save-fused", type=int, default=0,
                        help="是否保存原始和新外参下的融合场景点云（.ply），便于对比效果")
    args = parser.parse_args()

    # ── 校验参数 ──
    mesh_path = Path(args.mesh).expanduser().resolve()
    camera_ids = list(args.camera_ids)
    master_camera = str(args.master_camera)

    if master_camera not in camera_ids:
        raise ValueError(f"主相机 {master_camera} 不在 camera_ids 中: {camera_ids}")

    auto_mode = args.data_dir is not None
    explicit_mode = args.extrinsics is not None and args.point_cloud is not None

    if auto_mode and explicit_mode:
        raise ValueError("请选择一种模式：--data-dir（自动）或 --extrinsics + --point-cloud（显式），不能同时使用")
    if not auto_mode and not explicit_mode:
        raise ValueError("请提供 --data-dir（自动检测）或 --extrinsics + --point-cloud（显式传入）")

    # ── 输出路径 ──
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        default_output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else output_path.parent
        )
    else:
        default_output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else REPO_ROOT / "icp" / "output" / mesh_path.stem
        )
        output_path = default_output_dir / "refined_extrinsics.json"

    t0 = time.perf_counter()

    # ════════════════════════════════════════════
    # 1. 加载 mesh
    # ════════════════════════════════════════════
    print("=" * 60)
    print("1. 加载 mesh")
    print("=" * 60)
    mesh_pcd = load_mesh_pcd(mesh_path, args.num_mesh_points)
    print()

    # ════════════════════════════════════════════
    # 2. 加载数据
    # ════════════════════════════════════════════
    print("=" * 60)
    print("2. 加载数据")
    print("=" * 60)
    if auto_mode:
        print("  模式: 自动检测")
        data_dir = Path(args.data_dir).expanduser().resolve()
        print(f"  数据目录: {data_dir}")
        world_clouds, extrinsics, camera_info = auto_discover(data_dir, camera_ids)
    else:
        print("  模式: 显式传入")
        extrinsics_path = Path(args.extrinsics).expanduser().resolve()
        print(f"  外参: {extrinsics_path}")
        world_clouds, extrinsics = load_explicit(extrinsics_path, args.point_cloud, camera_ids)
        camera_info = {}
    print()

    # ════════════════════════════════════════════
    # 3. 执行配准
    # ════════════════════════════════════════════
    adj_results = run_registration(
        world_clouds=world_clouds,
        extrinsics=extrinsics,
        mesh_pcd=mesh_pcd,
        master_camera=master_camera,
        camera_ids=camera_ids,
        voxel_size=args.voxel_size,
        master_icp_dist=args.master_icp_dist,
        refine_icp_dist=args.refine_icp_dist,
        use_goicp=bool(args.use_goicp),
        refine_use_goicp=bool(args.refine_use_goicp),
        goicp_trim_fraction=(
            None
            if args.goicp_trim_fraction is None
            else float(args.goicp_trim_fraction)
        ),
        refine_goicp_trim_fraction=(
            None
            if args.refine_goicp_trim_fraction is None
            else float(args.refine_goicp_trim_fraction)
        ),
    )

    # ════════════════════════════════════════════
    # 4. 保存结果
    # ════════════════════════════════════════════
    elapsed = time.perf_counter() - t0
    save_results(output_path, master_camera, camera_ids, extrinsics, adj_results,
                 camera_info=camera_info, t_elapsed=elapsed)

    refined_extrinsics = build_refined_extrinsics(
        master_camera=master_camera,
        camera_ids=camera_ids,
        extrinsics=extrinsics,
        adj_results=adj_results,
    )
    save_default_output_artifacts(
        output_dir=default_output_dir,
        mesh_path=mesh_path,
        world_clouds=world_clouds,
        original_extrinsics=extrinsics,
        refined_extrinsics=refined_extrinsics,
        master_camera=master_camera,
        camera_ids=camera_ids,
        camera_info=camera_info,
        voxel_size=args.voxel_size,
    )

    # ════════════════════════════════════════════
    # 5. 保存融合场景点云（可选）
    # ════════════════════════════════════════════
    if args.save_fused:
        if auto_mode:
            data_dir = Path(args.data_dir).expanduser().resolve()
            fused_root = REPO_ROOT / "tests" / "outputs" / f"{data_dir.name}_refined"
        else:
            fused_root = output_path.parent
        save_fused_point_clouds(
            output_root=fused_root,
            world_clouds=world_clouds,
            extrinsics=extrinsics,
            master_camera=master_camera,
            camera_ids=camera_ids,
            adj_results=adj_results,
            voxel_size=args.voxel_size,
        )

    # ════════════════════════════════════════════
    # 6. 可视化（可选）
    # ════════════════════════════════════════════
    if args.visualize:
        print("\n打开可视化窗口...")
        master_pts = world_clouds[master_camera]
        mesh_pts = np.asarray(mesh_pcd.points, dtype=np.float64)
        master_down = np.asarray(
            o3d.geometry.PointCloud(
                points=o3d.utility.Vector3dVector(master_pts)
            ).voxel_down_sample(args.voxel_size).points
        )
        if args.use_goicp:
            if not _HAS_GOICP:
                raise RuntimeError("--use-goicp 已指定但 Go-ICP 未安装")
            goicp_cfg = _make_goicp_config(args.goicp_trim_fraction)
            goicp_res, _ = goicp_register(moving_points=master_down, reference_points=mesh_pts, config=goicp_cfg)
            T_MW_viz = goicp_res.transformation
        else:
            m_center = np.mean(master_down, axis=0)
            mesh_center = np.mean(mesh_pts, axis=0)
            T_init = np.eye(4)
            T_init[:3, 3] = mesh_center - m_center
            master_pcd = o3d.geometry.PointCloud()
            master_pcd.points = o3d.utility.Vector3dVector(master_down)
            mesh_temp = o3d.geometry.PointCloud()
            mesh_temp.points = o3d.utility.Vector3dVector(mesh_pts)
            result = o3d.pipelines.registration.registration_icp(
                source=master_pcd, target=mesh_temp,
                max_correspondence_distance=args.master_icp_dist,
                init=T_init,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=300),
            )
            T_MW_viz = result.transformation
        pcd_world_down = {}
        for cam_id in camera_ids:
            pts = world_clouds[cam_id]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            if args.voxel_size > 0:
                pcd = pcd.voxel_down_sample(args.voxel_size)
            pcd_world_down[cam_id] = pcd
        visualize(mesh_pcd, T_MW_viz, pcd_world_down, camera_ids)


if __name__ == "__main__":
    main()
