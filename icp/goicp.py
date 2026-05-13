"""可复用的 Go-ICP 点云配准接口。"""

from __future__ import annotations

import copy
import hashlib
import importlib
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import open3d as o3d


@dataclass
class GoICPConfig:
    voxel_size_ratio: float = 0.01
    goicp_max_corr_ratio: float = 0.05
    min_voxel_size: float = 0.001
    min_goicp_corr: float = 0.005
    goicp_module: str = ""
    goicp_quiet: bool = True
    goicp_dt_size: int = 300
    goicp_dt_factor: float = 2.0
    goicp_trim_fraction: float = 0.05
    goicp_mse_thresh: float = 3e-4
    goicp_epsilon: Optional[float] = None
    rotation_only_output: bool = False


@dataclass
class GoICPResult:
    transformation: np.ndarray
    fitness: float
    inlier_rmse: float
    elapsed_sec: float
    used_voxel_size: float
    used_goicp_max_corr: float
    built_dt_this_call: bool


@dataclass
class GoICPSolverContext:
    solver: Any
    point_cls: Any
    model: Any
    module_name: str
    reference_hash: str
    config_signature: Tuple[Any, ...]
    dt_built: bool = False


def validate_goicp_config(config: GoICPConfig) -> None:
    if config.voxel_size_ratio <= 0 or config.goicp_max_corr_ratio <= 0:
        raise ValueError("voxel_size_ratio/goicp_max_corr_ratio 必须为正")
    if config.min_voxel_size <= 0 or config.min_goicp_corr <= 0:
        raise ValueError("min_voxel_size/min_goicp_corr 必须为正")
    if config.goicp_dt_size <= 0 or config.goicp_dt_factor <= 0:
        raise ValueError("goicp_dt_size/goicp_dt_factor 必须为正")
    if not (0.0 <= config.goicp_trim_fraction < 1.0):
        raise ValueError("goicp_trim_fraction 必须在 [0, 1) 区间")
    if config.goicp_mse_thresh <= 0:
        raise ValueError("goicp_mse_thresh 必须为正")
    if config.goicp_epsilon is not None and config.goicp_epsilon <= 0:
        raise ValueError("goicp_epsilon 必须为正")


def _load_goicp_backend(module_name: Optional[str]) -> Tuple[str, Any]:
    candidates = []
    if module_name:
        candidates.append(module_name)
    candidates.extend(["py_goicp", "goicp"])

    tried = []
    for name in candidates:
        if name in tried:
            continue
        tried.append(name)
        try:
            return name, importlib.import_module(name)
        except Exception:
            continue

    raise RuntimeError("未找到 Go-ICP Python 绑定。请安装 py_goicp/goicp，或显式指定模块名。")


def _estimate_object_diameter(points: np.ndarray) -> float:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    diameter = float(np.max(maxs - mins))
    return max(diameter, 1e-6)


def _resolve_registration_scales(target_full: np.ndarray, config: GoICPConfig) -> Tuple[float, float]:
    diameter = _estimate_object_diameter(target_full)
    voxel_size = max(float(config.min_voxel_size), float(config.voxel_size_ratio) * diameter)
    goicp_corr = max(float(config.min_goicp_corr), float(config.goicp_max_corr_ratio) * diameter)
    return voxel_size, goicp_corr


def _to_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def _build_colored_pcd(points: np.ndarray, color: Tuple[float, float, float]) -> o3d.geometry.PointCloud:
    pcd = _to_pcd(points)
    pcd.paint_uniform_color(list(color))
    return pcd


def _build_colored_mesh(
    mesh_verts: np.ndarray,
    mesh_faces: np.ndarray,
    color: Tuple[float, float, float],
) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(mesh_verts)
    mesh.triangles = o3d.utility.Vector3iVector(mesh_faces)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(list(color))
    return mesh


def _translate_points(points: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return (points + offset[None, :]).astype(np.float64)


def _preprocess_for_global_registration(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
) -> o3d.geometry.PointCloud:
    pcd_down = pcd.voxel_down_sample(voxel_size)
    normal_radius = voxel_size * 2.0
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30),
    )
    return pcd_down


def _normalize_points_for_goicp(
    moving_points: np.ndarray,
    reference_points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    moving_center = np.mean(moving_points, axis=0).astype(np.float64)
    reference_center = np.mean(reference_points, axis=0).astype(np.float64)
    moving_centered = (moving_points - moving_center[None, :]).astype(np.float64)
    reference_centered = (reference_points - reference_center[None, :]).astype(np.float64)

    reference_min = np.min(reference_points, axis=0)
    reference_max = np.max(reference_points, axis=0)
    scale = float(np.linalg.norm(reference_max - reference_min))
    if scale <= 1e-12:
        scale = 1.0

    moving_norm = (moving_centered / scale).astype(np.float64)
    reference_norm = (reference_centered / scale).astype(np.float64)
    return moving_norm, reference_norm, moving_center, reference_center, scale


def _hash_points(points: np.ndarray) -> str:
    points32 = np.ascontiguousarray(points.astype(np.float32))
    digest = hashlib.blake2b(points32.view(np.uint8), digest_size=16).hexdigest()
    return f"{points32.shape}-{digest}"


def _config_signature(config: GoICPConfig) -> Tuple[Any, ...]:
    return (
        str(config.goicp_module),
        bool(config.goicp_quiet),
        int(config.goicp_dt_size),
        float(config.goicp_dt_factor),
        float(config.goicp_trim_fraction),
        float(config.goicp_mse_thresh),
        None if config.goicp_epsilon is None else float(config.goicp_epsilon),
    )


def _build_solver_context(reference_down_np: np.ndarray, config: GoICPConfig) -> GoICPSolverContext:
    module_name, goicp_mod = _load_goicp_backend(config.goicp_module)
    if not hasattr(goicp_mod, "GoICP") or not hasattr(goicp_mod, "POINT3D"):
        raise RuntimeError("Go-ICP 模块缺少 GoICP/POINT3D 接口")
    if not hasattr(goicp_mod, "ROTNODE") or not hasattr(goicp_mod, "TRANSNODE"):
        raise RuntimeError("Go-ICP 模块缺少 ROTNODE/TRANSNODE 接口")

    solver = goicp_mod.GoICP()
    point_cls = goicp_mod.POINT3D
    model = [point_cls(float(x), float(y), float(z)) for x, y, z in reference_down_np.tolist()]

    solver.setDTSizeAndFactor(int(config.goicp_dt_size), float(config.goicp_dt_factor))
    r = goicp_mod.ROTNODE()
    r.a = -3.1416
    r.b = -3.1416
    r.c = -3.1416
    r.w = 6.2832
    solver.setInitNodeRot(r)

    t = goicp_mod.TRANSNODE()
    t.x = -0.5
    t.y = -0.5
    t.z = -0.5
    t.w = 1.0
    solver.setInitNodeTrans(t)

    setattr(solver, "trimFraction", float(config.goicp_trim_fraction))
    if float(config.goicp_trim_fraction) < 0.001:
        setattr(solver, "doTrim", False)
    setattr(solver, "MSEThresh", float(config.goicp_mse_thresh))
    if config.goicp_epsilon is not None:
        epsilon_val = float(config.goicp_epsilon)
        if hasattr(solver, "setEpsilon"):
            solver.setEpsilon(epsilon_val)
        elif hasattr(solver, "epsilon"):
            setattr(solver, "epsilon", epsilon_val)

    return GoICPSolverContext(
        solver=solver,
        point_cls=point_cls,
        model=model,
        module_name=module_name,
        reference_hash=_hash_points(reference_down_np),
        config_signature=_config_signature(config),
        dt_built=False,
    )


def _prepare_solver_context(
    reference_down_np: np.ndarray,
    config: GoICPConfig,
    solver_ctx: Optional[GoICPSolverContext],
) -> GoICPSolverContext:
    ref_hash = _hash_points(reference_down_np)
    if (
        solver_ctx is None
        or solver_ctx.reference_hash != ref_hash
        or solver_ctx.module_name != (config.goicp_module or solver_ctx.module_name)
        or solver_ctx.config_signature != _config_signature(config)
    ):
        return _build_solver_context(reference_down_np=reference_down_np, config=config)
    return solver_ctx


def _run_goicp_registration(
    moving_down_np: np.ndarray,
    reference_down_np: np.ndarray,
    config: GoICPConfig,
    solver_ctx: Optional[GoICPSolverContext] = None,
) -> Tuple[np.ndarray, bool, GoICPSolverContext]:
    ctx = _prepare_solver_context(reference_down_np=reference_down_np, config=config, solver_ctx=solver_ctx)
    solver = ctx.solver
    point_cls = ctx.point_cls
    model = ctx.model
    data = [point_cls(float(x), float(y), float(z)) for x, y, z in moving_down_np.tolist()]
    solver.loadModelAndData(len(model), model, len(data), data)

    built_dt_this_call = False
    if not ctx.dt_built:
        solver.BuildDT()
        ctx.dt_built = True
        built_dt_this_call = True

    devnull_fd = None
    saved_stdout = None
    saved_stderr = None
    if bool(config.goicp_quiet):
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)

    try:
        solver.Register()
    finally:
        if bool(config.goicp_quiet):
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)
            os.close(devnull_fd)

    rotation = np.asarray(solver.optimalRotation(), dtype=np.float64)
    translation = np.asarray(solver.optimalTranslation(), dtype=np.float64).reshape(-1)
    if rotation.shape != (3, 3) or translation.shape[0] < 3:
        raise RuntimeError("Go-ICP 返回的 optimalRotation/optimalTranslation 形状异常")

    T_unit = np.eye(4, dtype=np.float64)
    T_unit[:3, :3] = rotation
    T_unit[:3, 3] = translation[:3]
    return T_unit, built_dt_this_call, ctx


def _orthonormalize_rotation(rot: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(rot)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1.0
        r = u @ vt
    return r


def transform_points(points: np.ndarray, transformation: np.ndarray) -> np.ndarray:
    transformed = points @ transformation[:3, :3].T + transformation[:3, 3][None, :]
    return transformed.astype(np.float64)


def register_point_clouds(
    moving_points: np.ndarray,
    reference_points: np.ndarray,
    config: Optional[GoICPConfig] = None,
    solver_ctx: Optional[GoICPSolverContext] = None,
) -> Tuple[GoICPResult, GoICPSolverContext]:
    config = config or GoICPConfig()
    validate_goicp_config(config)

    moving_points = np.asarray(moving_points, dtype=np.float64)
    reference_points = np.asarray(reference_points, dtype=np.float64)
    if moving_points.ndim != 2 or moving_points.shape[1] != 3:
        raise ValueError(f"moving_points 形状异常: {moving_points.shape}")
    if reference_points.ndim != 2 or reference_points.shape[1] != 3:
        raise ValueError(f"reference_points 形状异常: {reference_points.shape}")
    if moving_points.shape[0] < 4 or reference_points.shape[0] < 4:
        raise ValueError("moving/reference 点数都必须 >= 4")

    voxel_size, goicp_max_corr = _resolve_registration_scales(reference_points, config)
    moving_pcd = _to_pcd(moving_points)
    reference_pcd = _to_pcd(reference_points)
    moving_down = _preprocess_for_global_registration(moving_pcd, voxel_size)
    reference_down = reference_pcd

    moving_down_np = np.asarray(moving_down.points, dtype=np.float64)
    reference_down_np = np.asarray(reference_down.points, dtype=np.float64)
    if moving_down_np.shape[0] < 4 or reference_down_np.shape[0] < 4:
        raise RuntimeError("Go-ICP 配准失败：降采样后点数过少")

    moving_norm_np, reference_norm_np, moving_center, reference_center, norm_scale = _normalize_points_for_goicp(
        moving_points=moving_down_np,
        reference_points=reference_down_np,
    )

    t0 = time.perf_counter()
    T_unit, built_dt_this_call, solver_ctx = _run_goicp_registration(
        moving_down_np=moving_norm_np,
        reference_down_np=reference_norm_np,
        config=config,
        solver_ctx=solver_ctx,
    )
    elapsed_sec = float(time.perf_counter() - t0)

    transformation = np.eye(4, dtype=np.float64)
    transformation[:3, :3] = T_unit[:3, :3]
    transformation[:3, 3] = reference_center - T_unit[:3, :3] @ moving_center + norm_scale * T_unit[:3, 3]

    if config.rotation_only_output:
        transformation[:3, :3] = _orthonormalize_rotation(transformation[:3, :3])
        transformation[:3, 3] = 0.0

    evaluation = o3d.pipelines.registration.evaluate_registration(
        moving_down,
        reference_down,
        goicp_max_corr,
        transformation,
    )
    result = GoICPResult(
        transformation=transformation,
        fitness=float(evaluation.fitness),
        inlier_rmse=float(evaluation.inlier_rmse),
        elapsed_sec=elapsed_sec,
        used_voxel_size=float(voxel_size),
        used_goicp_max_corr=float(goicp_max_corr),
        built_dt_this_call=bool(built_dt_this_call),
    )
    return result, solver_ctx


def visualize_registration_result(
    reference_points: np.ndarray,
    moving_points: np.ndarray,
    transformation: np.ndarray,
    title_prefix: str = "goicp_registration",
    reference_mesh_verts: Optional[np.ndarray] = None,
    reference_mesh_faces: Optional[np.ndarray] = None,
    gap_scale: float = 1.6,
    point_size: float = 3.0,
    show_original: bool = True,
) -> None:
    moving_aligned = transform_points(np.asarray(moving_points, dtype=np.float64), transformation)
    reference_points = np.asarray(reference_points, dtype=np.float64)

    if show_original:
        all_pts = np.concatenate([reference_points, moving_points, moving_aligned], axis=0)
        mins = all_pts.min(axis=0)
        maxs = all_pts.max(axis=0)
        span = float(np.max(maxs - mins))
        if span <= 1e-8:
            span = 1.0
        step = span * gap_scale

        offset_left = np.array([-step, 0.0, 0.0], dtype=np.float64)
        offset_right = np.array([step, 0.0, 0.0], dtype=np.float64)
        geoms = [
            _build_colored_pcd(_translate_points(moving_points, offset_left), (0.9, 0.2, 0.2)),
            _build_colored_pcd(_translate_points(moving_aligned, offset_right), (0.2, 0.8, 0.2)),
        ]

        if reference_mesh_verts is not None and reference_mesh_faces is not None:
            base_mesh = _build_colored_mesh(reference_mesh_verts, reference_mesh_faces, (0.65, 0.65, 0.65))
            reference_left = copy.deepcopy(base_mesh)
            reference_right = copy.deepcopy(base_mesh)
            reference_left.translate(offset_left)
            reference_right.translate(offset_right)
            geoms.extend([reference_left, reference_right])
        else:
            geoms.extend(
                [
                    _build_colored_pcd(_translate_points(reference_points, offset_left), (0.65, 0.65, 0.65)),
                    _build_colored_pcd(_translate_points(reference_points, offset_right), (0.65, 0.65, 0.65)),
                ]
            )
        window_name = f"{title_prefix} | original(left) aligned(right)"
    else:
        geoms = [_build_colored_pcd(moving_aligned, (0.2, 0.8, 0.2))]
        if reference_mesh_verts is not None and reference_mesh_faces is not None:
            geoms.append(_build_colored_mesh(reference_mesh_verts, reference_mesh_faces, (0.65, 0.65, 0.65)))
        else:
            geoms.append(_build_colored_pcd(reference_points, (0.65, 0.65, 0.65)))
        window_name = f"{title_prefix} | aligned"

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    for geom in geoms:
        vis.add_geometry(geom)
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    render_option.point_size = float(point_size)
    vis.run()
    vis.destroy_window()
