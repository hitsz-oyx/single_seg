#!/usr/bin/env python3
"""Visualize target point clouds against the registered STL mesh.

The registration script estimates T_MW as world->mesh. For visualization in the
world frame, this script places the mesh with inv(T_MW), then optionally
reprojects each camera's target cloud with refined cam2world matrices.
"""

from __future__ import annotations

import argparse
import copy
import json
import struct
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from icp.register_to_mesh import read_ply, transform_world_to_camera  # noqa: E402

try:
    from icp.goicp import GoICPConfig, register_point_clouds as goicp_register  # noqa: E402
    _HAS_GOICP = True
except (ImportError, RuntimeError):
    GoICPConfig = None
    goicp_register = None
    _HAS_GOICP = False


CAMERA_COLORS_U8 = {
    "cam_00": np.array([40, 130, 255], dtype=np.uint8),
    "cam_01": np.array([255, 128, 40], dtype=np.uint8),
    "cam_02": np.array([40, 210, 90], dtype=np.uint8),
}
DEFAULT_POINT_COLOR_U8 = np.array([230, 230, 230], dtype=np.uint8)
MESH_COLOR_U8 = np.array([180, 180, 180], dtype=np.uint8)


def as_matrix(value: object) -> np.ndarray | None:
    try:
        mat = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if mat.shape != (4, 4):
        return None
    return mat


def matrix_from_entry(entry: object, prefer: str) -> np.ndarray | None:
    direct = as_matrix(entry)
    if direct is not None:
        return direct

    if not isinstance(entry, dict):
        return None

    if prefer == "original":
        keys = (
            "original_cam2world_4x4",
            "cam2world_4x4",
            "single_seg_gl_cam2world_4x4",
            "refined_cam2world_4x4",
            "new_cam2world",
        )
    else:
        keys = (
            "refined_cam2world_4x4",
            "new_cam2world",
            "cam2world_4x4",
            "single_seg_gl_cam2world_4x4",
            "original_cam2world_4x4",
        )

    for key in keys:
        if key in entry:
            mat = as_matrix(entry[key])
            if mat is not None:
                return mat

    pose_record = entry.get("pose_record")
    if isinstance(pose_record, dict):
        mat = as_matrix(pose_record.get("cam2world_4x4"))
        if mat is not None:
            return mat

    return None


def load_cam2worlds(path: Path, prefer: str = "refined") -> dict[str, np.ndarray]:
    with path.open() as f:
        data = json.load(f)

    result: dict[str, np.ndarray] = {}

    if isinstance(data, dict) and isinstance(data.get("cameras"), list):
        for item in data["cameras"]:
            if not isinstance(item, dict):
                continue
            cam_id = item.get("camera_id") or item.get("id")
            mat = matrix_from_entry(item, prefer)
            if cam_id and mat is not None:
                result[str(cam_id)] = mat

    if isinstance(data, dict) and isinstance(data.get("extrinsics"), dict):
        tmp = extract_top_level_cam2worlds(data["extrinsics"], prefer)
        if tmp:
            result.update(tmp)
        if result:
            return result
        raise ValueError(f"Could not parse extrinsics from {path}")

    if isinstance(data, dict):
        result.update(extract_top_level_cam2worlds(data, prefer))

    if not result and isinstance(data, dict):
        mat = matrix_from_entry(data, prefer)
        if mat is not None:
            result["cam_00"] = mat

    if not result:
        raise ValueError(f"Could not find cam2world matrices in {path}")
    return result


def load_world_to_mesh(path: Path) -> np.ndarray:
    with path.open() as f:
        data = json.load(f)

    candidates: list[object] = [data]
    if isinstance(data, dict):
        candidates.extend(
            data.get(key)
            for key in (
                "world_to_mesh_4x4",
                "T_MW",
                "t_mw",
                "world_to_mesh",
            )
        )

    for item in candidates:
        mat = as_matrix(item)
        if mat is not None:
            return mat
    raise ValueError(f"Could not find a 4x4 world-to-mesh matrix in {path}")


def extract_top_level_cam2worlds(data: dict, prefer: str) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for key, value in data.items():
        if not str(key).startswith("cam_"):
            continue
        mat = matrix_from_entry(value, prefer)
        if mat is not None:
            result[str(key)] = mat
    return result


def resolve_live_debug(data_dir: Path) -> Path:
    data_dir = data_dir.expanduser().resolve()
    if (data_dir / "live_rgbd_debug").is_dir():
        return data_dir / "live_rgbd_debug"
    if data_dir.name == "live_rgbd_debug" and data_dir.is_dir():
        return data_dir
    raise FileNotFoundError(f"{data_dir} is not an output root or live_rgbd_debug directory")


def read_point_cloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        return read_ply(path)
    except Exception:
        pcd = o3d.io.read_point_cloud(str(path))
        if pcd.is_empty():
            raise RuntimeError(f"Could not read point cloud: {path}")
        pts = np.asarray(pcd.points, dtype=np.float64)
        if pcd.has_colors():
            cols = np.clip(np.round(np.asarray(pcd.colors) * 255.0), 0, 255).astype(np.uint8)
        else:
            cols = np.tile(DEFAULT_POINT_COLOR_U8[None, :], (pts.shape[0], 1))
        return pts, cols


def load_auto_clouds(
    data_dir: Path,
    camera_ids: list[str],
    max_frames: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    live_debug = resolve_live_debug(data_dir)
    frame_dirs = sorted(p for p in live_debug.glob("frame_*") if p.is_dir())
    if max_frames > 0:
        frame_dirs = frame_dirs[:max_frames]
    if not frame_dirs:
        raise FileNotFoundError(f"No frame_* directories under {live_debug}")

    points: dict[str, list[np.ndarray]] = {cam_id: [] for cam_id in camera_ids}
    colors: dict[str, list[np.ndarray]] = {cam_id: [] for cam_id in camera_ids}

    for frame_dir in frame_dirs:
        for cam_id in camera_ids:
            ply_path = frame_dir / cam_id / "target_object_rgb.ply"
            if not ply_path.is_file():
                continue
            pts, cols = read_point_cloud(ply_path)
            if pts.shape[0] == 0:
                continue
            points[cam_id].append(pts)
            colors[cam_id].append(cols)

    world_clouds: dict[str, np.ndarray] = {}
    rgb_clouds: dict[str, np.ndarray] = {}
    for cam_id in camera_ids:
        if points[cam_id]:
            world_clouds[cam_id] = np.concatenate(points[cam_id], axis=0)
            rgb_clouds[cam_id] = np.concatenate(colors[cam_id], axis=0)
        else:
            world_clouds[cam_id] = np.empty((0, 3), dtype=np.float64)
            rgb_clouds[cam_id] = np.empty((0, 3), dtype=np.uint8)

    original_ext = load_payload_extrinsics(live_debug, camera_ids)
    return world_clouds, rgb_clouds, original_ext


def load_payload_extrinsics(live_debug: Path, camera_ids: list[str]) -> dict[str, np.ndarray]:
    for frame_dir in sorted(p for p in live_debug.glob("frame_*") if p.is_dir()):
        extrinsics: dict[str, np.ndarray] = {}
        for cam_id in camera_ids:
            payload_path = frame_dir / cam_id / "camera_payload.json"
            if not payload_path.is_file():
                continue
            with payload_path.open() as f:
                payload = json.load(f)
            mat = as_matrix(payload.get("pose_record", {}).get("cam2world_4x4"))
            if mat is not None:
                extrinsics[cam_id] = mat
        if len(extrinsics) == len(camera_ids):
            return extrinsics
    raise FileNotFoundError(f"Could not find camera_payload.json for all cameras under {live_debug}")


def parse_point_cloud_specs(specs: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Point cloud spec must be cam_id=/path/to/cloud.ply: {spec}")
        cam_id, path_str = spec.split("=", 1)
        result[cam_id.strip()] = Path(path_str).expanduser().resolve()
    return result


def load_explicit_clouds(
    specs: list[str],
    camera_ids: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    paths = parse_point_cloud_specs(specs)
    missing = [cam_id for cam_id in camera_ids if cam_id not in paths]
    if missing:
        raise ValueError(f"Missing --point-cloud entries for: {', '.join(missing)}")

    world_clouds: dict[str, np.ndarray] = {}
    rgb_clouds: dict[str, np.ndarray] = {}
    for cam_id in camera_ids:
        pts, cols = read_point_cloud(paths[cam_id])
        world_clouds[cam_id] = pts
        rgb_clouds[cam_id] = cols
    return world_clouds, rgb_clouds


def apply_cam2world(points_camera: np.ndarray, cam2world: np.ndarray) -> np.ndarray:
    return points_camera @ cam2world[:3, :3].T + cam2world[:3, 3][None, :]


def reproject_with_refined_extrinsics(
    world_clouds: dict[str, np.ndarray],
    original_ext: dict[str, np.ndarray],
    refined_ext: dict[str, np.ndarray],
    camera_ids: list[str],
) -> dict[str, np.ndarray]:
    missing_original = [cam_id for cam_id in camera_ids if cam_id not in original_ext]
    missing_refined = [cam_id for cam_id in camera_ids if cam_id not in refined_ext]
    if missing_original:
        raise ValueError(f"Missing original extrinsics for: {', '.join(missing_original)}")
    if missing_refined:
        raise ValueError(f"Missing refined extrinsics for: {', '.join(missing_refined)}")

    result: dict[str, np.ndarray] = {}
    for cam_id in camera_ids:
        pts = world_clouds[cam_id]
        if pts.shape[0] == 0:
            result[cam_id] = pts.copy()
            continue
        pts_camera = transform_world_to_camera(pts, original_ext[cam_id])
        result[cam_id] = apply_cam2world(pts_camera, refined_ext[cam_id])
    return result


def load_mesh(mesh_path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise RuntimeError(f"Could not load mesh: {mesh_path}")
    mesh.compute_vertex_normals()
    return mesh


def estimate_world_to_mesh(
    master_world_pts: np.ndarray,
    mesh_pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    max_correspondence_distance: float,
    backend: str,
) -> tuple[np.ndarray, float, float, int]:
    if master_world_pts.shape[0] == 0:
        raise ValueError("Master camera point cloud is empty")

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(master_world_pts)
    if voxel_size > 0:
        source = source.voxel_down_sample(voxel_size)

    source_pts = np.asarray(source.points, dtype=np.float64)
    target_pts = np.asarray(mesh_pcd.points, dtype=np.float64)

    if backend == "goicp":
        if not _HAS_GOICP:
            raise RuntimeError("Go-ICP backend is not available in this Python environment")
        goicp_cfg = GoICPConfig(
            voxel_size_ratio=0.01,
            goicp_max_corr_ratio=max_correspondence_distance
            / max(np.ptp(target_pts[:, 0]), np.ptp(target_pts[:, 1]), np.ptp(target_pts[:, 2]), 1e-6),
            min_voxel_size=0.001,
            min_goicp_corr=0.003,
            goicp_trim_fraction=0.0,
            goicp_mse_thresh=1e-3,
        )
        result, _ = goicp_register(
            moving_points=source_pts,
            reference_points=target_pts,
            config=goicp_cfg,
        )
        return result.transformation, float(result.fitness), float(result.inlier_rmse), len(source_pts)

    t_init = np.eye(4, dtype=np.float64)
    t_init[:3, 3] = np.mean(target_pts, axis=0) - np.mean(source_pts, axis=0)

    result = o3d.pipelines.registration.registration_icp(
        source=source,
        target=mesh_pcd,
        max_correspondence_distance=max_correspondence_distance,
        init=t_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=300),
    )
    return result.transformation, float(result.fitness), float(result.inlier_rmse), len(source_pts)


def make_pcd(
    points: np.ndarray,
    rgb: np.ndarray,
    cam_id: str,
    color_mode: str,
    voxel_size: float,
) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    if points.shape[0] == 0:
        return pcd

    pcd.points = o3d.utility.Vector3dVector(points)
    if color_mode == "rgb" and rgb.shape[0] == points.shape[0]:
        colors = rgb.astype(np.float64) / 255.0
    else:
        color = CAMERA_COLORS_U8.get(cam_id, DEFAULT_POINT_COLOR_U8).astype(np.float64) / 255.0
        colors = np.tile(color[None, :], (points.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    return pcd


def distance_report(
    mesh_world: o3d.geometry.TriangleMesh,
    pcd_by_cam: dict[str, o3d.geometry.PointCloud],
    sample_count: int,
) -> None:
    mesh_eval = mesh_world.sample_points_uniformly(number_of_points=sample_count)
    print("Distance to mesh sample (meters):")
    for cam_id, pcd in pcd_by_cam.items():
        if pcd.is_empty():
            print(f"  {cam_id}: empty")
            continue
        distances = np.asarray(pcd.compute_point_cloud_distance(mesh_eval), dtype=np.float64)
        if distances.size == 0:
            print(f"  {cam_id}: empty")
            continue
        print(
            f"  {cam_id}: n={distances.size} "
            f"mean={distances.mean():.5f} median={np.median(distances):.5f} "
            f"p90={np.percentile(distances, 90):.5f}"
        )


def make_point_splats(
    points: np.ndarray,
    colors: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points.shape[0] == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            np.empty((0, 3), dtype=np.int32),
        )

    dirs = np.array(
        [
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
        ],
        dtype=np.float32,
    )
    dirs /= np.float32(np.sqrt(3.0))

    vertices = points.astype(np.float32)[:, None, :] + dirs[None, :, :] * np.float32(radius)
    vertices = vertices.reshape(-1, 3)
    vertex_colors = np.repeat(colors.astype(np.uint8), 4, axis=0)

    local_faces = np.array(
        [
            [0, 1, 2],
            [0, 3, 1],
            [0, 2, 3],
            [1, 3, 2],
        ],
        dtype=np.int32,
    )
    base = (np.arange(points.shape[0], dtype=np.int32) * 4)[:, None, None]
    faces = (base + local_faces[None, :, :]).reshape(-1, 3)
    return vertices, vertex_colors, faces


def write_scene_ply(
    path: Path,
    mesh_world: o3d.geometry.TriangleMesh,
    pcd_by_cam: dict[str, o3d.geometry.PointCloud],
    point_render_mode: str = "splat",
    point_splat_size: float = 0.003,
) -> None:
    mesh_vertices = np.asarray(mesh_world.vertices, dtype=np.float32)
    mesh_triangles = np.asarray(mesh_world.triangles, dtype=np.int32)
    if mesh_world.has_vertex_colors():
        mesh_colors = np.clip(
            np.round(np.asarray(mesh_world.vertex_colors) * 255.0),
            0,
            255,
        ).astype(np.uint8)
    else:
        mesh_colors = np.tile(MESH_COLOR_U8[None, :], (mesh_vertices.shape[0], 1))

    point_vertices: list[np.ndarray] = []
    point_colors: list[np.ndarray] = []
    point_faces: list[np.ndarray] = []
    vertex_offset = int(mesh_vertices.shape[0])
    for pcd in pcd_by_cam.values():
        if pcd.is_empty():
            continue
        pts = np.asarray(pcd.points, dtype=np.float32)
        cols = np.clip(np.round(np.asarray(pcd.colors) * 255.0), 0, 255).astype(np.uint8)
        if point_render_mode == "splat":
            splat_vertices, splat_colors, splat_faces = make_point_splats(
                pts,
                cols,
                radius=float(point_splat_size),
            )
            point_vertices.append(splat_vertices)
            point_colors.append(splat_colors)
            point_faces.append(splat_faces + vertex_offset)
            vertex_offset += int(splat_vertices.shape[0])
        elif point_render_mode == "vertex":
            point_vertices.append(pts)
            point_colors.append(cols)
            vertex_offset += int(pts.shape[0])
        else:
            raise ValueError(f"Unsupported point render mode: {point_render_mode}")

    if point_vertices:
        vertices = np.vstack([mesh_vertices, *point_vertices]).astype(np.float32)
        colors = np.vstack([mesh_colors, *point_colors]).astype(np.uint8)
    else:
        vertices = mesh_vertices
        colors = mesh_colors
    if point_faces:
        faces = np.vstack([mesh_triangles, *point_faces]).astype(np.int32)
    else:
        faces = mesh_triangles

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        if point_render_mode == "splat":
            f.write(b"comment mesh vertices are first; point clouds are small tetrahedron faces\n")
        else:
            f.write(b"comment mesh vertices are first; point clouds are unreferenced vertices\n")
        f.write(f"element vertex {vertices.shape[0]}\n".encode("ascii"))
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        f.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {faces.shape[0]}\n".encode("ascii"))
        f.write(b"property list uchar int vertex_indices\n")
        f.write(b"end_header\n")

        vertex_dtype = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        )
        vertex_buf = np.empty(vertices.shape[0], dtype=vertex_dtype)
        vertex_buf["x"] = vertices[:, 0]
        vertex_buf["y"] = vertices[:, 1]
        vertex_buf["z"] = vertices[:, 2]
        vertex_buf["red"] = colors[:, 0]
        vertex_buf["green"] = colors[:, 1]
        vertex_buf["blue"] = colors[:, 2]
        f.write(vertex_buf.tobytes())

        for tri in faces:
            f.write(struct.pack("<Biii", 3, int(tri[0]), int(tri[1]), int(tri[2])))


def write_per_camera_scene_plys(
    output_dir: Path,
    mesh_world: o3d.geometry.TriangleMesh,
    pcd_by_cam: dict[str, o3d.geometry.PointCloud],
    point_render_mode: str,
    point_splat_size: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for cam_id, pcd in pcd_by_cam.items():
        write_scene_ply(
            output_dir / f"{cam_id}_mesh_and_registered_points.ply",
            mesh_world,
            {cam_id: pcd},
            point_render_mode=point_render_mode,
            point_splat_size=point_splat_size,
        )


def find_single_mesh(output_dir: Path) -> Path:
    candidates: list[Path] = []
    for suffix in ("*.stl", "*.STL", "*.obj", "*.OBJ", "*.ply", "*.PLY"):
        candidates.extend(output_dir.glob(suffix))
    candidates = [
        path
        for path in candidates
        if not path.name.endswith("_registered.ply")
        and path.name not in {"original_extrinsics.json", "refined_extrinsics.json"}
    ]
    if not candidates:
        raise FileNotFoundError(f"No mesh file found in {output_dir}")
    candidates.sort(key=lambda p: (p.suffix.lower() != ".stl", p.name))
    return candidates[0]


def has_registered_clouds(output_dir: Path, camera_ids: list[str]) -> bool:
    return any((output_dir / f"{cam_id}_registered.ply").is_file() for cam_id in camera_ids)


def resolve_output_scene_dir(output_dir: Path, camera_ids: list[str]) -> Path:
    """Accept either a concrete result dir or the icp/output root."""
    output_dir = output_dir.expanduser().resolve()
    if has_registered_clouds(output_dir, camera_ids):
        return output_dir

    child_dirs = [path for path in output_dir.iterdir() if path.is_dir()]
    candidates = [
        path
        for path in child_dirs
        if has_registered_clouds(path, camera_ids) and (path / "refined_extrinsics.json").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No register_to_mesh.py result directory found under {output_dir}"
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_output_dir_scene(
    output_dir: Path,
    camera_ids: list[str],
    voxel_size: float,
) -> tuple[o3d.geometry.TriangleMesh, dict[str, o3d.geometry.PointCloud], Path, Path]:
    output_dir = resolve_output_scene_dir(output_dir, camera_ids)
    mesh_path = find_single_mesh(output_dir)
    mesh = load_mesh(mesh_path)

    refined_json = output_dir / "refined_extrinsics.json"
    if refined_json.is_file():
        try:
            t_mw = load_world_to_mesh(refined_json)
            mesh.transform(np.linalg.inv(t_mw))
        except ValueError:
            pass
    mesh.paint_uniform_color((MESH_COLOR_U8.astype(np.float64) / 255.0).tolist())

    pcd_by_cam: dict[str, o3d.geometry.PointCloud] = {}
    for cam_id in camera_ids:
        ply_path = output_dir / f"{cam_id}_registered.ply"
        if not ply_path.is_file():
            continue
        pts, cols = read_point_cloud(ply_path)
        pcd_by_cam[cam_id] = make_pcd(
            points=pts,
            rgb=cols,
            cam_id=cam_id,
            color_mode="rgb",
            voxel_size=voxel_size,
        )
    if not pcd_by_cam:
        raise FileNotFoundError(f"No cam_*_registered.ply files found in {output_dir}")
    return mesh, pcd_by_cam, mesh_path, output_dir


def resolve_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show or export a mesh + registered target point-cloud scene.",
    )
    parser.add_argument("--mesh", type=Path, default=Path("icp/Register.STL"))
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Read register_to_mesh.py output directory directly")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--point-cloud", type=str, nargs="+", default=None)
    parser.add_argument("--original-extrinsics", type=Path, default=None)
    parser.add_argument("--refined-extrinsics", type=Path, default=None)
    parser.add_argument("--world-to-mesh-json", type=Path, default=None)
    parser.add_argument("--master-camera", type=str, default="cam_00")
    parser.add_argument("--camera-ids", type=str, nargs="+", default=["cam_00", "cam_01", "cam_02"])
    parser.add_argument("--view", choices=["original", "refined"], default=None)
    parser.add_argument("--color-mode", choices=["camera", "rgb"], default="camera")
    parser.add_argument("--voxel-size", type=float, default=0.003)
    parser.add_argument("--master-icp-dist", type=float, default=0.05)
    parser.add_argument("--mesh-pose-backend", choices=["open3d", "goicp"], default="open3d")
    parser.add_argument("--num-mesh-points", type=int, default=100000)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--export-ply", type=Path, default=None)
    parser.add_argument("--export-per-camera-dir", type=Path, default=None)
    parser.add_argument("--point-render-mode", choices=["splat", "vertex"], default="splat")
    parser.add_argument("--point-splat-size", type=float, default=0.003)
    parser.add_argument("--draw", type=int, default=1)
    parser.add_argument("--distance-report", type=int, default=1)
    parser.add_argument("--show-frame", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mesh_path = resolve_path(args.mesh)
    original_ext_path = resolve_path(args.original_extrinsics)
    refined_ext_path = resolve_path(args.refined_extrinsics)
    world_to_mesh_path = resolve_path(args.world_to_mesh_json)
    export_ply_path = resolve_path(args.export_ply)
    export_per_camera_dir = resolve_path(args.export_per_camera_dir)
    camera_ids = list(args.camera_ids)

    if args.master_camera not in camera_ids:
        raise ValueError(f"Master camera {args.master_camera} is not in camera_ids: {camera_ids}")

    output_dir = resolve_path(args.output_dir)
    if output_dir is not None:
        mesh_world, pcd_by_cam, output_mesh_path, scene_dir = load_output_dir_scene(
            output_dir=output_dir,
            camera_ids=camera_ids,
            voxel_size=args.voxel_size,
        )
        print(f"View: output-dir")
        print(f"Output dir: {scene_dir}")
        print(f"Mesh: {output_mesh_path}")
        for cam_id, pcd in pcd_by_cam.items():
            print(f"  {cam_id}: {len(pcd.points)} shown")
        if args.distance_report:
            distance_report(mesh_world, pcd_by_cam, args.num_mesh_points)
        if export_ply_path is not None:
            write_scene_ply(
                export_ply_path,
                mesh_world,
                pcd_by_cam,
                point_render_mode=args.point_render_mode,
                point_splat_size=args.point_splat_size,
            )
            print(f"Wrote: {export_ply_path}")
        if export_per_camera_dir is not None:
            write_per_camera_scene_plys(
                export_per_camera_dir,
                mesh_world,
                pcd_by_cam,
                point_render_mode=args.point_render_mode,
                point_splat_size=args.point_splat_size,
            )
            print(f"Wrote per-camera PLYs under: {export_per_camera_dir}")
        if args.draw:
            geoms: list[o3d.geometry.Geometry] = [mesh_world]
            geoms.extend(pcd_by_cam.values())
            if args.show_frame:
                extent = np.ptp(np.asarray(mesh_world.vertices), axis=0)
                frame_size = max(float(np.max(extent)) * 0.8, 0.03)
                geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size))
            o3d.visualization.draw_geometries(geoms, window_name=f"mesh registration: {output_dir.name}")
        return

    auto_mode = args.data_dir is not None
    explicit_mode = args.point_cloud is not None
    if auto_mode == explicit_mode:
        raise ValueError("Use exactly one input mode: --data-dir or --point-cloud")

    if auto_mode:
        world_clouds, rgb_clouds, original_ext = load_auto_clouds(
            data_dir=resolve_path(args.data_dir),
            camera_ids=camera_ids,
            max_frames=args.max_frames,
        )
    else:
        world_clouds, rgb_clouds = load_explicit_clouds(args.point_cloud, camera_ids)
        original_ext = {}

    if original_ext_path is not None:
        original_ext = load_cam2worlds(original_ext_path, prefer="original")

    refined_ext = None
    if refined_ext_path is not None:
        refined_ext = load_cam2worlds(refined_ext_path, prefer="refined")

    view = args.view or ("refined" if refined_ext is not None else "original")
    if view == "refined":
        if refined_ext is None:
            raise ValueError("--view refined requires --refined-extrinsics")
        view_clouds = reproject_with_refined_extrinsics(world_clouds, original_ext, refined_ext, camera_ids)
    else:
        view_clouds = {cam_id: world_clouds[cam_id].copy() for cam_id in camera_ids}

    mesh = load_mesh(mesh_path)
    mesh_pcd = mesh.sample_points_uniformly(number_of_points=args.num_mesh_points)
    if world_to_mesh_path is not None:
        t_mw = load_world_to_mesh(world_to_mesh_path)
        fitness = float("nan")
        rmse = float("nan")
        master_down_count = int(world_clouds[args.master_camera].shape[0])
        mesh_pose_label = f"provided {world_to_mesh_path.name}"
    else:
        t_mw, fitness, rmse, master_down_count = estimate_world_to_mesh(
            master_world_pts=world_clouds[args.master_camera],
            mesh_pcd=mesh_pcd,
            voxel_size=args.voxel_size,
            max_correspondence_distance=args.master_icp_dist,
            backend=args.mesh_pose_backend,
        )
        mesh_pose_label = args.mesh_pose_backend

    mesh_world = copy.deepcopy(mesh)
    mesh_world.transform(np.linalg.inv(t_mw))
    mesh_world.paint_uniform_color((MESH_COLOR_U8.astype(np.float64) / 255.0).tolist())

    pcd_by_cam = {
        cam_id: make_pcd(
            points=view_clouds[cam_id],
            rgb=rgb_clouds[cam_id],
            cam_id=cam_id,
            color_mode=args.color_mode,
            voxel_size=args.voxel_size,
        )
        for cam_id in camera_ids
    }

    print(f"View: {view}")
    print(f"Mesh: {mesh_path}")
    print(f"Master {mesh_pose_label}: n={master_down_count} fitness={fitness:.4f} rmse={rmse:.5f}")
    for cam_id in camera_ids:
        print(f"  {cam_id}: {world_clouds[cam_id].shape[0]} raw points -> {len(pcd_by_cam[cam_id].points)} shown")

    if args.distance_report:
        distance_report(mesh_world, pcd_by_cam, args.num_mesh_points)

    if export_ply_path is not None:
        write_scene_ply(
            export_ply_path,
            mesh_world,
            pcd_by_cam,
            point_render_mode=args.point_render_mode,
            point_splat_size=args.point_splat_size,
        )
        print(f"Wrote: {export_ply_path}")

    if export_per_camera_dir is not None:
        write_per_camera_scene_plys(
            export_per_camera_dir,
            mesh_world,
            pcd_by_cam,
            point_render_mode=args.point_render_mode,
            point_splat_size=args.point_splat_size,
        )
        print(f"Wrote per-camera PLYs under: {export_per_camera_dir}")

    if args.draw:
        geoms: list[o3d.geometry.Geometry] = [mesh_world]
        geoms.extend(pcd_by_cam.values())
        if args.show_frame:
            extent = np.ptp(np.asarray(mesh_world.vertices), axis=0)
            frame_size = max(float(np.max(extent)) * 0.8, 0.03)
            geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size))
        o3d.visualization.draw_geometries(geoms, window_name=f"mesh registration: {view}")


if __name__ == "__main__":
    main()
