#!/usr/bin/env python3
"""多帧 Go-ICP + Open3D ICP 配准测试：第一帧用 Go-ICP 获得 book 位姿，后续帧用 ICP 跟踪"""

import sys
import time
from pathlib import Path
import numpy as np
import open3d as o3d

REPO_ROOT = Path(__file__).resolve().parents[0]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def read_ply(path: Path) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points, dtype=np.float64)
    print(f"Loaded {path.name}: {len(points)} points")
    return points


def sample_asset_surface_points(mesh_path: Path, sample_points: int = 10000) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=True)
    if mesh.is_empty():
        raise ValueError(f"Mesh is empty: {mesh_path}")
    sampled = mesh.sample_points_uniformly(number_of_points=sample_points)
    points = np.asarray(sampled.points, dtype=np.float64)
    print(f"Sampled mesh {mesh_path.name}: {len(points)} points")
    return points


def run_open3d_icp(
    source_points: np.ndarray,
    target_points: np.ndarray,
    init_transformation: np.ndarray,
    max_correspondence_distance: float = 0.01,
) -> tuple:
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_points)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points)

    t0 = time.perf_counter()
    result = o3d.pipelines.registration.registration_icp(
        source=source_pcd,
        target=target_pcd,
        max_correspondence_distance=max_correspondence_distance,
        init=init_transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )
    elapsed = time.perf_counter() - t0
    return result, elapsed


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    return (points @ T[:3, :3].T + T[:3, 3][None, :]).astype(np.float64)


def invert_transform(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def compute_book_centroid(mesh_path: Path) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=True)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    return vertices.mean(axis=0)


def rotation_matrix_to_quaternion(R: np.ndarray) -> tuple:
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return w, x, y, z


def format_pose(R: np.ndarray, t: np.ndarray) -> str:
    qw, qx, qy, qz = rotation_matrix_to_quaternion(R)
    return (
        f"  position: [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]\n"
        f"  orientation (R):\n"
        f"    [{R[0,0]:.6f}, {R[0,1]:.6f}, {R[0,2]:.6f}]\n"
        f"    [{R[1,0]:.6f}, {R[1,1]:.6f}, {R[1,2]:.6f}]\n"
        f"    [{R[2,0]:.6f}, {R[2,1]:.6f}, {R[2,2]:.6f}]\n"
        f"  orientation (quat): [qw={qw:.6f}, qx={qx:.6f}, qy={qy:.6f}, qz={qz:.6f}]"
    )


def main():
    data_dir = REPO_ROOT / "tests/outputs/test_target"
    obj_path = REPO_ROOT / "assets/icp_assets/book.obj"

    ply_files = sorted(data_dir.glob("frame_*_target_only.ply"))
    if not ply_files:
        raise FileNotFoundError(f"No frame_*_target_only.ply found in {data_dir}")
    print(f"Found {len(ply_files)} frames\n")

    reference_points = sample_asset_surface_points(obj_path, sample_points=10000)
    book_centroid = compute_book_centroid(obj_path)
    print(f"Book centroid (book-local): [{book_centroid[0]:.6f}, {book_centroid[1]:.6f}, {book_centroid[2]:.6f}]")

    from icp.goicp import GoICPConfig, register_point_clouds

    config = GoICPConfig()
    first_frame_points = read_ply(ply_files[0])

    print("\n=== Frame 0: Go-ICP ===")
    t0 = time.perf_counter()
    goicp_result, _ = register_point_clouds(
        moving_points=first_frame_points,
        reference_points=reference_points,
        config=config,
    )
    goicp_elapsed = time.perf_counter() - t0

    T_cam_to_book = goicp_result.transformation.copy()
    T_book_in_world = invert_transform(T_cam_to_book)
    R_book = T_book_in_world[:3, :3]
    t_book = T_book_in_world[:3, 3]
    book_centroid_world = R_book @ book_centroid + t_book

    print(f"\nGo-ICP elapsed: {goicp_elapsed:.3f}s")
    print(f"Fitness: {goicp_result.fitness:.6f}")
    print(f"Inlier RMSE: {goicp_result.inlier_rmse:.6f}")
    print(f"\nBook pose in world (Go-ICP frame 0):")
    print(format_pose(R_book, t_book))
    print(f"  Book centroid in world: [{book_centroid_world[0]:.6f}, {book_centroid_world[1]:.6f}, {book_centroid_world[2]:.6f}]")

    frame_times = []

    print("\n=== Subsequent Frames: Open3D ICP (to obj reference) ===")
    for i, ply_file in enumerate(ply_files[1:], start=1):
        frame_points = read_ply(ply_file)

        icp_result, icp_elapsed = run_open3d_icp(
            source_points=frame_points,
            target_points=reference_points,
            init_transformation=T_cam_to_book,
            max_correspondence_distance=0.01,
        )

        T_cam_to_book = icp_result.transformation

        T_book_in_world = invert_transform(T_cam_to_book)
        R_book = T_book_in_world[:3, :3]
        t_book = T_book_in_world[:3, 3]
        book_centroid_world = R_book @ book_centroid + t_book

        frame_times.append({
            "frame": i,
            "filename": ply_file.name,
            "icp_elapsed_sec": icp_elapsed,
            "fitness": icp_result.fitness,
            "inlier_rmse": icp_result.inlier_rmse,
        })

        print(f"\nFrame {i} ({ply_file.name}): ICP {icp_elapsed:.4f}s, fitness={icp_result.fitness:.4f}, rmse={icp_result.inlier_rmse:.4f}")
        print(f"  Book pose in world:")
        print(format_pose(R_book, t_book))
        print(f"  Book centroid in world: [{book_centroid_world[0]:.6f}, {book_centroid_world[1]:.6f}, {book_centroid_world[2]:.6f}]")

    print("\n=== Summary ===")
    print(f"Go-ICP (frame 0): {goicp_elapsed:.3f}s")
    for ft in frame_times:
        print(f"  Frame {ft['frame']}: {ft['icp_elapsed_sec']:.4f}s, fitness={ft['fitness']:.4f}")
    total_icp = sum(ft['icp_elapsed_sec'] for ft in frame_times)
    print(f"Total Open3D ICP: {total_icp:.4f}s (avg: {total_icp/len(frame_times):.4f}s per frame)")


if __name__ == "__main__":
    main()
