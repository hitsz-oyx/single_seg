#!/usr/bin/env python3
"""Convert HDF5 robot data to live_rgbd_debug format.

Handles demo0_X_ds3.hdf5 files from the Demo directory.

Example:
    python utils/convert_hdf5_to_live_debug.py \\
        --hdf5-path /home/franka-client/oyx_ws/Demo/demo0_1_ds3.hdf5 \\
        --realsense-para-dir /home/franka-client/oyx_ws/realsense_para \\
        --output-dir tests/outputs/demo0_1_ds3_live \\
        --frame-start 0 --frame-end 10
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


IMAGE_INDEX_TO_CAM_ID = {
    "1": "cam_00",
    "3": "cam_01",
    "2": "cam_02",
}

CAM_ID_TO_REALSENSE_PARA_CAM = {
    "cam_00": "cam1",
    "cam_01": "cam3",
    "cam_02": "cam2",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HDF5 robot data to live_rgbd_debug format")
    parser.add_argument("--hdf5-path", type=Path, required=True,
                        help="Path to demo0_X_ds3.hdf5 file")
    parser.add_argument("--realsense-para-dir", type=Path, required=True,
                        help="Path to realsense_para directory (contains cam1/cam2/cam3 subdirs)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for live_rgbd_debug format")
    parser.add_argument("--frame-start", type=int, default=0,
                        help="Start frame index (default: 0)")
    parser.add_argument("--frame-end", type=int, default=None,
                        help="End frame index (exclusive, default: all frames)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite output directory if it exists")
    args = parser.parse_args()

    hdf5_path = args.hdf5_path.expanduser().resolve()
    para_dir = args.realsense_para_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"{output_dir} already exists. Use --overwrite to replace.")

    with h5py.File(hdf5_path, "r") as f:
        num_frames = f["timestamps"].shape[0]
        start = args.frame_start
        end = args.frame_end if args.frame_end is not None else num_frames
        end = min(end, num_frames)
        print(f"HDF5: {hdf5_path.name}, total frames: {num_frames}, converting [{start}, {end})")

        live_debug_dir = output_dir / "live_rgbd_debug"

        for frame_idx in range(start, end):
            frame_name = f"frame_{frame_idx:05d}"
            frame_dir = live_debug_dir / frame_name
            frame_dir.mkdir(parents=True, exist_ok=True)

            for img_idx, cam_id in IMAGE_INDEX_TO_CAM_ID.items():
                cam_output_dir = frame_dir / cam_id
                cam_output_dir.mkdir(parents=True, exist_ok=True)

                para_cam = CAM_ID_TO_REALSENSE_PARA_CAM[cam_id]
                para_payload = load_json(para_dir / para_cam / "camera_payload.json")

                rgb_data = f[f"observations/rgb/color_image{img_idx}"][frame_idx]
                Image.fromarray(rgb_data).save(cam_output_dir / "rgb.png")

                depth_data = f[f"observations/depth/depth_image{img_idx}"][frame_idx]
                depth_m = depth_data.astype(np.float32) / 1000.0
                np.save(cam_output_dir / "depth_aligned_m.npy", depth_m)

                ir_left_data = f[f"observations/infrared/infrared_image{img_idx}_1"][frame_idx]
                ir_right_data = f[f"observations/infrared/infrared_image{img_idx}_2"][frame_idx]
                Image.fromarray(ir_left_data).save(cam_output_dir / "ir_left_rect.png")
                Image.fromarray(ir_right_data).save(cam_output_dir / "ir_right_rect.png")

                camera_payload: dict = {
                    "camera_id": cam_id,
                    "depth_source": "fast",
                    "rgb_file": "rgb.png",
                    "depth_aligned_file": "depth_aligned_m.npy",
                    "color_intrinsics": dict(para_payload["color_intrinsics"]),
                    "depth_min": 0.1,
                    "depth_max": 3.0,
                    "stereo_rectification_mode": "opencv",
                    "fast_align_backend": "torch",
                    "ir_left_rect_file": "ir_left_rect.png",
                    "ir_right_rect_file": "ir_right_rect.png",
                    "rectified_k": para_payload["rectified_k"],
                    "rectified_to_color": para_payload["rectified_to_color"],
                    "baseline_m": para_payload["baseline_m"],
                    "left_ir_intrinsics": para_payload["left_ir_intrinsics"],
                    "right_ir_intrinsics": para_payload["right_ir_intrinsics"],
                    "left_to_right_4x4": para_payload["left_to_right_4x4"],
                }

                (cam_output_dir / "camera_payload.json").write_text(
                    json.dumps(camera_payload, indent=2),
                    encoding="utf-8",
                )
                (cam_output_dir / "depth_source.txt").write_text("fast", encoding="utf-8")

            print(f"  {frame_name}: {rgb_data.shape} RGB, depth range=[{depth_m.min():.3f}, {depth_m.max():.3f}]")

        timestamps = f["timestamps"][start:end]
        np.save(live_debug_dir / "timestamps.npy", timestamps)

    print()
    print(f"Done. Output written to {live_debug_dir}")
    print(f"Converted {end - start} frames.")


if __name__ == "__main__":
    main()
