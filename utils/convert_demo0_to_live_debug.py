#!/usr/bin/env python3
"""Convert demo0_frame_360 HDF5/npy data to live_rgbd_debug format.

This produces a directory that can be consumed by offline_replay.py.

Example:
    # Step 1: convert
    python utils/convert_demo0_to_live_debug.py \\
        --demo0-dir /home/franka-client/oyx_ws/demo0_frame_360 \\
        --realsense-para-dir /home/franka-client/oyx_ws/realsense_para \\
        --output-dir tests/outputs/demo0_frame_360_live

    # Step 2: Fast-Stereo + SAM3 (generic, works on any live_rgbd_debug input)
    python utils/offline_replay.py --config configs/offline_replay.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_INDEX_TO_CAM_ID = {
    "1": "cam_00",
    "3": "cam_01",
    "2": "cam_02",
}

CAM_ID_TO_SERIAL = {
    "cam_00": "243122075507",
    "cam_01": "148522074762",
    "cam_02": "845112070307",
}

CAM_ID_TO_REALSENSE_PARA_CAM = {
    "cam_00": "cam1",
    "cam_01": "cam3",
    "cam_02": "cam2",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert demo0_frame_360 to live_rgbd_debug format")
    parser.add_argument("--demo0-dir", type=Path, required=True,
                        help="Path to demo0_frame_360 directory")
    parser.add_argument("--realsense-para-dir", type=Path, required=True,
                        help="Path to realsense_para directory (contains cam1/cam2/cam3 subdirs)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for live_rgbd_debug format")
    parser.add_argument("--frame-name", type=str, default="frame_00000",
                        help="Frame directory name in output (default: frame_00000)")
    parser.add_argument("--depth-source", choices=("native", "fast"), default="fast",
                        help="Depth source: native (use HDF5 depth) or fast (use IR+Fast-Stereo)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite output directory if it exists")
    args = parser.parse_args()

    demo0_dir = args.demo0_dir.expanduser().resolve()
    para_dir = args.realsense_para_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"{output_dir} already exists. Use --overwrite to replace.")

    observations_dir = demo0_dir / "observations"

    frame_output_dir = output_dir / "live_rgbd_debug" / args.frame_name
    frame_output_dir.mkdir(parents=True, exist_ok=True)

    camera_ids = sorted(IMAGE_INDEX_TO_CAM_ID.values())
    print(f"Converting {len(camera_ids)} cameras from {demo0_dir}")
    print(f"Output: {frame_output_dir}")

    for cam_id in camera_ids:
        serial = CAM_ID_TO_SERIAL[cam_id]
        para_cam = CAM_ID_TO_REALSENSE_PARA_CAM[cam_id]

        cam_output_dir = frame_output_dir / cam_id
        cam_output_dir.mkdir(parents=True, exist_ok=True)

        para_payload = load_json(para_dir / para_cam / "camera_payload.json")

        img_idx = next(idx for idx, cid in IMAGE_INDEX_TO_CAM_ID.items() if cid == cam_id)

        rgb_npy = np.load(observations_dir / "rgb" / f"color_image{img_idx}.npy")
        Image.fromarray(rgb_npy).save(cam_output_dir / "rgb.png")
        print(f"  [{cam_id}] RGB: {rgb_npy.shape} -> rgb.png")

        depth_npy = np.load(observations_dir / "depth" / f"depth_image{img_idx}.npy")
        depth_m = depth_npy.astype(np.float32) / 1000.0
        np.save(cam_output_dir / "depth_aligned_m.npy", depth_m)
        print(f"  [{cam_id}] Depth: {depth_npy.shape} {depth_npy.dtype} -> depth_aligned_m.npy (meters)")

        ir_left_npy = np.load(observations_dir / "infrared" / f"infrared_image{img_idx}_1.npy")
        ir_right_npy = np.load(observations_dir / "infrared" / f"infrared_image{img_idx}_2.npy")
        Image.fromarray(ir_left_npy).save(cam_output_dir / "ir_left_rect.png")
        Image.fromarray(ir_right_npy).save(cam_output_dir / "ir_right_rect.png")
        print(f"  [{cam_id}] IR: left {ir_left_npy.shape}, right {ir_right_npy.shape} -> ir_left/right_rect.png")

        color_intrinsics = dict(para_payload["color_intrinsics"])
        camera_payload: dict = {
            "camera_id": cam_id,
            "serial_number": serial,
            "depth_source": str(args.depth_source),
            "rgb_file": "rgb.png",
            "depth_aligned_file": "depth_aligned_m.npy",
            "color_intrinsics": color_intrinsics,
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
        (cam_output_dir / "depth_source.txt").write_text(args.depth_source, encoding="utf-8")
        print(f"  [{cam_id}] camera_payload.json written (depth_source={args.depth_source})")
        print()

    print(f"Done. Output written to {frame_output_dir}")
    print()
    print("Next steps:")
    print("  1. Run Fast depth estimation:")
    print(f"     python utils/replay_fast_debug_dump.py \\")
    print(f"       --input-dir {output_dir} \\")
    print(f"       --output-dir <fast_output> --frame-index 0 --max-frames 1")
    print()
    print("  2. Run SAM3 segmentation:")
    print(f"     python utils/replay_sam3_segmenter_debug_dump.py \\")
    print(f"       --input-dir {output_dir} \\")
    print(f"       --output-dir <sam3_output> --target-name small \\")
    print(f"       --prompt-task-info assets/prompts/small/task_info.json \\")
    print(f"       --prompt-image-root assets/prompts/small/ \\")
    print(f"       --frame-index 0 --max-frames 1 --save-ply 1 --save-debug-2d 1")


if __name__ == "__main__":
    main()
