#!/usr/bin/env python3
import os
import shutil
import argparse
from pathlib import Path
"""
python utils/extract_fullscene.py \
    /home/franka-client/oyx_ws/single_seg/tests/outputs/mpc2_rollout/aggregate \
    /home/franka-client/oyx_ws/single_seg/tests/outputs/mpc2_rollout/fullscene
"""
def extract_fullscene(source_dir, output_dir):
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    frame_dirs = sorted(source_path.glob("frame_*"))
    count = 0

    for frame_dir in frame_dirs:
        for ply_file in frame_dir.glob("*_full_scene.ply"):
            shutil.copy2(ply_file, output_path / ply_file.name)
            count += 1

    print(f"Copied {count} full_scene.ply files to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract full_scene.ply files from frame directories")
    parser.add_argument("source_dir", help="Source directory containing frame_* subdirectories")
    parser.add_argument("output_dir", help="Output directory for full_scene.ply files")
    args = parser.parse_args()

    extract_fullscene(args.source_dir, args.output_dir)