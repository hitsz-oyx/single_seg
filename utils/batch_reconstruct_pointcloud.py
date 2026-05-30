#!/usr/bin/env python3
"""批量处理所有rollout目录的点云重建。"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def get_all_rollout_dirs(base_dir: Path) -> list[Path]:
    """获取所有rollout目录。"""
    rollout_dirs = []
    for item in sorted(base_dir.glob("*_demo_*")):
        if item.is_dir():
            rollout_dirs.append(item)
    return rollout_dirs


def main() -> None:
    script_path = Path(__file__).parent / "reconstruct_pointcloud_from_rgbd.py"
    rollout_base = Path("/home/franka-client/oyx_ws/rollout_results")
    
    rollout_dirs = get_all_rollout_dirs(rollout_base)
    
    if not rollout_dirs:
        print("未找到任何rollout目录")
        return
    
    print(f"找到 {len(rollout_dirs)} 个rollout目录")
    print("=" * 80)
    
    total = len(rollout_dirs)
    for idx, rollout_dir in enumerate(rollout_dirs, 1):
        print(f"\n[{idx}/{total}] 处理: {rollout_dir.name}")
        print("-" * 80)
        
        output_dir = rollout_dir / "pointcloud_output"
        
        cmd = [
            "python3",
            str(script_path),
            str(rollout_dir),
            str(output_dir),
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=False,
                text=True,
                cwd=str(script_path.parent.parent),
            )
            
            if result.returncode == 0:
                print(f"✓ 成功完成: {rollout_dir.name}")
            else:
                print(f"✗ 失败: {rollout_dir.name} (exit code: {result.returncode})")
                
        except Exception as e:
            print(f"✗ 错误: {e}")
        
        print("=" * 80)
    
    print(f"\n所有处理完成！")
    print(f"输出目录: {rollout_base}/*/pointcloud_output/")


if __name__ == "__main__":
    main()
