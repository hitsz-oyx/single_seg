#!/usr/bin/env python3
"""
RealSense 相机实时采集工具
实时显示相机图像，按键保存图片

支持两种显示模式：
1. 窗口显示模式（需要 DISPLAY 环境）- 默认
2. 虚拟显示模式（无 DISPLAY 时使用）- 使用 --virtual-display 启用

按键说明：
- 's' 或 'Space': 保存当前帧
- 'q' 或 'ESC': 退出程序

操作：
cd /home/franka-client/oyx_ws/single_seg
conda activate single-seg

# 1. 先列出所有相机
python utils/capture_realsense_images.py --list-cameras

# 2. 使用 D435i 的序列号运行（替换成你的 D435i 序列号）
python utils/capture_realsense_images.py \
--output-dir assets/prompts/bowl \
--serial <D435i序列号> \
--virtual-display 

默认命令：
python utils/capture_realsense_images.py \
--output-dir capture_realsense_images \
--serial 243122075507 \
--virtual-display 
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    print("错误: 需要安装 pyrealsense2")
    print("运行: pip install pyrealsense2")
    raise


def parse_args():
    parser = argparse.ArgumentParser(description="RealSense 相机实时采集工具")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("captured_images"),
        help="保存图片的目录 (默认: captured_images)",
    )
    parser.add_argument(
        "--virtual-display",
        action="store_true",
        help="使用虚拟显示模式（无 GUI 时使用，每隔一段时间刷新预览图）",
    )
    parser.add_argument(
        "--preview-interval",
        type=float,
        default=2,
        help="虚拟显示模式下的预览刷新间隔（秒，默认: 2）",
    )
    parser.add_argument(
        "--preview-name",
        type=str,
        default="latest_preview.png",
        help="虚拟显示模式下预览图的文件名（默认: latest_preview.png）",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="1280x720",
        help="分辨率，格式: WxH (默认: 640x480)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="帧率 (默认: 30)",
    )
    parser.add_argument(
        "--save-depth",
        action="store_true",
        help="同时保存深度图",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="保存文件名前缀",
    )
    parser.add_argument(
        "--serial",
        type=str,
        default="",
        help="指定相机序列号 (默认: 自动选择第一个可用相机)",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="列出所有可用相机并退出",
    )
    return parser.parse_args()


def has_display():
    """检查是否有可用的显示环境"""
    if os.name == 'nt':
        return True
    display = os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')
    return bool(display)


def list_cameras():
    """列出所有可用的 RealSense 相机"""
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("未检测到任何 RealSense 相机")
        return
    
    print(f"\n检测到 {len(devices)} 个 RealSense 相机:")
    print("=" * 60)
    
    for i, dev in enumerate(devices):
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)
        firmware = dev.get_info(rs.camera_info.firmware_version)
        print(f"  [{i}] {name}")
        print(f"      序列号: {serial}")
        print(f"      USB 类型: {usb_type}")
        print(f"      固件版本: {firmware}")
        print()
    
    print("=" * 60)
    print("使用 --serial <序列号> 指定相机")


def select_camera(config, serial):
    """选择指定序列号的相机"""
    if not serial:
        return
    
    ctx = rs.context()
    devices = ctx.query_devices()
    serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
    
    if serial not in serials:
        print(f"警告: 序列号 {serial} 未找到!")
        print(f"可用的序列号: {serials}")
        print("将使用默认相机...")
        return
    
    config.enable_device(serial)
    print(f"已选择相机序列号: {serial}")


def run_with_gui(args, output_dir, width, height):
    """窗口显示模式"""
    pipeline = rs.pipeline()
    config = rs.config()
    select_camera(config, args.serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, args.fps)
    if args.save_depth:
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, args.fps)
    
    print(f"\n{'='*50}")
    print("RealSense 相机实时采集工具 [窗口显示模式]")
    print(f"{'='*50}")
    print(f"分辨率: {width}x{height}")
    print(f"帧率: {args.fps} FPS")
    print(f"保存目录: {output_dir}")
    print(f"保存深度: {'是' if args.save_depth else '否'}")
    print(f"{'='*50}")
    print("\n按键说明:")
    print("  's' 或 'Space' - 保存当前帧")
    print("  'q' 或 'ESC'   - 退出程序")
    print(f"{'='*50}\n")
    
    try:
        profile = pipeline.start(config)
        
        device = profile.get_device()
        device_name = device.get_info(rs.camera_info.name) if device else "Unknown"
        serial = device.get_info(rs.camera_info.serial_number) if device else "Unknown"
        print(f"相机: {device_name}")
        print(f"序列号: {serial}\n")
        
        if args.save_depth:
            align = rs.align(rs.stream.color)
        
        saved_count = 0
        
        print("预热相机...")
        for i in range(3):
            try:
                pipeline.wait_for_frames(timeout_ms=5000)
                print(f"  预热帧 {i+1}/3 完成")
            except RuntimeError as e:
                print(f"  预热帧 {i+1}/3 超时: {e}")
        print("预热完成，开始显示...")
        
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError as e:
                print(f"等待帧超时: {e}, 重试中...")
                time.sleep(0.5)
                continue
            
            if args.save_depth:
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
            else:
                color_frame = frames.get_color_frame()
                depth_frame = None
            
            if not color_frame:
                continue
            
            color_image = np.asanyarray(color_frame.get_data())
            
            display_image = color_image.copy()
            cv2.putText(
                display_image,
                f"Saved: {saved_count} | Press 's' to save, 'q' to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            
            if args.save_depth and depth_frame:
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET,
                )
                display_image = np.hstack((display_image, depth_colormap))
            
            cv2.imshow("RealSense Capture", display_image)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') or key == ord(' '):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                prefix = f"{args.prefix}_" if args.prefix else ""
                color_filename = output_dir / f"{prefix}{timestamp}.png"
                cv2.imwrite(str(color_filename), color_image)
                saved_count += 1
                print(f"✓ 已保存: {color_filename}")
                
                if args.save_depth and depth_frame:
                    depth_filename = output_dir / f"{prefix}{timestamp}_depth.png"
                    depth_raw_filename = output_dir / f"{prefix}{timestamp}_depth_raw.npy"
                    cv2.imwrite(str(depth_filename), depth_colormap)
                    np.save(str(depth_raw_filename), depth_image)
                    print(f"✓ 已保存深度: {depth_filename}")
                    print(f"✓ 已保存深度数据: {depth_raw_filename}")
            
            elif key == ord('q') or key == 27:
                print("\n退出程序...")
                break
    
    except Exception as e:
        print(f"错误: {e}")
        raise
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    
    print(f"\n总计保存 {saved_count} 张图片到 {output_dir}")


def run_virtual_display(args, output_dir, width, height):
    """虚拟显示模式"""
    preview_path = output_dir / args.preview_name
    status_path = output_dir / "capture_status.txt"
    
    pipeline = rs.pipeline()
    config = rs.config()
    select_camera(config, args.serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, args.fps)
    if args.save_depth:
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, args.fps)
    
    print(f"\n{'='*50}")
    print("RealSense 相机实时采集工具 [虚拟显示模式]")
    print(f"{'='*50}")
    print(f"分辨率: {width}x{height}")
    print(f"帧率: {args.fps} FPS")
    print(f"保存目录: {output_dir}")
    print(f"预览文件: {preview_path}")
    print(f"刷新间隔: {args.preview_interval} 秒")
    print(f"保存深度: {'是' if args.save_depth else '否'}")
    print(f"{'='*50}")
    print("\n按键说明:")
    print("  's' 或 'Space' - 保存当前帧（通过标准输入）")
    print("  'q' 或 'ESC'   - 退出程序（通过标准输入）")
    print(f"{'='*50}")
    print("\n监控方法:")
    print(f"  预览图: ls -l {preview_path}")
    print(f"  状态文件: cat {status_path}")
    print(f"  所有图片: ls {output_dir}")
    print(f"{'='*50}\n")
    
    try:
        profile = pipeline.start(config)
        
        device = profile.get_device()
        device_name = device.get_info(rs.camera_info.name) if device else "Unknown"
        serial = device.get_info(rs.camera_info.serial_number) if device else "Unknown"
        print(f"相机: {device_name}")
        print(f"序列号: {serial}\n")
        
        if args.save_depth:
            align = rs.align(rs.stream.color)
        
        saved_count = 0
        last_preview_time = 0
        last_status_time = 0
        
        print("预热相机...")
        for i in range(3):
            try:
                pipeline.wait_for_frames(timeout_ms=5000)
                print(f"  预热帧 {i+1}/3 完成")
            except RuntimeError as e:
                print(f"  预热帧 {i+1}/3 超时: {e}")
        print("预热完成，开始采集...")
        
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError as e:
                print(f"等待帧超时: {e}, 重试中...")
                time.sleep(0.5)
                continue
            
            if args.save_depth:
                try:
                    aligned_frames = align.process(frames)
                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()
                except:
                    color_frame = frames.get_color_frame()
                    depth_frame = None
            else:
                color_frame = frames.get_color_frame()
                depth_frame = None
            
            if not color_frame:
                continue
            
            current_time = time.time()
            color_image = np.asanyarray(color_frame.get_data())
            
            if args.save_depth and depth_frame:
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET,
                )
                display_image = np.hstack((color_image, depth_colormap))
            else:
                display_image = color_image.copy()
            
            if current_time - last_preview_time >= args.preview_interval:
                display_with_info = display_image.copy()
                cv2.putText(
                    display_with_info,
                    f"Saved: {saved_count} | Preview: {time.strftime('%H:%M:%S')}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imwrite(str(preview_path), display_with_info)
                last_preview_time = current_time
            
            if current_time - last_status_time >= 1.0:
                with open(status_path, 'w') as f:
                    f.write(f"Last Update: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Camera: {device_name}\n")
                    f.write(f"Serial: {serial}\n")
                    f.write(f"Saved Images: {saved_count}\n")
                    f.write(f"Preview: {preview_path.name}\n")
                    f.write(f"Resolution: {width}x{height}\n")
                    f.write(f"FPS: {args.fps}\n")
                    f.write(f"Save Depth: {args.save_depth}\n")
                    f.write(f"\nPress Ctrl+C in terminal or send 'q' to quit\n")
                    f.write(f"Press 's' + Enter to save current frame\n")
                last_status_time = current_time
            
            try:
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    user_input = sys.stdin.readline().strip().lower()
                    if user_input == 'q' or user_input == 'quit' or user_input == 'exit':
                        print("\n退出程序...")
                        break
                    elif user_input == 's' or user_input == 'save':
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        prefix = f"{args.prefix}_" if args.prefix else ""
                        color_filename = output_dir / f"{prefix}{timestamp}.png"
                        cv2.imwrite(str(color_filename), color_image)
                        saved_count += 1
                        print(f"✓ 已保存: {color_filename}")
                        
                        if args.save_depth and depth_frame:
                            depth_filename = output_dir / f"{prefix}{timestamp}_depth.png"
                            depth_raw_filename = output_dir / f"{prefix}{timestamp}_depth_raw.npy"
                            cv2.imwrite(str(depth_filename), depth_colormap)
                            np.save(str(depth_raw_filename), depth_image)
                            print(f"✓ 已保存深度: {depth_filename}")
                            print(f"✓ 已保存深度数据: {depth_raw_filename}")
            except (ValueError, OSError):
                pass
    
    except KeyboardInterrupt:
        print("\n收到中断信号，退出程序...")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        pipeline.stop()
        if status_path.exists():
            status_path.unlink()
    
    print(f"\n总计保存 {saved_count} 张图片到 {output_dir}")


def main():
    args = parse_args()
    
    if args.list_cameras:
        list_cameras()
        return
    
    width, height = map(int, args.resolution.lower().split("x"))
    
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.virtual_display or not has_display():
        run_virtual_display(args, output_dir, width, height)
    else:
        run_with_gui(args, output_dir, width, height)


if __name__ == "__main__":
    main()
