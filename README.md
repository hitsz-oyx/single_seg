# single_seg

单目标在线 RGBD 点云分割仓库。当前默认面向 `libero_spatial` 三相机 episode。

仓库内已包含：

- `third_party/sam3` 子模块
- `assets/prompts/libero_spatial/semantic_split_parts` prompt 示例图
- `examples/data/libero_spatial/task_00_demo/episode_0001` 三帧最小 demo 数据

仓库内不包含：

- `sam3.pt` 权重文件

## 初始化

```bash
cd /home/oyx/wm_ws/single_seg
git submodule update --init --recursive
```

## 环境部署

仓库内提供了一份可部署环境：

- [environment.yml](/home/oyx/wm_ws/single_seg/environment.yml)

建议按下面顺序初始化：

```bash
cd /home/oyx/wm_ws/single_seg
conda env create -f environment.yml
conda activate single-seg
pip install -e ./third_party/sam3
pip install -e .
```

这份环境文件按当前已验证可用的版本整理，适合部署和复现。  
如果只想更新当前环境，也可以直接参考这个文件手动对齐依赖版本。

## 下载 SAM3 权重

ModelScope 页面：

- 模型页: https://www.modelscope.cn/models/facebook/sam3
- 文件页: https://www.modelscope.cn/models/facebook/sam3/files

推荐直接下载到 repo 内：

```bash
pip install modelscope
modelscope download --model facebook/sam3 sam3.pt --local_dir checkpoints
```

下载完成后，仓库默认会优先读取：

```text
checkpoints/sam3.pt
```

权重默认查找顺序：

1. 环境变量 `SAM3_CHECKPOINT`
2. `checkpoints/sam3.pt`
3. `~/.cache/modelscope/hub/facebook/sam3/sam3.pt`

这和 ModelScope 的默认缓存目录是一致的。

## 配置文件

常用配置放在：

```text
configs/
```

当前自带：

- `configs/default.yaml`
- `configs/fast_plate_demo.yaml`

配置文件只负责 `SingleObjectPointCloudSegmenter` 的初始化参数。路径默认按 repo 根目录解析。

## 最小运行

默认命令直接使用仓库内置 prompt 和 demo episode：

```bash
/home/oyx/miniconda3/envs/sam3/bin/python -m single_seg.single_object_segmenter \
  --target-name plate \
  --max-frames 3 \
  --save-ply \
  --save-debug-2d \
  --overwrite-output
```

输出默认写到：

```text
tests/outputs/demo_spatial_single_object
```

## 常用参数

- `--episode-dir`: 指向任意 LIBERO episode 目录
- `--prompt-task-info`: prompt 标注 json
- `--prompt-image-root`: prompt 图片目录
- `--checkpoint-path`: SAM3 权重
- `--video-backend tracker_only_stitched`: 当前默认推荐的快速路径
- `--tracker-image-size 896`: 当前默认输入尺寸
- `--output-format torch`: 返回 GPU tensor，仅在需要落盘时回 CPU
- `--save-ply`: 保存完整场景带标签点云
- `--save-debug-2d`: 保存逐帧 2D overlay

## Python 用法

```python
from pathlib import Path

from single_seg.single_object_segmenter import (
    SingleSegConfig,
    SingleObjectPointCloudSegmenter,
    collect_common_frame_names,
    load_episode_camera_records,
    load_episode_frame_inputs,
)

episode_dir = Path("examples/data/libero_spatial/task_00_demo/episode_0001")
camera_records = load_episode_camera_records(episode_dir)
camera_ids = [record["camera_id"] for record in camera_records]
frame_names = collect_common_frame_names(episode_dir, camera_ids)

config = SingleSegConfig.from_yaml("configs/fast_plate_demo.yaml")

with SingleObjectPointCloudSegmenter.from_config(
    config,
    save_ply=False,
    save_debug_2d=False,
) as segmenter:
    for frame_name in frame_names:
        camera_inputs = load_episode_frame_inputs(
            episode_dir=episode_dir,
            frame_name=frame_name,
            camera_records=camera_records,
            depth_scale=1000.0,
        )
        result = segmenter.process_frame(
            frame_name=frame_name,
            camera_inputs=camera_inputs,
            output_format="torch",
        )
        print(result["points_xyz"].shape, result["points_xyz"].device)
```

类接口：

- `SingleSegConfig`：初始化配置
- `SingleSegConfig.from_yaml(...)`：从 YAML 配置构造
- `SingleObjectPointCloudSegmenter.from_config(...)`：从配置对象构造
- `SingleObjectPointCloudSegmenter.from_config_file(...)`：从 YAML 配置构造
- `SingleObjectPointCloudSegmenter.process_frame(...)`：逐帧在线处理

## 3D 可视化

```bash
/home/oyx/miniconda3/envs/sam3/bin/python -m single_seg.view_ply_sequence \
  --input-dir tests/outputs/demo_spatial_single_object/frame_outputs
```

键位：

- `D` 下一帧
- `A` 上一帧
- `R` 重置视角
- `Q` 关闭
