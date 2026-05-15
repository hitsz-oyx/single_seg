# single_seg

单目标在线 RGBD 点云分割仓库。当前默认面向 `libero_spatial` 三相机 episode。

仓库内已包含：

- `third_party/sam3` 子模块
- `third_party/fastfoundationstereo` 子模块
- `assets/prompts/libero_spatial/semantic_split_parts` prompt 示例图
- `examples/data/libero_spatial/task_00_demo/episode_0001` 三帧最小 demo 数据

仓库内不包含：

- `sam3.pt` 权重文件
- `Fast-FoundationStereo` 权重文件

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

## 下载 Fast-FoundationStereo 权重

RealSense live 真 RGBD 流程默认依赖 `Fast-FoundationStereo` 的 `23-36-37` 官方权重：

```text
third_party/fastfoundationstereo/weights/23-36-37/
```

可以直接用 `gdown` 下载默认这组：

```bash
python -m pip install --user gdown
mkdir -p third_party/fastfoundationstereo/weights/23-36-37
python - <<'PY'
from pathlib import Path
import gdown

out = Path("third_party/fastfoundationstereo/weights/23-36-37")
out.mkdir(parents=True, exist_ok=True)
gdown.download(
    id="1GDBRYL-ZaLpXEtWfGFRJvkBc_2sywjgj",
    output=str(out / "cfg.yaml"),
    quiet=False,
    use_cookies=False,
)
gdown.download(
    id="1W1V1H64l9bAi97boEQQ2ueNzzGmSMz-E",
    output=str(out / "model_best_bp2_serialize.pth"),
    quiet=False,
    use_cookies=False,
)
PY
```

下载完成后，默认 live 入口会优先读取：

```text
third_party/fastfoundationstereo/weights/23-36-37/model_best_bp2_serialize.pth
```

## 配置文件

常用配置放在：

```text
configs/
```

当前自带：

- `configs/default.yaml`
- `configs/fast_plate_demo.yaml`
- `configs/realsense_d435_live.yaml`

其中 `realsense_d435_live.yaml` 同时覆盖三部分参数：

- `segmenter`: `SingleObjectPointCloudSegmenter` 初始化参数
- `realsense`: live 相机和运行参数
- `fast_stereo`: `Fast-FoundationStereo` 推理参数

路径默认按 repo 根目录解析。

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
- `--tracker-image-size 896`: 当前默认输入尺寸
- `--save-ply`: 保存完整场景带标签点云
- `--save-normal`: 保存 PLY 时写入估计法线，法线会翻转到朝向最近相机
- `--save-debug-2d`: 保存逐帧 2D overlay
- `--target-cluster-filter-enabled 1`: 对最终 3D 目标点做聚类去散点
- `--target-cluster-radius-m 0.013`: 3D 聚类邻域半径，单位米
- `--target-cluster-min-points 45`: 有效目标簇的最少点数
- `--target-cluster-keep-largest 1`: 单物体场景只保留最大目标簇

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
        )
        print(result["points_xyz"].shape, result["points_xyz"].device)
```

类接口：

- `SingleSegConfig`：初始化配置
- `SingleSegConfig.from_yaml(...)`：从 YAML 配置构造
- `SingleObjectPointCloudSegmenter.from_config(...)`：从配置对象构造
- `SingleObjectPointCloudSegmenter.from_config_file(...)`：从 YAML 配置构造
- `SingleObjectPointCloudSegmenter.process_frame(...)`：逐帧在线处理

## `task_info.json` 最小模板

当前 `assets/prompts/.../task_info.json` 里的字段比较多，但 `single_seg` 底层真正读取的只有这几个：

- 顶层 `assets`
- 每个 asset 的 `asset_name`
- 每个 asset 的 `image_path`
- 每个 asset 的 `bbox_xyxy`
- 可选的 `extra_views[].image_path`
- 可选的 `extra_views[].bbox_xyxy`

最小可用模板其实可以只有一个正样本：

```json
{
  "assets": [
    {
      "asset_name": "plate_0",
      "image_path": "plate_0.png",
      "bbox_xyxy": [120, 80, 420, 360]
    }
  ]
}
```

配套目录示例：

```text
assets/prompts/my_task/
├── plate_0.png
└── task_info.json
```

如果一个物体有多张参考图，推荐按当前仓库风格直接写成多条 `asset`：

```json
{
  "assets": [
    {
      "asset_name": "plate_0",
      "image_path": "plate_0.png",
      "bbox_xyxy": [120, 80, 420, 360]
    },
    {
      "asset_name": "plate_1",
      "image_path": "plate_1.png",
      "bbox_xyxy": [100, 70, 400, 350]
    },
    {
      "asset_name": "plate_2",
      "image_path": "plate_2.png",
      "bbox_xyxy": [110, 75, 410, 355]
    }
  ]
}
```

这时 `target_name` 仍然写 `plate`。代码会自动把 `plate_0`、`plate_1`、`plate_2` 归到同一个语义名 `plate`。

`extra_views` 也支持，但只是兼容写法，不是当前仓库默认风格：

```json
{
  "assets": [
    {
      "asset_name": "plate_0",
      "image_path": "plate_0.png",
      "bbox_xyxy": [120, 80, 420, 360],
      "extra_views": [
        {
          "image_path": "plate_1.png",
          "bbox_xyxy": [100, 70, 400, 350]
        }
      ]
    }
  ]
}
```

如果要加入负样本，也是在 `assets` 里再放别的语义对象：

```json
{
  "assets": [
    {
      "asset_name": "plate_0",
      "image_path": "plate_0.png",
      "bbox_xyxy": [120, 80, 420, 360]
    },
    {
      "asset_name": "bowl_0",
      "image_path": "bowl_0.png",
      "bbox_xyxy": [60, 90, 300, 340]
    }
  ]
}
```

几个容易踩坑的点：

- `target_name` 要和 `asset_name` 的“语义名”一致。代码会自动把结尾的 `_<数字>` 去掉，所以 `plate_0`、`plate_1` 都会被当成语义 `plate`。
- `image_path` 是相对 `prompt_image_root` 解析的，不是相对 `task_info.json`。
- `bbox_xyxy` 必须是原图像素坐标，不是归一化坐标。
- `bbox_xyxy` 的格式是 `[x0, y0, x1, y1]`。
- 原点在图像左上角，`x` 向右增大，`y` 向下增大。
- `x1`、`y1` 是包含在框内的右下角像素，不是开区间上界。
- 这只是 2D 图像坐标，不是 3D 右手系。
- `assets` 里至少要有一个和 `target_name` 匹配的正样本。
- 负样本不是必需的；当前实现允许 `negative_entries` 为空。

目前这份 `json` 仍然是必需输入，不是可选项。初始化时会先检查 `prompt_task_info` 文件存在，然后读取它来构造正负 prompt。  
也就是说，现阶段你可以把它精简到只剩上面这些必要字段，但还不能完全去掉。

### 手工标注 `bbox_xyxy`

如果 prompt 图是你自己拍的，仓库里现在有一个交互式标注脚本：

```bash
python utils/annotate_prompt_bboxes.py \
  --prompt-image-root assets/prompts/my_task \
  --semantic-name plate
```

这个脚本会：

- 扫描 `prompt-image-root` 下的图片
- 逐张弹出窗口让你拖一个框
- 按当前仓库风格把结果写进 `task_info.json`
- 自动生成 `plate_0`、`plate_1`、`plate_2` 这种 `asset_name`

默认输出位置：

- 标注结果写到 `assets/prompts/my_task/task_info.json`
- 如果传了 `--task-info /path/to/task_info.json`，就写到你指定的 JSON 文件
- 带框确认图默认写到 `assets/prompts/my_task/annotated/`
- 如果传了 `--annotated-dir /path/to/annotated`，确认图会写到你指定目录
- `--preview-dir` 仍然可用，是 `--annotated-dir` 的兼容别名

例如 `assets/prompts/my_task/plate_0.png` 的手工预览图会是：

```text
assets/prompts/my_task/annotated/plate_0.png
```

非 `--dry-run` 时，脚本也会在确认图目录里写一份同步后的 `task_info.json`：

```text
assets/prompts/my_task/annotated/task_info.json
```

命令结束时会打印实际目录：

```text
Annotated images dir: assets/prompts/my_task/annotated
```

常用参数：

- `--images ...`：只标指定图片
- `--skip-existing`：跳过 `task_info.json` 里已经有框的图片
- `--annotated-dir tmp/prompt_bbox_preview`：覆盖确认图目录
- `--dry-run`：只看结果，不写回 `task_info.json`

交互方式：

- 鼠标拖框
- `Enter` 或 `Space` 确认
- `c` 取消当前图片

这个脚本需要桌面显示环境；如果你在纯 headless 远程 shell 里跑，没有 GUI 就不能直接弹窗。

### 用分割结果自动取框

如果你不想手工框物体，可以直接用 `SAM3` 文本 prompt 在整张图上分割，再从最佳 mask 自动取 `bbox_xyxy`：

```bash
python utils/annotate_prompt_bboxes_with_sam3.py \
  --prompt-image-root assets/prompts/my_task \
  --semantic-name plate \
  --checkpoint-path checkpoints/sam3.pt
```

这条流程是：

- 默认把 `semantic-name` 作为文本 prompt 喂给 `SAM3`
- 如果 `semantic-name` 里有下划线，会自动转成空格
- 从 `SAM3` 返回的最佳 mask 里自动取最小外接框
- 可选再加一点 bbox padding
- 直接写入 `task_info.json`

默认输出位置：

- 自动取框结果写到 `assets/prompts/my_task/task_info.json`
- 如果传了 `--task-info /path/to/task_info.json`，就写到你指定的 JSON 文件
- 分割 overlay 确认图默认写到 `assets/prompts/my_task/annotated/`
- 如果传了 `--annotated-dir /path/to/annotated`，确认图会写到你指定目录
- `--preview-dir` 仍然可用，是 `--annotated-dir` 的兼容别名

例如 `assets/prompts/my_task/plate_0.png` 的 `SAM3` 预览图会是：

```text
assets/prompts/my_task/annotated/plate_0.png
```

非 `--dry-run` 时，脚本也会在确认图目录里写一份同步后的 `task_info.json`：

```text
assets/prompts/my_task/annotated/task_info.json
```

命令结束时会打印实际目录：

```text
Annotated images dir: assets/prompts/my_task/annotated
```

常用参数：

- `--skip-existing`：跳过已有框的图片
- `--annotated-dir tmp/prompt_sam3_preview`：覆盖确认图目录
- `--text-prompt "black bowl"`：覆盖默认文本 prompt
- `--bbox-pad-ratio 0.03`：给最终 bbox 加一点留白
- `--bbox-min-pad 2`：最少向外扩 2 像素
- `--min-mask-pixels 64`：过滤太小的噪声 mask

这条脚本不需要桌面显示环境，但要求 `SAM3` 权重可用。  
如果 `SAM3` 对某些图片找不到稳定目标，回退方案还是上面的手工框脚本 [utils/annotate_prompt_bboxes.py](/home/oyx/wm_ws/single_seg/utils/annotate_prompt_bboxes.py:1)。


## RealSense Live 真 RGBD

仓库内新增了一个 D435/D435i live 入口：

```text
single_seg/realsense_rgbd_segmenter.py
```

默认 `--depth-source fast` 时，这条链路不是直接用相机原生深度，而是：

1. 每个 D435 采集 `color + IR1 + IR2`
2. 用 `IR1/IR2` 经过 `Fast-FoundationStereo` 估计深度
3. 把深度从 rectified-left IR 坐标系重投影到 RGB 坐标系
4. 再把对齐后的 `RGBD` 送进现有 `SingleObjectPointCloudSegmenter`
5. 多个 D435 的点云最后再做融合

融合后的目标点可以再做一层轻量 3D 聚类过滤，用来去掉深度估计导致的孤立散点。相关参数在 `configs/realsense_d435_live.yaml` 的 `segmenter` 里：

```yaml
target_cluster_filter_enabled: true
target_cluster_radius_m: 0.013
target_cluster_min_points: 45
target_cluster_keep_largest: true
target_plane_filter_enabled: false
target_plane_filter_distance_m: 0.004
target_plane_filter_min_points: 80
target_plane_filter_min_inlier_ratio: 0.25
target_plane_filter_max_inlier_ratio: 0.85
target_plane_filter_max_planes: 1
target_plane_filter_ransac_iterations: 256
target_depth_band_filter_enabled: false
target_depth_band_filter_range_m: 0.015
target_depth_band_filter_min_valid_pixels: 50
target_depth_band_filter_min_keep_pixels: 20
target_3d_mask_erode_kernel: 0
```

`target_cluster_radius_m` 控制点之间多远算相邻；`target_cluster_min_points` 控制小到什么程度会被当成散点；单物体任务建议保持 `target_cluster_keep_largest: true`。
`target_3d_mask_erode_kernel` 只影响 3D 取点，不改变 2D mask 调试图；设为 `3` 或 `5` 可以先缩掉目标 mask 边界，减少深度断层和背景混入。
如果桌面点已经和目标在 3D 里连成一块，单纯聚类不会把它们分开，可以打开 `target_plane_filter_enabled`。这个开关会在每个相机的目标点云里拟合占比足够大的主平面，把主平面内点改回背景；`target_plane_filter_distance_m` 控制点离平面多近会被删除，`target_plane_filter_min/max_inlier_ratio` 用来避免没有明显平面或目标整体近似平面时误删，`target_plane_filter_max_planes` 默认 `1` 更保守，桌面残留明显时可临时调到 `2` 对比。
如果目标允许点少但要尽量少误点，优先试 `target_depth_band_filter_enabled`。它在反投影前统计目标 mask 内有效深度的中位数，只保留距离该中位数 `target_depth_band_filter_range_m` 以内的像素；红杯这类小物体可以从 `0.012` 到 `0.015` 试起。这个过滤比平面剔除更适合处理桌面碎点和深度边界误点。

也就是说，这里输出的是“真 RGBD”，不是把 IR 灰度图简单伪装成 RGB。

如果要和 D435 原生深度对比，可以改成 `--depth-source native`。这个模式只开 `color + depth`，用 RealSense SDK 把原生 depth 对齐到 color，不加载 `Fast-FoundationStereo` 模型。

### 单相机最小运行

推荐先用低带宽模式跑通。当前这台通过 USB/IP attach 的 D435，已验证下面这组更稳：

- `color`: `640x480`
- `stereo`: `480x270`
- `fps`: `6`

运行命令：

```bash
/home/oyx/miniconda3/envs/sam3/bin/python -m single_seg.realsense_rgbd_segmenter \
  --config configs/realsense_d435_live.yaml \
  --camera-count 1 \
  --max-frames 1 \
  --target-name plate \
  --save-live-debug 1 \
  --overwrite-output
```

如果已经 `pip install -e .`，也可以直接运行：

```bash
single-seg-realsense \
  --config configs/realsense_d435_live.yaml \
  --camera-count 1 \
  --max-frames 1 \
  --target-name plate \
  --save-live-debug 1 \
  --overwrite-output
```

如果不传额外覆盖参数，直接用配置文件也可以：

```bash
single-seg-realsense --config configs/realsense_d435_live.yaml
```

对比原生 D435 depth：

```bash
single-seg-realsense \
  --config configs/realsense_d435_live.yaml \
  --depth-source native \
  --overwrite-output
```

输出目录默认写到：

```text
tests/outputs/realsense_live
```

其中会额外保存一份 live RGBD 预处理调试结果：

```text
tests/outputs/realsense_live/live_rgbd_debug/
```

输出目录根部还会保存：

- `effective_config.yaml`：命令行覆盖 YAML 后的最终生效配置，可直接用于复跑
- `live_debug_config.yaml`：原始配置、最终配置、命令行参数、相机序列号和启动相机信息的调试快照

单相机调试图包括：

- `rgb.png`
- `ir_left_rect.png`：仅 `--depth-source fast` 时保存
- `ir_right_rect.png`：仅 `--depth-source fast` 时保存
- `depth_aligned_m.npy`
- `depth_aligned_vis.png`
- `depth_source.txt`
- `camera_payload.json`：保存离线重跑所需的相机内外参；`fast` 模式还包含 `rectified_k`、`baseline_m`、`rectified_to_color`

如果需要用 debug dump 离线 profile，不输出 `debug2d/ply`：

```bash
/home/oyx/miniconda3/envs/sam3/bin/python utils/profile_realsense_debug_dump.py \
  --input-dir tests/outputs/realsense_live/live_rgbd_debug \
  --depth-source saved
```

新 dump 有 `camera_payload.json` 后，也可以重新从 IR 图跑 Fast 前处理：

```bash
/home/oyx/miniconda3/envs/sam3/bin/python utils/profile_realsense_debug_dump.py \
  --input-dir tests/outputs/realsense_live/live_rgbd_debug \
  --depth-source fast
```

### Fast 深度速度调参

当前 D435 低带宽数据的 rectified IR 是 `480x270`，`rectified_fx * baseline ~= 12.1`。如果 `depth_min=0.1m`，最大视差约 `121px`，所以 `max_disp=192` 已经够用；`max_disp=256` 在这组数据上几乎没有收益，只会更慢。

离线 benchmark 使用 `tests/outputs/realsense_live_fast_tuned/live_rgbd_debug` 的 30 帧，以 `valid_iters=12,max_disp=256,scale=1.0` 作为参考，后 29 帧平均结果：

| Fast 参数 | depth 总耗时 | 推理耗时 | 相对参考 MAE | P90 误差 | 有效像素覆盖 |
|---|---:|---:|---:|---:|---:|
| `iters=12,max_disp=256` | `67.6 ms` | `66.0 ms` | `0.00 cm` | `0.00 cm` | `1.000` |
| `iters=12,max_disp=192` | `64.6 ms` | `63.1 ms` | `0.01 cm` | `0.02 cm` | `1.000` |
| `iters=8,max_disp=192` | `53.7 ms` | `52.2 ms` | `0.02 cm` | `0.04 cm` | `1.000` |
| `iters=6,max_disp=192` | `46.3 ms` | `44.8 ms` | `0.03 cm` | `0.06 cm` | `1.000` |
| `iters=4,max_disp=192` | `39.9 ms` | `38.3 ms` | `0.05 cm` | `0.10 cm` | `1.000` |

当前配置文件默认使用平衡档：

```yaml
fast_stereo:
  valid_iters: 4
  max_disp: 128
  scale: 0.75
  optimize_build_volume: pytorch1
  align_backend: open3d
  depth_edge_filter_enabled: false
  depth_edge_filter_threshold_m: 0.5
  depth_edge_filter_stage: rectified
```

在当前 `1280x720` rectified IR live dump 上，这组配置的 Fast 推理约在 `70 ms` 附近。`pytorch1` 是当前默认后端；`triton` 也可用，但首次运行会有编译/自调优开销，live 前几帧或离线 profile 首帧会明显慢一些；看稳定速度时应跳过首帧。
`depth_edge_filter_enabled` 打开后会做 Sobel 深度突变过滤，去掉边缘飞点；`depth_edge_filter_stage: rectified` 表示在 IR rectified depth 上过滤，通常比投影到 RGB 后再过滤更稳。阈值越小过滤越强，过小会吃掉真实物体边缘。

如果只追求速度，可以临时覆盖：

```bash
single-seg-realsense \
  --config configs/realsense_d435_live.yaml \
  --fast-valid-iters 4 \
  --fast-max-disp 192
```

不建议在当前 `480x270` IR 输入上直接降 `scale`。`scale=0.75/0.5` 在这组数据中没有明显加速，反而让投影到 RGB 后的有效深度覆盖下降明显。

### 时间线与性能定位

live 入口会在输出目录根部写：

```text
single_object_timeline.json
```

这个文件现在同时记录两类时间：

- `frame_runtime_sec`：只统计 `SingleObjectPointCloudSegmenter.process_frame()` 内部耗时。
- `loop_runtime_sec`：统计 live 主循环一帧端到端耗时，包含相机采集、RGBD 构建、FastStereo、SAM3/点云处理。

新增的 live 级字段：

```text
capture_time_sec
build_camera_inputs_time_sec
process_frame_wall_time_sec
loop_runtime_sec
live_timing_sec
```

其中 `live_timing_sec.rgbd_build` 会继续拆分：

```text
stereo_infer_time_sec
depth_align_time_sec
depth_filter_time_sec
live_debug_write_time_sec
per_camera[].stereo_infer_time_sec
per_camera[].depth_align_time_sec
per_camera[].depth_to_cpu_time_sec
per_camera[].open3d_align_time_sec
per_camera[].librealsense_align_time_sec
per_camera[].live_debug_write_time_sec
```

首帧还会额外记录 tracker 初始化：

```text
initialize_sessions_time_sec
initialize_sessions_breakdown_sec
```

`initialize_sessions_breakdown_sec` 里包含 prompt query、seed 拼接、tracker `start_session`、`add_prompt` 的耗时，用来解释第一帧为什么通常明显慢于后续帧。

默认 live 路径不强制 CUDA 同步，吞吐更好，但 GPU kernel 是异步排队的，某些前处理的等待时间可能被算进后面的 SAM3 `propagate_time_sec`。如果要做准确归因，用：

```bash
single-seg-realsense \
  --config configs/realsense_d435_live.yaml \
  --sync-timing 1
```

`--sync-timing 1` 会在关键计时点调用 `torch.cuda.synchronize()`，所以时间归因更清楚，但正式跑速度时建议仍保持默认 `0`。

看瓶颈时优先看这些字段：

- `loop_runtime_sec`：真实端到端每帧耗时。
- `live_timing_sec.rgbd_build.stereo_infer_time_sec`：Fast-FoundationStereo 推理耗时，三相机时通常是最大项。
- `live_timing_sec.rgbd_build.depth_align_time_sec`：Fast depth 对齐到 RGB 的总耗时。
- `live_timing_sec.rgbd_build.depth_to_cpu_time_sec`：仅 `align_backend=librealsense` 时有效，表示 CUDA depth 拷回 CPU 的耗时。
- `live_timing_sec.rgbd_build.open3d_align_time_sec`：仅 `align_backend=open3d` 时有效，表示 Open3D tensor 投影耗时。
- `live_timing_sec.rgbd_build.librealsense_align_time_sec`：仅 `align_backend=librealsense` 时有效，表示 `rs.align` 本身耗时。
- `propagate_time_sec`：SAM3 tracker 当前帧传播耗时。
- `initialize_sessions_time_sec`：首帧 prompt 和 tracker session 初始化耗时。
- `live_timing_sec.rgbd_build.live_debug_write_time_sec`：保存 live debug 图和 depth 文件的写盘耗时。

如果 `stereo_infer_time_sec` 最大，优先考虑：

- 改用 `--depth-source native`，直接绕过 FastStereo。
- 降低 `--fast-valid-iters`。
- 调小 `--fast-max-disp`，但要保证覆盖实际深度范围。
- 后续再考虑把多相机 FastStereo 从串行改成 batch 或多 CUDA stream。

如果 `propagate_time_sec` 最大，优先考虑降低 `--tracker-image-size`，例如 `784` 或 `672`；这个值必须是 `14` 的倍数。

如果 `live_debug_write_time_sec` 明显，正式跑时关掉：

```bash
single-seg-realsense \
  --config configs/realsense_d435_live.yaml \
  --save-live-debug 0
```

### 多相机参数

- `--camera-count`: 使用多少个 D435 逻辑相机
- `--camera-serials`: 指定串号列表，逗号分隔；不传时默认按枚举顺序取前 `N` 台
- `--camera-poses-json`: 多相机融合时提供每台 D435 的 `cam2world_4x4`
- `--depth-source`: `fast` 使用 `IR1/IR2 + Fast-FoundationStereo`；`native` 使用 D435 原生 depth
- `--stereo-rectification-mode`: Fast 路径的 IR 校正模式，`opencv` 为历史默认，`passthrough` 直接使用 RealSense 输出的 IR1/IR2
- `--emitter-enabled`: 设置 RealSense IR 投影器，`0` 关闭、`1` 开启；不传则保持相机当前设置
- `--fast-model-path`: 覆盖 `Fast-FoundationStereo` 权重路径
- `--fast-valid-iters`: Fast refine 迭代次数；越小越快
- `--fast-max-disp`: Fast 最大视差；当前低带宽 D435 数据默认 `192`
- `--fast-optimize-build-volume`: Fast cost-volume 后端，支持 `pytorch1` 或 `triton`
- `--fast-align-backend`: Fast depth 对齐到 RGB 的后端，`torch` 为默认 CUDA 投影，`open3d` 为 tensor 点云投影，`librealsense` 为实验性软件帧 + `rs.align`
- `--fast-depth-edge-filter-enabled`: 是否启用 Fast 深度边缘过滤
- `--fast-depth-edge-filter-threshold-m`: 深度边缘过滤阈值，默认 `0.5`
- `--fast-depth-edge-filter-stage`: 深度边缘过滤位置，推荐 `rectified`
- `--target-3d-mask-erode-kernel`: 仅用于 3D 取点的目标 mask 腐蚀核大小，`0/1` 关闭
- `--save-ply`: 保存融合后的点云输出
- `--sync-timing`: 调试性能时同步 CUDA 计时，`1` 更准但更慢，正式运行保持 `0`

当前代码内部的相机数量已经是动态的，不再假设固定 3 相机。  
不过目前只对“单台 D435 先把真 RGBD 预处理链跑通”做过实机验证；多 D435 融合接口已经接入，但还没有做完整现场联调。

### 多相机位姿文件格式

当 `--camera-count > 1` 时，当前入口要求有有效的 `camera_poses_json`。默认配置已经写到：

```yaml
realsense:
  camera_poses_json: tests/outputs/camera_poses_apriltag.json
```

如果这个文件还没有生成，单相机运行会回退到单位位姿；多相机运行会报错，避免多相机点云在没有外参时被错误融合。格式示例：

```json
{
  "cameras": [
    {
      "camera_id": "cam_00",
      "serial_number": "243122075507",
      "cam2world_4x4": [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
      ]
    },
    {
      "camera_id": "cam_01",
      "serial_number": "SECOND_SERIAL",
      "cam2world_4x4": [
        [1.0, 0.0, 0.0, 0.2],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
      ]
    }
  ]
}
```

单相机场景如果没有这个文件，默认使用单位位姿。

### AprilTag 外参标定

可以用 `utils/calibrate_realsense_apriltag_extrinsics.py` 从 AprilTag 直接生成 live 入口需要的 `camera_poses_json`。默认内置 `furniture-bench` 的 base tag 布局：`tag36h11` 的 `0/1/2/3` 四个 tag，边长 `0.048m`，中心分别在 `(-0.03,-0.03)`, `(0.03,-0.03)`, `(-0.03,0.03)`, `(0.03,0.03)`。

先列出相机：

```bash
/home/oyx/miniconda3/envs/sam3/bin/python utils/calibrate_realsense_apriltag_extrinsics.py \
  --list-cameras
```

标定两台相机并输出到配置文件默认路径：

```bash
/home/oyx/miniconda3/envs/sam3/bin/python utils/calibrate_realsense_apriltag_extrinsics.py \
  --serials 243122075507,SECOND_SERIAL \
  --num-frames 30 \
  --output tests/outputs/camera_poses_apriltag.json \
  --debug-dir tests/outputs/apriltag_calibration_debug
```

生成后直接按配置运行 live 即可：

```bash
single-seg-realsense \
  --config configs/realsense_d435_live.yaml \
  --camera-count 2
```

如果某台相机视野里完全没有 AprilTag，或者看到的 tag id 不在当前 layout 里，标定脚本会直接报错并打印该相机的 `camera_id/serial`、期望的 tag id、实际检测到的 tag id；如果传了 `--debug-dir`，还会保存最后一帧检测可视化图。

输出里的 `cam2world_4x4` 默认是 `single_seg` 点云反投影使用的 OpenGL 相机坐标约定；文件里也会额外写入 `opencv_cv_cam2world_4x4`，方便排查 AprilTag/RealSense 原始坐标。

## 3D 可视化

```bash
/home/oyx/miniconda3/envs/sam3/bin/python -m single_seg.view_ply_sequence \
  --input-dir tests/outputs/demo_spatial_single_object/frame_outputs
```

默认读取 `frame_*_instance_rgb.ply`。常用 PLY 模式：

- `frame_*_scene_rgb.ply`：原始 RGB 场景点云
- `frame_*_instance_rgb.ply`：目标实例高亮点云，目标点默认染成红色
- `frame_*_instance_label.ply`：包含实例标签字段，Open3D 不会自动把 label 映射成颜色

如果要查看目标实例随高度的颜色渐变，用 `--color-mode target-height`：

```bash
DISPLAY=localhost:10.0 /home/oyx/miniconda3/envs/sam3/bin/python \
  /home/oyx/wm_ws/single_seg/single_seg/view_ply_sequence.py \
  --input-dir tests/outputs/realsense_live_saved_depth_ply/frame_outputs \
  --pattern 'frame_*_instance_rgb.ply' \
  --color-mode target-height \
  --point-size 3
```

`target-height` 只会重着色 `instance_rgb.ply` 中的目标点；默认目标颜色是 `255,70,70`，可以用 `--target-color` 和 `--target-color-tolerance` 覆盖。

键位：

- `D` 下一帧
- `A` 上一帧
- `R` 重置视角
- `Q` 关闭

## 多相机外参微调（ICP 配准）

基于 mesh 标定物体，对多相机外参做微调。算法由用户设计：

1. **主相机配准** — 主相机的世界坐标点云配准到 mesh，得到 `T_MW` (world→mesh)
2. **定义 mesh 位姿** — `T_MW` 定义了 mesh 在世界坐标系中的位姿
3. **非主相机微调** — 对每个非主相机：
   - `predicted_T_MC = T_MW @ original_cam2world` (camera→mesh 空间的初始值)
   - 相机坐标点云 → ICP → mesh，以 `predicted_T_MC` 为初值做局部微调
   - `new_cam2world = inv(T_MW) @ refined_T_MC`

主相机配准支持两种后端，通过 `--use-goicp` 开关控制：

| 后端 | 特点 | 用法 |
|------|------|------|
| **Open3D ICP（默认）** | 质心对齐为初值，局部迭代，调整量小（~3-5°） | 不加 `--use-goicp` |
| **Go-ICP** | 全局搜索最优解，可能找到不同局部最优，调整量可能更大 | 加 `--use-goicp`，需 `pip install py_goicp` |

### 核心文件

| 文件 | 说明 |
|------|------|
| `icp/register_to_mesh.py` | 主脚本：配准+微调，输出 `refined_extrinsics.json` |
| `icp/goicp.py` | Go-ICP 后端的可复用接口（可选依赖） |
| `icp/Register.STL` | 标定物体的 mesh 文件 |
| `icp/config.yaml` | 配准参数配置文件，含详细注释 |

### 用法

**自动检测模式**（推荐 —— 给数据目录，自动寻找点云和外参）：

```bash
python icp/register_to_mesh.py \
  --data-dir tests/outputs/realsense_live_register_hand_three_cam_fast \
  --mesh icp/Register.STL \
  --master-camera cam_00
```

**使用 Go-ICP 全局配准**：

```bash
python icp/register_to_mesh.py \
  --data-dir tests/outputs/realsense_live_register_hand_three_cam_fast \
  --mesh icp/Register.STL \
  --master-camera cam_00 \
  --use-goicp
```

**显式模式**（手动指定外参 JSON 和各相机点云 PLY）：

```bash
python icp/register_to_mesh.py \
  --mesh icp/Register.STL \
  --extrinsics path/to/extrinsics.json \
  --point-cloud cam_00=/path/to/cam00.ply cam_01=/path/to/cam01.ply \
  --master-camera cam_00 \
  --output path/to/refined_extrinsics.json
```

**覆盖配准参数**：

```bash
python icp/register_to_mesh.py \
  --data-dir tests/outputs/realsense_live_register_hand_three_cam_fast \
  --mesh icp/Register.STL \
  --master-camera cam_00 \
  --voxel-size 0.005 \
  --master-icp-dist 0.08 \
  --refine-icp-dist 0.002
```

### 输出

输出 `refined_extrinsics.json`，包含每个相机的：

- `original_cam2world_4x4`：原始外参
- `refined_cam2world_4x4`：微调后的外参
- `adjustment_rotation_deg` / `adjustment_translation_m`：相对原始外参的调整量
- `icp_fitness` / `icp_inlier_rmse`：点云重叠质量指标
- `verify_vs_master_fitness` / `verify_vs_master_rmse`：微调后与主相机点云的重叠验证

输出目录结构：

- 外参 JSON → `configs/refined_extrinsics.json`
- 场景点云（`--save-fused 1` 时）→ `tests/outputs/{data_dir}_refined/fused/`

```text
configs/
└── refined_extrinsics.json      # 微调后的外参

tests/outputs/{data_dir}_refined/
└── fused/                        # --save-fused 时生成
    ├── original_fused.ply        # 原始外参融合场景点云
    ├── refined_fused.ply         # 新外参融合场景点云
    └── comparison_colored.ply    # 着色对比（蓝=新外参 橙=原始外参）
```

### 参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mesh` | — | 标定物体 .stl 文件 **（必填）** |
| `--master-camera` | `cam_00` | 主相机 ID |
| `--camera-ids` | `cam_00 cam_01 cam_02` | 所有相机 ID 列表 |
| `--output` | `configs/refined_extrinsics.json` | 输出 JSON 路径 |
| `--data-dir` | — | [自动检测] 数据目录 |
| `--extrinsics` | — | [显式] 原始外参 JSON |
| `--point-cloud` | — | [显式] 点云文件列表 |
| `--use-goicp` | `false` | Go-ICP 全局配准开关 |
| `--voxel-size` | `0.003` | 体素下采样大小（米） |
| `--master-icp-dist` | `0.05` | 主相机配准最大对应距离（米） |
| `--refine-icp-dist` | `0.003` | 非主相机微调最大对应距离（米） |
| `--num-mesh-points` | `100000` | mesh 采样点数 |
| `--visualize` | `0` | 显示 Open3D 可视化窗口 |
| `--save-fused` | `0` | 保存融合场景点云（.ply）用于对比 |
