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

这条链路不是直接用相机原生深度，而是：

1. 每个 D435 采集 `color + IR1 + IR2`
2. 用 `IR1/IR2` 经过 `Fast-FoundationStereo` 估计深度
3. 把深度从 rectified-left IR 坐标系重投影到 RGB 坐标系
4. 再把对齐后的 `RGBD` 送进现有 `SingleObjectPointCloudSegmenter`
5. 多个 D435 的点云最后再做融合

也就是说，这里输出的是“真 RGBD”，不是把 IR 灰度图简单伪装成 RGB。

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

输出目录默认写到：

```text
tests/outputs/realsense_live
```

其中会额外保存一份 live RGBD 预处理调试结果：

```text
tests/outputs/realsense_live/live_rgbd_debug/
```

单相机调试图包括：

- `rgb.png`
- `ir_left_rect.png`
- `ir_right_rect.png`
- `depth_aligned_m.npy`
- `depth_aligned_vis.png`

### 多相机参数

- `--camera-count`: 使用多少个 D435 逻辑相机
- `--camera-serials`: 指定串号列表，逗号分隔；不传时默认按枚举顺序取前 `N` 台
- `--camera-poses-json`: 多相机融合时提供每台 D435 的 `cam2world_4x4`
- `--fast-model-path`: 覆盖 `Fast-FoundationStereo` 权重路径
- `--save-ply`: 保存融合后的点云输出

当前代码内部的相机数量已经是动态的，不再假设固定 3 相机。  
不过目前只对“单台 D435 先把真 RGBD 预处理链跑通”做过实机验证；多 D435 融合接口已经接入，但还没有做完整现场联调。

### 多相机位姿文件格式

当 `--camera-count > 1` 时，当前入口要求传 `--camera-poses-json`。格式示例：

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

单相机场景如果不传这个文件，默认使用单位位姿。

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
