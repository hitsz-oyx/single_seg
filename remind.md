对，RealSense 里面有对应 API，而且 **D400 系列的左右 IR 图通常已经是 rectified 的**。关键是你要用对流。

RealSense 官方文档说：D400 使用 `depth + infrared` 流时，D4 ASIC 会生成“synthetic”左右红外视图，使它们看起来已经对齐；查询左右红外外参时，通常会得到 `R = I`，`t = (b, 0, 0)`，其中 `b` 就是 stereo baseline。官方也说明 rectified 的左右图和 depth frame 没有畸变参数，因为硬件已经在 rectification 过程中去畸变了。([RealSense][1])

所以你这里应该区分三件事：

### 1. RealSense 自带深度

如果你只是要重建，最稳的是直接用 RealSense 的 `depth_frame`：

```python
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

depth_image = np.asanyarray(depth_frame.get_data()) * depth_scale  # meters
```

这个深度已经是 RealSense ASIC 根据双目算出来的，不需要 FastFoundationStereo。

### 2. FastFoundationStereo 需要的是左右双目图

如果你想用 FastFoundationStereo，应该喂：

```text
left  = infrared stream index 1
right = infrared stream index 2
```

不要喂：

```text
color + depth
color + infrared
aligned depth + color
```

`rs.align` 是把 depth 对齐到 color，或者 color 对齐到 depth；它不是给第三方 stereo 网络准备左右双目输入的工具。RealSense 官方说明 stream alignment 是生成“从另一个 stream 视角看到的 synthetic view”，这主要用于 color-depth 对齐和贴纹理。([RealSense][1])

### 3. 你可以直接从 API 取 fx 和 baseline

示例：

```python
import pyrealsense2 as rs
import numpy as np

pipe = rs.pipeline()
cfg = rs.config()

# 常用分辨率可先用 848x480 或 1280x720
cfg.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 30)
cfg.enable_stream(rs.stream.infrared, 2, 848, 480, rs.format.y8, 30)

profile = pipe.start(cfg)

try:
    frames = pipe.wait_for_frames()

    left_frame = frames.get_infrared_frame(1)
    right_frame = frames.get_infrared_frame(2)

    left_img = np.asanyarray(left_frame.get_data())
    right_img = np.asanyarray(right_frame.get_data())

    left_profile = left_frame.profile.as_video_stream_profile()
    right_profile = right_frame.profile.as_video_stream_profile()

    intr = left_profile.get_intrinsics()
    extr = left_profile.get_extrinsics_to(right_profile)

    fx = intr.fx
    fy = intr.fy
    cx = intr.ppx
    cy = intr.ppy

    # translation 单位通常是 meter
    baseline = abs(extr.translation[0])

    print("fx:", fx)
    print("fy:", fy)
    print("cx:", cx)
    print("cy:", cy)
    print("baseline:", baseline)
    print("R:", extr.rotation)
    print("t:", extr.translation)

finally:
    pipe.stop()
```

然后 FastFoundationStereo 输出 disparity 后，用：

```python
depth = fx * baseline / disparity
```

注意：`baseline` 是米，`fx` 是像素，所以算出来的 `depth` 是米。

---

### 你现在最可能的问题

你说“仿真好，实拍很差”，用 RealSense 的话，我最怀疑这几个：

**第一，喂错图了。**
FastFoundationStereo 应该吃左右 IR，不是 RGB 和 depth。D435/D455 只有一个 RGB，相当于不是彩色双目。

**第二，resize 后没改 fx。**
如果你把 848×480 resize 到模型输入尺寸，比如 640×480 或 1024×768，`fx, fy, cx, cy` 要同步缩放。否则深度比例会错。

```python
scale_x = new_w / old_w
scale_y = new_h / old_h

fx_new = fx * scale_x
fy_new = fy * scale_y
cx_new = cx * scale_x
cy_new = cy * scale_y
```

**第三，IR projector 影响网络。**
RealSense D4xx 会投 IR speckle pattern 来帮助自带 stereo 匹配。官方说明 D400 的深度来自左右图匹配，投影器会给低纹理表面叠加半随机纹理，尤其对室内白墙有帮助。([RealSense][2])
但对 FoundationStereo 这类网络，IR speckle 有时可能帮忙，有时可能造成域差。建议你分别试：

```python
depth_sensor.set_option(rs.option.emitter_enabled, 0)  # 关闭
depth_sensor.set_option(rs.option.emitter_enabled, 1)  # 打开
```

看哪个对 FastFoundationStereo 更稳定。

**第四，RealSense 自己的标定可能漂了。**
官方自校准文档说，D4xx 性能下降时，平面点云会变得更“bumpy”，深度图噪声增加，严重时会出现 0 值和孔洞；也建议用有纹理的平面目标评估。([RealSense][3])
所以可以先用 RealSense Viewer 看自带 depth 是否正常。如果 Viewer 里的 depth/point cloud 都差，那不是 FastFoundationStereo 的问题，是相机设置、曝光、标定或环境问题。


updates:
能查到，但结论是：**直接“FastFoundationStereo + RealSense + 重建”的公开问题还不多**，因为 Fast-FoundationStereo 很新；但能找到三类很相关的公开经验：官方 Fast/FoundationStereo 说明、一个 Fast + RealSense D415 实时点云 demo、以及 Isaac ROS / RealSense 社区里大量“IR 双目 + 重建/SLAM”的坑。

## 我查到的关键问题和对应解决方法

| 问题                                            | 别人遇到的现象                                                                                                                                        | 解决/排查方法                                                                                                                                                 |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **RealSense IR 是灰度图，不是 RGB**                  | 有人在 FoundationStereo issue 里专门问“RealSense IR 灰度图怎么喂模型”。官方 README 也说 FoundationStereo 最适合 RGB stereo，但也测试过 RealSense D4xx 的 mono/IR stereo，能工作。 | 实用做法是把 `Y8 / mono8` 复制成 3 通道再走模型输入；不要用 RGB 单目图拼 IR。([GitHub][1])                                                                                        |
| **IR 输入质量不如 RGB 双目**                          | NVIDIA 论坛里有人用 RealSense D455 的 infra pair 跑 Bi3D，效果明显不如 RGB stereo；NVIDIA 回复指出模型训练在 `rgb8` 彩色图上，`mono8` 灰度图性能会下降。                              | 这说明 IR stereo 能跑，但模型域差真实存在。Fast/FoundationStereo 可能比 Bi3D 更泛化，但 IR speckle、灰度纹理、曝光仍然会影响。([NVIDIA Developer Forums][2])                                  |
| **左右图没有按模型要求 rectified/undistorted**          | FoundationStereo/FastFoundationStereo 官方都强调输入左右图必须已校正、去畸变、极线水平，不能左右反。                                                                          | RealSense 的 `/infra1/image_rect_raw` 和 `/infra2/image_rect_raw` 才是你应该用的；如果直接从 raw IR 自己处理，要确认同名点 y 坐标差接近 0。([GitHub][3])                                |
| **内参/基线文件写错**                                 | FoundationStereo issue 里有人问 `K.txt` 里的 `0.063` 是什么；官方 README 明确说 intrinsic 文件第一行是 3×3 内参，第二行是 baseline，单位米。                                    | 对 RealSense 要用左 IR 相机的 `fx, fy, cx, cy`，baseline 用 left IR 到 right IR 的外参平移。不要用 RGB 相机内参去投 IR 深度。([GitHub][4])                                          |
| **resize 后没同步改 K**                            | Fast 官方建议可以用 `--scale 0.5` 降分辨率，但这会改变像素焦距。                                                                                                     | 如果自己写 pipeline，resize 之后必须同步缩放 `fx, fy, cx, cy`。Fast 官方 demo 通过参数处理 scale；你自己重建时别直接拿原始 K。([GitHub][5])                                                  |
| **近距离物体视差超过 max_disp**                        | Fast 官方参数里 `MAX_DISP=192`，并说明如果非常近的物体，比如 `<0.1m`，要增大 `max_disp`。                                                                               | 你的场景如果有近距离桌面、夹爪、物体边缘，试 `max_disp=256/320`，同时注意显存和速度。([GitHub][5])                                                                                       |
| **IR projector / speckle pattern 影响模型或 SLAM** | RealSense 官方说明投影纹理的点密度、对比度、时间稳定性都会影响深度；激光 speckle 会带来时间噪声。Isaac ROS 也记录过多 RealSense 下 emitter 状态错乱会导致 tracking/reconstruction 差。               | 分别测试 `emitter off / on / 降 laser power`。模型输入给 Fast 时，不一定“打开投影器就更好”；对于弱纹理场景可能更好，对于深度网络域差可能更差。([RealSense][6])                                            |
| **RealSense IR 流拿不到或不稳定**                     | Isaac ROS troubleshooting 里记录了“看不到 IR 图但 depth 能出”“D455 infra fps 被限制到 15fps”“Failed to resolve request”等问题。                                   | 先用 `realsense-viewer` 确认 `infra1/infra2` 都能出；Linux/Jetson 下要检查 firmware、DKMS、librealsense 是否带 CUDA、USB3 带宽、udev rules。([nvidia-isaac-ros.github.io][7]) |
| **重建没有颜色或颜色对不准**                              | 一个 Fast-FoundationStereoPhysics demo 明确支持 Intel RealSense D415：用 IR stereo 估深度，再用 RGB colorization 给点云上色。                                      | 深度坐标是在左 IR 相机坐标系，不是 RGB 坐标系；如果要彩色点云，必须用 RealSense 的 IR-left ↔ RGB 外参重新投影/贴色。([GitHub][8])                                                               |
| **点云噪声、孔洞、边缘假面太多**                            | RealSense 官方后处理文档提到 temporal/spatial/persistence filter 能改善噪声和孔洞，但 hole filling 也可能把背景深度错误填到物体边缘。                                              | 用 Fast 输出重建时也类似：宁可保留洞，也不要把不可信深度融合进 TSDF/mesh；边缘、遮挡、小视差、异常深度要 mask。([RealSense][9])                                                                      |

## 公开案例里最接近你这个方向的

我找到一个第三方 demo：**Fast-FoundationStereoPhysics**，它直接写了支持 **Intel RealSense D415**，脚本是 `ffsd_demos/d415_ffs_realtime.py`，说明是“Uses IR stereo pair + RGB colorization”，并提供实时点云、Open3D 可视化、`VALID_ITERS / MAX_DISP / PCD_STRIDE / ZFAR / ZNEAR` 等参数。这个很接近你的任务，可以参考它的输入、点云过滤和上色方式。([GitHub][8])

它里面也有一个很容易忽略的 note：**部分脚本会把相机图像旋转 180 度**，如果你的相机不是倒装，应该去掉 `cv2.rotate(..., cv2.ROTATE_180)`。这类操作一旦左右图或 K 没同步，点云会明显错。([GitHub][8])

## 我觉得你现在最该重点查这几项

第一优先级：确认你喂的是：

```text
left  = /camera/infra1/image_rect_raw
right = /camera/infra2/image_rect_raw
```

而不是 color、aligned depth、raw infra、或者左右反了。官方 Fast/FoundationStereo 都强调输入要 rectified/undistorted 且不能左右互换。([GitHub][3])

第二优先级：确认你的 `K.txt` 类似这样：

```text
fx 0 cx 0 fy cy 0 0 1
baseline_in_meter
```

注意是**左 IR 相机内参**，不是 RGB 内参。Fast 官方 demo 的 intrinsic 文件就是这个格式，并且 baseline 单位是米。([GitHub][5])

第三优先级：如果你 resize 到模型输入，比如 848×480 → 640×480 或 640×360，一定同步改：

```python
fx *= new_w / old_w
fy *= new_h / old_h
cx *= new_w / old_w
cy *= new_h / old_h
```

否则点云比例和形状都会错。

第四优先级：试三组采集，对比 disparity 和点云：

```text
A: emitter off
B: emitter on
C: emitter on, laser power 降低
```

RealSense 官方文档说明投影纹理能帮助低纹理场景，但点的对比度、密度和时间稳定性都会影响深度质量；对神经网络来说，这个 IR pattern 也可能造成域差。([RealSense][6])

第五优先级：先别直接融合 mesh，先看单帧点云。Fast 官方 demo 有 `--denoise_cloud`、`--zfar`、`--remove_invisible` 等参数；你可以先限制：

```text
z_near = 0.15 或 0.2 m
z_far  = 2~5 m
disp > 0
depth finite
去掉边缘突变区域
统计离群点滤波
```

## 一个很实用的判断

如果 **RealSense Viewer 自带 depth 点云也差**：优先调 RealSense 相机设置、曝光、laser power、calibration、场景纹理。

如果 **RealSense Viewer 自带 depth 好，但 Fast 出来的差**：优先查 Fast 输入预处理：IR 灰度转 3 通道、左右顺序、K/baseline、resize、`max_disp`、emitter 域差。

如果 **Fast 单帧深度还行，但重建差**：问题大概率在后端融合：相机位姿、深度坐标系、RGB 上色外参、无效深度 mask、TSDF 参数。

[1]: https://github.com/NVlabs/FoundationStereo/issues/80 "How to process realsense 's IR image · Issue #80 · NVlabs/FoundationStereo · GitHub"
[2]: https://forums.developer.nvidia.com/t/bi3d-with-realsense-d455-on-agx-orin/289928 "Bi3d with Realsense D455 on AGX Orin - Isaac ROS - NVIDIA Developer Forums"
[3]: https://github.com/NVlabs/FoundationStereo "GitHub - NVlabs/FoundationStereo: [CVPR 2025 Best Paper Nomination] FoundationStereo: Zero-Shot Stereo Matching · GitHub"
[4]: https://github.com/NVlabs/FoundationStereo/issues/26 "camera intrinsics · Issue #26 · NVlabs/FoundationStereo · GitHub"
[5]: https://github.com/NVlabs/Fast-FoundationStereo "GitHub - NVlabs/Fast-FoundationStereo: [CVPR 2026] Fast-FoundationStereo: Real-Time Zero-Shot Stereo Matching · GitHub"
[6]: https://dev.realsenseai.com/docs/projectors/?utm_source=chatgpt.com "Projectors for D400 Series Depth Cameras"
[7]: https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_nvblox/isaac_ros_nvblox/troubleshooting/troubleshooting_nvblox_realsense.html "RealSense Issues — Isaac ROS"
[8]: https://github.com/Vector-Wangel/Fast-FoundationStereoPose "GitHub - Vector-Wangel/Fast-FoundationStereoPhysics: A standlone demo from ManiDreams: Fast-FoundationStereo + SAM2 + Newton, zero-shot real-time simulation-based world model · GitHub"
[9]: https://dev.realsenseai.com/docs/depth-post-processing-for-intel-realsense-depth-camera-d400-series/?utm_source=chatgpt.com "Depth Post-Processing for RealSense™ Depth Camera ..."
