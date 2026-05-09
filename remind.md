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


