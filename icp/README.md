# ICP 配准说明

这个目录里的脚本用于把三路相机分割出来的目标点云配准到 STL mesh，并输出调整后的相机外参。现在 ICP 内部优先使用左深度 / rectified-depth 外参（`depth_cam2world_4x4`）；如果输入里没有 depth 外参，才回退到 `cam2world_4x4`。

## 脚本

- `register_to_mesh.py`：执行配准，输出原始 mesh、原始外参、调整后外参和每个相机调整后的点云。
- `visualize_mesh_registration.py`：读取配准输出并显示 mesh 和点云，也可以导出合并后的可视化 PLY。
- `config.yaml`：当前验证过的参数记录。现在脚本以命令行参数为准，不会自动读取这个 YAML。

## 当前推荐命令

Go-ICP 在 `sam3` conda 环境里跑：

```bash
conda run -n sam3 python icp/register_to_mesh.py \
  --data-dir tests/outputs/realsense_live_small_three_cam_fast \
  --mesh icp/Register_small.STL \
  --master-camera cam_00 \
  --use-goicp \
  --refine-use-goicp \
  --num-mesh-points 30000
```

默认输出目录为：

```text
icp/output/<stl文件名不带后缀>/
```

例如 `icp/Register_small.STL` 会输出到：

```text
icp/output/Register_small/
```

## 输出内容

```text
icp/output/Register_small/
├── Register_small.STL
├── original_extrinsics.json
├── refined_extrinsics.json
├── cam_00_registered.ply
├── cam_01_registered.ply
└── cam_02_registered.ply
```

- `Register_small.STL`：原始 mesh 的拷贝。
- `original_extrinsics.json`：从数据里读到的原始外参；如果输入有 `depth_cam2world_4x4`，这里会一起保留。
- `refined_extrinsics.json`：调整后的外参，里面也包含 `world_to_mesh_4x4`；其中 `depth_cam2world_4x4` 是 ICP 实际优化的主外参，`cam2world_4x4` 是按 `depth_to_color_4x4` 反推出来的 RGB 外参。
- `cam_XX_registered.ply`：对应相机用调整后外参变换到世界坐标系的点云。

## 可视化

现在可以直接读 `icp/output` 下的配准输出。传具体结果目录：

```bash
DISPLAY=localhost:10.0 /home/oyx/miniconda3/envs/graspenv/bin/python icp/visualize_mesh_registration.py \
  --output-dir icp/output/Register_small
```

也可以传 `icp/output`，脚本会自动选择里面最近的有效结果目录：

```bash
DISPLAY=localhost:10.0 /home/oyx/miniconda3/envs/graspenv/bin/python icp/visualize_mesh_registration.py \
  --output-dir icp/output
```

无窗口检查距离统计：

```bash
/home/oyx/miniconda3/envs/graspenv/bin/python icp/visualize_mesh_registration.py \
  --output-dir icp/output/Register_small \
  --draw 0
```

导出一个包含 mesh 和三路点云的 PLY：

```bash
/home/oyx/miniconda3/envs/graspenv/bin/python icp/visualize_mesh_registration.py \
  --output-dir icp/output/Register_small \
  --draw 0 \
  --export-ply icp/output/Register_small/mesh_and_registered_points.ply
```

## 当前参数

这次效果较好的组合是主相机和从相机都使用 Go-ICP：

```text
use_goicp=true
refine_use_goicp=true
num_mesh_points=30000
voxel_size=0.003
```

Go-ICP 使用 `GoICPConfig()` 默认值，除非命令行显式传 `--goicp-trim-fraction` 或 `--refine-goicp-trim-fraction`：

```text
voxel_size_ratio=0.01
goicp_max_corr_ratio=0.05
min_voxel_size=0.001
min_goicp_corr=0.005
goicp_module=""
goicp_quiet=true
goicp_dt_size=300
goicp_dt_factor=2.0
goicp_trim_fraction=0.05
goicp_mse_thresh=3e-4
goicp_epsilon=None
rotation_only_output=false
```

从相机的 Go-ICP 是全局匹配，不再使用原始外参推出来的 `predicted_T_MC` 作为先验初值；这个矩阵只保留在日志和结果里方便排查。
