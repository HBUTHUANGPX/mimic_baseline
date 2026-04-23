# Motion Reconstruction HDF5 Human-Only Visualization

## Goal

这份文档解释 `hdf5_parse/out/annotation_soma.npz` 如何接入
`motion_reconstruction`，以及为什么 human-only 可视化里仍然会看到 robot。

## Data Path

当前链路分成两步：

1. `hdf5_parse/export_hdf5_to_soma_npz.py`
   - 从 `annotation.hdf5` 提取 `full_body_mocap + video + caption`
   - 走 `SMPL-H body -> SMPL -> SOMA`
   - 导出 human-only `.npz`

2. `motion_reconstruction`
   - 读取这个 human-only `.npz`
   - 构建 human feature
   - 只跑 `human encoder -> shared decoder`
   - 用 MuJoCo 播放原始 human skeleton 与 decoder 输出

## Why It Bypasses RawMotionLoader

`RawMotionLoader` 的设计目标是训练/评估原始 raw motion `.npz`，它要求同时存在：

- `joint_pos / joint_vel`
- `body_pos_w / body_quat_w`
- `human_*`

而 `hdf5_parse` 导出的文件是 human-only 数据，没有任何 `robot_*` 原始字段。
因此这条链路不会经过 `RawMotionLoader`，而是进入推理专用的 source adapter：

- `source=raw`
- `source=hdf5-human`

## What `human encoder -> decoder` Means

当前模型是“双 encoder、单 decoder”结构：

- robot encoder
- human encoder
- shared quantizer
- robot decoder

这里最关键的是：

- decoder 的输出始终是 robot feature

所以当我们在 human-only 模式下运行：

- 输入是 human feature
- latent 来自 human encoder
- 输出仍然是 robot motion

这就是为什么 viewer 里看到的是：

- 原始 human skeleton
- decoder 输出的 robot motion

而不是：

- 原始 human skeleton
- 重建后的 human skeleton

## Robot Anchor Rule In Human-Only Mode

human-only `.npz` 没有 robot 的原始世界轨迹，但 MuJoCo 里摆放 decoder 输出的
robot 仍然需要一个 anchor 世界位置。

当前实现采用：

- 用 human anchor body，默认是 `Hips`，的世界轨迹作为 `robot_anchor_pos_w`

这不是“真实 robot 轨迹”，只是为了让解码出来的 robot motion 能跟随 human 的
整体运动轨迹显示，便于观察：

- 姿态是不是合理
- 节奏是不是对
- 大致位移趋势是不是对

## Supported Commands

直接通过 `motion_reconstruction` CLI：

```bash
python3 -m motion_reconstruction.cli.visualize \
  --config motion_reconstruction/configs/dual_fsq.yaml \
  --checkpoint outputs/motion_reconstruction/test_run/checkpoints/latest.pt \
  --xml-path assets/unitree_g1/g1_29dof_rev_1_0.xml \
  --source hdf5-human \
  --motion-npz hdf5_parse/out/annotation_soma.npz \
  --inference-path human \
  --pair human
```

或者通过 `hdf5_parse` 的薄封装：

```bash
python hdf5_parse/visualize_hdf5_soma_npz.py \
  --config motion_reconstruction/configs/dual_fsq.yaml \
  --checkpoint outputs/motion_reconstruction/test_run/checkpoints/latest.pt \
  --xml-path assets/unitree_g1/g1_29dof_rev_1_0.xml
```

## Current Limits

- `source=hdf5-human` 只支持 `inference-path=human`
- 这种模式下推荐 `pair=human`
- 如果强行请求 `pair=robot` 或 `pair=both`，当前实现会直接报错
- 当前没有把 decoder 输出再映射回 human skeleton
