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

这里有一个语义边界需要特别注意：

- `soma-retargeter` 的参考播放器把 `human_local_transforms` 当作权威输入
- 然后按
  `human_local_transforms -> FK -> visualization frame`
  得到最终显示用的 `human_global_pos / human_global_quat`
- `hdf5-human source` 现在也采用同一条链路
- 只有当文件缺少 `human_local_transforms + human_parent_indices` 时，才回退到显式保存的
  `human_global_pos / human_global_quat`

这样做是为了让 `visualize_hdf5_soma_npz.py` 看到的人体骨架语义，与
`soma-retargeter/app/play_npz_mujoco.py` 和 `play_npz_newton.py` 保持一致。

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

在 `pair=human` 的 MuJoCo 播放里，机器人不是再画一套骨架，而是直接把传入的
`--xml-path` 当成 viewer 主模型。每一帧的更新规则是：

- decoder 输出的关节角直接写到 `qpos[7:]`
- 先在当前关节角下求出 `anchor` 在 `root` 下的相对位姿
- 再结合已知的 `anchor` 世界位姿，反解 `root` 在 world 下的平移和朝向
- 最后 `mj_forward`，让 MuJoCo 用 XML 自带的 body/geom/mesh 直接显示机器人

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
- 当前没有把 decoder 输出再映射回 human skeleton
