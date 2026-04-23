# hdf5_parse

这个目录现在主要承载两类工作：

- `annotation.hdf5` 的人体动作可视化
- `annotation.hdf5` 的人体动作与文本导出

当前和 `Xperience-10M` 相关的入口脚本如下：

- `smpl_body_mujoco_viewer.py`
  - 读取 `full_body_mocap`
  - 以 `SMPL-H` 或转换后的 `SMPL` 在 MuJoCo 中做轻量骨架可视化
- `export_hdf5_to_soma_npz.py`
  - 读取 `full_body_mocap + video + caption`
  - 调用 `SOMA-X` 的 `SMPL -> SOMA` 求逆
  - 输出 `save_retarget_npz()` 风格的人体 `.npz`
- `visualize_hdf5_soma_npz.py`
  - 不自己维护 viewer
  - 直接调用 `motion_reconstruction` 的包级可视化 API
  - 播放“原始 human skeleton vs human encoder 解码得到的 robot motion”

## 目录里的关键文件

- `hdf5_soma_export.py`
  - HDF5 解析、文本对齐、SOMA 求逆、人体骨架导出主逻辑
- `export_hdf5_to_soma_npz.py`
  - 命令行入口
- `smpl_motion_tools.py`
  - `SMPL-H / SMPL` 可视化用的姿态整理工具
- `smpl_visualization_notes.md`
  - `SMPL-H / SMPL` 可视化说明
- `visualize_hdf5_soma_npz.py`
  - human-only `.npz` 到 `motion_reconstruction` viewer 的薄封装

## 导出目标

`export_hdf5_to_soma_npz.py` 的目标不是导出标准 `SOMA-X save_soma_npz()` 文件，而是导出一个更贴近 `soma-retargeter` 人体侧结构的 `.npz`：

- `human_joint_names`
- `human_parent_indices`
- `human_reference_local_transforms`
- `human_local_transforms`
- `human_global_pos`
- `human_global_quat`

同时会额外保存：

- `timeline_frame_indices`
- `main_task_texts / sub_task_texts / current_action_texts / interaction_texts`
- 四类文本的逐帧索引数组
- `smpl_*` 中间结果
- `soma_*` 中间结果
- `per_vertex_error`

## 使用的数据字段

导出链路只依赖下面这些字段：

- `full_body_mocap/Ts_world_root`
- `full_body_mocap/body_quats`
- `full_body_mocap/betas`
- `full_body_mocap/frame_nums`
- `video/frame_number`
- `video/device_timestamp`
- `caption`

注意：

- 手部数据被明确忽略，不使用 `left_hand_quats/right_hand_quats`
- 只保留有效人体帧，不补齐缺失帧
- `timeline_frame_indices` 直接保存 HDF5 原始 `frame_nums`

## 文本对齐规则

四类文本分别独立维护文本池和逐帧索引：

- `Main Task`
- `Sub Task`
- `Current Action`
- `Interaction`

对齐方式：

- 文本时间基准来自 `video/device_timestamp`
- 有效动作帧先用 `frame_nums -> video/frame_number -> device_timestamp` 映射到时间戳
- `caption` 中的 `start_frame/end_frame` 被当作时间戳区间处理
- 没有文本覆盖的帧统一指向 `"UNKNOWN"`
- `"UNKNOWN"` 固定放在每个文本池的 `index 0`

## SOMA skeleton 筛选规则

导出结果保留整套 `SOMA` skeleton，但逐帧动态值只对以下 body 生效：

- `Hips`
- `motion_reconstruction/configs/dual_fsq.yaml` 里的 `human_body_names`

其他 joint 在逐帧数组中全部清零：

- `human_local_transforms`
- `human_global_pos`
- `human_global_quat`

静态参考姿态不会清零：

- `human_reference_local_transforms`

## 快速开始

环境：

```bash
conda activate mimic_baseline
```

查看命令行参数：

```bash
python hdf5_parse/export_hdf5_to_soma_npz.py --help
```

使用默认输入导出：

```bash
python hdf5_parse/export_hdf5_to_soma_npz.py \
  --smpl-model-path /home/hpx/HPX_LOCO_2/SOMA-X/assets/SMPL/SMPL_NEUTRAL.npz
```

只导出前 300 帧做快速检查：

```bash
python hdf5_parse/export_hdf5_to_soma_npz.py \
  --smpl-model-path /home/hpx/HPX_LOCO_2/SOMA-X/assets/SMPL/SMPL_NEUTRAL.npz \
  --end-frame 300 \
  --batch-size 32
```

默认输入输出：

- 输入：`hdf5_parse/hdf5/annotation.hdf5`
- 输出：`hdf5_parse/out/annotation_soma.npz`

## 复用 motion_reconstruction 可视化

导出后的 `.npz` 可以直接交给 `motion_reconstruction`，不需要在 `hdf5_parse`
里维护第二套推理和 MuJoCo 逻辑。

直接可视化：

```bash
python hdf5_parse/visualize_hdf5_soma_npz.py \
  --config motion_reconstruction/configs/dual_fsq.yaml \
  --checkpoint outputs/motion_reconstruction/test_run/checkpoints/latest.pt \
  --xml-path assets/unitree_g1/g1_29dof_rev_1_0.xml
```

这条链路内部做的是：

- 读取 `hdf5_parse/out/annotation_soma.npz`
- 构建 human-only source
- 只跑 `human encoder -> decoder`
- 在 MuJoCo 中显示“原始 human skeleton vs decoder 输出的 robot motion”

注意：

- 这里的 decoder 输出不是重建后的 human skeleton，而是 robot feature
- 因为 human-only 数据没有 robot 原始轨迹，viewer 会使用 `Hips` 的世界轨迹来摆放解码后的 robot

## 当前实现约束

- 只支持 `cuda`
- 依赖本地 `SOMA-X`
- 依赖可用的 `SMPL_NEUTRAL.npz` 或 `SMPL_NEUTRAL.pkl`
- 当前实现按 `SMPL-H body -> SMPL -> SOMA` 路线工作，不使用手部

## 已验证内容

本地已完成以下验证：

- `pytest tests/test_hdf5_soma_export.py tests/test_hdf5_soma_payload.py -q`
- `python hdf5_parse/export_hdf5_to_soma_npz.py --help`
- 真实烟雾测试：
  - `python hdf5_parse/export_hdf5_to_soma_npz.py --smpl-model-path /home/hpx/HPX_LOCO_2/SOMA-X/assets/SMPL/SMPL_NEUTRAL.npz --end-frame 8 --batch-size 2`
  - 成功输出 `hdf5_parse/out/annotation_soma.npz`

如果你要看字段语义、数组形状和设计细节，请继续看 `docs/hdf5_soma_export.md`。
