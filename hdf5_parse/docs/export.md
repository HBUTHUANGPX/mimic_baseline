# HDF5 Motion Export

这份文档说明当前 `hdf5_parse` 里的三条导出链：

- `export_hdf5_to_soma_npz.py`
- `export_hdf5_to_soma_bvh.py`
- `export_hdf5_segmented_motion.py`

以及它们现在各自负责什么。

## 当前职责划分

### 1. `export_hdf5_to_soma_npz.py`

这条链现在只负责导出：

- 有效人体帧时间线
- 对齐后的文本描述
- 原始 `caption` 备份

它默认输出：

- `hdf5_parse/out/annotation_soma.npz`

这个文件现在是元数据文件，不再是 human motion 文件。

保存字段：

- `fps`
- `num_frames`
- `timeline_frame_indices`
- `frame_timestamps`
- `source_caption`
- `main_task_texts`
- `sub_task_texts`
- `current_action_texts`
- `interaction_texts`
- `main_task_text_indices`
- `sub_task_text_indices`
- `current_action_text_indices`
- `interaction_text_indices`

它仍然会读取 `full_body_mocap`，但只用于筛出有效人体帧，不再做：

- `SMPL-H body -> SMPL`
- `SMPL -> SOMA`
- `human_* / smpl_* / soma_*` 保存

### 2. `export_hdf5_to_soma_bvh.py`

这条链负责真正的人体动作导出：

`annotation.hdf5 -> SMPL body -> SOMA-X -> SOMA BVH`

默认输出：

- `hdf5_parse/out/annotation_soma.bvh`

### 3. `export_hdf5_segmented_motion.py`

这条链在 `HDF5` 人体帧断续时使用：

- 先过滤无效帧
- 再按 `frame_nums` 连续性切段
- 每段导出一份 `SMPL npz`
- 每段导出一份 `SOMA BVH`

默认输出目录：

- `hdf5_parse/out/smpl`
- `hdf5_parse/out/soma_bvh`

分段 `SMPL npz` 默认保存为 `soma_y_up` 坐标语义，这一点和 `nymeria_parse` 以及后续 `SOMA BVH / soma-retargeter` 链路保持一致。原始 HDF5 body motion 本身更接近 `Z-up`；如果直接保存并拿到同一套 viewer 或下游链路里看，就会出现人体横卧。需要排查原始 HDF5 坐标时，可以显式传入 `--smpl-frame raw`。

## 输入字段

三条链共享的原始 HDF5 字段主要是：

- `full_body_mocap/Ts_world_root`
- `full_body_mocap/body_quats`
- `full_body_mocap/betas`
- `full_body_mocap/frame_nums`
- `video/frame_number`
- `video/device_timestamp`
- `caption`

其中：

- `Ts_world_root` 视为 world frame 下的 root 位姿
- `body_quats` 视为 local joint rotation
- 手部数据明确忽略

## 文本对齐

`annotation_soma.npz` 中的文本对齐规则是：

- 先把有效人体帧映射到 `video/device_timestamp`
- `caption` 里的 `start_frame/end_frame` 视为时间戳区间
- 四类文本独立存池与逐帧索引
- `"UNKNOWN"` 固定为每类文本池的 `index 0`

## BVH / SMPL 导出

真正的人体动作请走 BVH 或分段导出：

```bash
conda activate mimic_baseline

python hdf5_parse/scripts/export_hdf5_to_soma_bvh.py \
  --smpl-model-path /home/hpx/HPX_LOCO_2/SOMA-X/assets/SMPL/SMPL_NEUTRAL.npz

python hdf5_parse/scripts/export_hdf5_segmented_motion.py \
  --smpl-model-path /home/hpx/HPX_LOCO_2/SOMA-X/assets/SMPL/SMPL_NEUTRAL.npz \
  --smpl-frame soma_y_up
```

其中 `--smpl-frame` 支持：

- `soma_y_up`：默认值，保存出来的 `smpl_global_orient/smpl_transl` 与 Nymeria 和 SOMA-BVH 下游语义一致。
- `raw`：保留 HDF5 原始 SMPL root 坐标，不做可视化坐标转换，主要用于诊断。

## human motion npz 的正确来源

如果你后面需要 human motion npz，请不要再从 `annotation_soma.npz` 获取。

正确链路是：

`HDF5 -> SOMA BVH -> soma-retargeter/app/bvh_to_csv_converter.py -> human motion npz`

这个 human motion npz 才适合交给：

- `annotation_soma_mujoco_viewer.py`
- `soma_bvh_diagnostics.py`
- `visualize_hdf5_soma_npz.py`
- `motion_reconstruction --source hdf5-human`

## 代码结构

当前实现位于：

- `hdf5_parse/motion_export/core.py`
  - 文本与时间线元数据导出
- `hdf5_parse/motion_export/smpl_soma.py`
  - 共享的 `HDF5/SMPL/SOMA-X` 运动转换逻辑
- `hdf5_parse/motion_export/bvh.py`
  - BVH 导出
- `hdf5_parse/motion_export/segmented.py`
  - 分段导出
