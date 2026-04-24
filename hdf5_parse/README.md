# hdf5_parse

这个目录现在承载三类工作：

- 原始 `annotation.hdf5/full_body_mocap` 的可视化与校验
- `HDF5 -> SOMA BVH / SMPL` 的动作导出
- `HDF5 -> annotation_soma.npz` 的文本与时间线元数据导出

## 入口脚本

- `full_body_mocap_mujoco_viewer.py`
  - 直接看原始 HDF5 骨架
- `smpl_body_mujoco_viewer.py`
  - 以 `SMPL-H / SMPL` 轻量查看原始 body motion
- `export_hdf5_to_soma_npz.py`
  - 只导出有效人体帧时间线和文本标注
  - 默认输出 `hdf5_parse/out/annotation_soma.npz`
- `export_hdf5_to_soma_bvh.py`
  - 导出完整 `SOMA BVH`
  - 默认输出 `hdf5_parse/out/annotation_soma.bvh`
- `export_hdf5_segmented_motion.py`
  - 按有效骨骼连续区间切段
  - 每段输出一个 `SMPL npz` 和一个 `SOMA BVH`
- `annotation_soma_mujoco_viewer.py`
  - 可视化 human motion npz
  - 这个 npz 通常来自 `SOMA BVH -> soma-retargeter/app/bvh_to_csv_converter.py`
- `soma_bvh_diagnostics.py`
  - 对比 `SOMA BVH` 与 human motion npz 的人体语义是否一致
- `visualize_hdf5_soma_npz.py`
  - 复用 `motion_reconstruction` 的 human-only 推理/可视化链
  - 输入也应该是 human motion npz，而不是 `annotation_soma.npz`

## motion_export 结构

`motion_export/` 现在按职责拆成两层：

- `core.py`
  - 负责 `annotation_soma.npz` 的文本与时间线导出
  - 不再执行 `SMPL -> SOMA`，也不保存任何人体骨架字段
- `smpl_soma.py`
  - 负责共享的 `HDF5/SMPL/SOMA-X` 动作解析与转换
- `bvh.py`
  - 负责 `HDF5 -> SOMA BVH`
- `segmented.py`
  - 负责连续有效帧切段后的 `SMPL/BVH` 导出

这样现在的边界是明确的：

- `annotation_soma.npz` 是元数据文件
- `SOMA BVH` / 分段 `SMPL npz` 才是动作文件

## annotation_soma.npz 现在保存什么

`export_hdf5_to_soma_npz.py` 现在只保留这些内容：

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

这里仍然会读取 `full_body_mocap`，但用途只有一个：

- 用骨骼数据是否有限来筛出“有效人体帧”

它不会再：

- 运行 `SMPL -> SOMA`
- 保存 `human_*`
- 保存 `smpl_*`
- 保存 `soma_*`

## 文本对齐规则

- 有效人体帧先通过
  `frame_nums -> video/frame_number -> video/device_timestamp`
  映射到时间戳
- `caption` 里的 `start_frame/end_frame` 按时间戳区间处理
- 四类文本都采用“文本池 + 逐帧索引”
- `"UNKNOWN"` 固定在每个文本池的 `index 0`

## 动作导出

如果你要导出真正的人体 motion，请使用下面两条链之一。

### 单文件 BVH

```bash
conda activate mimic_baseline
python hdf5_parse/export_hdf5_to_soma_bvh.py \
  --smpl-model-path /home/hpx/HPX_LOCO_2/SOMA-X/assets/SMPL/SMPL_NEUTRAL.npz
```

输出：

- `hdf5_parse/out/annotation_soma.bvh`

### 分段导出

```bash
conda activate mimic_baseline
python hdf5_parse/export_hdf5_segmented_motion.py \
  --smpl-model-path /home/hpx/HPX_LOCO_2/SOMA-X/assets/SMPL/SMPL_NEUTRAL.npz
```

输出：

- `hdf5_parse/out/smpl/*.npz`
- `hdf5_parse/out/soma_bvh/*.bvh`

文件名用该段在原始 HDF5 中覆盖的时间戳范围命名。

## human motion npz 的来源

如果你后面要：

- 用 MuJoCo 看 human skeleton
- 走 `motion_reconstruction` 的 `source=hdf5-human`
- 做 BVH 与 npz 的一致性对照

那么请先走这条链：

`HDF5 -> SOMA BVH -> soma-retargeter/app/bvh_to_csv_converter.py -> human motion npz`

也就是说，这几个脚本期待的输入不是 `annotation_soma.npz`，而是由
`bvh_to_csv_converter.py` 生成的 npz：

- `annotation_soma_mujoco_viewer.py`
- `soma_bvh_diagnostics.py`
- `visualize_hdf5_soma_npz.py`

## 常用命令

导出文本与时间线：

```bash
python hdf5_parse/export_hdf5_to_soma_npz.py
```

导出 SOMA BVH：

```bash
python hdf5_parse/export_hdf5_to_soma_bvh.py \
  --smpl-model-path /home/hpx/HPX_LOCO_2/SOMA-X/assets/SMPL/SMPL_NEUTRAL.npz
```

导出分段 `SMPL + SOMA BVH`：

```bash
python hdf5_parse/export_hdf5_segmented_motion.py \
  --smpl-model-path /home/hpx/HPX_LOCO_2/SOMA-X/assets/SMPL/SMPL_NEUTRAL.npz
```

查看 human motion npz：

```bash
python hdf5_parse/annotation_soma_mujoco_viewer.py \
  --npz path/to/human_motion.npz
```
此处的npz是使用soma retargeter重映射后得到含有"human_global_pos","human_global_quat","fps","human_joint_names"等
例如：
```bash
python hdf5_parse/annotation_soma_mujoco_viewer.py \
  --npz hdf5_parse/out/soma_bvh_export/annotation_83581004785937_83595804786069.npz
```

对比 BVH 与 human motion npz：

```bash
python hdf5_parse/soma_bvh_diagnostics.py \
  --npz path/to/human_motion.npz \
  --bvh hdf5_parse/out/annotation_soma.bvh
```
例如：
```bash
python hdf5_parse/soma_bvh_diagnostics.py  \
 --npz hdf5_parse/out/soma_bvh_export/annotation_83581004785937_83595804786069.npz \
 --bvh hdf5_parse/out/soma_bvh/annotation_83581004785937_83595804786069.bvh
```
把 human motion npz 接到 `motion_reconstruction`：

```bash
python hdf5_parse/visualize_hdf5_soma_npz.py \
  --config motion_reconstruction/configs/dual_fsq.yaml \
  --checkpoint outputs/motion_reconstruction/test_run/checkpoints/latest.pt \
  --xml-path assets/unitree_g1/g1_29dof_rev_1_0.xml \
  --motion-npz path/to/human_motion.npz
```
