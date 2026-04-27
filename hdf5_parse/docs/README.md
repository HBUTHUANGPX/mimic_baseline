# hdf5_parse Docs

`hdf5_parse/` 现在按“库模块”和“可执行脚本”分开：

- `hdf5_parse/motion_export/`
  - 负责元数据、`SOMA BVH`、分段 `SMPL/BVH` 导出
- `hdf5_parse/utils/human_motion.py`
  - 负责 human motion npz 解析和 FK 工具
- `hdf5_parse/utils/smpl_motion_tools.py`
  - 负责 `SMPL-H / SMPL` 轻量解析工具
- `hdf5_parse/scripts/`
  - 放所有命令行入口
- `hdf5_parse/docs/`
  - 放说明文档

## 脚本索引

- `scripts/full_body_mocap_mujoco_viewer.py`
  - 直接看原始 HDF5 keypoints
- `scripts/smpl_body_mujoco_viewer.py`
  - 以 `SMPL-H / SMPL` 轻量查看原始 body motion
- `scripts/export_hdf5_to_soma_npz.py`
  - 只导出文本与时间线元数据
- `scripts/export_hdf5_to_soma_bvh.py`
  - 导出完整 `SOMA BVH`
- `scripts/export_hdf5_segmented_motion.py`
  - 导出分段 `SMPL npz + SOMA BVH`
- `scripts/batch_export_hdf5_motion.py`
  - 从 `test_data/<subset>/<ep>/annotation.hdf5` 自动批量导出
- `scripts/annotation_soma_mujoco_viewer.py`
  - 查看 human motion npz
- `scripts/soma_bvh_diagnostics.py`
  - 对比 `SOMA BVH` 与 human motion npz 是否一致
- `scripts/visualize_hdf5_soma_npz.py`
  - 把 human motion npz 接到 `motion_reconstruction`

## 文件边界

### `annotation_soma.npz`

现在只保存：

- `fps`
- `num_frames`
- `timeline_frame_indices`
- `frame_timestamps`
- `source_caption`
- 四类文本池与逐帧索引

它是元数据文件，不再保存人体骨架。

### human motion npz

如果你要：

- 看人体骨架
- 做 BVH/npz 一致性诊断
- 接到 `motion_reconstruction --source hdf5-human`

那应该使用：

`HDF5 -> SOMA BVH -> soma-retargeter/app/bvh_to_csv_converter.py -> human motion npz`

### segmented SMPL npz

`scripts/export_hdf5_segmented_motion.py` 同时会保存每段动作的 `SMPL npz`。这里默认使用 `--smpl-frame soma_y_up`，让 HDF5 导出的 SMPL root 坐标和 Nymeria、SOMA-BVH 下游保持同一套 `Y-up` 语义，避免直接查看时人体横卧。

如果需要保留 HDF5 原始坐标用于诊断，可以加 `--smpl-frame raw`。

## 常用命令

导出文本与时间线：

```bash
python hdf5_parse/scripts/export_hdf5_to_soma_npz.py
```

导出 `SOMA BVH`：

```bash
python hdf5_parse/scripts/export_hdf5_to_soma_bvh.py \
  --smpl-model-path /home/hpx/HPX_LOCO_2/SOMA-X/assets/SMPL/SMPL_NEUTRAL.npz
```

导出分段 `SMPL + SOMA BVH`：

```bash
python hdf5_parse/scripts/export_hdf5_segmented_motion.py \
  --smpl-model-path /home/hpx/HPX_LOCO_2/SOMA-X/assets/SMPL/SMPL_NEUTRAL.npz \
  --exports smpl soma-bvh \
  --smpl-frame soma_y_up
```

只导出分段 `SMPL npz`，不运行 SOMA-X：

```bash
python hdf5_parse/scripts/export_hdf5_segmented_motion.py \
  --exports smpl \
  --smpl-frame soma_y_up
```

批量导出 `test_data` 下全部 HDF5 episode 的文本与 SMPL：

```bash
python hdf5_parse/scripts/batch_export_hdf5_motion.py \
  --test-data-root hdf5_parse/test_data \
  --output-root hdf5_parse/out/batch \
  --exports annotation smpl \
  --workers 4 \
  --skip-existing \
  --summary-path hdf5_parse/out/batch/summary.jsonl
```

批量导出 SOMA BVH 时仍然是单进程顺序处理，即使传入 `--workers` 也不会并行抢 CUDA：

```bash
python hdf5_parse/scripts/batch_export_hdf5_motion.py \
  --exports soma-bvh \
  --smpl-model-path /home/hpx/HPX_LOCO_2/SOMA-X/assets/SMPL/SMPL_NEUTRAL.npz \
  --batch-size 128 \
  --skip-existing
```

看原始 HDF5 keypoints：

```bash
python hdf5_parse/scripts/full_body_mocap_mujoco_viewer.py --root-frame --slam-points 3000
```
使用soma-retargeter重映射，并将需要的数据保存为npz：
```bash
python soma-retargeter/app/bvh_to_csv_converter.py --config hdf5_parse/default_bvh_to_csv_converter_config.json
```
看使用soma-retargeter重映射保存的 human motion npz：

```bash
python hdf5_parse/scripts/annotation_soma_mujoco_viewer.py \
  --npz hdf5_parse/out/soma_bvh_export/annotation_83581004785937_83595804786069.npz
```

诊断 BVH 与 human motion npz：

```bash
python hdf5_parse/scripts/soma_bvh_diagnostics.py \
  --npz hdf5_parse/out/soma_bvh_export/annotation_83581004785937_83595804786069.npz \
  --bvh hdf5_parse/out/soma_bvh/annotation_83581004785937_83595804786069.bvh
```

接入 `motion_reconstruction`：

```bash
python hdf5_parse/scripts/visualize_hdf5_soma_npz.py \
  --config motion_reconstruction/configs/dual_fsq.yaml \
  --checkpoint outputs/motion_reconstruction/dual_fsq_ddp/checkpoints/epoch_0320.pt \
  --xml-path assets/unitree_g1/g1_29dof_rev_1_0.xml \
  --motion-npz hdf5_parse/out/soma_bvh_export/annotation_83581004785937_83595804786069.npz
```

## 详细文档

- [visualization.md](/home/hpx/HPX_LOCO_2/mimic_baseline/hdf5_parse/docs/visualization_notes.md)
- [smpl.md](/home/hpx/HPX_LOCO_2/mimic_baseline/hdf5_parse/docs/smpl_visualization_notes.md)
- [export.md](/home/hpx/HPX_LOCO_2/mimic_baseline/hdf5_parse/docs/export.md)
- [motion_reconstruction.md](/home/hpx/HPX_LOCO_2/mimic_baseline/hdf5_parse/docs/motion_reconstruction.md)
