# Motion Reconstruction Human-Only Visualization

这份文档解释 human-only 可视化链该怎么接。

## 关键变化

`hdf5_parse/out/annotation_soma.npz` 现在只保存：

- 有效人体帧时间线
- 文本池与逐帧文本索引
- 原始 `caption`

它不再保存任何 `human_*` 骨架字段，因此不能再直接喂给：

- `motion_reconstruction --source hdf5-human`
- `hdf5_parse/scripts/visualize_hdf5_soma_npz.py`

## 正确数据来源

如果你要跑 human-only 可视化，正确链路是：

`annotation.hdf5 -> export_hdf5_to_soma_bvh.py -> bvh_to_csv_converter.py -> human motion npz`

也就是说，`visualize_hdf5_soma_npz.py` 现在接收的是：

- `bvh_to_csv_converter.py` 生成的 human motion npz

而不是：

- `hdf5_parse/out/annotation_soma.npz`

## 可视化链

1. `hdf5_parse/scripts/export_hdf5_to_soma_bvh.py`
   - 从 HDF5 导出 `SOMA BVH`
2. `soma-retargeter/app/bvh_to_csv_converter.py`
   - 从 `SOMA BVH` 生成 human motion npz
3. `motion_reconstruction`
   - 读取这个 human motion npz
   - 构建 human feature
   - 只跑 `human encoder -> decoder`
   - 播放“原始 human skeleton vs decoder 输出的 robot motion”

## 示例命令

先生成 `SOMA BVH`：

```bash
python hdf5_parse/scripts/export_hdf5_to_soma_bvh.py \
  --smpl-model-path /home/hpx/HPX_LOCO_2/SOMA-X/assets/SMPL/SMPL_NEUTRAL.npz
```

再把 BVH 转成 human motion npz：

```bash
python soma-retargeter/app/bvh_to_csv_converter.py \
  ...
```

最后可视化：

```bash
python hdf5_parse/scripts/visualize_hdf5_soma_npz.py \
  --config motion_reconstruction/configs/dual_fsq.yaml \
  --checkpoint outputs/motion_reconstruction/test_run/checkpoints/latest.pt \
  --xml-path assets/unitree_g1/g1_29dof_rev_1_0.xml \
  --motion-npz path/to/human_motion.npz
```

或者直接用 `motion_reconstruction` CLI：

```bash
python3 -m motion_reconstruction.cli.visualize \
  --config motion_reconstruction/configs/dual_fsq.yaml \
  --checkpoint outputs/motion_reconstruction/test_run/checkpoints/latest.pt \
  --xml-path assets/unitree_g1/g1_29dof_rev_1_0.xml \
  --source hdf5-human \
  --motion-npz path/to/human_motion.npz \
  --inference-path human \
  --pair human
```

## 语义说明

- `human-only` 模式下，输入是 human feature
- `human encoder -> decoder` 的输出仍然是 robot motion
- `pair=human` 显示的是“原始 human skeleton vs decoder 输出的 robot”
