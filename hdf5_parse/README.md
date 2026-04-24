# hdf5_parse

说明文档已经整理到：

- [docs/README.md](/home/hpx/HPX_LOCO_2/mimic_baseline/hdf5_parse/docs/README.md)

当前目录职责：

- `motion_export/`
  - `HDF5 -> annotation_soma.npz` 元数据导出
  - `HDF5 -> SOMA BVH / segmented SMPL` 动作导出
- `utils/human_motion.py`
  - human motion npz 的解析与 FK / visualization-frame 工具
- `utils/smpl_motion_tools.py`
  - `SMPL-H / SMPL` 轻量解析工具
- `scripts/`
  - 所有可执行入口
- `docs/`
  - 本目录相关文档
