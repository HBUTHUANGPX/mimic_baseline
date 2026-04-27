# hdf5_parse

`hdf5_parse` is now a legacy-compatible implementation package. New batch conversion and deployment should use:

- [dataset_converter](../dataset_converter/README.md)
- `dataset-converter-hdf5-batch`
- `python -m dataset_converter.hdf5.cli.batch_export`

说明文档已经整理到：

- [docs/README.md](docs/README.md)

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

批量入口：

```bash
python hdf5_parse/scripts/batch_export_hdf5_motion.py \
  --test-data-root hdf5_parse/test_data \
  --output-root hdf5_parse/out/batch \
  --exports annotation smpl \
  --workers 4
```

`soma-bvh` 批量导出仍然单进程顺序处理，避免 CUDA 多进程抢卡。
