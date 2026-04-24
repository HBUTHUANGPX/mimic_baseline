# SMPL Viewer Notes

这份文档对应：

- [smpl_motion_tools.py](/home/hpx/HPX_LOCO_2/mimic_baseline/hdf5_parse/utils/smpl_motion_tools.py)
- [smpl_body_mujoco_viewer.py](/home/hpx/HPX_LOCO_2/mimic_baseline/hdf5_parse/scripts/smpl_body_mujoco_viewer.py)

## 作用

这条链用于把 `annotation.hdf5` 里的 body motion 整理成 `SMPL-H` 或 `SMPL`
可前向的姿态，再用 MuJoCo 轻量查看：

- joints
- bones
- 少量 mesh 采样点

它是“检查 `SMPL-H/SMPL` 解释是否合理”的工具，不是导出链本身。

## 使用的字段

- `full_body_mocap/Ts_world_root`
- `full_body_mocap/body_quats`
- `full_body_mocap/left_hand_quats`
- `full_body_mocap/right_hand_quats`
- `full_body_mocap/betas`
- `full_body_mocap/frame_nums`
- `video/length_sec`

语义上：

- `Ts_world_root` 是 world frame 下的 root 位姿
- `body_quats` / `left_hand_quats` / `right_hand_quats` 按 local joint rotation 使用

## 关键转换

- `SMPL-H body` 原始是 `63` 维 axis-angle
- 转 `SMPL` 时会在末尾补 `6` 个零，变成 `69` 维
- 手部 pose 在 `SMPL` 模式下被丢弃

## 推荐命令

查看 `SMPL-H`：

```bash
conda activate mimic_baseline
python hdf5_parse/scripts/smpl_body_mujoco_viewer.py
```

查看转换后的 `SMPL`：

```bash
conda activate mimic_baseline
python hdf5_parse/scripts/smpl_body_mujoco_viewer.py --model-type smpl
```

如果这条链正常，而 `SOMA BVH` 导出链不正常，问题通常就不在最初的
`SMPL-H/SMPL` 解析层。
