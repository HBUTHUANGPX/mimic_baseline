# Full-Body Viewer Notes

这份文档对应：

- [full_body_mocap_mujoco_viewer.py](/home/hpx/HPX_LOCO_2/mimic_baseline/hdf5_parse/scripts/full_body_mocap_mujoco_viewer.py)

它的目标很单纯：验证原始 `annotation.hdf5` 里的 `full_body_mocap/keypoints`
是不是正常。

## 使用的数据

脚本主要读取这些字段：

- `full_body_mocap/keypoints`
- `full_body_mocap/Ts_world_root`
- `full_body_mocap/frame_nums`
- `video/length_sec`
- `slam/point_cloud`
- `caption`

其中真正驱动骨架显示的是：

- `full_body_mocap/keypoints`

`Ts_world_root` 只用于 root 坐标轴和显示参考，不直接驱动关节链。

## 为什么手也来自 keypoints

当前 viewer 不直接叠加 `hand_mocap/left_joints_3d` 和 `right_joints_3d`。

原因是这两套手部数据与 `full_body_mocap/keypoints` 不在完全相同的显示坐标定义下，
直接叠加容易出现“手漂在身体上方”的错位。

所以当前实现统一使用：

- 身体：`keypoints[:20]`
- 左手：`keypoints[[20, 22:37]]`
- 右手：`keypoints[[21, 37:52]]`

这样整个人体都在一套坐标里。

## 非有限值处理

原始数据里存在少量 `NaN / Inf` 坏帧。viewer 会先做过滤，再在绘制时做二次保护。

如果某帧仍然含有非有限值：

- 该帧几何会被跳过
- 不会把整个 MuJoCo scene 拖坏

## 推荐命令

```bash
conda activate mimic_baseline
python hdf5_parse/scripts/full_body_mocap_mujoco_viewer.py --root-frame --slam-points 300
```

如果这条链看起来不对，问题就在原始 HDF5 数据或最基础的 keypoint 解析层。
