# SMPL-H / SMPL 可视化说明

本文档说明 `annotation.hdf5` 中哪些字段被用于 `SMPL-H` / `SMPL` 可视化，以及 `hdf5_parse/` 下新增脚本的实现方式。

## 相关脚本

- `hdf5_parse/smpl_motion_tools.py`
  负责从 HDF5 读取 `SMPL-H` 运动数据，过滤非有限值帧，并把四元数转成 `smplx` 需要的轴角表示。
- `hdf5_parse/smpl_body_mujoco_viewer.py`
  负责把 `SMPL-H` 或转换后的 `SMPL` motion clip 喂给 `smplx`，再用 MuJoCo 把关节、骨架和少量网格采样点画出来。

## 使用了哪些数据

可视化主链路只依赖 `full_body_mocap` 和少量 `video` 信息：

- `full_body_mocap/Ts_world_root`
  形状是 `[F, 7]`。前 4 维是 root 四元数，顺序为 `wxyz`；后 3 维是 root 平移 `xyz`。
  这里给的是 root 在 `world frame` 下的位姿。
- `full_body_mocap/body_quats`
  形状是 `[F, 21, 4]`。
  这是 `SMPL-H` body 部分 21 个关节的四元数，顺序为 `wxyz`。
  这里按 `local joint rotation` 使用，也就是“相对父关节”的旋转，不是每个 body 在 world 下的绝对姿态。
- `full_body_mocap/left_hand_quats`
  形状是 `[F, 15, 4]`，左手 15 个关节的局部四元数。
- `full_body_mocap/right_hand_quats`
  形状是 `[F, 15, 4]`，右手 15 个关节的局部四元数。
- `full_body_mocap/betas`
  形状是 `[F, 16]`，人体 shape 参数。
  当前实现会按 body model 支持的 `num_betas` 截断使用。
- `full_body_mocap/frame_nums`
  原始帧号，用于和数据集原始时间轴对齐。
- `video/length_sec`
  用来估算播放 `fps`。当前文件中算出来是 `20.0 fps`。

## 姿态是怎么整理的

`smpl_motion_tools.py` 做了下面几件事：

1. 从 `Ts_world_root` 里拆出 root 四元数和 root 平移。
2. 把所有 `wxyz` 四元数转成轴角 `rotvec`。
3. 生成 `SMPLMotionClip`：
   - `global_orient`: `[F, 3]`
   - `body_pose`: `SMPL-H` 时是 `[F, 63]`
   - `left_hand_pose`: `[F, 45]`
   - `right_hand_pose`: `[F, 45]`
   - `transl`: `[F, 3]`
4. 默认过滤包含 `NaN` / `Inf` 的帧，避免 `smplx` 前向和 MuJoCo 绘制时突然炸掉或整帧消失。

这意味着，如果你想直接拿“`SMPL-H` 关节旋转值”，最直接的入口就是：

- `global_orient`
  root 的世界系旋转，轴角表示。
- `body_pose`
  body 21 个局部关节旋转，按 3 维轴角拼接。
- `left_hand_pose`
  左手 15 个局部关节旋转，按 3 维轴角拼接。
- `right_hand_pose`
  右手 15 个局部关节旋转，按 3 维轴角拼接。

## SMPL-H -> SMPL 是怎么转换的

数据源本身是 `SMPL-H`，因为它包含手部关节。

如果切到 `SMPL` 模式，当前实现会：

1. 保留 `global_orient` 和 `transl`。
2. 保留前 21 个 body 关节的轴角。
3. 在 `body_pose` 末尾补两个单位旋转关节，也就是补 6 个零，得到 `[F, 69]`。
4. 丢弃左右手 pose。

这样做的含义是：

- `SMPL-H` 的 body 部分可以稳定映射到 `SMPL`。
- `SMPL` 没有独立手部关节，所以手部自由度会被折叠掉。

## MuJoCo 里是怎么画的

`smpl_body_mujoco_viewer.py` 的流程是：

1. 读取 HDF5，构建 `SMPLMotionClip`。
2. 根据 `--model-type smplh|smpl` 选择 body model。
3. 用 `smplx` 对每一帧做前向，拿到：
   - `vertices`
   - `joints`
   - `parents`
4. 用 MuJoCo `viewer.user_scn` 动态画：
   - 黄色球：关节
   - 青色 capsule：骨架连线
   - 橙色小球：从 body mesh 顶点里均匀采样出的少量表面点
   - 可选 root 坐标轴：红绿蓝三轴

注意：

- 这里没有把完整三角 mesh 注册成 MuJoCo geom，而是用“骨架 + 顶点采样点”来做轻量可视化。
- `SMPL-H` 模式下，手也包含在 `smplx` 输出的 joints / vertices 里，所以不会再出现“只有下面人体，橙色手不见了”的旧逻辑问题。

## 命令行用法

默认查看 `SMPL-H`：

```bash
conda activate mimic_baseline
python hdf5_parse/smpl_body_mujoco_viewer.py
```

查看转换后的 `SMPL`：

```bash
conda activate mimic_baseline
python hdf5_parse/smpl_body_mujoco_viewer.py --model-type smpl
```

只看一段区间，并显示 root 坐标系：

```bash
conda activate mimic_baseline
python hdf5_parse/smpl_body_mujoco_viewer.py --start 0 --end 300 --root-frame
```

循环播放并减少表面采样点：

```bash
conda activate mimic_baseline
python hdf5_parse/smpl_body_mujoco_viewer.py --loop --mesh-points 200
```

## 默认 body model 路径

当前代码会优先使用这些本地文件：

- `SMPL-H`: `/home/hpx/HPX_Loco/loco-mujoco/datasets/smplh/SMPLH_NEUTRAL.pkl`
- `SMPL`: `/home/hpx/HPX_LOCO_2/SOMA-X/assets/SMPL/SMPL_NEUTRAL.npz`

如果你后面下载了新的模型文件，也可以手动指定：

```bash
python hdf5_parse/smpl_body_mujoco_viewer.py --model-type smplh --smplh-model-path /path/to/SMPLH_NEUTRAL.pkl
python hdf5_parse/smpl_body_mujoco_viewer.py --model-type smpl --smpl-model-path /path/to/SMPL_NEUTRAL.npz
```

## 关于非有限值

`annotation.hdf5` 里少量帧存在 `NaN` / `Inf`。当前实现默认直接丢掉这些帧。

原因通常是上游 mocap 拟合在个别帧没有稳定收敛，比如：

- 某些关节被严重遮挡
- 手或身体关键点暂时丢失
- 传感器同步或重建阶段产生了无效解

如果不先过滤，这些帧会导致：

- `smplx` 前向输出异常
- MuJoCo 某一帧整个人消失
- 相机或 scene geom 更新出现跳变

## 当前验证结果

本地已经验证过两条链路都能完成首帧前向：

- `SMPL-H`: `body_pose.shape == (2, 63)`，输出 `vertices.shape == (6890, 3)`，`joints.shape == (73, 3)`
- `SMPL`: `body_pose.shape == (2, 69)`，输出 `vertices.shape == (6890, 3)`，`joints.shape == (45, 3)`

对应测试：

```bash
conda activate mimic_baseline
pytest tests/test_smpl_motion_tools.py tests/test_smpl_body_mujoco_viewer.py -q
```
