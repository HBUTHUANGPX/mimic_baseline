# `annotation.hdf5` 可视化说明

这份说明对应 [full_body_mocap_mujoco_viewer.py](/home/hpx/HPX_LOCO_2/mimic_baseline/hdf5_parse/full_body_mocap_mujoco_viewer.py)，目标是讲清楚两件事：

- 这个脚本实际读取了哪些数据
- 这些数据是怎样被组织成 MuJoCo 里的身体骨架、手部和附加可视化的

## 1. 文件与入口

- 数据文件：`hdf5_parse/hdf5/annotation.hdf5`
- 可视化脚本：`hdf5_parse/full_body_mocap_mujoco_viewer.py`
- 说明文档：`hdf5_parse/visualization_notes.md`

推荐运行方式：

```bash
source /home/hpx/miniconda3/etc/profile.d/conda.sh
conda activate mimic_baseline
python hdf5_parse/full_body_mocap_mujoco_viewer.py --root-frame --slam-points 300
```

## 2. 实际使用的数据字段

脚本目前是一个 “keypoint / skeleton viewer”，不是完整的人体动力学模型。因此它只直接使用了少数几类字段。

### 主体数据

- `full_body_mocap/keypoints`
  - shape: `(5858, 52, 3)`
  - 作用：整套可视化的核心输入。每帧 52 个三维关键点。

- `full_body_mocap/Ts_world_root`
  - shape: `(5858, 7)`
  - 作用：提供 root 的世界位姿，用于 root 坐标轴显示和相机参考。
  - 解析方式：`[qw, qx, qy, qz, tx, ty, tz]`

- `video/length_sec`
  - 值：`292.9`
  - 作用：计算播放 FPS。当前脚本得到 `5858 / 292.9 = 20.0 FPS`。

### 辅助数据

- `slam/point_cloud`
  - shape: `(6665, 3)`
  - 作用：在 `--slam-points N` 打开时，作为静态点云叠加显示。

- `caption`
  - 作用：终端里打印一行摘要，帮助确认片段语义，不参与绘制。

### 当前会读取但不直接绘制的字段

- `full_body_mocap/body_quats`
- `full_body_mocap/left_hand_quats`
- `full_body_mocap/right_hand_quats`
- `full_body_mocap/contacts`
- `full_body_mocap/frame_nums`

这些字段目前主要用于保留原始注释和后续扩展，比如把 `body_quats` 驱动到真正的 MuJoCo articulation。

## 3. 为什么主体使用 `full_body_mocap/keypoints`

一开始尝试过同时使用：

- `full_body_mocap/keypoints`
- `hand_mocap/left_joints_3d`
- `hand_mocap/right_joints_3d`

但实际检查后发现，`hand_mocap/*_joints_3d` 和 `full_body_mocap/keypoints` 不在同一显示坐标定义下，直接叠加会出现：

- 手漂在身体上方
- 手和身体明显错位

所以当前实现统一采用 `full_body_mocap/keypoints` 这一套空间定义：

- 身体来自 `keypoints`
- 手也从 `keypoints` 中切分出来

这样可以保证整个人体都在同一套坐标系里显示。

## 4. 当前骨架是怎么拆分的

`full_body_mocap/keypoints` 一共有 52 个点。脚本把它拆成两层显示：

### Body 层

- body 只显示前 20 个“身体本体”关键点
- 不再显示手指链条

对应常量：

- `BODY_VISUAL_KEYPOINT_INDICES = [0..19]`

这样做的目的，是避免 body 层把手部再次盖住，让橙色手可以单独看清。

### Hand 层

手部不是从 `hand_mocap` 读，而是从 `full_body_mocap/keypoints` 里切出来：

- 左手：`[20, 22:37]`
- 右手：`[21, 37:52]`

对应常量：

- `LEFT_HAND_KEYPOINT_INDICES`
- `RIGHT_HAND_KEYPOINT_INDICES`

这意味着：

- 手腕位置和身体天然对齐
- 手指链条和 body 属于同一套 keypoints
- MuJoCo 里看到的橙色手，不是另一份独立 MANO 世界坐标，而是 body keypoints 里的手部分支

## 5. 骨骼连线是怎么构建的

### 身体骨架

脚本使用官方 parent indices 常量：

- `SMPL_H_BODY_PARENT_INDICES`

流程是：

1. 先根据 parent indices 得到 body 全部连接关系
2. 再过滤掉 hand finger chains
3. 保留“身体本体”的骨骼线段

对应函数：

- `build_segment_index_pairs()`
- `build_body_visual_bone_segments()`

### 手部骨架

手部连接使用：

- `SMPLH_HAND_PARENT_INDICES`

对应函数：

- `build_visual_hand_bone_segments()`

手部骨骼会以单独的橙色 / 浅橙色层绘制，因此现在能和 body 清晰区分。

## 6. MuJoCo 场景是怎么构造的

脚本没有导入复杂的人体 XML，而是创建了一个最小场景：

- 一个平面地面
- 两盏灯
- 一个自由相机

对应函数：

- `make_viewer_model()`

每一帧的显示流程在 `populate_scene()` 中完成：

1. 读取当前帧 `clip.keypoints[frame_idx]`
2. 统一应用显示偏移 `display_offset`
3. 绘制 body 关键点与 body 骨骼
4. 如果 `--hands` 开启，绘制左右手关键点与手部骨骼
5. 如果 `--root-frame` 开启，绘制 root 坐标轴
6. 如果 `--slam-points N` 开启，绘制抽样后的点云

当前颜色约定：

- 身体关键点：黄色球
- 身体骨骼：青色 capsule
- 手部关键点：橙色球
- 手部骨骼：浅橙色 capsule

## 7. 为什么需要 `display_offset`

原始 `full_body_mocap/keypoints` 的 `z` 值整体在 MuJoCo 地板下方。直接绘制时，人会像是埋在平面里。

因此脚本会计算一个仅用于显示的偏移量：

1. 取第一帧 root 的 `x, y`
2. 找到整段 keypoints 的最小 `z`
3. 把整段 motion 一起抬到地板之上

对应函数：

- `compute_display_offset()`
- `offset_points()`

注意：

- 这个偏移只影响显示
- 不会改写原始 HDF5 文件

## 8. `NaN` / 非有限值是怎么处理的

这份数据里有一些连续坏帧，里面会出现 `NaN` 或其它非有限值。如果不处理，MuJoCo 里会出现：

- 骨架突然消失
- 部分球点或线段绘制失败

当前分两层保护：

### 加载阶段

对应函数：

- `frame_is_finite()`
- `filter_valid_frames()`
- `apply_frame_valid_mask()`

逻辑是：

1. 对每个逐帧数组执行 `np.isfinite(...)`
2. 求联合有效 mask
3. 仅保留所有关键数组都有限的帧

### 绘制阶段

对应函数：

- `draw_sphere()`
- `draw_line()`
- `draw_axes()`

如果某个点、线段端点或位姿仍然不是有限值，该几何会被直接跳过，避免把整帧场景拖坏。

## 9. 命令行参数

当前 CLI 只保留最常用的参数：

- `--hdf5-path`
  - 指定输入 HDF5 文件
- `--start`
  - 起始帧
- `--end`
  - 结束帧，`-1` 表示读到末尾
- `--stride`
  - 每隔多少帧采样一次
- `--loop`
  - 是否循环播放
- `--hands` / `--no-hands`
  - 是否显示手部，默认开启
- `--root-frame`
  - 是否显示 root 坐标轴
- `--slam-points N`
  - 显示最多 `N` 个 SLAM 点，`0` 表示关闭

## 10. 当前实现刻意没有做的事

这个脚本目前专注于“看清数据”，没有尝试一步做到完整人体仿真。它没有做：

- 用 `body_quats` 驱动一个真实关节链的 MuJoCo humanoid
- 把 `contacts` 画成接触提示
- 使用 `cpf` 做额外坐标系显示
- 直接把 `hand_mocap/*` 作为显示输入

如果后面要继续扩展，比较自然的方向是：

- 把 `body_quats + Ts_world_root` 映射到一个可运动的人形模型
- 为坏帧做插值，而不是简单过滤
- 把 `contacts` 画成脚底 / 手掌接触标记

## 11. 参考

- Xperience-10M 数据集：<https://huggingface.co/datasets/ropedia-ai/xperience-10m>
- HOMIE-toolkit：<https://github.com/Ropedia/HOMIE-toolkit>
- 骨架常量参考：<https://raw.githubusercontent.com/Ropedia/HOMIE-toolkit/main/utils/constants_utils.py>
