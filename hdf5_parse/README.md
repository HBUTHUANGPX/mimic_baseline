# hdf5_parse

这个目录现在主要承载两类工作：

- `annotation.hdf5` 的人体动作可视化
- `annotation.hdf5` 的人体动作与文本导出

当前和 `Xperience-10M` 相关的入口脚本如下：

- `full_body_mocap_mujoco_viewer.py`
  - 直接读取 `annotation.hdf5/full_body_mocap/keypoints`
  - 这是当前用来核对原始 HDF5 人体骨架是否正常的基线查看器
- `smpl_body_mujoco_viewer.py`
  - 读取 `full_body_mocap`
  - 以 `SMPL-H` 或转换后的 `SMPL` 在 MuJoCo 中做轻量骨架可视化
- `annotation_soma_mujoco_viewer.py`
  - 直接读取 `annotation_soma.npz`
  - 严格按 `soma-retargeter/app/play_npz_mujoco.py` 的人体链路
    `human_local_transforms -> FK -> apply_visualization_frame -> draw_animation_frame`
    来画骨架和 joint 坐标轴
- `soma_bvh_diagnostics.py`
  - 读取 `annotation_soma.npz` 和导出的 `SOMA BVH`
  - 严格复用 `soma-retargeter` 官方 BVH 解析链路
  - 比较两边得到的人体 `global pos/global quat` 是否一致
- `export_hdf5_to_soma_npz.py`
  - 读取 `full_body_mocap + video + caption`
  - 调用 `SOMA-X` 的 `SMPL -> SOMA` 求逆
  - 输出 `save_retarget_npz()` 风格的人体 `.npz`
- `export_hdf5_to_soma_bvh.py`
  - 读取 `full_body_mocap`
  - 调用 `SOMA-X` 的 `SMPL -> SOMA` 求逆
  - 输出完整 SOMA skeleton 的 `.bvh`
- `export_hdf5_segmented_motion.py`
  - 读取 `full_body_mocap`
  - 按原始 HDF5 中有效骨骼帧的连续区间切段
  - 每段同时输出一个 `SMPL npz` 和一个 `SOMA BVH`
- `visualize_hdf5_soma_npz.py`
  - 不自己维护 viewer
  - 直接调用 `motion_reconstruction` 的包级可视化 API
  - 播放“原始 human skeleton vs human encoder 解码得到的 robot motion”

## 目录里的关键文件

- `motion_export/`
  - 当前所有 motion 导出实现的统一目录
  - `core.py` 负责 `HDF5 -> SOMA-style npz` 与共享 HDF5/SMPL/SOMA 逻辑
  - `bvh.py` 负责 `HDF5 -> SOMA BVH`
  - `segmented.py` 负责连续有效帧切段后的 `SMPL/BVH` 分段导出
- `export_hdf5_to_soma_npz.py`
  - 命令行入口
- `export_hdf5_to_soma_bvh.py`
  - BVH 导出命令行入口
- `export_hdf5_segmented_motion.py`
  - 分段导出命令行入口
- `annotation_soma_mujoco_viewer.py`
  - `annotation_soma.npz` 的 human-only MuJoCo 骨架/坐标轴播放器
- `soma_bvh_diagnostics.py`
  - `annotation_soma.npz` 与 `SOMA BVH` 对齐诊断脚本
- `smpl_motion_tools.py`
  - `SMPL-H / SMPL` 可视化用的姿态整理工具
- `smpl_visualization_notes.md`
  - `SMPL-H / SMPL` 可视化说明
- `visualize_hdf5_soma_npz.py`
  - human-only `.npz` 到 `motion_reconstruction` viewer 的薄封装

## 导出目标

`export_hdf5_to_soma_npz.py` 的目标不是导出标准 `SOMA-X save_soma_npz()` 文件，而是导出一个更贴近 `soma-retargeter` 人体侧结构的 `.npz`：

- `human_joint_names`
- `human_parent_indices`
- `human_reference_local_transforms`
- `human_local_transforms`
- `human_global_pos`
- `human_global_quat`

同时会额外保存：

- `timeline_frame_indices`
- `main_task_texts / sub_task_texts / current_action_texts / interaction_texts`
- 四类文本的逐帧索引数组
- `smpl_*` 中间结果
- `soma_*` 中间结果
- `per_vertex_error`

如果你想直接得到可以被 `soma-retargeter/assets/bvh.py` 或其他 BVH 工具读取的动作文件，也可以使用：

- `export_hdf5_to_soma_bvh.py`

这条链会保留完整 `SOMA` skeleton，不按 `dual_fsq` 做 joint 清零。
同时会把 `Root/Hips` 的通道语义对齐到官方 SOMA BVH：

- `Root` 的 6 通道保持零姿态
- 用于 pre-visualization frame 的固定 root 旋转会吸收到 `Hips` 运动通道
- 因此回读到 `bvh_to_csv_converter.py` 或 `play_npz_mujoco.py` 时，人体会保持直立，而不是沿 `-Y` 横卧

如果你想处理 `HDF5` 里“骨骼有时消失”的情况，并按有效动作段分文件保存，可以使用：

- `export_hdf5_segmented_motion.py`

这条链会：

- 先过滤掉非有限值人体帧
- 再按原始 `frame_nums` 的连续性切段
- 每段保存一份 `SMPL npz`
- 每段保存一份 `SOMA BVH`
- 文件名使用该段在原始 HDF5 中覆盖的时间戳范围
- 其中每段 `SOMA BVH` 和 `export_hdf5_to_soma_bvh.py` 使用同一套 `Root/Hips` 规范化逻辑

这里的语义约定和 `soma-retargeter` 参考播放器保持一致：

- `human_local_transforms` 保存的是 visualization frame 之前的 local skeleton
- `human_global_pos / human_global_quat` 保存的是
  `human_local_transforms -> FK -> visualization frame`
  之后的结果

导出时还会做一步自动归一化：

- 如果 `SOMA-X` 返回的 local skeleton 主干方向已经是 `Z-up`
- 会自动把 root 旋回到参考播放器期望的 pre-visualization frame

这样 `annotation_soma_mujoco_viewer.py` 和 `soma-retargeter/app/play_npz_mujoco.py`
都只需要执行同一套
`human_local_transforms -> FK -> apply_visualization_frame`
就能看到站立的人体。

这样 `play_npz_mujoco.py / play_npz_newton.py` 和 `motion_reconstruction` 读取当前导出的
`.npz` 时，看到的是同一套人体骨架语义。

## BVH 诊断

如果你怀疑导出的 `SOMA BVH` 在 `Root/Hips` 通道语义上和 `annotation_soma.npz`
不一致，可以直接运行：

```bash
python hdf5_parse/soma_bvh_diagnostics.py \
  --npz hdf5_parse/out/annotation_soma.npz \
  --bvh hdf5_parse/out/soma_bvh/annotation_83581004785937_83582554784896.bvh
```

这个脚本会：

- 用 `soma-retargeter/assets/bvh.py` 官方解析器读取 `.bvh`
- 用 `play_npz_mujoco.py` 同款
  `compute_global_joint_transforms -> apply_visualization_frame`
  计算人体全局骨架
- 和 `.npz` 中的 `human_global_pos / human_global_quat` 逐帧逐关节比较

如果 `quat diff` 很小而 `pos diff` 明显偏大，通常说明问题更像是
`Hips` 平移通道或局部位移语义不一致，而不是单纯的欧拉角显示看起来“像 -90 度”。

## 使用的数据字段

导出链路只依赖下面这些字段：

- `full_body_mocap/Ts_world_root`
- `full_body_mocap/body_quats`
- `full_body_mocap/betas`
- `full_body_mocap/frame_nums`
- `video/frame_number`
- `video/device_timestamp`
- `caption`

注意：

- 手部数据被明确忽略，不使用 `left_hand_quats/right_hand_quats`
- 只保留有效人体帧，不补齐缺失帧
- `timeline_frame_indices` 直接保存 HDF5 原始 `frame_nums`

## 文本对齐规则

四类文本分别独立维护文本池和逐帧索引：

- `Main Task`
- `Sub Task`
- `Current Action`
- `Interaction`

对齐方式：

- 文本时间基准来自 `video/device_timestamp`
- 有效动作帧先用 `frame_nums -> video/frame_number -> device_timestamp` 映射到时间戳
- `caption` 中的 `start_frame/end_frame` 被当作时间戳区间处理
- 没有文本覆盖的帧统一指向 `"UNKNOWN"`
- `"UNKNOWN"` 固定放在每个文本池的 `index 0`

## SOMA skeleton 筛选规则

导出结果保留整套 `SOMA` skeleton，但逐帧动态值只对以下 body 生效：

- `Hips`
- `motion_reconstruction/configs/dual_fsq.yaml` 里的 `human_body_names`

其他 joint 在逐帧数组中全部清零：

- `human_local_transforms`
- `human_global_pos`
- `human_global_quat`

静态参考姿态不会清零：

- `human_reference_local_transforms`

## 快速开始

环境：

```bash
conda activate mimic_baseline
```

查看命令行参数：

```bash
python hdf5_parse/export_hdf5_to_soma_npz.py --help
python hdf5_parse/export_hdf5_to_soma_bvh.py --help
python hdf5_parse/export_hdf5_segmented_motion.py --help
```

使用默认输入导出：

```bash
python hdf5_parse/export_hdf5_to_soma_npz.py \
  --smpl-model-path /home/hpx/HPX_LOCO_2/SOMA-X/assets/SMPL/SMPL_NEUTRAL.npz
```

只导出前 300 帧做快速检查：

```bash
python hdf5_parse/export_hdf5_to_soma_npz.py \
  --smpl-model-path /home/hpx/HPX_LOCO_2/SOMA-X/assets/SMPL/SMPL_NEUTRAL.npz \
  --end-frame 300 \
  --batch-size 32
```

默认输入输出：

- 输入：`hdf5_parse/hdf5/annotation.hdf5`
- 输出：`hdf5_parse/out/annotation_soma.npz`

直接导出 BVH：

```bash
python hdf5_parse/export_hdf5_to_soma_bvh.py \
  --smpl-model-path /home/hpx/HPX_LOCO_2/SOMA-X/assets/SMPL/SMPL_NEUTRAL.npz
```

默认 BVH 输出：

- `hdf5_parse/out/annotation_soma.bvh`

按连续有效动作段导出 `SMPL + SOMA BVH`：

```bash
python hdf5_parse/export_hdf5_segmented_motion.py \
  --smpl-model-path /home/hpx/HPX_LOCO_2/SOMA-X/assets/SMPL/SMPL_NEUTRAL.npz
```

默认分段输出目录：

- `hdf5_parse/out/smpl`
- `hdf5_parse/out/soma_bvh`

文件名示例：

- `annotation_1740000010000_1740000010100.npz`
- `annotation_1740000010000_1740000010100.bvh`

## 用参考播放器语义检查 annotation_soma.npz

如果你想先独立验证 `SMPL -> SOMA -> annotation_soma.npz` 这一步，而不引入
`motion_reconstruction` 的 encoder/decoder 和机器人 XML，可直接使用：

```bash
python hdf5_parse/annotation_soma_mujoco_viewer.py \
  --npz hdf5_parse/out/annotation_soma.npz
```

这条查看链只做两件事：

- 读取 `human_local_transforms / human_parent_indices / human_joint_names / fps`
- 严格按 `soma-retargeter/app/play_npz_mujoco.py` 的同款流程算出人体每个 joint 的
  world-frame 位置与姿态后画骨架和坐标轴

默认会显示每个 joint 的局部坐标轴；如果临时只想看骨架，可以加：

```bash
python hdf5_parse/annotation_soma_mujoco_viewer.py \
  --npz hdf5_parse/out/annotation_soma.npz \
  --hide-axes
```

也就是说，这个查看器的目标是单独回答一个问题：

- `annotation_soma.npz` 的人体骨架语义本身对不对

如果这条链看起来正常，再去看 `visualize_hdf5_soma_npz.py` 里的
human encoder / decoder / robot XML，就能把问题范围收得很小。

## 复用 motion_reconstruction 可视化

导出后的 `.npz` 可以直接交给 `motion_reconstruction`，不需要在 `hdf5_parse`
里维护第二套推理和 MuJoCo 逻辑。

直接可视化：

```bash
python hdf5_parse/visualize_hdf5_soma_npz.py \
  --config motion_reconstruction/configs/dual_fsq.yaml \
  --checkpoint outputs/motion_reconstruction/test_run/checkpoints/latest.pt \
  --xml-path assets/unitree_g1/g1_29dof_rev_1_0.xml
```

这条链路内部做的是：

- 读取 `hdf5_parse/out/annotation_soma.npz`
- 构建 human-only source
- 只跑 `human encoder -> decoder`
- 在 MuJoCo 中显示“原始 human skeleton vs decoder 输出的 robot motion”

注意：

- 这里的 decoder 输出不是重建后的 human skeleton，而是 robot feature
- 因为 human-only 数据没有 robot 原始轨迹，viewer 会使用 `Hips` 的世界轨迹来摆放解码后的 robot
- `pair=human` 下机器人直接使用 `--xml-path` 指定的 MuJoCo XML 作为主模型显示
- decoder 输出的关节角直接写入 `qpos[7:]`，anchor 的世界位姿只用来反解 XML 根节点的 world 姿态
- `motion_reconstruction` 读取这类 human-only `.npz` 时，会优先走
  `human_local_transforms + human_parent_indices -> FK -> visualization frame`
  这条参考播放器同款链路；只有缺少 local transforms 时，才回退到
  文件里显式保存的 `human_global_pos / human_global_quat`

## 当前实现约束

- 只支持 `cuda`
- 依赖本地 `SOMA-X`
- 依赖可用的 `SMPL_NEUTRAL.npz` 或 `SMPL_NEUTRAL.pkl`
- 当前实现按 `SMPL-H body -> SMPL -> SOMA` 路线工作，不使用手部

## 已验证内容

本地已完成以下验证：

- `pytest tests/test_hdf5_soma_export.py tests/test_hdf5_soma_payload.py -q`
- `python hdf5_parse/export_hdf5_to_soma_npz.py --help`
- `python hdf5_parse/export_hdf5_to_soma_bvh.py --help`
- `python hdf5_parse/export_hdf5_segmented_motion.py --help`
- 真实烟雾测试：
  - `python hdf5_parse/export_hdf5_to_soma_npz.py --smpl-model-path /home/hpx/HPX_LOCO_2/SOMA-X/assets/SMPL/SMPL_NEUTRAL.npz --end-frame 8 --batch-size 2`
  - 成功输出 `hdf5_parse/out/annotation_soma.npz`
  - `python hdf5_parse/export_hdf5_to_soma_bvh.py --smpl-model-path /home/hpx/HPX_LOCO_2/SOMA-X/assets/SMPL/SMPL_NEUTRAL.npz --end-frame 8 --batch-size 2`
  - 成功输出 `hdf5_parse/out/annotation_soma.bvh`
  - `python hdf5_parse/export_hdf5_segmented_motion.py --smpl-model-path /home/hpx/HPX_LOCO_2/SOMA-X/assets/SMPL/SMPL_NEUTRAL.npz --end-frame 32 --batch-size 4`
  - 成功输出 `hdf5_parse/out/smpl/*.npz` 和 `hdf5_parse/out/soma_bvh/*.bvh`

如果你要看字段语义、数组形状和设计细节，请继续看 `docs/hdf5_soma_export.md`。
