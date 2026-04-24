# HDF5 to SOMA Export

本文档说明：

- `hdf5_parse/export_hdf5_to_soma_npz.py`
- `hdf5_parse/export_hdf5_to_soma_bvh.py`
- `hdf5_parse/export_hdf5_segmented_motion.py`
- `hdf5_parse/soma_bvh_diagnostics.py`

是怎么把 `annotation.hdf5` 中的人体动作分别导出成训练友好的 `.npz`、标准 `SOMA BVH`，以及按连续有效动作段切开的 `SMPL + SOMA BVH` 文件集合的。

## 目标

输出一个“人体侧 `save_retarget_npz()` 风格”的数据包，满足下面几点：

- 只保留有效人体帧
- 保留原始 HDF5 时间线索引
- 保留四类文本描述和逐帧文本索引
- 用 `SOMA-X` 把 `SMPL` 风格 body motion 转成 `SOMA skeleton`
- 不输出任何 `robot_*` 字段

同时也支持输出一份 `SOMA BVH`，便于直接和参考 `BVH` 工具链对照。

如果原始 `HDF5` 中人体骨骼会间断消失，还支持按连续有效帧分段导出：

- 每段一份 `SMPL npz`
- 每段一份 `SOMA BVH`

另外提供了一个只读诊断脚本：

- `soma_bvh_diagnostics.py`

它会把导出的 `SOMA BVH` 重新交给 `soma-retargeter` 官方解析器读取，再和
`annotation_soma.npz` 里的人体 `global pos/global quat` 语义做逐帧比较。

## 输入字段

### 动作字段

- `full_body_mocap/Ts_world_root`
  - `[F, 7]`
  - 前四维是 root quaternion，顺序 `wxyz`
  - 后三维是 root translation，顺序 `xyz`
- `full_body_mocap/body_quats`
  - `[F, 21, 4]`
  - 21 个 body joint 的局部四元数，顺序 `wxyz`
- `full_body_mocap/betas`
  - `[F, 16]`
  - body shape 参数
- `full_body_mocap/frame_nums`
  - `[F]`
  - HDF5 原始动作帧索引

### 时间对齐字段

- `video/frame_number`
- `video/device_timestamp`

这里用于建立：

`frame_nums -> device_timestamp`

### 文本字段

- `caption`

`caption` 不是 HDF5 group，而是一个 JSON 字符串。当前实现使用其中这些内容：

- `config["Main Task"]`
- `segments[*]["Sub Task"]`
- `segments[*]["Current Action"]`
- `segments[*]["interaction"]`

## 关键语义

### root 和 body rotation

- `Ts_world_root` 按 world frame 下的 root 位姿使用
- `body_quats` 按 local joint rotation 使用，不是每个 body 在 world frame 下的绝对姿态

### SMPL-H body 到 SMPL

原始 HDF5 更接近 `SMPL-H body`：

- `body_quats` 只覆盖 21 个 body joints
- 忽略手部以后，旋转向量维度是 `63`

但 `SMPL` body 前向需要 `69` 维，所以导出时会：

- 把 `63` 维 body pose 末尾补 `6` 个零
- 得到标准 `SMPL body_pose [F, 69]`

### betas 维度

HDF5 的 `betas` 是 `16` 维，但具体 `SMPL` 模型支持多少维，要看模型文件本身。

当前实现会：

- 根据 `SMPL_NEUTRAL.npz/.pkl` 自动读取可用 `num_betas`
- 如果 HDF5 维度更大，则截断
- 如果 HDF5 维度更小，则补零

这样做是为了同时兼容：

- `SOMA-X` 内部 identity model
- `smplx` 前向

## 转换链路

整体流程如下：

1. 读取 HDF5 动作字段和文本字段
2. 过滤 `NaN / Inf` 帧
3. 生成有效帧的：
   - `smpl_global_orient`
   - `smpl_body_pose`
   - `smpl_transl`
   - `smpl_betas`
4. 调用 `SOMA-X`：
   - `smplx.SMPL(...)`
   - `SOMALayer(...)`
   - `PoseInversion.fit(...)`
5. 恢复每帧 `SOMA` 的局部骨架变换
6. 转成 `save_retarget_npz()` 风格的人体字段
7. 叠加文本池、文本索引和追踪字段
8. 保存为 `npz`

如果走 `BVH` 导出，则第 6 到 8 步变成：

6. 保留完整 `SOMA` skeleton，不做 `dual_fsq` joint mask
7. 将 local transforms 写成 `BVH HIERARCHY + MOTION`
8. 保存为 `annotation_soma.bvh`

如果走分段导出，则会在第 2 步之后额外做一次：

- 按过滤后的 `frame_nums` 连续性切段

然后对每个 segment 分别保存：

- `hdf5_parse/out/smpl/<timestamp_range>.npz`
- `hdf5_parse/out/soma_bvh/<timestamp_range>.bvh`

这里每个 segment 的 `SOMA BVH` 会复用单文件 BVH 导出同一套规范化逻辑：

- `reference_local_transforms` 保持 `SOMA-X` 原始静态 skeleton
- 动态 `local_transforms` 先转到 pre-visualization frame
- 再把固定 root 运动吸收到 `Hips`，让 `Root` 的 6 通道保持为全 0

在第 6 步里，当前实现会额外检查 `SOMA-X` 返回的 local skeleton 主朝向：

- 如果它已经是 `Y-up` 的 pre-visualization 语义，就原样保留
- 如果它已经落在 `Z-up` 的 visualization 语义，就自动把 root 旋回去

这样导出的 `human_local_transforms` 始终满足参考播放器约定：

`human_local_transforms -> FK -> apply_visualization_frame`

之后得到的才是最终在 MuJoCo / Newton 里显示的 world-frame 人体骨架。

## 文本对齐规则

### 文本池

四类文本都使用“文本池 + 逐帧索引”的形式：

- `main_task_texts`
- `sub_task_texts`
- `current_action_texts`
- `interaction_texts`

配套逐帧数组：

- `main_task_text_indices`
- `sub_task_text_indices`
- `current_action_text_indices`
- `interaction_text_indices`

### 去重与默认值

- 每类文本池完全去重
- `"UNKNOWN"` 固定在 `index 0`
- 没命中的帧默认填 `0`

### 覆盖规则

- `Main Task`
  - 全片常量覆盖
- `Sub Task`
  - 按 segment 的 `start_frame/end_frame` 时间戳区间覆盖
- `Current Action`
  - 按 action 自己的 `start_frame/end_frame` 时间戳区间覆盖
- `Interaction`
  - 从某条 interaction 时间戳开始，持续到下一条 interaction 时间戳前

## SOMA skeleton 输出规则

输出保留完整 `SOMA` skeleton。

但逐帧动态值只保留：

- `Hips`
- `motion_reconstruction/configs/dual_fsq.yaml` 中的 `human_body_names`

其他 joint 在下列数组中全部清零：

- `human_local_transforms`
- `human_global_pos`
- `human_global_quat`

静态参考姿态保留真实值：

- `human_reference_local_transforms`

当前 `.npz` 和 `.bvh` 都保留 `SOMA` 虚拟根节点 `Root`，因为参考 `SOMA BVH` 本来就是：

- `ROOT Root`
- 子节点 `JOINT Hips`

## 输出字段

### 基础字段

- `fps`
- `num_frames`
- `scalar_first`

### 人体骨架字段

- `human_joint_names`
- `human_parent_indices`
- `human_up_axis`
- `human_forward_axis`
- `human_reference_local_transforms`
- `human_local_transforms`
- `human_global_pos`
- `human_global_quat`

这里有一个很重要的约定：

- `human_local_transforms`
  - 按 `soma-retargeter` / `play_npz_mujoco.py` / `play_npz_newton.py` 的语义保存
  - 也就是先处在“visualization frame 之前”的 skeleton frame
- `human_global_pos / human_global_quat`
  - 按参考链路
    `human_local_transforms -> FK -> visualization frame`
    计算后保存

这样做有两个目的：

- 用 `play_npz_*` 播放当前导出的 `.npz` 时，人体显示流程和参考工程一致
- `motion_reconstruction` 的 `hdf5-human source` 可以按同一套 local->global 语义构造
  human feature，避免训练和推理坐标系漂移

## 独立可视化核对

为了把问题拆开定位，当前工程里另外补了一个只看人体骨架的查看器：

- `hdf5_parse/annotation_soma_mujoco_viewer.py`

它不走 `motion_reconstruction`，也不显示机器人，只复刻参考播放器的人体部分：

1. 读取 `human_local_transforms`
2. 调用 `compute_global_joint_transforms`
3. 调用 `apply_visualization_frame`
4. 在 MuJoCo 中用 sphere + capsule 画 joint、bone 和 joint 坐标轴

快速命令：

```bash
conda activate mimic_baseline
python hdf5_parse/annotation_soma_mujoco_viewer.py \
  --npz hdf5_parse/out/annotation_soma.npz
```

如果临时只想看骨架，不看坐标轴：

```bash
python hdf5_parse/annotation_soma_mujoco_viewer.py \
  --npz hdf5_parse/out/annotation_soma.npz \
  --hide-axes
```

这一步的判断标准很简单：

- 如果这个查看器里的骨架正常，说明 `annotation_soma.npz` 的人体侧语义是自洽的
- 如果这里就不正常，问题就收敛到 `SMPL -> SOMA -> export_hdf5_to_soma_npz.py`
  这一段，而不是后面的 `motion_reconstruction`

### 追踪与调试字段

- `timeline_frame_indices`
- `smpl_global_orient`
- `smpl_body_pose`
- `smpl_transl`
- `smpl_betas`
- `soma_poses`
- `soma_transl`
- `soma_joint_orient`
- `per_vertex_error`

### 文本字段

- `main_task_texts`
- `sub_task_texts`
- `current_action_texts`
- `interaction_texts`
- `main_task_text_indices`
- `sub_task_text_indices`
- `current_action_text_indices`
- `interaction_text_indices`

## BVH 输出格式

`export_hdf5_to_soma_bvh.py` 会写出和参考 `SOMA BVH` 一致的结构：

- `Root` 和 `Hips` 都带 6 通道
- 其他 joints 只带旋转通道
- 旋转顺序固定为 `Zrotation Yrotation Xrotation`
- `OFFSET` 和每帧平移都写成厘米
- `Root` 的 motion channels 保持为全 0
- 为了和官方 SOMA BVH 的 root 语义一致，pre-visualization frame 的固定 root 旋转会吸收到 `Hips` motion channels

这和 `soma_retargeter.assets.bvh.load_bvh()` 的读取约定是配套的：

- 读回时会把位置从厘米还原成米
- 旋转会按 `ZYX` Euler 恢复为局部四元数

## 分段 SMPL 输出格式

`export_hdf5_segmented_motion.py` 里的每个 `SMPL npz` 会保存：

- `fps`
- `num_frames`
- `frame_nums`
- `frame_timestamps`
- `smpl_global_orient`
- `smpl_body_pose`
- `smpl_transl`
- `smpl_betas`

文件名使用该段首尾时间戳：

- `annotation_<start_timestamp>_<end_timestamp>.npz`

## 本地烟雾测试结果

以下命令已实际跑通：

```bash
conda activate mimic_baseline
python hdf5_parse/export_hdf5_to_soma_npz.py \
  --smpl-model-path /home/hpx/HPX_LOCO_2/SOMA-X/assets/SMPL/SMPL_NEUTRAL.npz \
  --end-frame 8 \
  --batch-size 2
python hdf5_parse/export_hdf5_to_soma_bvh.py \
  --smpl-model-path /home/hpx/HPX_LOCO_2/SOMA-X/assets/SMPL/SMPL_NEUTRAL.npz \
  --end-frame 8 \
  --batch-size 2
python hdf5_parse/export_hdf5_segmented_motion.py \
  --smpl-model-path /home/hpx/HPX_LOCO_2/SOMA-X/assets/SMPL/SMPL_NEUTRAL.npz \
  --end-frame 32 \
  --batch-size 4
```

输出：

- `hdf5_parse/out/annotation_soma.npz`
- `hdf5_parse/out/annotation_soma.bvh`
- `hdf5_parse/out/smpl/*.npz`
- `hdf5_parse/out/soma_bvh/*.bvh`

关键信息：

- `human_joint_names.shape == (77,)`
- `human_local_transforms.shape == (8, 77, 7)`
- `human_global_pos.shape == (8, 77, 3)`
- `human_global_quat.shape == (8, 77, 4)`
- `timeline_frame_indices.shape == (8,)`
- `smpl_body_pose.shape == (8, 69)`
- `soma_poses.shape == (8, 77, 3)`

并且已经确认：

- `Hips / Spine / 四肢` 等关注 joint 保留真实动态值
- 对这些关注 joint，`play_npz_*` 使用的
  `human_local_transforms -> FK -> visualization frame`
  与文件里的 `human_global_pos / human_global_quat` 数值一致
- 像 `Jaw` 这种不在 `human_body_names` 里的 joint，在动态数组中全零

## 测试

单元测试：

```bash
conda activate mimic_baseline
pytest tests/test_hdf5_soma_export.py tests/test_hdf5_soma_payload.py -q
pytest tests/test_hdf5_soma_bvh_export.py -q
pytest tests/test_hdf5_segmented_export.py -q
```

当前通过结果：

- `19 passed`
- `6 passed`

## 已知约束

- 只支持 `cuda`
- 依赖本地 `SOMA-X`
- 依赖本地 `SMPL_NEUTRAL` 模型
- 当前只走 body 链路，不使用手部
- 输出目标是“训练/重定向友好的人体包”，不是标准官方 `SOMA pose npz`
