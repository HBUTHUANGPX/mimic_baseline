# HDF5 to SOMA Export

本文档说明 `hdf5_parse/export_hdf5_to_soma_npz.py` 是怎么把 `annotation.hdf5` 中的人体动作和文本描述导出成训练友好的 `.npz` 的。

## 目标

输出一个“人体侧 `save_retarget_npz()` 风格”的数据包，满足下面几点：

- 只保留有效人体帧
- 保留原始 HDF5 时间线索引
- 保留四类文本描述和逐帧文本索引
- 用 `SOMA-X` 把 `SMPL` 风格 body motion 转成 `SOMA skeleton`
- 不输出任何 `robot_*` 字段

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

此外，导出时会去掉 `SOMA` 虚拟根节点 `Root`，让输出 skeleton 直接以 `Hips` 作为根。

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

## 本地烟雾测试结果

以下命令已实际跑通：

```bash
conda activate mimic_baseline
python hdf5_parse/export_hdf5_to_soma_npz.py \
  --smpl-model-path /home/hpx/HPX_LOCO_2/SOMA-X/assets/SMPL/SMPL_NEUTRAL.npz \
  --end-frame 8 \
  --batch-size 2
```

输出：

- `hdf5_parse/out/annotation_soma.npz`

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
- 像 `Jaw` 这种不在 `human_body_names` 里的 joint，在动态数组中全零

## 测试

单元测试：

```bash
conda activate mimic_baseline
pytest tests/test_hdf5_soma_export.py tests/test_hdf5_soma_payload.py -q
```

当前通过结果：

- `10 passed`

## 已知约束

- 只支持 `cuda`
- 依赖本地 `SOMA-X`
- 依赖本地 `SMPL_NEUTRAL` 模型
- 当前只走 body 链路，不使用手部
- 输出目标是“训练/重定向友好的人体包”，不是标准官方 `SOMA pose npz`
