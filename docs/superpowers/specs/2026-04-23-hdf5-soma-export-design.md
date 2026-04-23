# HDF5 to SOMA Export Design

## Goal

在 `hdf5_parse/` 中新增一条稳定的导出链路，把 `annotation.hdf5` 中的 `full_body_mocap` 和 `caption` 数据解析后，转换成兼容 `save_retarget_npz()` 人体侧结构的 `.npz` 文件，并额外保存原始 HDF5 时间线索引与四类文本标注索引。

目标输出服务于两类场景：

- 检查 `SMPL-H body -> SOMA skeleton` 解析是否正确；
- 为后续 `motion_reconstruction` 提供干净、可追踪、可复现的人体动作数据包。

## Constraints

- 只处理人体 body，不使用 `left_hand_quats` / `right_hand_quats`。
- 只做 CUDA 路线，不提供 CPU fallback。
- 一个 `annotation.hdf5` 输出一个 `.npz`。
- 只保存有效人体帧，不补齐缺失帧。
- 逐帧时间线索引直接对应 HDF5 原始 `frame_nums`。
- 文本池完全去重。
- 没有文本覆盖的帧统一指向 `"UNKNOWN"`，且其索引固定为 `0`。
- 最终输出不要任何 `robot_*` 字段。

## Input Data

### Motion fields

从 `annotation.hdf5` 中读取：

- `full_body_mocap/Ts_world_root`
- `full_body_mocap/body_quats`
- `full_body_mocap/betas`
- `full_body_mocap/frame_nums`
- `video/frame_number`
- `video/device_timestamp`

动作侧采用以下解释：

- `Ts_world_root` 提供根节点世界位姿，拆分为根平移和根旋转；
- `body_quats` 表示 body joints 的局部旋转；
- `betas` 作为 identity shape 参数；
- `frame_nums` 是与视频时间线对齐的原始帧号。

### Text fields

`caption` 是一个标量 JSON 字符串，关键层级为：

- `config["Main Task"]`
- `segments[*]["Sub Task"]`
- `segments[*]["Current Action"]`
- `segments[*]["interaction"]`

时间语义：

- `segments[*].start_frame/end_frame` 是时间戳区间；
- `Current Action[*].start_frame/end_frame` 也是时间戳区间；
- `interaction` 是稀疏时间戳到文本的映射。

## Conversion Architecture

### 1. HDF5 parsing layer

新增 `hdf5_parse/hdf5_soma_export.py`，负责：

- 读取所需 HDF5 字段；
- 对 `Ts_world_root`、`body_quats`、`betas` 做非有限值过滤；
- 只保留有效人体帧；
- 维护 `timeline_frame_indices`，内容即有效帧对应的原始 `frame_nums`；
- 用 `frame_nums -> video/device_timestamp` 建立文本对齐时间基准。

### 2. SMPL body preparation layer

沿用现有 `hdf5_parse/smpl_motion_tools.py` 的思路，但为导出链路明确产出：

- `smpl_global_orient`
- `smpl_body_pose`
- `smpl_transl`
- `smpl_betas`

这里的策略是：

- 把 HDF5 中的 `SMPL-H body` 视为 `SMPL` body 输入；
- 手部数据直接忽略，不参与后续求逆。

### 3. SOMA inversion layer

直接在 `hdf5_parse` 中复用 `SOMA-X` 的核心对象，而不走中间 BVH：

- `smplx.create(model_type="smpl", ...)`
- `SOMALayer(...)`
- `PoseInversion.fit(...)`

求逆输入是 SMPL mesh vertices，输出保留：

- `soma_poses`
- `soma_transl`
- `soma_joint_orient`
- `per_vertex_error`

同时通过 `SOMALayer.batched_skinning.pose(..., absolute_pose=True, return_transforms=True)` 恢复每帧 SOMA joint transforms，再转成 `save_retarget_npz()` 需要的 `human_local_transforms`。

### 4. Skeleton export layer

输出保留完整 SOMA skeleton，但动态值只对以下 joints 生效：

- `Hips`
- `motion_reconstruction/configs/dual_fsq.yaml` 中的 `human_body_names`

其余 joints 在逐帧数据中清零：

- `human_local_transforms`
- `human_global_pos`
- `human_global_quat`

静态 skeleton 信息保留真实值：

- `human_joint_names`
- `human_parent_indices`
- `human_reference_local_transforms`
- `human_up_axis`
- `human_forward_axis`

默认去掉 SOMA 虚拟 `Root`，以 `Hips` 作为导出骨架根节点。

### 5. Text alignment layer

为四类文本分别维护：

- 文本池数组
- 有效帧级别的索引数组

具体字段：

- `main_task_texts`
- `sub_task_texts`
- `current_action_texts`
- `interaction_texts`
- `main_task_text_indices`
- `sub_task_text_indices`
- `current_action_text_indices`
- `interaction_text_indices`

对齐规则：

- `Main Task`：全序列常量覆盖；
- `Sub Task`：按 segment 时间戳区间覆盖；
- `Current Action`：按 action 自己的时间戳区间覆盖；
- `Interaction`：从某条 interaction 时间戳开始，一直持续到下一条 interaction 时间戳前；
- 所有文本池第 0 项固定为 `"UNKNOWN"`；
- 所有逐帧索引数组默认填 0。

## Output Format

默认输出到：

- `hdf5_parse/out/annotation_soma.npz`

输出内容由三部分构成。

### Human motion payload

与 `save_retarget_npz()` 的人体侧字段对齐：

- `fps`
- `num_frames`
- `scalar_first`
- `human_joint_names`
- `human_parent_indices`
- `human_up_axis`
- `human_forward_axis`
- `human_reference_local_transforms`
- `human_local_transforms`
- `human_global_pos`
- `human_global_quat`

### Tracking / debug payload

- `timeline_frame_indices`
- `smpl_global_orient`
- `smpl_body_pose`
- `smpl_transl`
- `smpl_betas`
- `soma_poses`
- `soma_transl`
- `soma_joint_orient`
- `per_vertex_error`

### Text payload

- `main_task_texts`
- `sub_task_texts`
- `current_action_texts`
- `interaction_texts`
- `main_task_text_indices`
- `sub_task_text_indices`
- `current_action_text_indices`
- `interaction_text_indices`

## Validation Strategy

需要验证以下几类行为：

- `frame_nums` 与 `video/frame_number` 一一对应；
- `caption` 时间戳能正确映射到 `video/device_timestamp`；
- `num_frames == len(timeline_frame_indices)`；
- 四类文本索引长度都等于有效人体帧数；
- 非关注 joints 在动态数组里为全零；
- `Hips + human_body_names` 至少有真实非零动态数据；
- 输出 skeleton 中能找到 `Hips` 和配置里的全部 body names；
- 输出 `.npz` 不包含任何 `robot_*` 字段。

## Documentation Scope

需要同时补齐三类文档：

- 工程级 `readme.md`：增加 HDF5->SOMA 导出入口和依赖说明；
- `hdf5_parse` 专项说明：介绍输入字段、转换链路、输出字段；
- `docs/superpowers` 文档：保留这次设计与实施计划，方便后续维护。
