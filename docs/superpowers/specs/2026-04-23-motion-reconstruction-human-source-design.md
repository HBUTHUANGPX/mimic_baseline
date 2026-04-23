# Motion Reconstruction Human-Only Source Design

## Goal

让 `motion_reconstruction` 支持两类新能力：

- 用模块化方式读取不同来源的 `.npz`，不再把 `cli.visualize` 绑死在训练原始 raw motion 上；
- 用模块化方式选择 dual encoder 的推理链路，至少支持 `human encoder -> shared decoder`。

在这条基础上，`hdf5_parse` 可以把自己导出的 human-only `.npz` 交给
`motion_reconstruction` 以包调用方式完成可视化，而不再维护一套独立 viewer
逻辑。

## Approved Scope

用户已确认本轮采用：

- 输入来源：`hdf5_parse` 导出的 human-only `.npz`
- 推理路径：`human encoder -> decoder`
- 可视化内容：`原始 human skeleton` vs `decoder 输出的 robot motion`
- 文档需要同步更新

## Existing Constraints

- 当前 `DualFSQAutoEncoder` 的 decoder 始终输出 robot feature，而不是 human feature。
- 当前 `cli.visualize` 只能走：
  `config -> build_motion_runtime -> reconstruct_motion -> play_reconstruction`
- 当前 `RawMotionLoader` 严格要求 robot 和 human raw 字段都存在，因此无法直接读取
  `hdf5_parse/out/*.npz`
- `hdf5_parse` 处理的是 human-only 数据，没有 robot 原始轨迹

## Design

### 1. Source adapter layer

新增独立的 source adapter 层，负责把不同来源的数据整理成统一的推理输入。

至少支持两类 source：

- `raw`
  - 复用现有 `build_motion_runtime`
  - 读取完整训练/评估 raw motion `.npz`
- `hdf5-human`
  - 直接读取 `hdf5_parse` 导出的 human-only `.npz`
  - 从 `human_global_pos/human_global_quat` 构建 human feature
  - 不要求任何 `robot_*` 字段

统一输出的内容应至少包含：

- `fps`
- `center_indices`
- `window_offsets`
- `human_features`
- `robot_features` 或 `None`
- `robot_anchor_pos_w`
- `human_body_pos_w`
- `robot_joint_names`
- `robot_body_names`
- `human_body_names`
- `robot_anchor_body`
- `human_anchor_body`
- `display_human_body_names`

### 2. Inference path layer

新增推理路径选择，不再默认总是同时跑两条 encoder 路径。

本轮至少支持：

- `robot`
  - `robot encoder -> decoder`
- `human`
  - `human encoder -> decoder`

在 `hdf5-human` source 下：

- 只允许 `human` 路径
- 如果请求 `robot` 路径，直接报错

### 3. Human-only robot anchor rule

`human encoder -> decoder` 的输出是 robot feature，MuJoCo 可视化时仍需要
`robot_anchor_pos_w` 来摆放机器人。

由于 `hdf5_parse` 没有 robot 轨迹，本轮采用以下规则：

- 用 human anchor body（默认 `Hips`）的世界坐标作为 `robot_anchor_pos_w`

这不是“真实机器人轨迹”，而是为了让 decoder 输出的 robot motion 能稳定跟随 human
轨迹显示，服务于“人到机重构是否合理”的判断。

### 4. Reconstruction result shape

扩展 `ReconstructionResult`，让它支持部分字段缺失：

- `original_robot_feature` 可以为空
- `recon_from_robot_feature` 可以为空
- `recon_from_human_feature` 保留

对应地：

- `metrics()` 只对可用字段计算
- `save_npz()` 只写存在的字段
- viewer 在 `pair=human` 时不依赖 robot 原始分支

### 5. Viewer payload builder

可视化依然复用现有 MuJoCo viewer，但不再要求上游一定来自完整 raw runtime。

需要保证：

- `pair=human` 在 human-only source 下可工作
- `pair=robot/both` 在缺少 robot 原始分支时给出清晰错误
- human skeleton 仍只显示配置里的 anchor + selected body names

### 6. Package API

在 `motion_reconstruction` 包内新增稳定入口，供其它工程直接调用。

至少提供：

- 一个“构建重构结果”的包级 API
- 一个“直接播放 human-only 重构”的包级 API

这样 `hdf5_parse` 只需要做很薄的参数转发。

### 7. CLI updates

`motion_reconstruction.cli.visualize` 需要新增：

- `--source {raw,hdf5-human}`
- `--motion-npz PATH`
- `--inference-path {robot,human}`

行为规则：

- `source=raw` 时继续兼容现有配置驱动流程
- `source=hdf5-human` 时通过 `--motion-npz` 指定 human-only `.npz`
- 默认 `source=raw`
- 默认 `inference-path=human`

## Validation Strategy

需要覆盖以下行为：

- `hdf5-human` source 可以在没有 `robot_*` 字段时正常加载
- human feature 构建结果与现有 `FeatureBuilder` 语义一致
- `human` 路径可以在没有 robot 原始特征时完成推理
- `robot` 路径在 human-only source 下会报明确错误
- `pair=human` 可视化在 human-only result 下不崩溃
- `hdf5_parse` 的薄封装脚本能调通包级 API

## Documentation Scope

需要同步更新：

- `motion_reconstruction/README.md`
- `motion_reconstruction/docs/usage.md`
- 顶层 `readme.md`
- `hdf5_parse/README.md`

并新增一份面向此次集成的专项说明文档，说明：

- 为什么 human-only `.npz` 不经过 `RawMotionLoader`
- 为什么 `human encoder -> decoder` 的输出仍是 robot motion
- 为什么 human-only 可视化时使用 human anchor 作为 robot anchor
