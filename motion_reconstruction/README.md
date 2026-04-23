# motion_reconstruction

`motion_reconstruction` 是一个独立的动作重构训练包，用于从大量 `.npz`
motion 文件中加载数据，构建 robot/human 语义特征，并训练基于 FSQ 或 iFSQ
的双编码器自编码器。它的设计目标不是一次性脚本，而是一个能被多工程复用的
小型训练与评估库。

当前版本已经覆盖：

- 兼容旧工程 `motion_file.yaml` 和直接给定的 `.npz` 文件、目录。
- 保留参考 motion loader 的原始字段语义，并统一 quaternion 顺序。
- 用独立 `FeatureBuilder` 构建网络输入，方便跨工程复用。
- 使用 robot encoder、human encoder、共享 quantizer、robot decoder 训练。
- 在 device 上构建 `history + current + future` 窗口并直接采样。
- 支持单卡训练，也支持单节点 `torch.distributed.run` 多进程训练。
- 写入 TensorBoard 日志，保存包含 normalizer、schema、quantizer 配置的 checkpoint。
- 导出 current 帧重构误差和 `reconstruction.npz`。
- 使用 MuJoCo 播放原始/重构结果。

## 当前边界

当前多卡训练采用：

- 按 motion 文件的合法中心帧数量做 rank 间分片
- 每个 rank 只加载自己的文件子集
- 每个 rank 在本地 device 上维护窗口缓冲
- normalizer 统计通过 `all_reduce` 合并为全局统计

这意味着它已经能把大数据集摊到多张卡上，但当前还不是 CPU/memmap streaming
方案。也就是说，每个 rank 分到的那份数据，仍然需要能放进该 rank 的显存或
对应 device 内存中。

## 包结构

| 模块 | 说明 |
| --- | --- |
| `pipeline.py` | 训练、评估、可视化共享的构建流程。 |
| `data/` | motion 文件解析、raw `.npz` 加载、metadata 扫描、窗口缓冲与分片。 |
| `features/` | raw motion 到网络输入 feature 的语义转换层。 |
| `models/` | FSQ/iFSQ quantizer、双编码器、单解码器。 |
| `training/` | normalizer、loss、checkpoint、distributed runtime、trainer。 |
| `evaluation/` | 从 checkpoint 生成重构结果和基础误差。 |
| `visualization/` | MuJoCo 播放原始动作和重构动作。 |
| `config/` | YAML 配置 schema 和加载逻辑。 |
| `cli/` | 训练、评估、可视化入口。 |

更细的说明见：

- [使用命令](docs/usage.md)
- [架构说明](docs/architecture.md)
- [开发约定](docs/development.md)

## 安装依赖

```bash
python3 -m pip install -r motion_reconstruction/requirements.txt
```

如果当前环境没有 `tensorboard` 命令：

```bash
python3 -m pip install tensorboard
```

如果需要 MuJoCo 可视化：

```bash
python3 -m pip install mujoco
```

## 快速开始

### 单卡训练

```bash
python3 -m motion_reconstruction.cli.train \
  --config motion_reconstruction/configs/dual_fsq.yaml \
  --device cuda \
  --run-name test_run
```

关闭终端 tqdm：

```bash
python3 -m motion_reconstruction.cli.train \
  --config motion_reconstruction/configs/dual_fsq.yaml \
  --device cuda \
  --run-name test_run \
  --no-progress
```

### 单节点多卡训练

```bash
python3 -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  -m motion_reconstruction.cli.train \
  --config motion_reconstruction/configs/dual_fsq.yaml \
  --device cuda \
  --run-name dual_fsq_ddp
```

说明：

- `train.batch_size` 表示每个 rank 的本地 batch size，不是全局 batch size。
- 全局 batch size 约等于 `batch_size * world_size`。
- 只有 `rank 0` 会写 TensorBoard、保存 checkpoint、显示主进度条。
- 即使当前只有一张卡，也可以用 `--nproc_per_node=1` 先走通分布式代码路径。

### TensorBoard

```bash
tensorboard --logdir outputs/motion_reconstruction
```

如果当前环境没有 `tensorboard` 命令：

```bash
python3 -m tensorboard.main --logdir outputs/motion_reconstruction
```

默认输出目录：

```text
outputs/motion_reconstruction/<run_name>/
```

其中通常包含：

- `checkpoints/latest.pt`
- `checkpoints/epoch_XXXX.pt`
- `tb/`

## 评估与导出

训练完成后，可以导出基础误差和重构结果：

```bash
python3 -m motion_reconstruction.cli.evaluate \
  --config motion_reconstruction/configs/dual_fsq.yaml \
  --checkpoint outputs/motion_reconstruction/test_run/checkpoints/latest.pt \
  --output outputs/motion_reconstruction/test_run/eval \
  --device cuda
```

输出包括：

- `metrics.json`
- `reconstruction.npz`

当前评估默认：

- 只在合法中心帧上进行
- decoder 虽然重构完整 robot window，但导出和误差计算默认取 `current` 帧
- 指标是 current 帧 robot feature 和 joint 子集的基础 MSE

## MuJoCo 可视化

```bash
python3 -m motion_reconstruction.cli.visualize \
  --config motion_reconstruction/configs/dual_fsq.yaml \
  --checkpoint outputs/motion_reconstruction/test_run/checkpoints/latest.pt \
  --xml-path assets/unitree_g1/g1_29dof_rev_1_0.xml \
  --pair both \
  --max-frames 1000 \
  --loop
```

可视化中你会看到：

- `--pair robot`：原始 robot 与 robot encoder 重构出的 robot
- `--pair human`：原始 human 骨架与 human encoder 重构出的 robot
- `--pair both`：同时显示两组对比

默认会按 anchor 居中显示，便于观察姿态和关节；如需保留世界坐标轨迹：

```bash
--keep-world
```

机器人骨架显示时，会以 MuJoCo XML/URDF 的根节点作为自由基座根节点，而不是
直接把 anchor body 当作根节点。人体骨架显示时，只显示
`features.human_anchor_body` 和 `features.human_body_names` 指定的必要节点，
不会把原始 human 全量 body 一股脑全部画出来。

## 原始 `.npz` 语义

loader 会按参考 `_motion_data_np_list_to_tensor` 的语义输出以下字段：

| 规范字段 | 兼容别名 |
| --- | --- |
| `joint_pos` | `joint_pos`, `robot_joint_pos` |
| `joint_vel` | `joint_vel`, `robot_joint_vel` |
| `body_pos_w` | `body_pos_w`, `robot_body_pos` |
| `body_quat_w` | `body_quat_w`, `robot_body_quat` |
| `body_lin_vel_w` | `body_lin_vel_w`, `robot_body_lin_vel` |
| `body_ang_vel_w` | `body_ang_vel_w`, `robot_body_ang_vel` |
| `human_body_pos_w` | `human_body_pos_w`, `human_global_pos` |
| `human_body_quat_w` | `human_body_quat_w`, `human_global_quat` |

metadata 至少需要包含：

- `fps`
- `robot_joint_names` 或 `joint_names`
- `robot_body_names` 或 `body_names`
- `human_body_names` 或 `human_joint_names`

quaternion 输入顺序由每个 `.npz` 内的 `scalar_first` 判断：

- 缺省或 `true`：认为输入为 `wxyz`
- `false`：认为输入为 `xyzw`，加载时会转成内部统一的 `wxyz`

同一次训练内，所有 motion 文件的 joint/body 名字和顺序必须一致。当前版本不做
隐式重排，避免静默训练错误 schema。

## FeatureBuilder 输出

网络不会直接接收 raw motion 字段，而是接收二次处理后的 feature。

robot 单帧 feature：

```text
[robot_anchor_rot6d, robot_joint_pos]
```

其中：

- `robot_anchor_rot6d` 是指定 robot anchor body 在 world frame 下的姿态，
  使用 6D rotation 表示
- `robot_joint_pos` 使用 `.npz` 中全部 `robot_joint_names` 对应的关节角

human 单帧 feature：

```text
[human_anchor_rot6d, selected_human_body_pos_in_anchor_frame]
```

其中：

- `human_anchor_rot6d` 是指定 human anchor body 在 world frame 下的姿态，
  使用 6D rotation 表示
- `selected_human_body_pos_in_anchor_frame` 是指定 human body 在 anchor body
  frame 下的 xyz 位移

默认 human body 列表参考：

```text
Spine1, Spine2, Chest, Neck1, Neck2, Head, HeadEnd,
LeftShoulder, LeftArm, LeftForeArm, LeftHand,
RightShoulder, RightArm, RightForeArm, RightHand,
LeftLeg, LeftShin, LeftFoot, LeftToeBase, LeftToeEnd,
RightLeg, RightShin, RightFoot, RightToeBase, RightToeEnd
```

`FeatureSchema` 会写入 checkpoint，方便后续复用网络或解释输入维度。

## 窗口采样与归一化

训练使用中心帧池：

- 每个 motion clip 独立计算合法中心帧
- 合法中心帧必须满足 `history` 和 `future` 不跨 clip 边界
- 每个 epoch 前，对中心帧池做随机打乱
- decoder 的训练目标是完整 robot window，而不是只有 current 帧

在单进程模式下，epoch 内直接遍历本地中心帧池。  
在分布式模式下：

- 先按文件做 rank 间分片
- 每个 rank 只打乱自己的本地中心帧池
- 为了让各 rank 的 backward 次数一致，会把本地 epoch batch 数补齐到全局最大值

归一化方面：

- robot normalizer 基于 robot 单帧 feature 统计
- human normalizer 基于 human 单帧 feature 统计
- 统计结果会 repeat 到 window 维度
- 在分布式模式下，各 rank 的统计会通过 `all_reduce` 合并成全局统计

robot normalizer 被 robot encoder 输入、decoder 输出目标和反归一化共享。

## 模型结构与损失

模型包含：

- robot encoder
- human encoder
- 共享 quantizer
- robot decoder

两个 encoder 都直接输出 `latent_dim`，然后进入同一个 FSQ/iFSQ quantizer。
decoder 的输入是量化后的 latent，重构目标始终是完整 robot feature window。

损失当前只使用四项 MSE：

```text
total =
  w_robot_recon  * mse(recon_from_robot, robot_target)
+ w_human_recon  * mse(recon_from_human, robot_target)
+ w_latent_align * mse(q_human, q_robot)
+ w_cycle_latent * mse(q_cycle, q_human)
```

其中 cycle 路径为：

```text
human encoder -> quantizer -> decoder -> robot encoder -> quantizer
```

这条路径不做 detach，latent loss 使用量化后的 latent。

## FSQ / iFSQ

`FSQQuantizer` 的核心标量量化逻辑对齐 lucidrains 的 FSQ 实现：

- bound 到量化区间
- 使用 STE round
- 偶数 level 使用 `0.5` offset
- 输出归一化到近似 `[-1, 1]`

`IFSQQuantizer` 参考 Tencent-Hunyuan iFSQ 的 simple bound 方案，当前默认使用：

```text
2 * sigmoid(1.6 * x) - 1
```

## 配置速查

参考配置文件：

```text
motion_reconstruction/configs/dual_fsq.yaml
```

常用字段：

- `data.motion_yaml`
- `data.files`
- `data.dirs`
- `features.robot_anchor_body`
- `features.human_anchor_body`
- `features.human_body_names`
- `model.latent_dim`
- `model.robot_encoder_hidden_dims`
- `model.human_encoder_hidden_dims`
- `model.decoder_hidden_dims`
- `model.quantizer.type`
- `train.history`
- `train.future`
- `train.batch_size`
- `train.progress`
- `train.distributed.enabled`
- `train.distributed.backend`
- `train.distributed.find_unused_parameters`

## Checkpoint 内容

checkpoint 包含：

- model state
- optimizer state
- epoch
- global_step
- config
- robot/human normalizer 统计
- feature schema
- quantizer config

这使得训练后的网络可以被其它工程直接复用，也为后续评估、误差导出和可视化
提供了足够上下文。

## 验证

代码改动后，至少建议运行：

```bash
python3 -m pytest tests/motion_reconstruction -q
```

如果改动涉及 CLI 入口，再补充：

```bash
python3 -m motion_reconstruction.cli.train --help
python3 -m motion_reconstruction.cli.evaluate --help
python3 -m motion_reconstruction.cli.visualize --help
```

## 后续遗留项

- 更完整的评估协议与分关节指标
- 更系统的可视化分析工具
- 真正面向超大数据集的 CPU/memmap streaming 数据管线
