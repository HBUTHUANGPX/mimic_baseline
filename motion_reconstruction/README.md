# motion_reconstruction

`motion_reconstruction` 是一个独立的动作重构训练包，用于从大量 `.npz`
motion 文件中加载数据，构建 robot/human 语义特征，并训练 FSQ 或 iFSQ
双编码器自编码器。

当前实现聚焦训练主链路：

- 加载旧工程 `motion_file.yaml` 或直接传入 `.npz` 文件、目录。
- 保留参考 motion loader 中的原始字段语义。
- 构建可复用的 `FeatureBuilder`。
- 将 frame feature 常驻 GPU，并在 GPU 上按中心帧采样
  `history + current + future` window。
- 使用 robot encoder、human encoder、共享 quantizer 和 robot decoder 训练。
- 写入 TensorBoard log，并保存包含 schema、normalizer、quantizer config 的 checkpoint。

## 包结构

| 模块 | 说明 |
| --- | --- |
| `data/` | motion 文件收集、raw `.npz` 加载、GPU window buffer。 |
| `features/` | raw motion 到网络输入 feature 的语义转换层。 |
| `models/` | FSQ/iFSQ quantizer、双 encoder、单 decoder。 |
| `training/` | normalizer、loss、checkpoint、trainer、TensorBoard。 |
| `config/` | YAML 配置 schema 和加载逻辑。 |
| `cli/` | 训练入口。 |

## 快速启动

若环境缺少 tqdm，先安装：

```bash
python3 -m pip install -r motion_reconstruction/requirements.txt
```

```bash
python3 -m motion_reconstruction.cli.train \
  --config motion_reconstruction/configs/dual_fsq.yaml \
  --device cuda \
  --run-name smoke
```

训练时终端会显示 tqdm 进度条。若只想写 TensorBoard 和 checkpoint，不显示终端进度：

```bash
python3 -m motion_reconstruction.cli.train \
  --config motion_reconstruction/configs/dual_fsq.yaml \
  --device cuda \
  --run-name smoke \
  --no-progress
```

输出目录默认位于：

```text
outputs/motion_reconstruction/<run_name>/
```

其中包含：

- `checkpoints/latest.pt`
- `checkpoints/epoch_XXXX.pt`
- `tb/`

查看 TensorBoard：

```bash
tensorboard --logdir outputs/motion_reconstruction
```

## 原始 `.npz` 字段

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

metadata 需要包含：

- `fps`
- `robot_joint_names` 或 `joint_names`
- `robot_body_names` 或 `body_names`
- `human_body_names` 或 `human_joint_names`

quaternion 输入顺序由每个 `.npz` 内的 `scalar_first` 判断：

- 缺省或 `true`：认为输入为 `wxyz`
- `false`：认为输入为 `xyzw`，加载时会转成内部统一的 `wxyz`

同一次训练内，所有 motion 文件的 joint/body 名字和顺序必须一致。当前版本不做隐式重排，避免静默训练错误 schema。

## FeatureBuilder

网络不会直接接收 raw motion 字段，而是接收二次处理后的 feature。

robot frame feature：

```text
[robot_anchor_rot6d, robot_joint_pos]
```

其中 `robot_anchor_rot6d` 是指定 robot anchor body 在 world frame 下的姿态，
使用 6D rotation 表示；`robot_joint_pos` 使用 `.npz` 中全部
`robot_joint_names` 对应的转动关节角度。

human frame feature：

```text
[human_anchor_rot6d, selected_human_body_pos_in_anchor_frame]
```

其中 `human_anchor_rot6d` 是指定 human anchor body 在 world frame 下的姿态，
使用 6D rotation 表示；`selected_human_body_pos_in_anchor_frame` 是指定 human
body 在 anchor body frame 下的 xyz 位移。

默认 human body 列表为：

```python
[
    "Spine1", "Spine2", "Chest",
    "Neck1", "Neck2", "Head", "HeadEnd",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase", "RightToeEnd",
]
```

`FeatureSchema` 会写入 checkpoint，供其它工程复用网络或解释输入维度。

## Window 采样

训练目标使用中心帧池：

- 每个 motion clip 独立计算合法中心帧。
- 合法中心帧必须满足 `history` 和 `future` 不越过 clip 边界。
- 每个 epoch 前，对完整中心帧池做一次 GPU 上的无放回随机打乱。
- mini batch 保留最后不足 `batch_size` 的 batch。
- window 张量形状为 `[B, W, D]`，进入 MLP 前展平为 `[B, W * D]`。

重构目标是完整 robot window，而不仅是 current frame。

## 模型结构

模型包含：

- robot encoder
- human encoder
- 共享 quantizer
- robot decoder

robot encoder 只接收 robot feature window；human encoder 只接收 human feature
window。两个 encoder 直接输出 `latent_dim`，然后进入同一个 FSQ/iFSQ quantizer。

decoder 的输入是量化后的 latent，重构目标始终是完整 robot feature window。

## FSQ/iFSQ

`FSQQuantizer` 的核心标量量化逻辑与 lucidrains FSQ 对齐：

- tanh bound
- STE round
- 偶数 level 使用 `0.5` offset
- 输出归一化到近似 `[-1, 1]`

`IFSQQuantizer` 默认采用 Tencent-Hunyuan iFSQ 的常用 simple bound：

```text
2 * sigmoid(1.6 * x) - 1
```

当前版本保留 per-dimension `level_indices`，暂不引入 token、multi-codebook 或混合进制 global index。

## 损失函数

当前只使用 MSE 组合：

```text
total =
  w_robot_recon  * mse(recon_from_robot, robot_target)
+ w_human_recon  * mse(recon_from_human, robot_target)
+ w_latent_align * mse(q_human, q_robot)
+ w_cycle_latent * mse(q_cycle, q_human)
```

latent loss 使用量化后的 latent。

cycle 路径为：

```text
human encoder -> quantizer -> decoder -> robot encoder -> quantizer
```

该路径不做 detach。

## 归一化

训练启动时一次性用全量 frame feature 统计 mean/std：

- robot normalizer 基于 robot frame feature 统计。
- human normalizer 基于 human frame feature 统计。
- 统计结果 repeat 到 window 维度。

robot normalizer 被 robot encoder 输入、decoder target/output 和后续反归一化共享。

## 配置

参考配置：

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

## Checkpoint

checkpoint 内容包括：

- model state
- optimizer state
- epoch
- global_step
- config
- robot/human normalizer 统计
- feature schema
- quantizer config

这使训练后的网络可以被其它工程复用，也能为后续导出重构误差或可视化做准备。

## 测试

运行：

```bash
python3 -m pytest tests/motion_reconstruction -q
```

覆盖内容包括 raw loader、FeatureBuilder、GPU window buffer、FSQ/iFSQ、模型
forward/loss、normalizer、checkpoint 和 1 epoch smoke train。

## 后续遗留项

- 重构误差导出格式。
- 原始/重构轨迹可视化方案。
- 评估协议与指标。
