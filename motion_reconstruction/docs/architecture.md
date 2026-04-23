# 架构说明

## 数据流

单进程模式：

```text
npz / motion yaml
  -> MotionSourceResolver
  -> RawMotionLoader
  -> FeatureBuilder
  -> MotionWindowBuffer
  -> WindowFeatureNormalizer
  -> DualFSQAutoEncoder
  -> ReconstructionResult
  -> metrics / npz / MuJoCo viewer
```

单节点分布式模式：

```text
npz / motion yaml
  -> MotionSourceResolver
  -> RawMotionLoader.scan
  -> MotionShardPlan
  -> 每个 rank 只加载自己的文件子集
  -> FeatureBuilder
  -> MotionWindowBuffer
  -> 全局统计 all_reduce
  -> DDP 包装的 DualFSQAutoEncoder
```

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `config/` | 配置 schema 和 YAML 加载。 |
| `data/raw_motion.py` | raw 字段加载、schema 校验、metadata 扫描。 |
| `data/sharding.py` | 按合法中心帧数量做 rank 间文件分片。 |
| `data/gpu_buffer.py` | 在 device 上构建窗口缓冲并采样 batch。 |
| `features/` | 将 raw motion 转成网络 feature。 |
| `models/` | FSQ/iFSQ 量化器、双编码器、单解码器。 |
| `training/distributed.py` | 分布式运行时、collective 和 rank 信息。 |
| `training/` | normalizer、loss、checkpoint 和训练循环。 |
| `evaluation/` | 从 checkpoint 生成重构结果和基础误差。 |
| `visualization/` | 使用 MuJoCo 播放评估结果。 |
| `pipeline.py` | 训练、评估、可视化共享的构建流程。 |
| `cli/` | 命令行薄入口。 |

## 可复用边界

`RawMotionLoader` 负责保留参考 motion loader 的 raw 字段语义，并统一
quaternion 顺序。它不做网络输入处理。

`FeatureBuilder` 是 motion 语义进入网络前的唯一转换层。其它工程如果需要兼容
不同 motion 来源，优先在这里扩展，而不是改模型。

`DualFSQAutoEncoder` 只接收已经归一化并展平的 robot/human window feature。
它不依赖 npz 字段、body 名字或 MuJoCo。

`WindowFeatureNormalizer` 只关注 feature 统计，不关心 motion 文件来源；因此
训练和评估都可以复用它。

`pipeline.py` 负责把配置连接到数据、feature、buffer 和模型。训练和评估都走
这里，避免两边逻辑漂移。

`visualization/` 只消费 `ReconstructionResult`。MuJoCo 依赖不进入训练主链路。

## 原始数据语义

loader 输出字段语义对齐参考 `_motion_data_np_list_to_tensor`，包括：

- `joint_pos`
- `joint_vel`
- `body_pos_w`
- `body_quat_w`
- `body_lin_vel_w`
- `body_ang_vel_w`
- `human_body_pos_w`
- `human_body_quat_w`

它做的事情只有：

- 字段别名兼容
- joint/body 名字一致性校验
- 按 `scalar_first` 统一 quaternion 到内部 `wxyz`

它不会：

- 隐式重排 joint/body 顺序
- 直接拼出网络输入

## FeatureBuilder 语义

`FeatureBuilder` 把 raw motion 映射成单帧特征：

robot 单帧 feature：

```text
[robot_anchor_rot6d, robot_joint_pos]
```

human 单帧 feature：

```text
[human_anchor_rot6d, selected_human_body_pos_in_anchor_frame]
```

其中：

- robot 使用全部 `robot_joint_names`
- human 只使用配置里声明的 `human_body_names`
- human body 位移在 `human_anchor_body` 的局部坐标系中表达

`FeatureSchema` 会和 checkpoint 一起保存，供其它工程复用网络或解释维度。

## 窗口采样

`MotionWindowBuffer` 负责把单帧 feature 组织成窗口：

```text
history + current + future
```

关键约束：

- 合法中心帧必须不跨 clip 边界
- window 索引构造发生在 device 上
- 输出窗口形状为 `[B, W, D]`
- 进入 MLP 前再展平为 `[B, W * D]`

训练目标是完整 robot window，而不是只有 current 帧。  
评估和导出默认再从完整 window 中取中心帧。

## 分布式训练结构

`MotionReconstructionTrainer` 在分布式模式下会做这些事：

1. 根据环境变量初始化 `DistributedRuntime`
2. 用 `RawMotionLoader.scan()` 扫描所有文件 metadata
3. 用 `MotionShardPlan` 按合法中心帧数量均衡分片
4. 每个 rank 只加载自己的文件子集
5. 每个 rank 本地构建 `FeatureBuilder` 和 `MotionWindowBuffer`
6. 每个 rank 统计本地 feature 的 `count / sum / sumsq`
7. 通过 `all_reduce` 得到全局 normalizer
8. 用 DDP 包装模型

为了让所有 rank 的 backward 次数一致：

- 每个 rank 先计算本地 epoch batch 数
- 再取全局最大 batch 数
- 本地样本不足时，循环补齐到统一 batch 数

这样可以避免有的 rank 提前结束而导致 DDP 卡住。

## 归一化

训练启动时一次性用单帧 feature 统计 mean/std：

- robot normalizer 基于 robot 单帧 feature 统计
- human normalizer 基于 human 单帧 feature 统计
- 统计结果 repeat 到 window 维度

robot normalizer 会被这些位置共享：

- robot encoder 输入
- decoder 的重构目标
- 评估时的反归一化

在分布式模式下，normalizer 不是“每个 rank 各算各的”，而是通过 `all_reduce`
合成全局统计。

## 模型结构

模型包含：

- robot encoder
- human encoder
- 共享 quantizer
- robot decoder

两个 encoder 都直接输出 `latent_dim`，然后进入同一个 FSQ/iFSQ quantizer。
decoder 的重构目标始终是完整 robot feature window。

## 损失

当前只使用四项 MSE：

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

## 重构结果

`ReconstructionResult` 保存：

- 原始 robot feature
- robot encoder 重构 robot feature
- human encoder 重构 robot feature
- robot anchor 世界坐标
- 原始 human body 世界坐标
- joint/body 名字和 anchor 名字
- MuJoCo 可视化使用的人体 body 名字

## MuJoCo 可视化边界

MuJoCo viewer 会用 robot feature 中的 6D rotation 和 joint pos 构建 `qpos`。

对于 free-base 机器人：

- 先根据当前 joint pos 求出 `XML root body -> anchor body` 的局部变换
- 再由 anchor body 的世界位姿反解真正的 XML 根节点 `qpos`

因此骨架显示时，根节点是 MuJoCo XML/URDF 的根节点，不是训练 feature 里的
anchor body。

人体骨架显示时，只使用 `features.human_anchor_body` 和
`features.human_body_names` 指定的节点；导出的 `human_body_pos_w` 仍保留原始
全量数据。
