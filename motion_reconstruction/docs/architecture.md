# 架构说明

## 数据流

```text
npz / motion yaml
  -> MotionSourceResolver
  -> RawMotionLoader
  -> FeatureBuilder
  -> MotionWindowBuffer
  -> DualFSQAutoEncoder
  -> ReconstructionResult
  -> metrics / npz / MuJoCo viewer
```

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `config/` | 配置 schema 和 YAML 加载。 |
| `data/` | motion 文件解析、raw 字段加载、GPU window buffer。 |
| `features/` | 将 raw motion 转成网络 feature。 |
| `models/` | FSQ/iFSQ 量化器、双编码器、单解码器。 |
| `training/` | normalizer、loss、checkpoint 和训练循环。 |
| `evaluation/` | 从 checkpoint 生成重构结果和基础误差。 |
| `visualization/` | 使用 MuJoCo 播放评估结果。 |
| `pipeline.py` | 训练、评估、可视化共享的构建流程。 |
| `cli/` | 命令行薄入口。 |

## 可复用边界

`RawMotionLoader` 负责保留参考 motion loader 的 raw 字段语义，并统一
quaternion 顺序。它不做网络输入处理。

`FeatureBuilder` 是 motion 语义进入网络前的唯一转换层。其它工程如果需要
兼容不同 motion 来源，优先在这里扩展，而不是改模型。

`DualFSQAutoEncoder` 只接收已经归一化并展平的 robot/human window feature。
它不依赖 npz 字段、body 名字或 MuJoCo。

`pipeline.py` 负责把配置连接到数据、feature、buffer 和模型。训练和评估都
走这里，避免两边逻辑漂移。

`visualization/` 只消费 `ReconstructionResult`。MuJoCo 依赖不进入训练主链路。

## 重构结果

训练时 decoder 重构完整 robot window。评估导出时默认取窗口中心帧：

```text
history + current + future -> current
```

`ReconstructionResult` 保存：

- 原始 robot feature
- robot encoder 重构 robot feature
- human encoder 重构 robot feature
- robot anchor 世界坐标
- 原始 human body 世界坐标
- joint/body 名字和 anchor 名字

MuJoCo viewer 会用 robot feature 中的 6D rotation 和 joint pos 构建 qpos。
