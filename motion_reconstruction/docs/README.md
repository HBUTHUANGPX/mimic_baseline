# 文档索引

这里记录 `motion_reconstruction` 的稳定使用方式、模块边界和开发约定。根目录
`README.md` 偏向总览和快速入口；本目录补充更细的工程说明。

## 文档列表

- [使用命令](usage.md)：安装依赖、单卡训练、单节点多卡训练、评估、MuJoCo 可视化。
- [架构说明](architecture.md)：数据流、分片与归一化、模块职责、可复用边界。
- [开发约定](development.md)：新增模块、文档同步规则、测试与验证要求。

## 当前实现边界

这个包的核心目标是独立训练 FSQ/iFSQ 动作重构网络，同时保证：

- motion 加载语义可以和旧工程兼容
- 网络结构、checkpoint 和重构结果可以被其它工程复用

当前多卡训练能力是：

- 单节点 `torch.distributed.run` / `torchrun`
- 按文件合法中心帧数量做 rank 间分片
- 每个 rank 本地维护自己的窗口缓冲
- 全局 normalizer 统计通过 `all_reduce` 合并

当前还没有实现面向超大数据集的 CPU/memmap streaming 数据管线，因此每个 rank
分到的数据仍然需要能放进该 rank 的 device 内存中。

MuJoCo 不进入训练主链路。训练与评估只依赖 tensor、schema 和 checkpoint；
MuJoCo 只在 `visualization/` 中消费 `ReconstructionResult`。
