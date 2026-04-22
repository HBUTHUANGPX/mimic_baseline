# 文档索引

这里记录 `motion_reconstruction` 的稳定使用方式和模块边界。根目录
`README.md` 保留快速入口；更细的说明放在本目录，避免主文档变得臃肿。

## 文档列表

- [使用命令](usage.md)：训练、评估、MuJoCo 可视化和 TensorBoard。
- [架构说明](architecture.md)：数据流、模块职责和可复用边界。
- [开发约定](development.md)：新增模块、测试和文档维护规则。

## 当前边界

这个包的核心目标是独立训练 FSQ/iFSQ 动作重构网络，同时保证 motion
加载和网络结构能被其它工程复用。

当前不会在训练主链路中绑定 MuJoCo。MuJoCo 只在 `visualization/` 中使用，
评估结果通过 `ReconstructionResult` 和导出的 `reconstruction.npz` 传递。
