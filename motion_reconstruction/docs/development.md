# 开发约定

## 代码边界

新增 motion 字段兼容逻辑时，优先放在 `data/raw_motion.py`。

新增 metadata 扫描或文件分片逻辑时，优先放在：

- `data/raw_motion.py`
- `data/sharding.py`

新增网络输入语义时，优先放在 `features/builder.py`，并同步更新
`FeatureSchema`。

新增模型结构时，优先放在 `models/`，保持模型层只依赖 tensor 维度，不依赖
npz、body 名字或 MuJoCo。

训练、评估、可视化共同需要的构建逻辑放在 `pipeline.py`。

分布式运行时、rank 行为和 collective 封装放在 `training/distributed.py`，
不要把这些细节散落到各个 CLI 或数据模块里。

MuJoCo 相关代码只放在 `visualization/`，不要让训练入口导入 MuJoCo。

## 文档同步

文档和注释使用中文。英文保留在代码标识符、命令、文件名、配置键和公认技术名
中。

新增或修改 CLI 参数时，需要同步更新：

- `motion_reconstruction/README.md`
- `motion_reconstruction/docs/usage.md`

新增或修改数据语义、模型语义、分片逻辑、分布式训练行为时，需要同步更新：

- `motion_reconstruction/README.md`
- `motion_reconstruction/docs/architecture.md`
- `motion_reconstruction/docs/README.md`

新增或修改开发流程、验证要求时，需要同步更新：

- `motion_reconstruction/docs/development.md`

## 测试

每次修改后至少运行：

```bash
python3 -m pytest tests/motion_reconstruction -q
```

如果改动涉及命令行入口，再运行：

```bash
python3 -m motion_reconstruction.cli.train --help
python3 -m motion_reconstruction.cli.evaluate --help
python3 -m motion_reconstruction.cli.visualize --help
```

如果改动涉及分布式训练，至少再确认其中一条：

```bash
python3 -m pytest tests/motion_reconstruction/test_distributed_training.py -q
```

或：

```bash
python3 -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=1 \
  -m motion_reconstruction.cli.train \
  --config motion_reconstruction/configs/dual_fsq.yaml \
  --device cuda \
  --run-name ddp_smoke
```

如果改动涉及 MuJoCo XML 或 viewer，至少确认：

```bash
python3 - <<'PY'
import mujoco
import mujoco.viewer
print(mujoco.__version__)
PY
```

实际打开 viewer 的检查需要在带显示环境的终端中进行。

## 变更原则

优先保持下面几条：

- raw loader 只做 raw 语义和 schema 校验
- FeatureBuilder 是 motion 语义到网络输入的唯一转换层
- 模型层不依赖具体工程里的 body 名字或可视化库
- 训练器负责编排，不在内部偷偷复制另一套评估或可视化逻辑
- 分布式行为集中封装，避免主链路里到处散落 `rank == 0` 判断
