# 开发约定

## 代码边界

新增 motion 字段兼容逻辑时，优先放在 `data/raw_motion.py`。

新增网络输入语义时，优先放在 `features/builder.py`，并同步更新
`FeatureSchema`。

新增模型结构时，优先放在 `models/`，保持模型层只依赖 tensor 维度，不依赖
npz、body 名字或 MuJoCo。

训练、评估、可视化共同需要的构建逻辑放在 `pipeline.py`。

MuJoCo 相关代码只放在 `visualization/`，不要让训练入口导入 MuJoCo。

## 文档

文档和注释使用中文。英文保留在代码标识符、命令、文件名、配置键和公认技术名
中。

新增 CLI 参数时，需要同步更新：

- `motion_reconstruction/docs/usage.md`
- 根目录 `README.md` 中的快速入口

新增数据语义或模型语义时，需要同步更新：

- `motion_reconstruction/docs/architecture.md`
- 根目录 `README.md` 中对应章节

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

如果改动涉及 MuJoCo XML 或 viewer，至少确认：

```bash
python3 - <<'PY'
import mujoco
import mujoco.viewer
print(mujoco.__version__)
PY
```

实际打开 viewer 的检查需要在带显示环境的终端中进行。
