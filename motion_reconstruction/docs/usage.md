# 使用命令

## 安装依赖

```bash
python3 -m pip install -r motion_reconstruction/requirements.txt
```

如果只训练，不做 MuJoCo 可视化，`mujoco` 可以暂时不安装；如果当前环境已经
安装过，也可以直接使用。

## 训练

```bash
python3 -m motion_reconstruction.cli.train \
  --config motion_reconstruction/configs/dual_fsq.yaml \
  --device cuda \
  --run-name test_run
```

关闭终端进度条：

```bash
python3 -m motion_reconstruction.cli.train \
  --config motion_reconstruction/configs/dual_fsq.yaml \
  --device cuda \
  --run-name test_run \
  --no-progress
```

输出目录：

```text
outputs/motion_reconstruction/<run_name>/
```

常用文件：

- `checkpoints/latest.pt`
- `checkpoints/epoch_XXXX.pt`
- `tb/`

## TensorBoard

```bash
tensorboard --logdir outputs/motion_reconstruction
```

如果没有 `tensorboard` 命令：

```bash
python3 -m pip install tensorboard
python3 -m tensorboard.main --logdir outputs/motion_reconstruction
```

## 评估

```bash
python3 -m motion_reconstruction.cli.evaluate \
  --config motion_reconstruction/configs/dual_fsq.yaml \
  --checkpoint outputs/motion_reconstruction/test_run/checkpoints/latest.pt \
  --output outputs/motion_reconstruction/test_run/eval \
  --device cuda
```

输出：

- `metrics.json`
- `reconstruction.npz`

当前指标是 current 帧 robot feature 的基础 MSE，包含完整 robot feature 和
joint 部分。

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

`--pair robot` 显示原始 robot 和 robot encoder 重构 robot。

`--pair human` 显示原始 human 和 human encoder 重构 robot。

`--pair both` 同时显示两组对比。

默认按 anchor 居中显示，便于观察姿态和关节；如需保留世界坐标轨迹：

```bash
--keep-world
```
