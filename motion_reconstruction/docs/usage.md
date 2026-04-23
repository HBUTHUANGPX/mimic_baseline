# 使用命令

## 安装依赖

```bash
python3 -m pip install -r motion_reconstruction/requirements.txt
```

如果当前环境没有 TensorBoard：

```bash
python3 -m pip install tensorboard
```

如果需要 MuJoCo 可视化：

```bash
python3 -m pip install mujoco
```

## 训练

### 单卡训练

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

- `train.batch_size` 表示每个 rank 的本地 batch size
- 全局 batch size 约等于 `batch_size * world_size`
- 只有 `rank 0` 会写 TensorBoard、保存 checkpoint、显示主进度条
- 即使当前只有一张卡，也可以先用 `--nproc_per_node=1` 验证分布式入口

如果想用 CPU 两进程做最小 smoke：

```bash
python3 -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
  -m motion_reconstruction.cli.train \
  --config motion_reconstruction/configs/dual_fsq.yaml \
  --device cpu \
  --run-name cpu_ddp_smoke
```

### 输出目录

默认输出目录：

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

当前评估默认：

- 只在合法中心帧上进行
- decoder 虽然重构完整 robot window，但导出和误差统计默认取 `current` 帧
- 指标是 current 帧 robot feature 与 joint 子集的基础 MSE

如果只需要 `metrics.json`：

```bash
python3 -m motion_reconstruction.cli.evaluate \
  --config motion_reconstruction/configs/dual_fsq.yaml \
  --checkpoint outputs/motion_reconstruction/test_run/checkpoints/latest.pt \
  --output outputs/motion_reconstruction/test_run/eval \
  --device cuda \
  --no-npz
```

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

显示模式：

- `--pair robot`：原始 robot 和 robot encoder 重构出的 robot
- `--pair human`：原始 human 骨架和 human encoder 重构出的 robot
- `--pair both`：同时显示两组对比

默认会按 anchor 居中显示，便于观察姿态和关节；如需保留世界坐标轨迹：

```bash
--keep-world
```

当前 viewer 的关键行为：

- 机器人 free-base 始终以 MuJoCo XML 根节点作为根节点
- anchor body 的世界位姿只用来反解 XML 根节点 `qpos`
- human 骨架只显示 `features.human_anchor_body` 和 `features.human_body_names`
  指定的节点

## 配置入口

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
- `train.distributed.enabled`
- `train.distributed.backend`
- `train.distributed.find_unused_parameters`
- `train.distributed.timeout_minutes`
