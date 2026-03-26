# awesome_deploy

`awesome_deploy` 现在同时支持两种参考运动源：

- `offline`
  - 读取现有 `.npz` motion 数据集
- `realtime`
  - 通过 ZMQ 订阅 Xsens `link_states`
  - 使用 GMR 在线重映射为机器人 `joint_pos/joint_vel/body_*`
  - 以 `MotionLoader` 兼容接口供 `deploy_g1_mujoco.py` 使用

## 默认行为

默认仍是离线模式，因此原命令不变：

```bash
source ~/miniconda3/bin/activate mimic_baseline
cd /home/hpx/HPX_LOCO_2/mimic_baseline
python awesome_deploy/awesome_deploy/scripts/deploy_g1_mujoco.py --robot_name q1
```

## 切到 Realtime

不需要改 Python 代码，直接用环境变量覆盖：

```bash
source ~/miniconda3/bin/activate mimic_baseline
cd /home/hpx/HPX_LOCO_2/mimic_baseline

export AWESOME_DEPLOY_MOTION_SOURCE=realtime
export AWESOME_DEPLOY_MOTION_SOURCE_URI=tcp://127.0.0.1:5555
export AWESOME_DEPLOY_MOTION_SOURCE_TOPIC=xsens.link_states.v1
export AWESOME_DEPLOY_MOTION_SOURCE_BUFFER_SIZE=16
export AWESOME_DEPLOY_GMR_ROBOT=Q1
export AWESOME_DEPLOY_GMR_HUMAN_HEIGHT=1.66

python awesome_deploy/awesome_deploy/scripts/deploy_g1_mujoco.py --robot_name q1
```

如果只设置：

```bash
export AWESOME_DEPLOY_MOTION_SOURCE=realtime
```

则会自动使用这些 realtime 默认值：

```text
AWESOME_DEPLOY_MOTION_SOURCE_URI=tcp://127.0.0.1:5555
AWESOME_DEPLOY_MOTION_SOURCE_TOPIC=xsens.link_states.v1
AWESOME_DEPLOY_MOTION_SOURCE_BUFFER_SIZE=16
```

## Realtime 可视化叠加

在 `motion_source=realtime` 时，deploy 会默认把 Xsens 原始 `link_states` 的坐标系直接画到 MuJoCo viewer 上，效果与：

- [zmq_mujoco_link_visualizer.py](/home/hpx/HPX_LOCO_2/mimic_baseline/xsens_mvn_refactor_build/examples/zmq_mujoco_link_visualizer.py)

一致，但这里是直接叠加在 deploy 的在线仿真窗口里。

默认行为：

- `AWESOME_DEPLOY_REALTIME_DRAW_XSENS_FRAMES=1`
- `AWESOME_DEPLOY_REALTIME_DRAW_XSENS_LABELS=0`
- `AWESOME_DEPLOY_REALTIME_XSENS_FRAME_AXIS_LENGTH=0.08`
- `AWESOME_DEPLOY_REALTIME_XSENS_FRAME_SHAFT_WIDTH=0.006`

如果你想手动控制：

```bash
export AWESOME_DEPLOY_REALTIME_DRAW_XSENS_FRAMES=1
export AWESOME_DEPLOY_REALTIME_DRAW_XSENS_LABELS=0
export AWESOME_DEPLOY_REALTIME_XSENS_FRAME_AXIS_LENGTH=0.08
export AWESOME_DEPLOY_REALTIME_XSENS_FRAME_SHAFT_WIDTH=0.006
```

如果你想关闭这层原始动捕坐标轴叠加：

```bash
export AWESOME_DEPLOY_REALTIME_DRAW_XSENS_FRAMES=0
```

## 前置条件

- 先启动 Xsens ZMQ 发布侧，例如 `xsens_mvn_refactor_build` 里的 `xsens_link_states_publisher`
- `mimic_baseline` 环境里需要可用的：
  - `mujoco`
  - `zmq`
  - `protobuf`
  - `general_motion_retargeting`
- 运行时会优先导入 `link_states_pb2`
- 如果本地没有生成的 `link_states_pb2.py`，会自动调用本机 `protoc`，从：
  - `/home/hpx/HPX_LOCO_2/mimic_baseline/xsens_mvn_refactor_build/proto/link_states.proto`
    生成到临时目录后再导入

## 限制

- `motion_play=True` 现在同时支持离线和 realtime 模式
- 离线 `motion_play` 会按参考轨迹顺序逐帧播放，不经过 policy 推理
- realtime `motion_play` 会在每个播放周期先抓取最新一帧，再直接把该帧写进 MuJoCo
- `motion_source=realtime` 且 `motion_play=False` 时，`deploy_g1_mujoco.py` 会在每次推理前抓取最新一帧
- 如果某次播放或推理前没有收到新 ZMQ 数据，realtime 源会重复上一帧推进窗口
