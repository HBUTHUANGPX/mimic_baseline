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

## 常用命令

下面这些命令形态都已经在参数解析测试里覆盖过。

### 1. 默认离线

使用 `Q1RobotCfg` 中默认的离线 motion 文件：

```bash
python awesome_deploy/awesome_deploy/scripts/deploy_g1_mujoco.py \
  --robot_name q1
```

### 2. 显式指定离线模式

```bash
python awesome_deploy/awesome_deploy/scripts/deploy_g1_mujoco.py \
  --robot_name q1 \
  --motion-source offline
```

### 3. 离线模式 + 自定义 motion 文件

```bash
python awesome_deploy/awesome_deploy/scripts/deploy_g1_mujoco.py \
  --robot_name q1 \
  --motion-source offline \
  --motion-source-uri /abs/path/to/your_motion.npz
```

### 4. 离线模式 + 直接播放参考轨迹

不经过 policy 推理，直接逐帧播放离线 motion：

```bash
python awesome_deploy/awesome_deploy/scripts/deploy_g1_mujoco.py \
  --robot_name q1 \
  --motion-source offline \
  --motion-play
```

## 切到 Realtime

推荐直接用脚本参数，不再需要先导出一串环境变量：

```bash
source ~/miniconda3/bin/activate mimic_baseline
cd /home/hpx/HPX_LOCO_2/mimic_baseline

python awesome_deploy/awesome_deploy/scripts/deploy_g1_mujoco.py \
  --robot_name q1 \
  --motion-source realtime \
  --motion-source-uri tcp://127.0.0.1:5555 \
  --motion-source-topic xsens.link_states.v1 \
  --motion-source-buffer-size 16 \
  --gmr-robot Q1 \
  --gmr-human-height 1.66
```

如果只写：

```bash
python awesome_deploy/awesome_deploy/scripts/deploy_g1_mujoco.py \
  --robot_name q1 \
  --motion-source realtime
```

则会自动使用这些 realtime 默认值：

```text
motion_source_uri=tcp://127.0.0.1:5555
motion_source_topic=xsens.link_states.v1
motion_source_buffer_size=16
```

### 5. Realtime 最简用法

只切换到 realtime，其余使用默认值：

```bash
python awesome_deploy/awesome_deploy/scripts/deploy_g1_mujoco.py \
  --robot_name q1 \
  --motion-source realtime
```

### 6. Realtime 全参数

```bash
python awesome_deploy/awesome_deploy/scripts/deploy_g1_mujoco.py \
  --robot_name q1 \
  --motion-source realtime \
  --motion-source-uri tcp://127.0.0.1:5555 \
  --motion-source-topic xsens.link_states.v1 \
  --motion-source-buffer-size 16 \
  --gmr-robot Q1 \
  --gmr-human-height 1.66
```

### 7. Realtime + 直接播放最新动捕帧

不经过 policy 推理，直接把最新一帧重映射结果写进 MuJoCo：

```bash
python awesome_deploy/awesome_deploy/scripts/deploy_g1_mujoco.py \
  --robot_name q1 \
  --motion-source realtime \
  --motion-play
```

对应的脚本参数接口都由 [cfg.py](/home/hpx/HPX_LOCO_2/mimic_baseline/awesome_deploy/awesome_deploy/utils/cfg.py) 中的 `BaseRobotCfg` 承接，当前支持：

```text
--motion-source {offline,realtime}
--motion-source-uri <uri_or_npz_path>
--motion-source-topic <topic>
--motion-source-buffer-size <int>
--gmr-robot <robot_name>
--gmr-human-height <float>
--motion-play / --no-motion-play
--draw-xsens-frames / --no-draw-xsens-frames
--draw-xsens-labels / --no-draw-xsens-labels
--xsens-frame-axis-length <float>
--xsens-frame-shaft-width <float>
```

## Realtime 可视化叠加

在 `motion_source=realtime` 时，deploy 会默认把 Xsens 原始 `link_states` 的坐标系直接画到 MuJoCo viewer 上，效果与：

- [zmq_mujoco_link_visualizer.py](/home/hpx/HPX_LOCO_2/mimic_baseline/xsens_mvn_refactor_build/examples/zmq_mujoco_link_visualizer.py)

一致，但这里是直接叠加在 deploy 的在线仿真窗口里。

默认行为：

- `draw_xsens_frames=True`
- `draw_xsens_labels=False`
- `xsens_frame_axis_length=0.08`
- `xsens_frame_shaft_width=0.006`

如果你想手动控制：

```bash
python awesome_deploy/awesome_deploy/scripts/deploy_g1_mujoco.py \
  --robot_name q1 \
  --motion-source realtime \
  --draw-xsens-frames \
  --no-draw-xsens-labels \
  --xsens-frame-axis-length 0.08 \
  --xsens-frame-shaft-width 0.006
```

如果你想关闭这层原始动捕坐标轴叠加：

```bash
python awesome_deploy/awesome_deploy/scripts/deploy_g1_mujoco.py \
  --robot_name q1 \
  --motion-source realtime \
  --no-draw-xsens-frames
```

如果你想显式关闭布尔开关，也可以这样写：

```bash
python awesome_deploy/awesome_deploy/scripts/deploy_g1_mujoco.py \
  --robot_name q1 \
  --motion-source realtime \
  --no-motion-play \
  --no-draw-xsens-frames \
  --no-draw-xsens-labels
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

当前运行时配置来源只有两层：

- 脚本参数
- `BaseRobotCfg` 默认值
