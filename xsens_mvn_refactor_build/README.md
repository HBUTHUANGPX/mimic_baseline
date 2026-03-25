# xsens_mvn_refactor_build

`xsens_mvn_ros2` 的并行非 ROS2 重构工程，当前已完成：

- `xsens_sdk`：Xsens MVN SDK 解析层
- `xsens_core`：纯 C++ 数据采集/帧抽象
- `xsens_transport`：Protobuf + ZMQ PUB/SUB 传输
- `src/apps/`：非 ROS2 的本地发布/订阅可执行程序

## 构建

```bash
source ~/miniconda3/bin/activate mimic_baseline
cd /home/hpx/HPX_LOCO_2/mimic_baseline/xsens_mvn_refactor_build
cmake -S . -B build
cmake --build build
```

## 自动化验证

```bash
source ~/miniconda3/bin/activate mimic_baseline
cd /home/hpx/HPX_LOCO_2/mimic_baseline/xsens_mvn_refactor_build
ctest --test-dir build --output-on-failure
```

当前自动化测试覆盖：

- `test_parsermanager_cache`
- `test_human_data_handler`
- `test_link_state_proto_serializer`
- `test_zmq_loopback`
- `test_link_states_test_publisher_process`
- `test_xsens_link_state_runner`

## 本地回环验证

推荐直接运行自动化回环测试：

```bash
ctest --test-dir build -R 'test_zmq_loopback|test_link_states_test_publisher_process' --output-on-failure
```

也可以手工启动发布端：

```bash
./build/src/link_states_test_publisher tcp://*:5555 30 100 1000 xsens.link_states.v1
```

再从另一个程序订阅：

```bash
./build/src/link_states_zmq_subscriber tcp://127.0.0.1:5555 xsens.link_states.v1 5000 2
```

说明：

- 发布端参数：`bind_address publish_count period_ms warmup_ms topic`
- 订阅端参数：`connect_address topic timeout_ms receive_count`
- `warmup_ms` 用于给 ZMQ 订阅建立时间，局域网环境建议保守取 `1000` ms 起步

## 真实 XSens 运行主程序

真实 UDP 采集发布程序：

```bash
./build/src/xsens_link_states_publisher 8001 tcp://*:5555 xsens.link_states.v1 240
```

参数：

- `udp_port`
- `zmq_bind_address`
- `zmq_topic`
- `publish_rate_hz`

配套订阅验证：

```bash
./build/src/link_states_zmq_subscriber tcp://127.0.0.1:5555 xsens.link_states.v1 5000 2
```

说明：

- 该程序依赖真实 XSens MVN UDP 数据源
- 本仓库自动化测试验证了 runner、protobuf、ZMQ 回环与进程级订阅行为
- 未接入真实 XSens 设备时，无法在当前环境完成真正的 live UDP 采集验收
- 发布语义是：只有收到新的 UDP 更新后，才发布一帧新的 ZMQ `link_states`
- 发布前会对 `link_states` 做裁剪，当前过滤规则为：
  - 去掉 `prop1` 到 `prop4`
  - 去掉 `left_carpus` / `right_carpus`
  - 去掉所有手指 segment：
    - `left_first_*` 到 `left_fifth_*`
    - `right_first_*` 到 `right_fifth_*`
  - 保留 `left_hand` / `right_hand`
- 因此当前真实发布结果会保留人体主骨架和手掌 link，但不会发送 props、腕部 carpus 和各手指细分 link

## Python 订阅示例

先生成 Python protobuf 代码。注意 `--python_out` 后面必须是目录，不是 `.proto` 文件路径。

```bash
mkdir -p /tmp/xsens_proto
protoc \
  --proto_path /home/hpx/HPX_LOCO_2/mimic_baseline/xsens_mvn_refactor_build/proto \
  --python_out /tmp/xsens_proto \
  /home/hpx/HPX_LOCO_2/mimic_baseline/xsens_mvn_refactor_build/proto/link_states.proto
```

准备 Python 依赖：

```bash
source ~/miniconda3/bin/activate mimic_baseline
pip install pyzmq protobuf
```

当前环境说明：

- 本机 `protoc` 版本是 `3.12.4`
- 若 Python 环境中的 `protobuf` 版本较新，直接导入生成的 `link_states_pb2.py` 可能报兼容错误
- 当前环境下已经实测可用的方式是额外设置：
  `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`

运行订阅示例：

```bash
PYTHONPATH=/tmp/xsens_proto \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
python /home/hpx/HPX_LOCO_2/mimic_baseline/xsens_mvn_refactor_build/examples/python_subscriber.py \
  --connect tcp://127.0.0.1:5555 \
  --topic xsens.link_states.v1 \
  --count 2
```

该脚本会输出：

- `topic=xsens.link_states.v1`
- `schema_version=1`
- `frame_id=...`
- `states_size=...`
- 每个 `state[i]` 的名称和位置

## Python MuJoCo 可视化

本工程还提供了一个只做 `ZMQ SUB + protobuf 解析 + MuJoCo 绘制` 的脚本：

- `examples/zmq_mujoco_link_visualizer.py`

功能：

- 订阅 `xsens.link_states.v1`
- 加载 Q1 的 MuJoCo XML 作为背景场景
- 对收到的全部 `states` 画出 link 坐标系
- 不依赖 ROS2
- 不依赖 GMR

默认使用的 XML：

- `/home/hpx/HPX_LOCO_2/mimic_baseline/general_motion_tracker_whole_body_teleoperation/general_motion_tracker_whole_body_teleoperation/assets/Q1/mjcf/Q1_wo_hand.xml`

运行方式：

```bash
PYTHONPATH=/tmp/xsens_proto \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
python /home/hpx/HPX_LOCO_2/mimic_baseline/xsens_mvn_refactor_build/examples/zmq_mujoco_link_visualizer.py \
  --connect tcp://127.0.0.1:5555 \
  --topic xsens.link_states.v1
```

可选参数：

- `--xml-path`：指定 MuJoCo XML
- `--axis-length`：坐标轴长度
- `--shaft-width`：箭头粗细
- `--timeout-ms`：ZMQ 接收超时

## 局域网跨机测试步骤

假设发布机 IP 是 `192.168.1.10`。

发布机：

```bash
./build/src/link_states_test_publisher tcp://*:5555 100 50 1000
```

订阅机 C++：

```bash
./build/src/link_states_zmq_subscriber tcp://192.168.1.10:5555 xsens.link_states.v1 5000 3
```

订阅机 Python：

```bash
PYTHONPATH=/tmp/xsens_proto \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
python examples/python_subscriber.py \
  --connect tcp://192.168.1.10:5555 \
  --topic xsens.link_states.v1 \
  --count 3
```

网络前提：

- 两台机器互相可达：`ping 192.168.1.10`
- 发布机防火墙放行 `5555/tcp`
- Topic 保持完全一致：`xsens.link_states.v1`
- 使用同一份 `link_states.proto` 生成代码

验收清单：

- 订阅端打印 `schema_version=1`
- 订阅端打印 `frame_id=world`
- `states_size=2`
- 至少看到 `pelvis` 和 `left_hand`
- `pelvis` 位置为 `1.1,2.2,3.3`
- `left_hand` 位置为 `-1,-2,-3`
- 多个订阅端可同时收到同一发布端数据

## 目录说明

- `include/xsens_mvn_sdk` / `src/xsens_mvn_sdk`：SDK 解析代码
- `include/xsens_core` / `src/xsens_core`：核心采集与帧模型
- `include/xsens_transport` / `src/xsens_transport`：ZMQ + Protobuf 传输
- `include/xsens_apps` / `src/apps`：非 ROS2 应用层
- `proto/link_states.proto`：跨语言协议定义
- `examples/python_subscriber.py`：Python 订阅示例
- `examples/zmq_mujoco_link_visualizer.py`：Python MuJoCo 可视化示例
