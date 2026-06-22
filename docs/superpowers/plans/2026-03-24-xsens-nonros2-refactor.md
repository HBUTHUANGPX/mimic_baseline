# XSens Non-ROS2 Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a parallel non-ROS2 version of `xsens_mvn_ros2` under `~/HPX_LOCO_2/mimic_baseline/xsens_mvn_refactor_build` with standard CMake, protobuf + ZMQ transport, and continuous module-level simulation tests.

**Architecture:** Create a standalone CMake project that reuses the validated XSens parsing and model-building logic but removes ROS2 dependencies entirely. Split the new project into `xsens_sdk`, `xsens_core`, `xsens_transport`, and `apps` so each layer has a narrow responsibility and can be tested independently before full loopback integration.

**Tech Stack:** C++14, CMake, Eigen3, Boost, Protobuf, ZeroMQ, GTest, Conda environment `mimic_baseline`

---

### Task 1: Scaffold the standalone CMake project

**Files:**
- Create: `/home/jerry_huang/HPX_Loco/mimic_baseline/xsens_mvn_refactor_build/CMakeLists.txt`
- Create: `/home/jerry_huang/HPX_Loco/mimic_baseline/xsens_mvn_refactor_build/cmake/`
- Create: `/home/jerry_huang/HPX_Loco/mimic_baseline/xsens_mvn_refactor_build/include/`
- Create: `/home/jerry_huang/HPX_Loco/mimic_baseline/xsens_mvn_refactor_build/src/`
- Create: `/home/jerry_huang/HPX_Loco/mimic_baseline/xsens_mvn_refactor_build/tests/`
- Create: `/home/jerry_huang/HPX_Loco/mimic_baseline/xsens_mvn_refactor_build/proto/`

- [ ] **Step 1: Write the top-level standalone CMakeLists**

Include:
- `project(xsens_mvn_refactor_build)`
- `set(CMAKE_CXX_STANDARD 14)`
- `find_package(Eigen3 REQUIRED)`
- `find_package(Boost REQUIRED COMPONENTS system thread chrono)`
- `find_package(Protobuf REQUIRED)`
- `find_package(PkgConfig REQUIRED)`
- `pkg_check_modules(ZMQ REQUIRED libzmq)`
- `find_package(GTest REQUIRED)`
- `enable_testing()`

- [ ] **Step 2: Run CMake configure to verify dependency discovery**

Run:
```bash
cd /home/jerry_huang/HPX_Loco/mimic_baseline/xsens_mvn_refactor_build
source ~/miniconda3/bin/activate mimic_baseline
cmake -S . -B build
```

Expected: configure succeeds and prints found Eigen3 / Boost / Protobuf / ZMQ / GTest.

- [ ] **Step 3: Commit**

```bash
git add xsens_mvn_refactor_build/CMakeLists.txt xsens_mvn_refactor_build/cmake
git commit -m "build: scaffold standalone xsens refactor project"
```

### Task 2: Port `xsens_sdk` and lock parser behavior with tests

**Files:**
- Create: `/home/jerry_huang/HPX_Loco/mimic_baseline/xsens_mvn_refactor_build/include/xsens_sdk/...`
- Create: `/home/jerry_huang/HPX_Loco/mimic_baseline/xsens_mvn_refactor_build/src/xsens_sdk/...`
- Test: `/home/jerry_huang/HPX_Loco/mimic_baseline/xsens_mvn_refactor_build/tests/test_parsermanager_cache.cpp`

- [ ] **Step 1: Copy the SDK headers and sources from the ROS2 package**

Source of truth:
- `/home/hpx/HPX_Loco_3/pack/Q1_control_0203_1944/src/xsens_mvn_ros2/xsens_mvn_sdk/include/xsens_mvn_sdk`
- `/home/hpx/HPX_Loco_3/pack/Q1_control_0203_1944/src/xsens_mvn_ros2/xsens_mvn_sdk/src`

- [ ] **Step 2: Add a standalone `xsens_sdk` library target to CMake**

Link:
- `Boost::system`
- `Boost::thread`
- `Boost::chrono`
- `Eigen3`

- [ ] **Step 3: Write the failing parser cache regression test**

Test cases:
- Quaternion cache survives later COM datagram
- Joint-angle cache survives later quaternion datagram

Use the existing validated logic from:
- `/home/hpx/HPX_Loco_3/pack/Q1_control_0203_1944/src/xsens_mvn_ros2/test/test_parsermanager_cache.cpp`

- [ ] **Step 4: Run the parser cache test**

Run:
```bash
cd /home/jerry_huang/HPX_Loco/mimic_baseline/xsens_mvn_refactor_build
source ~/miniconda3/bin/activate mimic_baseline
cmake -S . -B build
cmake --build build --target test_parsermanager_cache
ctest --test-dir build -R test_parsermanager_cache --output-on-failure
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add xsens_mvn_refactor_build/include/xsens_sdk xsens_mvn_refactor_build/src/xsens_sdk xsens_mvn_refactor_build/tests/test_parsermanager_cache.cpp xsens_mvn_refactor_build/CMakeLists.txt
git commit -m "feat: port xsens sdk with parser cache tests"
```

### Task 3: Port `xsens_core` and remove ROS2 types

**Files:**
- Create: `/home/jerry_huang/HPX_Loco/mimic_baseline/xsens_mvn_refactor_build/include/xsens_core/HumanDataHandler.h`
- Create: `/home/jerry_huang/HPX_Loco/mimic_baseline/xsens_mvn_refactor_build/include/xsens_core/Socket.h`
- Create: `/home/jerry_huang/HPX_Loco/mimic_baseline/xsens_mvn_refactor_build/include/xsens_core/XSensClient.h`
- Create: `/home/jerry_huang/HPX_Loco/mimic_baseline/xsens_mvn_refactor_build/include/xsens_core/XsensFrame.h`
- Create: `/home/jerry_huang/HPX_Loco/mimic_baseline/xsens_mvn_refactor_build/src/xsens_core/HumanDataHandler.cpp`
- Create: `/home/jerry_huang/HPX_Loco/mimic_baseline/xsens_mvn_refactor_build/src/xsens_core/Socket.cpp`
- Create: `/home/jerry_huang/HPX_Loco/mimic_baseline/xsens_mvn_refactor_build/src/xsens_core/XSensClient.cpp`
- Test: `/home/jerry_huang/HPX_Loco/mimic_baseline/xsens_mvn_refactor_build/tests/test_human_data_handler.cpp`

- [ ] **Step 1: Copy `HumanDataHandler`, `Socket`, and `XSensClient` into the new tree**

Source of truth:
- `/home/hpx/HPX_Loco_3/pack/Q1_control_0203_1944/src/xsens_mvn_ros2/include/xsens_mvn_ros2`
- `/home/hpx/HPX_Loco_3/pack/Q1_control_0203_1944/src/xsens_mvn_ros2/src/xsens_client`

- [ ] **Step 2: Replace ROS2 include paths and package-local include names**

Examples:
- replace `xsens_mvn_ros2/HumanDataHandler.h` with `xsens_core/HumanDataHandler.h`
- remove any `rclcpp` dependency from the library

- [ ] **Step 3: Introduce a pure C++ frame snapshot type**

Add `XsensFrame.h` with:
- `struct LinkSample`
- `struct JointSample`
- `struct ComSample`
- `struct XsensFrame`

The frame should carry:
- timestamp or monotonic sample id
- link name, pose, twist, accel
- joint name, angles
- COM

- [ ] **Step 4: Add minimal snapshot extraction methods to `XSensClient`**

Example interface:
```cpp
bool copyFrame(XsensFrame& frame) const;
```

- [ ] **Step 5: Write the failing `HumanDataHandler` unit test**

Test cases:
- set/get joint angles
- set/get link pose and state
- COM setter/getter

- [ ] **Step 6: Run the focused core tests**

Run:
```bash
cmake --build build --target test_human_data_handler
ctest --test-dir build -R "test_human_data_handler|test_parsermanager_cache" --output-on-failure
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add xsens_mvn_refactor_build/include/xsens_core xsens_mvn_refactor_build/src/xsens_core xsens_mvn_refactor_build/tests/test_human_data_handler.cpp xsens_mvn_refactor_build/CMakeLists.txt
git commit -m "feat: port xsens core without ros2 types"
```

### Task 4: Port protobuf + ZMQ transport to pure C++

**Files:**
- Create: `/home/jerry_huang/HPX_Loco/mimic_baseline/xsens_mvn_refactor_build/proto/link_states.proto`
- Create: `/home/jerry_huang/HPX_Loco/mimic_baseline/xsens_mvn_refactor_build/include/xsens_transport/LinkStateProtoSerializer.h`
- Create: `/home/jerry_huang/HPX_Loco/mimic_baseline/xsens_mvn_refactor_build/include/xsens_transport/LinkStatePublisher.h`
- Create: `/home/jerry_huang/HPX_Loco/mimic_baseline/xsens_mvn_refactor_build/src/xsens_transport/LinkStateProtoSerializer.cpp`
- Create: `/home/jerry_huang/HPX_Loco/mimic_baseline/xsens_mvn_refactor_build/src/xsens_transport/LinkStatePublisher.cpp`
- Test: `/home/jerry_huang/HPX_Loco/mimic_baseline/xsens_mvn_refactor_build/tests/test_link_state_proto_serializer.cpp`

- [ ] **Step 1: Copy the validated `.proto` schema into the new project**

Source of truth:
- `/home/hpx/HPX_Loco_3/pack/Q1_control_0203_1944/src/xsens_mvn_ros2/proto/link_states.proto`

- [ ] **Step 2: Write the failing serializer test against `XsensFrame`**

Test cases:
- schema version == 1
- top-level frame id matches input
- link names and positions survive serialization mapping

- [ ] **Step 3: Implement serializer helpers against pure C++ frame types**

Do not depend on ROS messages.

- [ ] **Step 4: Add a small `LinkStatePublisher` wrapper over ZeroMQ `PUB`**

Requirements:
- bind address parameter
- topic string parameter
- multipart send: topic + protobuf payload
- default `conflate = false`

- [ ] **Step 5: Run serializer tests**

Run:
```bash
cmake --build build --target test_link_state_proto_serializer
ctest --test-dir build -R test_link_state_proto_serializer --output-on-failure
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add xsens_mvn_refactor_build/proto xsens_mvn_refactor_build/include/xsens_transport xsens_mvn_refactor_build/src/xsens_transport xsens_mvn_refactor_build/tests/test_link_state_proto_serializer.cpp xsens_mvn_refactor_build/CMakeLists.txt
git commit -m "feat: add pure c++ protobuf zmq transport"
```

### Task 5: Build non-ROS2 apps and local loopback validation

**Files:**
- Create: `/home/jerry_huang/HPX_Loco/mimic_baseline/xsens_mvn_refactor_build/src/apps/xsens_zmq_publisher.cpp`
- Create: `/home/jerry_huang/HPX_Loco/mimic_baseline/xsens_mvn_refactor_build/src/apps/link_states_test_publisher.cpp`
- Create: `/home/jerry_huang/HPX_Loco/mimic_baseline/xsens_mvn_refactor_build/src/apps/link_states_zmq_subscriber.cpp`
- Create: `/home/jerry_huang/HPX_Loco/mimic_baseline/xsens_mvn_refactor_build/tests/test_zmq_loopback.cpp`
- Create: `/home/jerry_huang/HPX_Loco/mimic_baseline/xsens_mvn_refactor_build/README.md`

- [ ] **Step 1: Implement the production non-ROS2 publisher app**

Responsibilities:
- parse CLI flags (`--udp-port`, `--bind`, `--topic`)
- create `XSensClient`
- initialize the suit connection
- poll snapshots
- publish protobuf over ZMQ

- [ ] **Step 2: Implement standalone loopback tools**

Tools:
- `link_states_test_publisher`: emits deterministic fake `XsensFrame`
- `link_states_zmq_subscriber`: decodes and prints received protobuf

- [ ] **Step 3: Write the failing ZMQ loopback integration test**

Test cases:
- publisher emits topic `xsens.link_states.v1`
- subscriber receives a payload
- decoded link count and names match the fake frame

- [ ] **Step 4: Run the full local loopback suite**

Run:
```bash
cmake --build build
ctest --test-dir build --output-on-failure
```

Expected: all tests pass, including loopback integration.

- [ ] **Step 5: Execute a manual smoke test**

Run in separate terminals:
```bash
./build/bin/link_states_zmq_subscriber tcp://127.0.0.1:5555 xsens.link_states.v1 15000
./build/bin/link_states_test_publisher
```

Expected subscriber output contains:
```text
topic=xsens.link_states.v1
schema_version=1
states_size=2
state[0].name=pelvis
```

- [ ] **Step 6: Document build and run commands in README**

Include:
- dependency install commands
- CMake build commands
- loopback commands
- note that multipart PUB/SUB must keep `conflate` disabled

- [ ] **Step 7: Commit**

```bash
git add xsens_mvn_refactor_build/src/apps xsens_mvn_refactor_build/tests/test_zmq_loopback.cpp xsens_mvn_refactor_build/README.md xsens_mvn_refactor_build/CMakeLists.txt
git commit -m "feat: add non-ros2 apps and loopback validation"
```
