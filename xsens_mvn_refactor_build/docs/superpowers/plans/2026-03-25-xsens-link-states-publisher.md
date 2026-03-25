# XSens Link States Publisher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a non-ROS2 runtime executable that reads real XSens MVN UDP data through `XSensClient` and publishes `link_states` snapshots over Protobuf + ZMQ PUB/SUB.

**Architecture:** Keep the transport contract unchanged and add a thin app-layer runner between `XSensClient` and `LinkStatePublisher`. The runner owns the main loop, rate limiting, empty-frame filtering, and logging; `XSensClient` remains the UDP acquisition source and `xsens_transport` remains the wire serializer/publisher.

**Tech Stack:** C++17, CMake, Eigen, Boost, existing `XSensClient`, existing `LinkStatePublisher`, GTest, ZMQ, Protobuf

---

### Task 1: Add a Testable App Runner

**Files:**
- Create: `include/xsens_apps/XsensLinkStateRunner.h`
- Create: `src/apps/XsensLinkStateRunner.cpp`
- Modify: `src/CMakeLists.txt`
- Test: `tests/test_xsens_link_state_runner.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write the failing test**

```cpp
TEST(XsensLinkStateRunnerTest, PublishesNonEmptyFrameFromClientSnapshot)
{
  FakeFrameSource frame_source;
  FakeFrameSink frame_sink;
  frame_source.frame.frame_id = "world";
  frame_source.frame.links.push_back(xsens::core::LinkSample{});

  xsens::apps::XsensLinkStateRunner runner(frame_source, frame_sink, 240.0);

  EXPECT_TRUE(runner.runOnce());
  EXPECT_EQ(frame_sink.publish_count, 1);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source ~/miniconda3/bin/activate mimic_baseline && cmake --build build && ctest --test-dir build -R test_xsens_link_state_runner --output-on-failure`
Expected: FAIL because `XsensLinkStateRunner` and test doubles do not exist yet

- [ ] **Step 3: Write minimal implementation**

```cpp
struct FrameSource {
  virtual ~FrameSource() = default;
  virtual bool copyFrame(xsens::core::XsensFrame& frame) = 0;
};

struct FrameSink {
  virtual ~FrameSink() = default;
  virtual void publish(const xsens::core::XsensFrame& frame) = 0;
};

class XsensLinkStateRunner {
public:
  bool runOnce();
};
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source ~/miniconda3/bin/activate mimic_baseline && cmake --build build && ctest --test-dir build -R test_xsens_link_state_runner --output-on-failure`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add include/xsens_apps/XsensLinkStateRunner.h src/apps/XsensLinkStateRunner.cpp src/CMakeLists.txt tests/test_xsens_link_state_runner.cpp tests/CMakeLists.txt
git commit -m "feat: add xsens link state runner"
```

### Task 2: Wire Real XSensClient and ZMQ Publisher Into a Runtime Executable

**Files:**
- Create: `src/apps/xsens_link_states_publisher.cpp`
- Modify: `src/CMakeLists.txt`
- Test: `tests/test_xsens_link_state_runner.cpp`

- [ ] **Step 1: Write the failing test**

```cpp
TEST(XsensLinkStateRunnerTest, SkipsPublishWhenFrameIsMissingOrEmpty)
{
  FakeFrameSource frame_source;
  FakeFrameSink frame_sink;
  xsens::apps::XsensLinkStateRunner runner(frame_source, frame_sink, 240.0);

  EXPECT_FALSE(runner.runOnce());
  EXPECT_EQ(frame_sink.publish_count, 0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source ~/miniconda3/bin/activate mimic_baseline && cmake --build build && ctest --test-dir build -R test_xsens_link_state_runner --output-on-failure`
Expected: FAIL because empty/missing frame handling is not implemented yet

- [ ] **Step 3: Write minimal implementation**

```cpp
int main(int argc, char** argv)
{
  XSensClient client(udp_port);
  client.init();
  xsens::transport::LinkStatePublisher publisher(bind_address, topic);
  xsens::apps::RealXsensFrameSource frame_source(client);
  xsens::apps::ZmqFrameSink frame_sink(publisher);
  xsens::apps::XsensLinkStateRunner runner(frame_source, frame_sink, publish_rate_hz);
  return runner.runForever();
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source ~/miniconda3/bin/activate mimic_baseline && cmake --build build && ctest --test-dir build -R test_xsens_link_state_runner --output-on-failure`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/apps/xsens_link_states_publisher.cpp src/CMakeLists.txt tests/test_xsens_link_state_runner.cpp
git commit -m "feat: add real xsens link states publisher app"
```

### Task 3: Verify End-to-End Local Runtime Behavior

**Files:**
- Modify: `README.md`
- Test: `tests/test_xsens_link_state_runner.cpp`

- [ ] **Step 1: Write the failing test**

```cpp
TEST(XsensLinkStateRunnerTest, SleepsAccordingToConfiguredPublishRate)
{
  FakeFrameSource frame_source;
  FakeFrameSink frame_sink;
  FakeSleeper sleeper;
  frame_source.frame.links.push_back(xsens::core::LinkSample{});

  xsens::apps::XsensLinkStateRunner runner(frame_source, frame_sink, 240.0, sleeper);
  runner.runOnce();

  EXPECT_EQ(sleeper.last_sleep_ns, std::chrono::nanoseconds(4166666));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source ~/miniconda3/bin/activate mimic_baseline && cmake --build build && ctest --test-dir build -R test_xsens_link_state_runner --output-on-failure`
Expected: FAIL because no explicit rate pacing hook exists yet

- [ ] **Step 3: Write minimal implementation**

```cpp
class SleepStrategy {
public:
  virtual ~SleepStrategy() = default;
  virtual void sleepFor(std::chrono::nanoseconds duration) = 0;
};
```

Update `README.md` with the real runtime commands:

```bash
./build/src/xsens_link_states_publisher 8001 tcp://*:5555 xsens.link_states.v1 240
./build/src/link_states_zmq_subscriber tcp://127.0.0.1:5555 xsens.link_states.v1 2
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source ~/miniconda3/bin/activate mimic_baseline && cmake --build build && ctest --test-dir build -R test_xsens_link_state_runner --output-on-failure`
Expected: PASS

- [ ] **Step 5: Run focused and full verification**

Run: `source ~/miniconda3/bin/activate mimic_baseline && ctest --test-dir build -R 'test_xsens_link_state_runner|test_zmq_loopback|test_link_states_test_publisher_process|test_link_states_zmq_subscriber_process' --output-on-failure`
Expected: PASS

Run: `source ~/miniconda3/bin/activate mimic_baseline && ctest --test-dir build --output-on-failure`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add README.md tests/test_xsens_link_state_runner.cpp
git commit -m "test: verify xsens link states runtime loop"
```
