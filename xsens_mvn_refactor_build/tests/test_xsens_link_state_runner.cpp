#include <chrono>

#include <gtest/gtest.h>

#include "xsens_apps/XsensLinkStateRunner.h"

namespace
{
class FakeFrameSource : public xsens::apps::FrameSource
{
public:
  bool should_succeed = true;
  xsens::core::XsensFrame frame;

  bool copyFrame(xsens::core::XsensFrame& output) override
  {
    if (!should_succeed)
    {
      return false;
    }

    output = frame;
    return true;
  }
};

class FakeFrameSink : public xsens::apps::FrameSink
{
public:
  int publish_count = 0;
  xsens::core::XsensFrame last_frame;

  void publish(const xsens::core::XsensFrame& frame) override
  {
    ++publish_count;
    last_frame = frame;
  }
};

class FakeSleepStrategy : public xsens::apps::SleepStrategy
{
public:
  int sleep_call_count = 0;
  std::chrono::nanoseconds last_duration{0};

  void sleepFor(std::chrono::nanoseconds duration) override
  {
    ++sleep_call_count;
    last_duration = duration;
  }
};
}  // namespace

TEST(XsensLinkStateRunnerTest, PublishesNonEmptyFrameFromClientSnapshot)
{
  FakeFrameSource frame_source;
  FakeFrameSink frame_sink;

  frame_source.frame.frame_id = "world";
  xsens::core::LinkSample link_sample;
  link_sample.name = "pelvis";
  frame_source.frame.links.push_back(link_sample);

  xsens::apps::XsensLinkStateRunner runner(frame_source, frame_sink, 240.0);

  EXPECT_TRUE(runner.runOnce());
  EXPECT_EQ(frame_sink.publish_count, 1);
  EXPECT_EQ(frame_sink.last_frame.frame_id, "world");
  ASSERT_EQ(frame_sink.last_frame.links.size(), 1U);
  EXPECT_EQ(frame_sink.last_frame.links.front().name, "pelvis");
}

TEST(XsensLinkStateRunnerTest, SkipsPublishWhenFrameIsMissingOrEmpty)
{
  FakeFrameSource frame_source;
  FakeFrameSink frame_sink;

  xsens::apps::XsensLinkStateRunner runner(frame_source, frame_sink, 240.0);

  frame_source.should_succeed = false;
  EXPECT_FALSE(runner.runOnce());
  EXPECT_EQ(frame_sink.publish_count, 0);

  frame_source.should_succeed = true;
  frame_source.frame = xsens::core::XsensFrame{};
  EXPECT_FALSE(runner.runOnce());
  EXPECT_EQ(frame_sink.publish_count, 0);
}

TEST(XsensLinkStateRunnerTest, SleepsAccordingToConfiguredPublishRate)
{
  FakeFrameSource frame_source;
  FakeFrameSink frame_sink;
  FakeSleepStrategy sleep_strategy;

  xsens::core::LinkSample link_sample;
  link_sample.name = "pelvis";
  frame_source.frame.links.push_back(link_sample);

  xsens::apps::XsensLinkStateRunner runner(frame_source, frame_sink, 240.0, sleep_strategy);

  EXPECT_TRUE(runner.runOnce());
  EXPECT_EQ(sleep_strategy.sleep_call_count, 1);
  EXPECT_EQ(sleep_strategy.last_duration.count(), 4166666);
}
