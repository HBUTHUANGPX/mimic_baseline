#ifndef XSENS_APPS_XSENS_LINK_STATE_RUNNER_H
#define XSENS_APPS_XSENS_LINK_STATE_RUNNER_H

#include <chrono>

#include "xsens_core/XsensFrame.h"

class XSensClient;

namespace xsens
{
namespace transport
{
class LinkStatePublisher;
}  // namespace transport
}  // namespace xsens

namespace xsens
{
namespace apps
{
class FrameSource
{
public:
  virtual ~FrameSource() = default;
  virtual bool copyFrame(xsens::core::XsensFrame& frame) = 0;
};

class FrameSink
{
public:
  virtual ~FrameSink() = default;
  virtual void publish(const xsens::core::XsensFrame& frame) = 0;
};

class SleepStrategy
{
public:
  virtual ~SleepStrategy() = default;
  virtual void sleepFor(std::chrono::nanoseconds duration) = 0;
};

class ThreadSleepStrategy : public SleepStrategy
{
public:
  void sleepFor(std::chrono::nanoseconds duration) override;
};

class XSensClientFrameSource : public FrameSource
{
public:
  explicit XSensClientFrameSource(::XSensClient& client);
  bool copyFrame(xsens::core::XsensFrame& frame) override;

private:
  ::XSensClient& client_;
};

class ZmqFrameSink : public FrameSink
{
public:
  explicit ZmqFrameSink(xsens::transport::LinkStatePublisher& publisher);
  void publish(const xsens::core::XsensFrame& frame) override;

private:
  xsens::transport::LinkStatePublisher& publisher_;
};

class XsensLinkStateRunner
{
public:
  XsensLinkStateRunner(FrameSource& frame_source, FrameSink& frame_sink, double publish_rate_hz);
  XsensLinkStateRunner(
    FrameSource& frame_source,
    FrameSink& frame_sink,
    double publish_rate_hz,
    SleepStrategy& sleep_strategy);

  bool runOnce();
  std::chrono::nanoseconds publishPeriod() const;

private:
  FrameSource& frame_source_;
  FrameSink& frame_sink_;
  double publish_rate_hz_;
  ThreadSleepStrategy owned_sleep_strategy_;
  SleepStrategy* sleep_strategy_;
};
}  // namespace apps
}  // namespace xsens

#endif
