#include "xsens_apps/XsensLinkStateRunner.h"

#include <thread>

#include "xsens_core/XSensClient.h"
#include "xsens_transport/LinkStatePublisher.h"

namespace xsens
{
namespace apps
{
void ThreadSleepStrategy::sleepFor(std::chrono::nanoseconds duration)
{
  std::this_thread::sleep_for(duration);
}

XSensClientFrameSource::XSensClientFrameSource(::XSensClient& client)
  : client_(client)
{
}

bool XSensClientFrameSource::copyFrame(xsens::core::XsensFrame& frame, std::uint64_t& frame_sequence)
{
  return client_.copyFrame(frame, frame_sequence);
}

ZmqFrameSink::ZmqFrameSink(xsens::transport::LinkStatePublisher& publisher)
  : publisher_(publisher)
{
}

void ZmqFrameSink::publish(const xsens::core::XsensFrame& frame)
{
  publisher_.publish(frame);
}

XsensLinkStateRunner::XsensLinkStateRunner(
  FrameSource& frame_source,
  FrameSink& frame_sink,
  double publish_rate_hz)
  : frame_source_(frame_source),
    frame_sink_(frame_sink),
    publish_rate_hz_(publish_rate_hz),
    last_published_sequence_(0),
    sleep_strategy_(&owned_sleep_strategy_)
{
}

XsensLinkStateRunner::XsensLinkStateRunner(
  FrameSource& frame_source,
  FrameSink& frame_sink,
  double publish_rate_hz,
  SleepStrategy& sleep_strategy)
  : frame_source_(frame_source),
    frame_sink_(frame_sink),
    publish_rate_hz_(publish_rate_hz),
    last_published_sequence_(0),
    sleep_strategy_(&sleep_strategy)
{
}

bool XsensLinkStateRunner::runOnce()
{
  xsens::core::XsensFrame frame;
  std::uint64_t frame_sequence = 0;
  if (!frame_source_.copyFrame(frame, frame_sequence))
  {
    sleep_strategy_->sleepFor(publishPeriod());
    return false;
  }

  if (frame.links.empty())
  {
    sleep_strategy_->sleepFor(publishPeriod());
    return false;
  }

  if (frame_sequence <= last_published_sequence_)
  {
    sleep_strategy_->sleepFor(publishPeriod());
    return false;
  }

  frame_sink_.publish(frame);
  last_published_sequence_ = frame_sequence;
  sleep_strategy_->sleepFor(publishPeriod());
  return true;
}

std::chrono::nanoseconds XsensLinkStateRunner::publishPeriod() const
{
  if (publish_rate_hz_ <= 0.0)
  {
    return std::chrono::nanoseconds(0);
  }

  const auto period_ns = static_cast<long long>(1000000000.0 / publish_rate_hz_);
  return std::chrono::nanoseconds(period_ns);
}
}  // namespace apps
}  // namespace xsens
