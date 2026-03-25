#include <atomic>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "xsens_apps/XsensLinkStateRunner.h"
#include "xsens_apps/LinkStateFrameFilter.h"
#include "xsens_core/XSensClient.h"
#include "xsens_transport/LinkStatePublisher.h"

namespace
{
std::atomic_bool g_should_stop{false};

void handleSignal(int)
{
  g_should_stop.store(true);
}
}  // namespace

int main(int argc, char** argv)
{
  const int udp_port = argc > 1 ? std::atoi(argv[1]) : 8001;
  const std::string bind_address = argc > 2 ? argv[2] : "tcp://*:5555";
  const std::string topic = argc > 3 ? argv[3] : "xsens.link_states.v1";
  const double publish_rate_hz = argc > 4 ? std::atof(argv[4]) : 240.0;

  if (publish_rate_hz <= 0.0)
  {
    std::cerr << "invalid_publish_rate_hz=" << publish_rate_hz << std::endl;
    return 1;
  }

  std::signal(SIGINT, handleSignal);
  std::signal(SIGTERM, handleSignal);

  try
  {
    std::cout << "xsens_udp_port=" << udp_port << std::endl;
    std::cout << "zmq_bind_address=" << bind_address << std::endl;
    std::cout << "zmq_topic=" << topic << std::endl;
    std::cout << "publish_rate_hz=" << publish_rate_hz << std::endl;

    auto client = std::make_shared<XSensClient>(udp_port);
    if (!client->init())
    {
      std::cerr << "xsens_client_init_failed=true" << std::endl;
      return 1;
    }

    xsens::transport::LinkStatePublisher publisher(bind_address, topic);
    xsens::apps::XSensClientFrameSource frame_source(*client);
    class FilteringFrameSink final : public xsens::apps::FrameSink
    {
    public:
      explicit FilteringFrameSink(xsens::transport::LinkStatePublisher& publisher)
        : publisher_(publisher)
      {
      }

      void publish(const xsens::core::XsensFrame& frame) override
      {
        xsens::core::XsensFrame filtered_frame = frame;
        xsens::apps::filterOutHandLinks(filtered_frame);
        if (!filtered_frame.links.empty())
        {
          publisher_.publish(filtered_frame);
        }
      }

    private:
      xsens::transport::LinkStatePublisher& publisher_;
    };

    FilteringFrameSink frame_sink(publisher);
    xsens::apps::XsensLinkStateRunner runner(frame_source, frame_sink, publish_rate_hz);

    std::cout << "publisher_ready=true" << std::endl;
    while (!g_should_stop.load())
    {
      runner.runOnce();
    }
  }
  catch (const std::exception& exception)
  {
    std::cerr << "xsens_link_states_publisher_error=" << exception.what() << std::endl;
    return 1;
  }

  std::cout << "publisher_stopped=true" << std::endl;
  return 0;
}
