#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

#include "xsens_apps/TestFrameFactory.h"
#include "xsens_transport/LinkStatePublisher.h"

int main(int argc, char** argv)
{
  const std::string bind_address = argc > 1 ? argv[1] : "tcp://*:5555";
  const int publish_count = argc > 2 ? std::atoi(argv[2]) : 30;
  const int period_ms = argc > 3 ? std::atoi(argv[3]) : 100;
  const int warmup_ms = argc > 4 ? std::atoi(argv[4]) : 500;
  const std::string topic = argc > 5 ? argv[5] : "xsens.link_states.v1";

  try
  {
    xsens::transport::LinkStatePublisher publisher(bind_address, topic);
    std::cout << "publisher_bind=" << bind_address << std::endl;
    std::cout << "publisher_topic=" << publisher.topic() << std::endl;
    std::cout << "publisher_warmup_ms=" << warmup_ms << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(warmup_ms));

    const auto frame = xsens::apps::buildDeterministicTestFrame();
    for (int index = 0; index < publish_count; ++index)
    {
      publisher.publish(frame);
      std::cout << "published_frame=" << (index + 1)
                << " frame_id=" << frame.frame_id
                << " states_size=" << frame.links.size()
                << std::endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(period_ms));
    }
  }
  catch (const std::exception& exception)
  {
    std::cerr << "publisher_error=" << exception.what() << std::endl;
    return 1;
  }

  return 0;
}
