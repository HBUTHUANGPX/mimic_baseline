#include <chrono>
#include <cstdlib>
#include <future>
#include <sstream>
#include <string>
#include <thread>

#include <gtest/gtest.h>
#include <zmq.hpp>

#include "xsens_apps/TestFrameFactory.h"
#include "xsens_transport/LinkStateProtoSerializer.h"

namespace
{
using namespace std::chrono_literals;
}

TEST(LinkStatesZmqSubscriberProcessTest, RejectsUnexpectedTopicFrames)
{
  const std::string expected_topic = "xsens.link_states.v1";
  const std::string wrong_topic = "xsens.link_states.v1_extra";
  const std::string connect_address = "tcp://127.0.0.1:5566";
  const std::string bind_address = "tcp://*:5566";

  std::ostringstream command;
  command << LINK_STATES_ZMQ_SUBSCRIBER_PATH << " "
          << connect_address << " " << expected_topic << " 3000 1";
  const std::string command_string = command.str();

  auto subscriber_future = std::async(
    std::launch::async,
    [command_string]() {
      return std::system(command_string.c_str());
    });

  std::this_thread::sleep_for(300ms);

  zmq::context_t context(1);
  zmq::socket_t publisher(context, zmq::socket_type::pub);
  publisher.set(zmq::sockopt::linger, 0);
  publisher.bind(bind_address);
  std::this_thread::sleep_for(1000ms);

  const auto payload = xsens::transport::serializeLinkStateArrayProto(
    xsens::apps::buildDeterministicTestFrame());
  ASSERT_TRUE(publisher.send(zmq::buffer(wrong_topic), zmq::send_flags::sndmore).has_value());
  ASSERT_TRUE(publisher.send(zmq::buffer(payload), zmq::send_flags::none).has_value());

  const int exit_code = subscriber_future.get();
  EXPECT_NE(exit_code, 0);
}
