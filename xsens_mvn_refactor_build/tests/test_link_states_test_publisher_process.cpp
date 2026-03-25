#include <chrono>
#include <cstdlib>
#include <sstream>
#include <string>
#include <thread>

#include <gtest/gtest.h>
#include <zmq.hpp>

#include "link_states.pb.h"

namespace
{
using namespace std::chrono_literals;
}

TEST(LinkStatesTestPublisherProcessTest, PublishedMessageIsVisibleToExternalSubscriber)
{
  const std::string address = "tcp://127.0.0.1:5562";
  const std::string bind_address = "tcp://*:5562";
  const std::string topic = "xsens.link_states.v1";

  zmq::context_t context(1);
  zmq::socket_t subscriber(context, zmq::socket_type::sub);
  subscriber.set(zmq::sockopt::subscribe, topic);
  subscriber.set(zmq::sockopt::rcvtimeo, 5000);
  subscriber.connect(address);
  std::this_thread::sleep_for(300ms);

  std::ostringstream command;
  command << LINK_STATES_TEST_PUBLISHER_PATH << " "
          << bind_address << " 1 0 1000";

  const int exit_code = std::system(command.str().c_str());
  ASSERT_EQ(exit_code, 0);

  zmq::message_t topic_frame;
  zmq::message_t payload_frame;
  ASSERT_TRUE(subscriber.recv(topic_frame, zmq::recv_flags::none).has_value());
  ASSERT_TRUE(subscriber.recv(payload_frame, zmq::recv_flags::none).has_value());
  EXPECT_EQ(topic_frame.to_string(), topic);

  xsens::transport::LinkStateArray proto_msg;
  ASSERT_TRUE(
    proto_msg.ParseFromArray(payload_frame.data(), static_cast<int>(payload_frame.size())));
  EXPECT_EQ(proto_msg.header().schema_version(), 1);
  EXPECT_EQ(proto_msg.header().frame_id(), "world");
  ASSERT_EQ(proto_msg.states_size(), 2);
  EXPECT_EQ(proto_msg.states(0).name(), "pelvis");
  EXPECT_EQ(proto_msg.states(1).name(), "left_hand");
}
