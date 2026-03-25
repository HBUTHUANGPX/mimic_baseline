#include <chrono>
#include <string>
#include <thread>

#include <gtest/gtest.h>
#include <zmq.hpp>

#include "link_states.pb.h"
#include "xsens_apps/TestFrameFactory.h"
#include "xsens_transport/LinkStatePublisher.h"

namespace
{
using namespace std::chrono_literals;
}

TEST(ZmqLoopbackTest, PublishesDeterministicFrameOverMultipartPubSub)
{
  const std::string address = "tcp://127.0.0.1:5561";
  const std::string topic = "xsens.link_states.v1";

  zmq::context_t context(1);
  zmq::socket_t subscriber(context, zmq::socket_type::sub);
  subscriber.set(zmq::sockopt::subscribe, topic);
  subscriber.set(zmq::sockopt::rcvtimeo, 3000);
  subscriber.connect(address);

  xsens::transport::LinkStatePublisher publisher(address, topic);
  std::this_thread::sleep_for(500ms);

  const auto frame = xsens::apps::buildDeterministicTestFrame();
  publisher.publish(frame);

  zmq::message_t topic_frame;
  zmq::message_t payload_frame;
  ASSERT_TRUE(subscriber.recv(topic_frame, zmq::recv_flags::none).has_value());
  ASSERT_TRUE(subscriber.recv(payload_frame, zmq::recv_flags::none).has_value());

  EXPECT_EQ(topic_frame.to_string(), topic);

  xsens::transport::LinkStateArray proto_msg;
  ASSERT_TRUE(
    proto_msg.ParseFromArray(payload_frame.data(), static_cast<int>(payload_frame.size())));
  EXPECT_EQ(proto_msg.header().frame_id(), "world");
  ASSERT_EQ(proto_msg.states_size(), 2);
  EXPECT_EQ(proto_msg.states(0).name(), "pelvis");
  EXPECT_DOUBLE_EQ(proto_msg.states(0).pose().position().x(), 1.1);
  EXPECT_EQ(proto_msg.states(1).name(), "left_hand");
  EXPECT_DOUBLE_EQ(proto_msg.states(1).pose().position().z(), -3.0);
}
