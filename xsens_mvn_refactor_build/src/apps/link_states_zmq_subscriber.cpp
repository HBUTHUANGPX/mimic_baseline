#include <cstdlib>
#include <iostream>
#include <string>

#include <google/protobuf/stubs/common.h>
#include <zmq.hpp>

#include "link_states.pb.h"

int main(int argc, char** argv)
{
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  const std::string connect_address = argc > 1 ? argv[1] : "tcp://127.0.0.1:5555";
  const std::string topic = argc > 2 ? argv[2] : "xsens.link_states.v1";
  const int timeout_ms = argc > 3 ? std::atoi(argv[3]) : 5000;
  const int receive_count = argc > 4 ? std::atoi(argv[4]) : 2;

  zmq::context_t context(1);
  zmq::socket_t subscriber(context, zmq::socket_type::sub);
  subscriber.set(zmq::sockopt::subscribe, topic);
  subscriber.set(zmq::sockopt::rcvtimeo, timeout_ms);
  subscriber.connect(connect_address);

  std::cout << "subscriber_connect=" << connect_address << std::endl;
  std::cout << "subscriber_topic=" << topic << std::endl;
  std::cout << "subscriber_receive_count=" << receive_count << std::endl;

  for (int received = 0; received < receive_count; ++received)
  {
    zmq::message_t topic_frame;
    zmq::message_t payload_frame;
    if (!subscriber.recv(topic_frame, zmq::recv_flags::none))
    {
      std::cerr << "recv_error=topic_timeout index=" << received << std::endl;
      return 1;
    }
    if (!subscriber.recv(payload_frame, zmq::recv_flags::none))
    {
      std::cerr << "recv_error=payload_timeout index=" << received << std::endl;
      return 1;
    }
    const std::string received_topic = topic_frame.to_string();
    if (received_topic != topic)
    {
      std::cerr << "recv_error=unexpected_topic index=" << received
                << " expected=" << topic
                << " actual=" << received_topic << std::endl;
      return 1;
    }

    xsens::transport::LinkStateArray proto_msg;
    if (!proto_msg.ParseFromArray(payload_frame.data(), static_cast<int>(payload_frame.size())))
    {
      std::cerr << "recv_error=parse_failed index=" << received << std::endl;
      return 1;
    }

    std::cout << "topic=" << received_topic << std::endl;
    std::cout << "schema_version=" << proto_msg.header().schema_version() << std::endl;
    std::cout << "frame_id=" << proto_msg.header().frame_id() << std::endl;
    std::cout << "states_size=" << proto_msg.states_size() << std::endl;

    for (int index = 0; index < proto_msg.states_size(); ++index)
    {
      const auto& state = proto_msg.states(index);
      std::cout << "state[" << index << "].name=" << state.name() << std::endl;
      std::cout << "state[" << index << "].position="
                << state.pose().position().x() << ","
                << state.pose().position().y() << ","
                << state.pose().position().z() << std::endl;
    }
  }

  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
