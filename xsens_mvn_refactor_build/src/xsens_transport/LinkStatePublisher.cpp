#include "xsens_transport/LinkStatePublisher.h"

#include <stdexcept>

#include "xsens_transport/LinkStateProtoSerializer.h"

namespace xsens
{
namespace transport
{
LinkStatePublisher::LinkStatePublisher(
  const std::string& bind_address,
  const std::string& topic,
  int sndhwm,
  bool conflate)
  : context_(1),
    socket_(context_, zmq::socket_type::pub),
    topic_(topic)
{
  socket_.set(zmq::sockopt::sndhwm, sndhwm);
  socket_.set(zmq::sockopt::linger, 0);
  if (conflate)
  {
    socket_.set(zmq::sockopt::conflate, true);
  }
  socket_.bind(bind_address);
}

void LinkStatePublisher::publish(const xsens::core::XsensFrame& frame)
{
  const std::string payload = serializeLinkStateArrayProto(frame);
  const auto topic_sent = socket_.send(zmq::buffer(topic_), zmq::send_flags::sndmore);
  const auto payload_sent = socket_.send(zmq::buffer(payload), zmq::send_flags::none);
  if (!topic_sent || !payload_sent)
  {
    throw std::runtime_error("Failed to publish LinkStateArray over ZMQ");
  }
}

const std::string& LinkStatePublisher::topic() const
{
  return topic_;
}
}  // namespace transport
}  // namespace xsens
