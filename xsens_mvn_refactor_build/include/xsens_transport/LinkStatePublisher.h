#ifndef XSENS_TRANSPORT_LINK_STATE_PUBLISHER_H
#define XSENS_TRANSPORT_LINK_STATE_PUBLISHER_H

#include <string>

#include <zmq.hpp>

#include "xsens_core/XsensFrame.h"

namespace xsens
{
namespace transport
{
class LinkStatePublisher
{
public:
  LinkStatePublisher(
    const std::string& bind_address,
    const std::string& topic = "xsens.link_states.v1",
    int sndhwm = 5,
    bool conflate = false);

  void publish(const xsens::core::XsensFrame& frame);
  const std::string& topic() const;

private:
  zmq::context_t context_;
  zmq::socket_t socket_;
  std::string topic_;
};
}  // namespace transport
}  // namespace xsens

#endif
