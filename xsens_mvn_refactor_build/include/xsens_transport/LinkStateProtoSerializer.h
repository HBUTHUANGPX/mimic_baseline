#ifndef XSENS_TRANSPORT_LINK_STATE_PROTO_SERIALIZER_H
#define XSENS_TRANSPORT_LINK_STATE_PROTO_SERIALIZER_H

#include <string>

#include "link_states.pb.h"
#include "xsens_core/XsensFrame.h"

namespace xsens
{
namespace transport
{
constexpr unsigned int kLinkStatesSchemaVersion = 1U;

::xsens::transport::LinkStateArray buildLinkStateArrayProto(
  const xsens::core::XsensFrame& frame);

std::string serializeLinkStateArrayProto(const xsens::core::XsensFrame& frame);
}  // namespace transport
}  // namespace xsens

#endif
