#include "xsens_transport/LinkStateProtoSerializer.h"

#include <stdexcept>

namespace xsens
{
namespace transport
{
namespace
{
void fillVector3(::xsens::transport::Vector3* proto_vector, const Eigen::Vector3d& vector)
{
  proto_vector->set_x(vector.x());
  proto_vector->set_y(vector.y());
  proto_vector->set_z(vector.z());
}
}  // namespace

::xsens::transport::LinkStateArray buildLinkStateArrayProto(
  const xsens::core::XsensFrame& frame)
{
  ::xsens::transport::LinkStateArray proto_msg;

  auto* header = proto_msg.mutable_header();
  header->set_schema_version(kLinkStatesSchemaVersion);
  header->set_stamp_sec(frame.stamp_sec);
  header->set_stamp_nanosec(frame.stamp_nanosec);
  header->set_frame_id(frame.frame_id);

  for (const auto& link : frame.links)
  {
    auto* proto_state = proto_msg.add_states();
    proto_state->set_name(link.name);

    auto* pose = proto_state->mutable_pose();
    fillVector3(pose->mutable_position(), link.position);
    pose->mutable_orientation()->set_x(link.orientation.x());
    pose->mutable_orientation()->set_y(link.orientation.y());
    pose->mutable_orientation()->set_z(link.orientation.z());
    pose->mutable_orientation()->set_w(link.orientation.w());

    auto* twist = proto_state->mutable_twist();
    fillVector3(twist->mutable_linear(), link.linear_velocity);
    fillVector3(twist->mutable_angular(), link.angular_velocity);

    auto* accel = proto_state->mutable_accel();
    fillVector3(accel->mutable_linear(), link.linear_acceleration);
    fillVector3(accel->mutable_angular(), link.angular_acceleration);
  }

  return proto_msg;
}

std::string serializeLinkStateArrayProto(const xsens::core::XsensFrame& frame)
{
  const auto proto_msg = buildLinkStateArrayProto(frame);
  std::string payload;
  if (!proto_msg.SerializeToString(&payload))
  {
    throw std::runtime_error("Failed to serialize LinkStateArray protobuf payload");
  }
  return payload;
}
}  // namespace transport
}  // namespace xsens
