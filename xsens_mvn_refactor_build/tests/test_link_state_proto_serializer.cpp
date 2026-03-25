#include <gtest/gtest.h>

#include "xsens_transport/LinkStateProtoSerializer.h"

TEST(LinkStateProtoSerializerTest, ConvertsFrameToProtobuf)
{
  xsens::core::XsensFrame frame;
  frame.frame_id = "world";
  frame.stamp_sec = 123;
  frame.stamp_nanosec = 456U;

  xsens::core::LinkSample pelvis;
  pelvis.name = "pelvis";
  pelvis.position = Eigen::Vector3d(1.0, 2.0, 3.0);
  pelvis.orientation = Eigen::Quaterniond(0.4, 0.1, 0.2, 0.3);
  pelvis.linear_velocity = Eigen::Vector3d(4.0, 5.0, 6.0);
  pelvis.angular_velocity = Eigen::Vector3d(7.0, 8.0, 9.0);
  pelvis.linear_acceleration = Eigen::Vector3d(10.0, 11.0, 12.0);
  pelvis.angular_acceleration = Eigen::Vector3d(13.0, 14.0, 15.0);
  frame.links.push_back(pelvis);

  xsens::core::LinkSample hand;
  hand.name = "left_hand";
  hand.position = Eigen::Vector3d(-1.0, -2.0, -3.0);
  hand.orientation = Eigen::Quaterniond::Identity();
  frame.links.push_back(hand);

  const auto proto_msg = xsens::transport::buildLinkStateArrayProto(frame);

  EXPECT_EQ(proto_msg.header().schema_version(), xsens::transport::kLinkStatesSchemaVersion);
  EXPECT_EQ(proto_msg.header().frame_id(), "world");
  EXPECT_EQ(proto_msg.header().stamp_sec(), 123);
  EXPECT_EQ(proto_msg.header().stamp_nanosec(), 456U);
  ASSERT_EQ(proto_msg.states_size(), 2);

  const auto& proto_pelvis = proto_msg.states(0);
  EXPECT_EQ(proto_pelvis.name(), "pelvis");
  EXPECT_DOUBLE_EQ(proto_pelvis.pose().position().x(), 1.0);
  EXPECT_DOUBLE_EQ(proto_pelvis.pose().position().y(), 2.0);
  EXPECT_DOUBLE_EQ(proto_pelvis.pose().position().z(), 3.0);
  EXPECT_DOUBLE_EQ(proto_pelvis.twist().linear().x(), 4.0);
  EXPECT_DOUBLE_EQ(proto_pelvis.accel().angular().z(), 15.0);

  const auto& proto_hand = proto_msg.states(1);
  EXPECT_EQ(proto_hand.name(), "left_hand");
  EXPECT_DOUBLE_EQ(proto_hand.pose().position().x(), -1.0);
  EXPECT_DOUBLE_EQ(proto_hand.pose().position().y(), -2.0);
  EXPECT_DOUBLE_EQ(proto_hand.pose().position().z(), -3.0);
  EXPECT_DOUBLE_EQ(proto_hand.pose().orientation().w(), 1.0);
}
