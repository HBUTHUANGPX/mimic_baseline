#include <gtest/gtest.h>

#include "xsens_core/HumanDataHandler.h"

TEST(HumanDataHandlerTest, StoresAndReadsJointAngles)
{
  hrii::ergonomics::HumanDataHandler handler;
  const Eigen::Vector3d angles(1.0, 2.0, 3.0);

  EXPECT_TRUE(handler.setJointAngles("right_elbow", angles));

  hrii::ergonomics::Joint joint;
  ASSERT_TRUE(handler.getJoint("right_elbow", joint));
  EXPECT_EQ(joint.name, "right_elbow");
  EXPECT_DOUBLE_EQ(joint.state.angles.x(), 1.0);
  EXPECT_DOUBLE_EQ(joint.state.angles.y(), 2.0);
  EXPECT_DOUBLE_EQ(joint.state.angles.z(), 3.0);
}

TEST(HumanDataHandlerTest, StoresAndReadsLinkPoseAndState)
{
  hrii::ergonomics::HumanDataHandler handler;

  const Eigen::Vector3d position(4.0, 5.0, 6.0);
  const Eigen::Quaterniond orientation(0.4, 0.1, 0.2, 0.3);
  ASSERT_TRUE(handler.setLinkPose("pelvis", position, orientation));

  hrii::ergonomics::Link link;
  ASSERT_TRUE(handler.getLink("pelvis", link));
  EXPECT_EQ(link.name, "pelvis");
  EXPECT_DOUBLE_EQ(link.state.position.x(), 4.0);
  EXPECT_DOUBLE_EQ(link.state.position.y(), 5.0);
  EXPECT_DOUBLE_EQ(link.state.position.z(), 6.0);
  EXPECT_NEAR(link.state.orientation.norm(), 1.0, 1e-9);

  hrii::ergonomics::LinkState link_state;
  link_state.position = Eigen::Vector3d(7.0, 8.0, 9.0);
  link_state.orientation = Eigen::Quaterniond::Identity();
  link_state.velocity.linear = Eigen::Vector3d(1.0, 0.0, 0.0);
  link_state.velocity.angular = Eigen::Vector3d(0.0, 1.0, 0.0);
  link_state.acceleration.linear = Eigen::Vector3d(0.0, 0.0, 1.0);
  link_state.acceleration.angular = Eigen::Vector3d(1.0, 1.0, 1.0);
  ASSERT_TRUE(handler.setLinkState("pelvis", link_state));

  ASSERT_TRUE(handler.getLink("pelvis", link));
  EXPECT_DOUBLE_EQ(link.state.position.x(), 7.0);
  EXPECT_DOUBLE_EQ(link.state.velocity.angular.y(), 1.0);
  EXPECT_DOUBLE_EQ(link.state.acceleration.angular.z(), 1.0);
}

TEST(HumanDataHandlerTest, StoresAndReadsCenterOfMass)
{
  hrii::ergonomics::HumanDataHandler handler;
  const Eigen::Vector3d com(0.1, 0.2, 0.3);

  handler.setCOM(com);
  const auto read_com = handler.getCOM();

  EXPECT_DOUBLE_EQ(read_com.x(), 0.1);
  EXPECT_DOUBLE_EQ(read_com.y(), 0.2);
  EXPECT_DOUBLE_EQ(read_com.z(), 0.3);
}
