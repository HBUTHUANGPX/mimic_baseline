#include "xsens_apps/TestFrameFactory.h"

namespace xsens
{
namespace apps
{
xsens::core::XsensFrame buildDeterministicTestFrame()
{
  xsens::core::XsensFrame frame;
  frame.frame_id = "world";
  frame.stamp_sec = 123;
  frame.stamp_nanosec = 456U;

  xsens::core::LinkSample pelvis;
  pelvis.name = "pelvis";
  pelvis.position = Eigen::Vector3d(1.1, 2.2, 3.3);
  pelvis.orientation = Eigen::Quaterniond(0.4, 0.1, 0.2, 0.3);
  pelvis.linear_velocity = Eigen::Vector3d(4.4, 5.5, 6.6);
  pelvis.angular_velocity = Eigen::Vector3d(7.7, 8.8, 9.9);
  pelvis.linear_acceleration = Eigen::Vector3d(10.1, 11.1, 12.1);
  pelvis.angular_acceleration = Eigen::Vector3d(13.1, 14.1, 15.1);
  frame.links.push_back(pelvis);

  xsens::core::LinkSample left_hand;
  left_hand.name = "left_hand";
  left_hand.position = Eigen::Vector3d(-1.0, -2.0, -3.0);
  left_hand.orientation = Eigen::Quaterniond::Identity();
  frame.links.push_back(left_hand);

  return frame;
}
}  // namespace apps
}  // namespace xsens
