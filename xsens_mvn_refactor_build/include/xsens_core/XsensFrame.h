#ifndef XSENS_CORE_XSENS_FRAME_H
#define XSENS_CORE_XSENS_FRAME_H

#include <string>
#include <vector>

#include <Eigen/Dense>

namespace xsens
{
namespace core
{
struct JointSample
{
  std::string name;
  Eigen::Vector3d angles = Eigen::Vector3d::Zero();
};

struct LinkSample
{
  std::string name;
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
  Eigen::Vector3d linear_velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d linear_acceleration = Eigen::Vector3d::Zero();
  Eigen::Vector3d angular_acceleration = Eigen::Vector3d::Zero();
};

struct XsensFrame
{
  std::string frame_id;
  int stamp_sec = 0;
  unsigned int stamp_nanosec = 0U;
  std::vector<JointSample> joints;
  std::vector<LinkSample> links;
  Eigen::Vector3d center_of_mass = Eigen::Vector3d::Zero();
};
}  // namespace core
}  // namespace xsens

#endif
