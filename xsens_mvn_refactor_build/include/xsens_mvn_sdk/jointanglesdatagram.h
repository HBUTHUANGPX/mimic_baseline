/*! \file
  \section FileCopyright Copyright Notice
  This is free and unencumbered software released into the public domain.

  Anyone is free to copy, modify, publish, use, compile, sell, or
  distribute this software, either in source code form or as a compiled
  binary, for any purpose, commercial or non-commercial, and by any
  means.

  In jurisdictions that recognize copyright laws, the author or authors
  of this software dedicate any and all copyright interest in the
  software to the public domain. We make this dedication for the benefit
  of the public at large and to the detriment of our heirs and
  successors. We intend this dedication to be an overt act of
  relinquishment in perpetuity of all present and future rights to this
  software under copyright law.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
  IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
  OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
  ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
  OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef JOINTANGLESDATAGRAM_H
#define JOINTANGLESDATAGRAM_H

#include "datagram.h"
#include <Eigen/Geometry>
#include <map>
#include "xsens_mvn_sdk/MvnModel.h" // Include directly

struct JointAngle
{
  int32_t parent;
  int32_t child;
  int32_t parentSegmentId;
  int32_t childSegmentId;
  float rotation[3];

  JointAngle(int32_t parent = 0, int32_t child = 0, float rotx = 0, float roty = 0, float rotz = 0);
  JointAngle operator+(const JointAngle &j) const;
};

struct FingerJointAngle
{
  int32_t parentSegmentId;
  int32_t childSegmentId;
  std::string jointName;
  float rotation[3]; // x, y, z angles in degrees
};

class JointAnglesDatagram : public Datagram
{

public:
  JointAnglesDatagram();
  virtual ~JointAnglesDatagram();
  virtual void printData() const;
  virtual void printCSVData() const;
  const std::vector<JointAngle>& getData() const;
  JointAngle getItem(int32_t pointparent, int32_t pointchild);

  // Methods for finger joint angles
  bool hasFingerData() const;
  void calculateFingerJointAngles(const std::vector<Eigen::Quaternionf> &leftHandQuats,
                                  const std::vector<Eigen::Quaternionf> &rightHandQuats,
                                  int propCount = 0);
  std::vector<FingerJointAngle> getFingerJointAngles() const;
  void printFingerJointAngles() const;

protected:
  virtual void deserializeData(Streamer &inputStreamer);

private:
  std::vector<JointAngle> m_data;
  std::vector<FingerJointAngle> m_fingerJointAngles;
  bool m_hasFingerData;
  MvnModelNames m_modelNames; // Instance of MvnModelNames

  // Helper methods for finger joint angle calculation
  void setEulerXZYwithYUp(const Eigen::Quaternionf &quat, Eigen::Vector3f &euler) const;

  // calculating joint angles between quaternions
  Eigen::Vector3f calculateJointAngles(
      const Eigen::Quaternionf &qParent,
      const Eigen::Quaternionf &qChild,
      const std::string &jointName,
      bool isLeftHand = true,
      int jointIndex = -1,
      bool verbose = false);
};

#endif