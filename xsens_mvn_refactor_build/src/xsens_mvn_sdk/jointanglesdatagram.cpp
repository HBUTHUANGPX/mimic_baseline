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

#include "xsens_mvn_sdk/jointanglesdatagram.h"
#include <cstdlib>

namespace {
bool xsensVerboseLoggingEnabled()
{
  const char* value = std::getenv("XSENS_MVN_VERBOSE");
  return value != nullptr && std::string(value) == "1";
}
}
#include <boost/concept_check.hpp>


JointAngle::JointAngle(int32_t parent, int32_t child, float rotx, float roty, float rotz)
  : parent(parent), child(child)
{
  // std::cout << "DEBUG: JointAngle constructor called" << std::endl;
  rotation[0] = rotx;
  rotation[1] = roty;
  rotation[2] = rotz;
}

JointAngle JointAngle::operator+(const JointAngle& j) const
{ 
  // std::cout << "DEBUG: JointAngle operator+ called" << std::endl;
  if ((child != j.parent) && (parent != j.child))
    return JointAngle(parent,child,rotation[0],rotation[1],rotation[2]);
  else {
    return JointAngle(
            (child == j.parent) ? parent : j.parent,
            (parent == j.child) ? child : j.child,
            rotation[0]+j.rotation[0],
            rotation[1]+j.rotation[1],
            rotation[2]+j.rotation[2]);
  }
}

/*! \class JointAnglesDatagram
	\brief a Joint Angle datagram (type 0x20)

	Information about each joint is sent as follows.

	4 bytes parent connection identifier: 256 * segment ID + point ID
	4 bytes child connection identifer: 256 * segment ID + point ID
	4 bytes x rotation
	4 bytes y rotation
	4 bytes z rotation

	Total: 20 bytes per joint

  The coordinates use a Z-Up, right-handed coordinate system.
*/

/*! Constructor */
JointAnglesDatagram::JointAnglesDatagram()
  : Datagram(), m_hasFingerData(false)
{
  // std::cout << "DEBUG: JointAnglesDatagram constructor called" << std::endl;
  setType(SPJointAngles);
}

/*! Destructor */
JointAnglesDatagram::~JointAnglesDatagram()
{
  // std::cout << "DEBUG: JointAnglesDatagram destructor called" << std::endl;
}

/*! Deserialize the data from \a arr
  \sa serializeData
*/
void JointAnglesDatagram::deserializeData(Streamer &inputStreamer)
{
  // std::cout << "DEBUG: deserializeData called" << std::endl;
  Streamer* streamer = &inputStreamer;

  for (int i = 0; i < dataCount(); i++)
  {
    // std::cout << "DEBUG: deserializing joint " << i << " of " << dataCount() << std::endl;
    JointAngle joint;

    // Parent Connection ID  -> 4 byte 
    streamer->read(joint.parent);
    joint.parentSegmentId = joint.parent / 256;

    // Child Connection ID -> 4 byte
    streamer->read(joint.child);
    joint.childSegmentId = joint.child / 256;

    // Store the Rotation in a Vector -> 12 byte	(3 x 4 byte)
    for (int k = 0; k < 3; k++)
      streamer->read(joint.rotation[k]);

    m_data.push_back(joint);
  }
  // std::cout << "DEBUG: deserializeData finished, processed " << m_data.size() << " joints" << std::endl;
}

/*! Print Data datagram in a formated why
*/
void JointAnglesDatagram::printData() const
{
  // std::cout << "DEBUG: printData called" << std::endl;
  std::cout << "*********************** DATA CONTENT ***********************" <<  std::endl <<  std::endl; 

  for (int i = 0; i < m_data.size(); i++)
  {
    std::cout << "Parent Connection ID (256 * segment ID + point ID): " << m_data.at(i).parent << std::endl;
    std::cout << "Child Connection ID (256 * segment ID + point ID): " << m_data.at(i).child << std::endl;
    // Rotation
    std::cout << "Rotation: " << "(";
    std::cout << "x: " << m_data.at(i).rotation[0] << ", ";
    std::cout << "y: " << m_data.at(i).rotation[1] << ", ";
    std::cout << "z: " << m_data.at(i).rotation[2] << ")"<< std::endl << std::endl;
  }
  
  // Print finger joint angles if available
  if (m_hasFingerData) {
    printFingerJointAngles();
  }
}

/*! Print Data datagram in CSV format
*/
void JointAnglesDatagram::printCSVData() const
{
  // std::cout << "DEBUG: printCSVData called" << std::endl;
  for (int i = 0; i < m_data.size(); i++)
  {
    // Rotation
    std::cout << m_data.at(i).rotation[0] << ",";
    std::cout << m_data.at(i).rotation[1] << ",";
    std::cout << m_data.at(i).rotation[2] << ",";
  }
  std::cout << std::endl;
  
  // Print finger joint angles if available
  if (m_hasFingerData) {
    for (const auto& joint : m_fingerJointAngles) {
      std::cout << joint.rotation[0] << ",";
      std::cout << joint.rotation[1] << ",";
      std::cout << joint.rotation[2] << ",";
    }
    std::cout << std::endl;
  }
}

/*! Get the datagram value
*/
const std::vector<JointAngle>& JointAnglesDatagram::getData() const
{
  // std::cout << "DEBUG: getData called" << std::endl;
  return m_data;  // Now returns a const reference
}

/*! Get the requested item of the datagram data
*/
JointAngle JointAnglesDatagram::getItem(int32_t parentSegmentId, int32_t childSegmentId)
{  
  // std::cout << "DEBUG: getItem called with parentSegmentId=" << parentSegmentId 
            // << ", childSegmentId=" << childSegmentId << std::endl;
  
  JointAngle req_joint_angle;
  std::vector<JointAngle>::iterator it;
  
  //TODO: For now we are considering that the input is segmentID and not pointID.
  //TODO: We are not considering the possibility of other pointID, only pointID 0. 
	//Identifier = 256 * segment ID + point ID
  /*pointparent = pointparent*256 + 0;
  pointchild = pointchild*256 + 0;*/
  
  struct FindByParentAndChild {
      int32_t point_parent;
      int32_t point_child;
      FindByParentAndChild(int32_t pparent, int32_t pchild) : point_parent(pparent), point_child(pchild) {}
      bool operator()(const JointAngle& j) const { 
          return (j.parent/256 == point_parent && j.child/256 == point_child); 
      }
  };
  
  it = std::find_if(m_data.begin(), m_data.end(), FindByParentAndChild(parentSegmentId, childSegmentId));
  
  if (it != m_data.end()) {
    // std::cout << "DEBUG: getItem found matching joint" << std::endl;
    req_joint_angle = *it;
  } else {
    // std::cout << "DEBUG: getItem did not find matching joint" << std::endl;
    std::memset(&req_joint_angle, 0, sizeof req_joint_angle);
  }
  
  return req_joint_angle;
}

/*! Check if finger data is available
*/
bool JointAnglesDatagram::hasFingerData() const
{
  // std::cout << "DEBUG: hasFingerData called, returning " << (m_hasFingerData ? "true" : "false") << std::endl;
  return m_hasFingerData;
}

/*! Calculate finger joint angles from quaternion data
  \param leftHandQuats Quaternions for left hand finger segments
  \param rightHandQuats Quaternions for right hand finger segments
  \param propCount Number of props in use (affects segment IDs)
*/
void JointAnglesDatagram::calculateFingerJointAngles(
  const std::vector<Eigen::Quaternionf>& leftHandQuats,
  const std::vector<Eigen::Quaternionf>& rightHandQuats,
  int propCount)
{
  // std::cout << "DEBUG: calculateFingerJointAngles called with leftHandQuats.size()=" << leftHandQuats.size() 
            // << ", rightHandQuats.size()=" << rightHandQuats.size() 
            // << ", propCount=" << propCount << std::endl;
  
  // Clear previous finger joint angles
  m_fingerJointAngles.clear();
  
  // Check if we have finger data
  if (leftHandQuats.empty() && rightHandQuats.empty()) {
      if (xsensVerboseLoggingEnabled()) {
          std::cerr << "JointAnglesDatagram: No finger quaternion data provided" << std::endl;
      }
      m_hasFingerData = false;
      return;
  }
  
  if (xsensVerboseLoggingEnabled()) {
    std::cout << "JointAnglesDatagram: Calculating finger joint angles from " 
              << leftHandQuats.size() << " left quats and " 
              << rightHandQuats.size() << " right quats" << std::endl;
  }
  
  m_hasFingerData = true;
  
  // Base segment IDs (adjusted by prop count)
  int leftFingerBase = 24 + propCount; 
  int rightFingerBase = leftFingerBase + 20;
  
  // std::cout << "DEBUG: leftFingerBase=" << leftFingerBase << ", rightFingerBase=" << rightFingerBase << std::endl;
  
  // Get the finger hierarchy from MvnModelNames
  // std::cout << "DEBUG: About to call getFingerHierarchy" << std::endl;
  std::vector<std::pair<int, int>> fingerHierarchy = m_modelNames.getFingerHierarchy(propCount);
  // std::cout << "DEBUG: Got fingerHierarchy with " << fingerHierarchy.size() << " pairs" << std::endl;
  
  // Process left hand joints
  if (!leftHandQuats.empty()) {
    // std::cout << "DEBUG: Processing left hand joints, count=" << m_modelNames.left_finger_joints.size() << std::endl;
    for (size_t i = 0; i < m_modelNames.left_finger_joints.size(); i++) {
      const std::string& jointName = m_modelNames.left_finger_joints[i];
      // std::cout << "DEBUG: Processing left joint " << i << ": " << jointName << std::endl;
      
      // Find the parent-child segment pair for this joint from the hierarchy
      int parentId = -1;
      int childId = -1;
      
      for (const auto& pair : fingerHierarchy) {
        // Skip if not in left hand range
        if (pair.second < leftFingerBase || pair.second >= leftFingerBase + 20) {
          continue;
        }
        
        // std::cout << "DEBUG: Checking hierarchy pair: " << pair.first << " -> " << pair.second << std::endl;
        std::string parentName = m_modelNames.getSegmentNameFromId(pair.first, propCount);
        std::string childName = m_modelNames.getSegmentNameFromId(pair.second, propCount);
        // std::cout << "DEBUG: Segment names: " << parentName << " -> " << childName << std::endl;
        
        // Extract segment names from joint name (removing "left_" prefix)
        std::string jointNameWithoutPrefix = jointName.substr(5); // Remove "left_"
        
        // Joint names follow pattern: "parent_child" like "first_metacarpal_proximal"
        size_t underscorePos = jointNameWithoutPrefix.find('_');
        if (underscorePos != std::string::npos) {
          std::string firstPart = jointNameWithoutPrefix.substr(0, underscorePos);
          std::string remainingPart = jointNameWithoutPrefix.substr(underscorePos + 1);
          
          // Check if parent segment name contains the first part and child contains the last part
          if (parentName.find(firstPart) != std::string::npos && 
              childName.find(remainingPart) != std::string::npos) {
            parentId = pair.first;
            childId = pair.second;
            // std::cout << "DEBUG: Found match! parentId=" << parentId << ", childId=" << childId << std::endl;
            break;
          }
        }
      }
      
      // If we found a valid parent-child pair
      if (parentId != -1 && childId != -1) {
        // Convert to local indices for quaternion array
        int parentIdx = (parentId == 15) ? 0 : (parentId - leftFingerBase); // 15 is left_hand
        int childIdx = childId - leftFingerBase;
        
        // std::cout << "DEBUG: Converted to local indices: parentIdx=" << parentIdx << ", childIdx=" << childIdx << std::endl;
        
        // Check bounds
        if (parentIdx >= 0 && parentIdx < leftHandQuats.size() && 
            childIdx >= 0 && childIdx < leftHandQuats.size()) {
          
          // std::cout << "DEBUG: About to calculate joint angles" << std::endl;
          // Calculate joint angles
          Eigen::Vector3f angles = calculateJointAngles(
            leftHandQuats[parentIdx], 
            leftHandQuats[childIdx], 
            jointName, 
            true,  // isLeftHand
            i      // jointIndex
          );
          
          // std::cout << "DEBUG: Calculated angles: [" << angles[0] << ", " << angles[1] << ", " << angles[2] << "]" << std::endl;
          
          // Create and add finger joint angle
          FingerJointAngle fingerJoint;
          fingerJoint.parentSegmentId = parentId;
          fingerJoint.childSegmentId = childId;
          fingerJoint.jointName = jointName;
          fingerJoint.rotation[0] = angles[0]; // abduction/adduction
          fingerJoint.rotation[1] = angles[1]; // internal/external rotation
          fingerJoint.rotation[2] = angles[2]; // flexion/extension
          
          m_fingerJointAngles.push_back(fingerJoint);
        } else {
         // std::cerr << "DEBUG: Index out of bounds: parentIdx=" << parentIdx 
                    // << "(max=" << leftHandQuats.size() - 1 << "), childIdx=" << childIdx 
                    // << "(max=" << leftHandQuats.size() - 1 << ")" << std::endl;
        }
      } else {
       // std::cerr << "DEBUG: Could not find parent-child pair for joint " << jointName << std::endl;
      }
    }
  }
  
  // Process right hand joints
  if (!rightHandQuats.empty()) {
    // std::cout << "DEBUG: Processing right hand joints, count=" << m_modelNames.right_finger_joints.size() << std::endl;
    for (size_t i = 0; i < m_modelNames.right_finger_joints.size(); i++) {
      const std::string& jointName = m_modelNames.right_finger_joints[i];
      // std::cout << "DEBUG: Processing right joint " << i << ": " << jointName << std::endl;
      
      // Find the parent-child segment pair for this joint from the hierarchy
      int parentId = -1;
      int childId = -1;
      
      for (const auto& pair : fingerHierarchy) {
        // Skip if not in right hand range
        if (pair.second < rightFingerBase || pair.second >= rightFingerBase + 20) {
          continue;
        }
        
        // std::cout << "DEBUG: Checking hierarchy pair: " << pair.first << " -> " << pair.second << std::endl;
        std::string parentName = m_modelNames.getSegmentNameFromId(pair.first, propCount);
        std::string childName = m_modelNames.getSegmentNameFromId(pair.second, propCount);
        // std::cout << "DEBUG: Segment names: " << parentName << " -> " << childName << std::endl;
        
        // Extract segment names from joint name (removing "right_" prefix)
        std::string jointNameWithoutPrefix = jointName.substr(6); // Remove "right_"
        
        // Joint names follow pattern: "parent_child" like "first_metacarpal_proximal"
        size_t underscorePos = jointNameWithoutPrefix.find('_');
        if (underscorePos != std::string::npos) {
          std::string firstPart = jointNameWithoutPrefix.substr(0, underscorePos);
          std::string remainingPart = jointNameWithoutPrefix.substr(underscorePos + 1);
          
          // Check if parent segment name contains the first part and child contains the last part
          if (parentName.find(firstPart) != std::string::npos && 
              childName.find(remainingPart) != std::string::npos) {
            parentId = pair.first;
            childId = pair.second;
            // std::cout << "DEBUG: Found match! parentId=" << parentId << ", childId=" << childId << std::endl;
            break;
          }
        }
      }
      
      // If we found a valid parent-child pair
      if (parentId != -1 && childId != -1) {
        // Convert to local indices for quaternion array
        int parentIdx = (parentId == 11) ? 0 : (parentId - rightFingerBase); // 11 is right_hand
        int childIdx = childId - rightFingerBase;
        
        // std::cout << "DEBUG: Converted to local indices: parentIdx=" << parentIdx << ", childIdx=" << childIdx << std::endl;
        
        // Check bounds
        if (parentIdx >= 0 && parentIdx < rightHandQuats.size() && 
            childIdx >= 0 && childIdx < rightHandQuats.size()) {
          
          // std::cout << "DEBUG: About to calculate joint angles" << std::endl;
          // Calculate joint angles
          Eigen::Vector3f angles = calculateJointAngles(
            rightHandQuats[parentIdx], 
            rightHandQuats[childIdx], 
            jointName, 
            false,                 // isLeftHand
            i + m_modelNames.left_finger_joints.size()  // jointIndex (offset for right hand)
          );
          
          // std::cout << "DEBUG: Calculated angles: [" << angles[0] << ", " << angles[1] << ", " << angles[2] << "]" << std::endl;
          
          // Create and add finger joint angle
          FingerJointAngle fingerJoint;
          fingerJoint.parentSegmentId = parentId;
          fingerJoint.childSegmentId = childId;
          fingerJoint.jointName = jointName;
          fingerJoint.rotation[0] = angles[0]; // abduction/adduction
          fingerJoint.rotation[1] = angles[1]; // internal/external rotation
          fingerJoint.rotation[2] = angles[2]; // flexion/extension
          
          m_fingerJointAngles.push_back(fingerJoint);
        } else {
         // std::cerr << "DEBUG: Index out of bounds: parentIdx=" << parentIdx 
                    // << "(max=" << rightHandQuats.size() - 1 << "), childIdx=" << childIdx 
                    // << "(max=" << rightHandQuats.size() - 1 << ")" << std::endl;
        }
      } else {
       // std::cerr << "DEBUG: Could not find parent-child pair for joint " << jointName << std::endl;
      }
    }
  }
  
  // std::cout << "DEBUG: Finished calculating finger joint angles, total: " 
            // << m_fingerJointAngles.size() << std::endl;
}

/*! Calculate joint angles between parent and child quaternions
  \param qParent Parent quaternion
  \param qChild Child quaternion
  \param jointName Name of the joint for reference
  \param isLeftHand Whether this is a left hand joint
  \param jointIndex Index of the joint in the model
  \param verbose Whether to print verbose output
  \return Vector of Euler angles in 'rgb' order (abduction, internal/external, flexion/extension)
*/
Eigen::Vector3f JointAnglesDatagram::calculateJointAngles(
  const Eigen::Quaternionf& qParent,
  const Eigen::Quaternionf& qChild,
  const std::string& jointName,
  bool isLeftHand,
  int jointIndex,
  bool verbose)
{
  // std::cout << "DEBUG: calculateJointAngles called for joint " << jointName << std::endl;
  
  // Check for identity quaternions (may indicate missing data)
  if (qParent.isApprox(Eigen::Quaternionf::Identity()) || 
      qChild.isApprox(Eigen::Quaternionf::Identity())) {
      if (verbose) {
          std::cerr << "Skipping joint " << jointName << " due to identity quaternions" << std::endl;
      }
      return Eigen::Vector3f::Zero();
  }
  
  // Ensure quaternions are normalized before operations
  Eigen::Quaternionf qParentNorm = qParent.normalized();
  Eigen::Quaternionf qChildNorm = qChild.normalized();

  // Calculate relative rotation
  Eigen::Quaternionf qDiff = qParentNorm.inverse() * qChildNorm;
  qDiff.normalize(); // Normalize result as well
  
  // Convert to Euler angles (XZY order with Y-up)
  Eigen::Vector3f euler;
  setEulerXZYwithYUp(qDiff, euler);
  
  // Store original angles for comparison if verbose
  Eigen::Vector3f originalEuler = euler;
  
  // Adjust angles for left hand
  if (isLeftHand) {
      euler[0] = -euler[0]; // Negate flexion/extension for left hand
      euler[1] = -euler[1]; // Negate internal/external rotation for left hand
  }
  
  // Create the final result in 'rgb' order: abduction(aa), internal/external(ie), flexion/extension(fe)
  Eigen::Vector3f result;
  result[0] = euler[2];  // aa - abduction/adduction (was at index 2)
  result[1] = euler[1];  // ie - internal/external rotation
  result[2] = euler[0];  // fe - flexion/extension (was at index 0)
  
  // Check if this joint should have its abduction angle negated
  bool shouldNegateAbduction = true; // Default: negate abduction
  
  if (isLeftHand) {
      // 11-18 means the 4th and 5th fingers should not be negated
      // This corresponds to joints 11-18 in the updated indexing
      if (jointIndex > 10 && jointIndex <= 18) {
          shouldNegateAbduction = false;
      }
  } else {
      // 0-10(indices 20-30) means for right hand, the 1-3 fingers should be not negated.
      // This corresponds to joints 0-10 in the updated indexing
      if (jointIndex >= 20 && jointIndex <= 30) {
          shouldNegateAbduction = false;
      }
  }
  
  // Apply negation to abduction angle if needed
  if (shouldNegateAbduction) {
      result[0] = -result[0];
  }
  
  if (verbose) {
      // Print results
      std::cout << "----------------------------------------" << std::endl;
      std::cout << "Joint: " << jointName << " (isLeftHand: " << (isLeftHand ? "true" : "false") 
                << ", jointIndex: " << jointIndex << ")" << std::endl;
      std::cout << "Original Euler angles XZY (degrees): [" 
              << originalEuler[0] << ", " << originalEuler[1] << ", " << originalEuler[2] << "]" << std::endl;
      std::cout << "Adjusted Euler angles (degrees): [" 
              << euler[0] << ", " << euler[1] << ", " << euler[2] << "]" << std::endl;
      std::cout << "Final angles in 'rgb' order (degrees): [" 
              << result[0] << ", " << result[1] << ", " << result[2] << "]" << std::endl;
      std::cout << "Abduction negation applied: " << (shouldNegateAbduction ? "Yes" : "No") << std::endl;
      std::cout << "Interpretation:" << std::endl;
      std::cout << "  Abduction/adduction (aa): " << result[0] << " degrees" << std::endl;
      std::cout << "  Internal/external rotation (ie): " << result[1] << " degrees" << std::endl;
      std::cout << "  Flexion/extension (fe): " << result[2] << " degrees" << std::endl;
  }
  
  return result;
}

/*! Convert quaternion to Euler angles in XZY order with Y-up
  \param quat Input quaternion
  \param euler Output Euler angles in degrees
*/
void JointAnglesDatagram::setEulerXZYwithYUp(const Eigen::Quaternionf& quat, Eigen::Vector3f& euler) const
{
  // std::cout << "DEBUG: setEulerXZYwithYUp called" << std::endl;
  
  // Convert quaternion to rotation matrix
  Eigen::Matrix3f m = quat.toRotationMatrix();

  // Calculate z angle first (rotation around Z axis)
  // Use asin with clamping to avoid numerical issues
  float z1 = std::asin(-m(0, 1));
  
  // Handle edge cases for z angle
  if (z1 < -M_PI/2.0f)
    z1 = -M_PI - z1;
  else if (z1 > M_PI/2.0f)
    z1 = M_PI - z1;

  // Extract the sines and cosines for x and y angles
  float sx = m(2, 1);
  float cx = m(1, 1);
  float sy = m(0, 2);
  float cy = m(0, 0);

  // Calculate x and y angles using atan2
  euler[0] = std::atan2(sx, cx) * 180.0f / M_PI;  // Convert to degrees
  euler[1] = std::atan2(sy, cy) * 180.0f / M_PI;  // Convert to degrees
  euler[2] = z1 * 180.0f / M_PI;                  // Convert to degrees
  
  // std::cout << "DEBUG: setEulerXZYwithYUp calculated angles: [" 
            // << euler[0] << ", " << euler[1] << ", " << euler[2] << "]" << std::endl;
}

/*! Get the finger joint angles
  \return Vector of finger joint angles
*/
std::vector<FingerJointAngle> JointAnglesDatagram::getFingerJointAngles() const
{
  // std::cout << "DEBUG: getFingerJointAngles called, returning " << m_fingerJointAngles.size() << " joints" << std::endl;
  return m_fingerJointAngles;
}

/*! Print the finger joint angles
*/
void JointAnglesDatagram::printFingerJointAngles() const
{
  // std::cout << "DEBUG: printFingerJointAngles called" << std::endl;
  
  if (!m_hasFingerData || m_fingerJointAngles.empty()) {
    std::cout << "No finger joint angle data available" << std::endl;
    return;
  }
  
  std::cout << "*********************** FINGER JOINT ANGLES ***********************" << std::endl << std::endl;
  
  for (const auto& joint : m_fingerJointAngles) {
    std::cout << "Joint: " << joint.jointName << std::endl;
    std::cout << "Parent Segment ID: " << joint.parentSegmentId << ", Child Segment ID: " << joint.childSegmentId << std::endl;
    std::cout << "Rotation: (";
    std::cout << "x (abduction/adduction): " << joint.rotation[0] << ", ";
    std::cout << "y (internal/external): " << joint.rotation[1] << ", ";
    std::cout << "z (flexion/extension): " << joint.rotation[2] << ")" << std::endl << std::endl;
  }
}
