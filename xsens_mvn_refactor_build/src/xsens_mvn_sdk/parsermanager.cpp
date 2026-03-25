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

#include "xsens_mvn_sdk/parsermanager.h"


ParserManager::ParserManager(bool printData /*=false*/, bool printHeader /*=false*/)
  : m_printdata(false),
    m_printheader(false),
    m_datagram(NULL),
    m_type(SPPoseEuler),
    m_quaternion_datagram(NULL),
    m_joint_angles_datagram(NULL),
    m_linear_segment_kinematics_datagram(NULL),
    m_angular_segment_kinematics_datagram(NULL),
    m_center_of_mass_datagram(NULL),
    m_time_code_datagram(NULL)
{
  // std::cout << "DEBUG: ParserManager constructor called" << std::endl;
  if (printData)
    m_printdata = true;

  if (printHeader)
    m_printheader = true;
}

/*! Destructor */
ParserManager::~ParserManager()
{
   // std::cout << "DEBUG: ParserManager destructor called" << std::endl;
   if (m_datagram != NULL)
   {
      // std::cout << "DEBUG: ParserManager deleting datagram of type " << m_type << std::endl;
      delete m_datagram;
      m_datagram = NULL;
   }
   delete m_quaternion_datagram;
   delete m_joint_angles_datagram;
   delete m_linear_segment_kinematics_datagram;
   delete m_angular_segment_kinematics_datagram;
   delete m_center_of_mass_datagram;
   delete m_time_code_datagram;
}

void ParserManager::cacheCurrentDatagram()
{
  switch (m_type)
  {
  case SPPoseQuaternion:
    delete m_quaternion_datagram;
    m_quaternion_datagram = new QuaternionDatagram(
      *static_cast<QuaternionDatagram*>(m_datagram));
    break;
  case SPJointAngles:
    delete m_joint_angles_datagram;
    m_joint_angles_datagram = new JointAnglesDatagram(
      *static_cast<JointAnglesDatagram*>(m_datagram));
    break;
  case SPLinearSegmentKinematics:
    delete m_linear_segment_kinematics_datagram;
    m_linear_segment_kinematics_datagram = new LinearSegmentKinematicsDatagram(
      *static_cast<LinearSegmentKinematicsDatagram*>(m_datagram));
    break;
  case SPAngularSegmentKinematics:
    delete m_angular_segment_kinematics_datagram;
    m_angular_segment_kinematics_datagram = new AngularSegmentKinematicsDatagram(
      *static_cast<AngularSegmentKinematicsDatagram*>(m_datagram));
    break;
  case SPCenterOfMass:
    delete m_center_of_mass_datagram;
    m_center_of_mass_datagram = new CenterOfMassDatagram(
      *static_cast<CenterOfMassDatagram*>(m_datagram));
    break;
  case SPTimeCode:
    delete m_time_code_datagram;
    m_time_code_datagram = new TimeCodeDatagram(
      *static_cast<TimeCodeDatagram*>(m_datagram));
    break;
  default:
    break;
  }
}

Datagram* ParserManager::createDgram(StreamingProtocol proto)
{
  // std::cout << "DEBUG: ParserManager::createDgram called with protocol " << proto << std::endl;
  
  Datagram* datagram = NULL;
  
  switch (proto)
  {
  case SPPoseEuler:   
    datagram = new EulerDatagram;
    break;
  case SPPoseQuaternion:
    // std::cout << "DEBUG: Creating QuaternionDatagram" << std::endl;
    datagram = new QuaternionDatagram;
    break;
  case SPPosePositions: 
    datagram = new PositionDatagram;
    break;
  case SPMetaScaling:   
    datagram = new ScaleDatagram;
    break;
  case SPMetaMoreMeta:  
    datagram = new MetaDatagram;
    break;
  case SPJointAngles:  
    // std::cout << "DEBUG: Creating JointAnglesDatagram" << std::endl; 
    datagram = new JointAnglesDatagram;
    break;
  case SPLinearSegmentKinematics:   
    datagram = new LinearSegmentKinematicsDatagram;
    break;
  case SPAngularSegmentKinematics:  
    datagram = new AngularSegmentKinematicsDatagram;
    break;
  case SPTrackerKinematics:   
    datagram = new TrackerKinematicsDatagram;
    break;
  case SPCenterOfMass:        
    datagram = new CenterOfMassDatagram;
    break;
  case SPTimeCode:            
    datagram = new TimeCodeDatagram;
    break;

  default:
    // std::cout << "DEBUG: Unknown protocol: " << proto << std::endl;
    return NULL;
  }
  
  if (datagram != NULL) {
    // std::cout << "DEBUG: Successfully created datagram of type " << proto << std::endl;
  } else {
    // std::cout << "DEBUG: Failed to create datagram of type " << proto << std::endl;
  }
  
  return datagram;
}

/*! Read single datagram from the incoming stream */
void ParserManager::readDatagram(const char* data)
{
  // std::cout << "DEBUG: ParserManager::readDatagram called" << std::endl;
  
  try {
    // Get message type from the data
    StreamingProtocol type = static_cast<StreamingProtocol>(Datagram::messageType(data));
    // std::cout << "DEBUG: Detected message type: " << type << std::endl;
    
    // Delete existing datagram if there is one
    if (m_datagram != NULL) {
       // std::cout << "DEBUG: Deleting existing datagram of type " << m_type << std::endl;
       delete m_datagram;
       m_datagram = NULL;
    }
    
    // Set the current type
    m_type = type;
    
    // Create new datagram based on the type
    m_datagram = createDgram(m_type);
    
    if (m_datagram != NULL)
    {
      // std::cout << "DEBUG: About to deserialize datagram of type " << m_type << std::endl;
      
      // Deserialize the data
      m_datagram->deserialize(data);
      // std::cout << "DEBUG: Datagram deserialized successfully" << std::endl;
      
      // Print header and data if requested
      if (m_printheader) {
        m_datagram->printHeader();
      }
      if (m_printdata) {
        m_datagram->printData();
      }

      cacheCurrentDatagram();
      
      // Extra validation for QuaternionDatagram
      if (m_type == SPPoseQuaternion) {
        QuaternionDatagram* quat_dgram = static_cast<QuaternionDatagram*>(m_datagram);
        if (quat_dgram) {
          try {
            const auto& quat_data = quat_dgram->getData();
            size_t size = quat_data.size();
            
            if (size > 100) {
              std::cerr << "ERROR: Quaternion datagram has suspicious size: " << size << std::endl;
            } else {
              // std::cout << "DEBUG: Quaternion datagram has " << size << " elements" << std::endl;
              
              // Print first few elements
              for (size_t i = 0; i < std::min(size, size_t(3)); i++) {
                // std::cout << "DEBUG: Element " << i << " - segmentId: " << quat_data[i].segmentId << std::endl;
              }
            }
          } catch (const std::exception& e) {
            std::cerr << "ERROR: Exception when checking quaternion datagram: " << e.what() << std::endl;
          }
        }
      }
      
    } else {
     // std::cerr << "DEBUG: Failed to create datagram for type " << m_type << std::endl;
    }
  } catch (const std::exception& e) {
    std::cerr << "ERROR: Exception in readDatagram: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "ERROR: Unknown exception in readDatagram" << std::endl;
  }
}

CenterOfMassDatagram* ParserManager::getCenterOfMassDatagram()
{
    // std::cout << "DEBUG: getCenterOfMassDatagram called, current type: " << m_type << std::endl;
    if (m_center_of_mass_datagram != NULL) {
      // std::cout << "DEBUG: Returning CenterOfMassDatagram" << std::endl;
      return m_center_of_mass_datagram;
    }
    // std::cout << "DEBUG: Not a CenterOfMassDatagram, returning NULL" << std::endl;
    return NULL;
}


TimeCodeDatagram* ParserManager::getTimeCodeDatagram() {
   // std::cout << "DEBUG: getTimeCodeDatagram called, current type: " << m_type << std::endl;
   if (m_time_code_datagram != NULL) {
     // std::cout << "DEBUG: Returning TimeCodeDatagram" << std::endl;
     return m_time_code_datagram;
   }
   // std::cout << "DEBUG: Not a TimeCodeDatagram, returning NULL" << std::endl;
   return NULL;
}

/* Return JointAnglesDatagram else return null */
JointAnglesDatagram* ParserManager::getJointAnglesDatagram() {
   // std::cout << "DEBUG: getJointAnglesDatagram called, current type: " << m_type << std::endl;
   if (m_joint_angles_datagram != NULL) {
     // std::cout << "DEBUG: Returning JointAnglesDatagram" << std::endl;
     return m_joint_angles_datagram;
   }
   // std::cout << "DEBUG: Not a JointAnglesDatagram, returning NULL" << std::endl;
   return NULL;
}

/* Return QuaternionDatagram else return null */
QuaternionDatagram* ParserManager::getQuaternionDatagram() {
  // std::cout << "DEBUG: getQuaternionDatagram called, current type: " << m_type << std::endl;
  if (m_quaternion_datagram != NULL) {
    QuaternionDatagram* quat_dgram = m_quaternion_datagram;
    // std::cout << "DEBUG: Returning QuaternionDatagram at address " << quat_dgram << std::endl;
    
    // Validate the quaternion datagram before returning it
    if (quat_dgram) {
      try {
        const auto& quat_data = quat_dgram->getData();
        size_t size = quat_data.size();
        
        if (size > 100) {
          std::cerr << "ERROR: Quaternion datagram has suspicious size: " << size << std::endl;
          std::cerr << "ERROR: This is likely corrupted data, returning NULL instead" << std::endl;
          return NULL;
        } else {
          // std::cout << "DEBUG: Quaternion datagram has " << size << " elements" << std::endl;
        }
      } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception when checking quaternion datagram: " << e.what() << std::endl;
        return NULL;
      }
    }
    
    return quat_dgram;
  }
  // std::cout << "DEBUG: Not a QuaternionDatagram, returning NULL" << std::endl;
  return NULL;
}

/* Return AngularSegmentKinematicsDatagram else return null */
AngularSegmentKinematicsDatagram* ParserManager::getAngularSegmentKinematicsDatagram() {
   // std::cout << "DEBUG: getAngularSegmentKinematicsDatagram called, current type: " << m_type << std::endl;
   if (m_angular_segment_kinematics_datagram != NULL) {
     // std::cout << "DEBUG: Returning AngularSegmentKinematicsDatagram" << std::endl;
     return m_angular_segment_kinematics_datagram;
   }
   // std::cout << "DEBUG: Not an AngularSegmentKinematicsDatagram, returning NULL" << std::endl;
   return NULL;
}

/* Return LinearSegmentKinematicsDatagram else return null */
LinearSegmentKinematicsDatagram *ParserManager::getLinearSegmentKinematicsDatagram() {
  // std::cout << "DEBUG: getLinearSegmentKinematicsDatagram called, current type: " << m_type << std::endl;
  if (m_linear_segment_kinematics_datagram != NULL) {
    // std::cout << "DEBUG: Returning LinearSegmentKinematicsDatagram" << std::endl;
    return m_linear_segment_kinematics_datagram;
  }
  // std::cout << "DEBUG: Not a LinearSegmentKinematicsDatagram, returning NULL" << std::endl;
  return NULL;
}
