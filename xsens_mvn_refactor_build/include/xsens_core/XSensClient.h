#ifndef __XSENS_MVN_CLIENT_H__
#define __XSENS_MVN_CLIENT_H__

#define MAX_MVN_DATAGRAM_SIZE 5000

#include <thread>
#include <memory>
#include <vector>
#include <string>
#include "xsens_core/Socket.h"
#include "xsens_core/HumanDataHandler.h"
#include "xsens_core/XsensFrame.h"
#include "xsens_mvn_sdk/parsermanager.h"
#include "xsens_mvn_sdk/MvnModel.h"

class XSensClient
{
public:
    XSensClient(const int& udp_port);
    virtual ~XSensClient();
    bool init();

    hrii::ergonomics::HumanDataHandler::Ptr getHumanData();
    bool copyFrame(xsens::core::XsensFrame& frame) const;
    std::string getURDFString();

private:
    int udp_port_;
    std::shared_ptr<Socket> udp_socket_;
    ParserManager parser_manager_;
    hrii::ergonomics::HumanDataHandler::Ptr human_data_;
    std::vector<std::string> link_name_list_, joint_name_list_;
    bool has_finger_data_;
    int prop_count_;
    std::vector<std::string> finger_link_name_list_;
    std::vector<std::string> finger_joint_name_list_;
    
    std::thread data_acquisition_thread_;
    void dataAcquisitionCallback();
    char data_buffer_[MAX_MVN_DATAGRAM_SIZE];
    bool client_active_;

    std::vector<Eigen::Quaternionf> m_leftFingerQuats;
    std::vector<Eigen::Quaternionf> m_rightFingerQuats;
    bool m_hasStoredFingerQuats;
    
    bool buildXSensModel();
    JointAnglesDatagram* waitForJointAnglesDatagram(bool required = true);
    QuaternionDatagram* waitForQuaternionDatagram();

    QuaternionDatagram* quaternion_datagram_ptr_;
    JointAnglesDatagram* joint_angles_datagram_ptr_;
    
    bool readData();
    void updateJointAngles();
    void updateLinkPoses();
    void updateLinkLinearTwists();
    void updateLinkAngularTwists();
    void updateCOM();
    void updateFingerJointAngles();

    std::pair<std::vector<Eigen::Quaternionf>, std::vector<Eigen::Quaternionf>> extractFingerQuaternions();
    
    Eigen::Vector3d jointAngleToEigenVector3d(const JointAngle& joint_angle, 
                                            const double& x_axis, 
                                            const double& y_axis, 
                                            const double& z_axis);
    void rotateLink(const std::string& link_name, const Eigen::Quaterniond& quat);
    void rotateJoint(const std::string& joint_name, const Eigen::Quaterniond& quat);

    int detectPropCount(QuaternionDatagram& quaternion_datagram);
    bool detectFingerData(QuaternionDatagram& quaternion_datagram);
    void setupFingerJoints(const MvnModelNames& xsens_model_names);
};

#endif // __XSENS_MVN_CLIENT_H__
