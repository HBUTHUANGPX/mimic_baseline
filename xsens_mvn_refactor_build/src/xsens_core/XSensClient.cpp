#include "xsens_core/XSensClient.h"
#include <Eigen/Geometry>
#include <xsens_mvn_sdk/jointanglesdatagram.h>
#include <chrono>
#include <cstdlib>

namespace {
bool xsensVerboseLoggingEnabled()
{
    const char* value = std::getenv("XSENS_MVN_VERBOSE");
    return value != nullptr && std::string(value) == "1";
}
}

XSensClient::XSensClient(const int& udp_port) :
    udp_port_(udp_port),
    client_active_(false),
    frame_sequence_(0),
    has_finger_data_(false),
    prop_count_(0),
    m_hasStoredFingerQuats(false),
    quaternion_datagram_ptr_(nullptr),
    joint_angles_datagram_ptr_(nullptr)
{
    // std::cout << "DEBUG: XSensClient constructor" << std::endl;
    m_leftFingerQuats.clear();
    m_rightFingerQuats.clear();
}

XSensClient::~XSensClient()
{
    // std::cout << "DEBUG: XSensClient destructor called" << std::endl;
    client_active_ = false;
    if (data_acquisition_thread_.joinable()) {
        data_acquisition_thread_.join();
    }
}

bool XSensClient::init()
{
    // std::cout << "DEBUG: XSensClient::init() starting" << std::endl;
    udp_socket_ = std::make_shared<Socket>(IP_UDP);
    if (!udp_socket_->bind(udp_port_))
    {
        std::cout << "Error binding XSens port." << std::endl;
        return false;
    }
    
    data_acquisition_thread_ = std::thread(&XSensClient::dataAcquisitionCallback, this);

    while (!client_active_)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    std::cout << "XSens client initialized." << std::endl;

    return true;
}

void XSensClient::dataAcquisitionCallback()
{
    // std::cout << "DEBUG: dataAcquisitionCallback starting" << std::endl;
    std::cout << "XSens client start reading." << std::endl << std::endl;
    
    if (!buildXSensModel())
    {
        std::cerr << "Failure building human model." << std::endl;
        client_active_ = false;
    }
    else
    {
        if (xsensVerboseLoggingEnabled()) {
            std::cout << "Human model built." << std::endl;
        }
        client_active_ = true;
    }

    while (client_active_)
    {
        if (readData())
        {
            // std::cout << "DEBUG: Updating data cycle started" << std::endl;
            updateJointAngles();
            updateLinkPoses();
            updateLinkLinearTwists();
            updateLinkAngularTwists();
            updateCOM();
            
            // Update finger data if available
            if (has_finger_data_)
            {
                // std::cout << "DEBUG: About to update finger joint angles" << std::endl;
                updateFingerJointAngles();
                // std::cout << "DEBUG: Finished updating finger joint angles" << std::endl;
            }
            ++frame_sequence_;
            // std::cout << "DEBUG: Updating data cycle completed" << std::endl;
        }
    }
}

bool XSensClient::readData()
{
    int readed_bytes = udp_socket_->read(data_buffer_, MAX_MVN_DATAGRAM_SIZE);
    if (readed_bytes > 0)
    {
        parser_manager_.readDatagram(data_buffer_);
        return true;
    }
    std::cerr << "Error reading data" << std::endl;
    client_active_ = false;
    return false;
}

bool XSensClient::buildXSensModel()
{
    // std::cout << "DEBUG: buildXSensModel starting" << std::endl;
    human_data_ = std::make_shared<hrii::ergonomics::HumanDataHandler>();
    link_name_list_.clear();
    joint_name_list_.clear();
    finger_link_name_list_.clear();
    finger_joint_name_list_.clear();
    
    m_leftFingerQuats.clear();
    m_rightFingerQuats.clear();
    m_hasStoredFingerQuats = false;

    MvnModelNames xsens_model_names;

    if (quaternion_datagram_ptr_) {
        delete quaternion_datagram_ptr_;
        quaternion_datagram_ptr_ = nullptr;
    }
    quaternion_datagram_ptr_ = waitForQuaternionDatagram();
    if (quaternion_datagram_ptr_ == nullptr)
    {
        std::cerr << "Failed to receive quaternion datagram during model initialization" << std::endl;
        return false;
    }
    
    if (joint_angles_datagram_ptr_) {
        delete joint_angles_datagram_ptr_;
        joint_angles_datagram_ptr_ = nullptr;
    }
    joint_angles_datagram_ptr_ = waitForJointAnglesDatagram(false);
    const bool has_initial_joint_angles = (joint_angles_datagram_ptr_ != nullptr);
    if (!has_initial_joint_angles)
    {
        std::cerr << "Joint angles datagram not available during initialization, continuing with static joint model" << std::endl;
        joint_angles_datagram_ptr_ = new JointAnglesDatagram();
    }

    // std::cout << "DEBUG: About to detect props" << std::endl;
    prop_count_ = detectPropCount(*quaternion_datagram_ptr_);
    // std::cout << "DEBUG: Detected " << prop_count_ << " props" << std::endl;

    // std::cout << "DEBUG: About to detect finger data" << std::endl;
    has_finger_data_ = detectFingerData(*quaternion_datagram_ptr_);
    // std::cout << "DEBUG: Finger data available: " << (has_finger_data_ ? "Yes" : "No") << std::endl;

    const auto& quaternion_data = quaternion_datagram_ptr_->getData();
    if (xsensVerboseLoggingEnabled()) {
        std::cout << "Available links: " << quaternion_data.size() << std::endl;
    }

    for (auto xsens_link: quaternion_data)
    {
        std::string link_name = xsens_model_names.getSegmentNameFromId(xsens_link.segmentId, prop_count_);
        
        int finger_id = xsens_model_names.getFingerSegmentId(link_name, prop_count_);
        if (finger_id != -1 && has_finger_data_)
        {
            if(!human_data_->setLink(link_name, hrii::ergonomics::Link(link_name)))
            {
                std::cerr << "Error inserting finger link " << link_name << ". Exiting...";
                return false;
            }
            else
            {
                finger_link_name_list_.push_back(link_name);
                if (xsensVerboseLoggingEnabled()) {
                    std::cout << "Added finger link: " << link_name << " (ID: " << finger_id << ")" << std::endl;
                }
            }
        }
        else if (xsens_link.segmentId < xsens_model_names.links.size())
        {
            if(!human_data_->setLink(xsens_model_names.links[xsens_link.segmentId], hrii::ergonomics::Link(xsens_model_names.links[xsens_link.segmentId])))
            {
                std::cerr << "Error inserting link " << xsens_model_names.links[xsens_link.segmentId] << ". Exiting...";
                return false;
            }
            else
            {
                link_name_list_.push_back(xsens_model_names.links[xsens_link.segmentId]);
            }
        }
        else if (24 <= xsens_link.segmentId && xsens_link.segmentId < 24 + prop_count_)
        {
            int prop_index = xsens_link.segmentId - 24;
            std::string prop_name = "prop" + std::to_string(prop_index + 1);
            
            if(!human_data_->setLink(prop_name, hrii::ergonomics::Link(prop_name)))
            {
                std::cerr << "Error inserting prop link " << prop_name << ". Exiting...";
                return false;
            }
            else
            {
                link_name_list_.push_back(prop_name);
                if (xsensVerboseLoggingEnabled()) {
                    std::cout << "Added prop: " << prop_name << " (ID: " << xsens_link.segmentId << ")" << std::endl;
                }
            }
        }
        else
        {
            std::cerr << "Segment ID " << xsens_link.segmentId << " is out of bounds. Max index is " << (xsens_model_names.links.size()-1) << std::endl;
            continue;
        }
    }
    
    if (human_data_->getLinks().size() == 0)
    {
        std::cerr << "No link elements found." << std::endl;
        return false;
    }

    if (has_initial_joint_angles)
    {
        const auto& joint_data = joint_angles_datagram_ptr_->getData();
        if (xsensVerboseLoggingEnabled()) {
            std::cout << "Available joints: " << joint_data.size() << std::endl;
        }

        for (size_t joint_cnt = 0; joint_cnt < joint_data.size() && joint_cnt < xsens_model_names.joints.size(); joint_cnt++)
        {
            auto xsens_joint = joint_data[joint_cnt];
            if (xsens_joint.parentSegmentId > 0 && xsens_joint.parentSegmentId - 1 < link_name_list_.size() &&
                xsens_joint.childSegmentId > 0 && xsens_joint.childSegmentId - 1 < link_name_list_.size())
            {
                if (xsensVerboseLoggingEnabled()) {
                    std::cout << joint_cnt << ") " << link_name_list_[xsens_joint.parentSegmentId - 1]
                              << "(" << xsens_joint.parentSegmentId - 1 << ") -> "
                              << link_name_list_[xsens_joint.childSegmentId - 1]
                              << "(" << xsens_joint.childSegmentId - 1 << ")" << std::endl;
                }
            }
            else
            {
                std::cerr << "Invalid segment IDs for joint " << joint_cnt << ": parent="
                          << xsens_joint.parentSegmentId << ", child=" << xsens_joint.childSegmentId << std::endl;
            }
            if (xsensVerboseLoggingEnabled()) {
                std::cout << xsens_model_names.joints[joint_cnt] << std::endl;
            }
        }
    }

    for (const auto& joint_name : xsens_model_names.joints)
    {
        if(!human_data_->setJoint(joint_name, hrii::ergonomics::Joint(joint_name)))
        {
            std::cerr << "Error inserting joint " << joint_name << ". Exiting...";
            return false;
        }
        joint_name_list_.push_back(joint_name);
    }
    
    if (has_finger_data_)
    {
        // std::cout << "DEBUG: About to set up finger joints" << std::endl;
        setupFingerJoints(xsens_model_names);
        // std::cout << "DEBUG: Finished setting up finger joints" << std::endl;
    }
    
    if (human_data_->getJoints().size() == 0)
    {
        std::cerr << "No joint elements found." << std::endl;
        return false;
    }

    // std::cout << "DEBUG: buildXSensModel completed successfully" << std::endl;
    return true;
}

int XSensClient::detectPropCount(QuaternionDatagram& quaternion_datagram)
{
    // std::cout << "DEBUG: detectPropCount starting" << std::endl;
    
    try {
        const std::vector<quaternionKinematics>& data = quaternion_datagram.getData();
        // std::cout << "DEBUG: Quaternion data size: " << data.size() << std::endl;
        
        int max_prop_id = 23;
        
        for (const auto& xsens_link : data)
        {
            int segment_id = xsens_link.segmentId;
            
            if (segment_id >= 24 && segment_id <= 27) 
            {
                if (segment_id > max_prop_id) {
                    max_prop_id = segment_id;
                }
            }
        }
        
        if (max_prop_id > 23) {
            int detected_props = max_prop_id - 23;
            // std::cout << "DEBUG: Detected " << detected_props 
                    //  << " props (max prop ID: " << max_prop_id << ")" << std::endl;
            return detected_props;
        }
        
        // std::cout << "DEBUG: No props detected" << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR in detectPropCount: " << e.what() << std::endl;
        return 0;
    }
}

bool XSensClient::detectFingerData(QuaternionDatagram& quaternion_datagram)
{
    MvnModelNames xsens_model_names;

    try {
        auto quaternion_data = quaternion_datagram.getData();
        // std::cout << "DEBUG: Got quaternion data in detectFingerData, size: " << quaternion_data.size() << std::endl;
        
        for (auto xsens_link: quaternion_data)
        {
            std::string link_name = xsens_model_names.getSegmentNameFromId(xsens_link.segmentId, prop_count_);
            
            bool is_left_finger = std::find(xsens_model_names.left_finger_links.begin(),
                                  xsens_model_names.left_finger_links.end(),
                                  link_name) != xsens_model_names.left_finger_links.end();
                                  
            bool is_right_finger = std::find(xsens_model_names.right_finger_links.begin(),
                                    xsens_model_names.right_finger_links.end(),
                                    link_name) != xsens_model_names.right_finger_links.end();
                                    
            if (is_left_finger || is_right_finger)
            {
                return true;
            }
        }
        
        return false;
    }
    catch (const std::bad_alloc& e) {
       // std::cerr << "DEBUG: Memory allocation error in detectFingerData: " << e.what() << std::endl;
        return false;
    }
    catch (const std::exception& e) {
       // std::cerr << "DEBUG: Exception in detectFingerData: " << e.what() << std::endl;
        return false;
    }
}

void XSensClient::setupFingerJoints(const MvnModelNames& xsens_model_names)
{
    // std::cout << "DEBUG: setupFingerJoints starting" << std::endl;
    auto finger_hierarchy = xsens_model_names.getFingerHierarchy(prop_count_);
    // std::cout << "DEBUG: Got finger hierarchy with " << finger_hierarchy.size() << " entries" << std::endl;
    
    // std::cout << "DEBUG: Adding " << xsens_model_names.left_finger_joints.size() << " left finger joints" << std::endl;
    for (size_t i = 0; i < xsens_model_names.left_finger_joints.size(); i++)
    {
        std::string joint_name = xsens_model_names.left_finger_joints[i];
        if (!human_data_->setJoint(joint_name, hrii::ergonomics::Joint(joint_name)))
        {
            std::cerr << "Error inserting finger joint " << joint_name << std::endl;
        }
        else
        {
            finger_joint_name_list_.push_back(joint_name);
            if (xsensVerboseLoggingEnabled()) {
                std::cout << "Added finger joint: " << joint_name << std::endl;
            }
        }
    }
    
    // std::cout << "DEBUG: Adding " << xsens_model_names.right_finger_joints.size() << " right finger joints" << std::endl;
    for (size_t i = 0; i < xsens_model_names.right_finger_joints.size(); i++)
    {
        std::string joint_name = xsens_model_names.right_finger_joints[i];
        if (!human_data_->setJoint(joint_name, hrii::ergonomics::Joint(joint_name)))
        {
            std::cerr << "Error inserting finger joint " << joint_name << std::endl;
        }
        else
        {
            finger_joint_name_list_.push_back(joint_name);
            if (xsensVerboseLoggingEnabled()) {
                std::cout << "Added finger joint: " << joint_name << std::endl;
            }
        }
    }
    // std::cout << "DEBUG: setupFingerJoints completed, added " << finger_joint_name_list_.size() << " finger joints" << std::endl;
}

QuaternionDatagram* XSensClient::waitForQuaternionDatagram()
{
    // std::cout << "DEBUG: waitForQuaternionDatagram starting" << std::endl;
    std::cout << "Waiting for quaternion datagram..." << std::endl;
    int max_attempts = 100;
    int attempts = 0;
    
    while (parser_manager_.getQuaternionDatagram() == NULL && attempts < max_attempts)
    {
        if(!readData()) 
        {
            std::cerr << "Failed to read data while waiting for quaternion datagram" << std::endl;
            return nullptr;
        }
        
        attempts++;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    if (attempts >= max_attempts)
    {
        std::cerr << "Exceeded maximum attempts waiting for quaternion datagram" << std::endl;
        return nullptr;
    }
    
    std::cout << "Quaternion datagram received." << std::endl;
    
    QuaternionDatagram* source_datagram = parser_manager_.getQuaternionDatagram();
    
    QuaternionDatagram* copy_datagram = nullptr;
    if (source_datagram) {
        try {
            copy_datagram = new QuaternionDatagram(*source_datagram);
            // std::cout << "DEBUG: Created copy of quaternion datagram" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Failed to copy quaternion datagram: " << e.what() << std::endl;
            return nullptr;
        }
    }
    
    return copy_datagram;
}

JointAnglesDatagram* XSensClient::waitForJointAnglesDatagram(bool required)
{
    // std::cout << "DEBUG: waitForJointAnglesDatagram starting" << std::endl;
    std::cout << "Waiting for joint angles..." << std::endl;
    int max_attempts = 100;
    int attempts = 0;
    
    while (parser_manager_.getJointAnglesDatagram() == NULL && attempts < max_attempts)
    {
        if(!readData()) 
        {
            std::cerr << "Failed to read data while waiting for joint angles datagram" << std::endl;
            return nullptr;
        }
        
        attempts++;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    if (attempts >= max_attempts)
    {
        if (required)
        {
            std::cerr << "Exceeded maximum attempts waiting for joint angles datagram" << std::endl;
        }
        else
        {
            std::cerr << "Joint angles datagram not received within initialization window" << std::endl;
        }
        return nullptr;
    }
    
    std::cout << "Joint angles datagram received." << std::endl;
    
    JointAnglesDatagram* source_datagram = parser_manager_.getJointAnglesDatagram();
    
    JointAnglesDatagram* copy_datagram = nullptr;
    if (source_datagram) {
        try {
            copy_datagram = new JointAnglesDatagram(*source_datagram);
            // std::cout << "DEBUG: Created copy of joint angles datagram" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Failed to copy joint angles datagram: " << e.what() << std::endl;
            return nullptr;
        }
    }
    
    return copy_datagram;
}

void XSensClient::updateJointAngles()
{
    // std::cout << "DEBUG: updateJointAngles starting" << std::endl;
    auto joint_angles = parser_manager_.getJointAnglesDatagram();
    if(joint_angles != NULL)
    {
        human_data_->setJointAngles("l5_s1",             jointAngleToEigenVector3d(joint_angles->getItem(1, 2), 1, 1, 1));
        human_data_->setJointAngles("l4_l3",             jointAngleToEigenVector3d(joint_angles->getItem(2, 3), 1, 1, 1));
        human_data_->setJointAngles("l1_t12",            jointAngleToEigenVector3d(joint_angles->getItem(3, 4), 1, 1, 1));
        human_data_->setJointAngles("t9_t8",             jointAngleToEigenVector3d(joint_angles->getItem(4, 5), 1, 1, 1));
        human_data_->setJointAngles("t1_c7",             jointAngleToEigenVector3d(joint_angles->getItem(5, 6), 1, 1, 1));
        human_data_->setJointAngles("c1_head",           jointAngleToEigenVector3d(joint_angles->getItem(6, 7), 1, 1, 1));
        human_data_->setJointAngles("right_c7_shoulder", jointAngleToEigenVector3d(joint_angles->getItem(5, 8), -1, 1, -1));
        human_data_->setJointAngles("right_shoulder",    jointAngleToEigenVector3d(joint_angles->getItem(8, 9), -1, 1, 1));
        human_data_->setJointAngles("right_elbow",       jointAngleToEigenVector3d(joint_angles->getItem(9, 10), -1, 1, -1));
        human_data_->setJointAngles("right_wrist",       jointAngleToEigenVector3d(joint_angles->getItem(10, 11), -1, 1, -1));
        human_data_->setJointAngles("left_c7_shoulder",  jointAngleToEigenVector3d(joint_angles->getItem(5, 12), 1, -1, -1));
        human_data_->setJointAngles("left_shoulder",     jointAngleToEigenVector3d(joint_angles->getItem(12, 13), 1, -1, -1));
        human_data_->setJointAngles("left_elbow",        jointAngleToEigenVector3d(joint_angles->getItem(13, 14), 1, -1, -1));
        human_data_->setJointAngles("left_wrist",        jointAngleToEigenVector3d(joint_angles->getItem(14, 15), 1, -1, -1));
        human_data_->setJointAngles("right_hip",         jointAngleToEigenVector3d(joint_angles->getItem(1, 16), -1, -1, 1));
        human_data_->setJointAngles("right_knee",        jointAngleToEigenVector3d(joint_angles->getItem(16, 17), -1, 1, -1));
        human_data_->setJointAngles("right_ankle",       jointAngleToEigenVector3d(joint_angles->getItem(17, 18), -1, 1, -1));
        human_data_->setJointAngles("right_ballfoot",    jointAngleToEigenVector3d(joint_angles->getItem(18, 19), -1, 1, -1));
        human_data_->setJointAngles("left_hip",          jointAngleToEigenVector3d(joint_angles->getItem(1, 20), 1, 1, -1));
        human_data_->setJointAngles("left_knee",         jointAngleToEigenVector3d(joint_angles->getItem(20, 21), 1, -1, 1));
        human_data_->setJointAngles("left_ankle",        jointAngleToEigenVector3d(joint_angles->getItem(21, 22), 1, -1, -1));
        human_data_->setJointAngles("left_ballfoot",     jointAngleToEigenVector3d(joint_angles->getItem(22, 23), 1, -1, -1));

        // hrii::ergonomics::Joint elbow_joint;
        // human_data_->getJoint("right_elbow", elbow_joint);
        // elbow_joint.state.angles[2] -= 90.0;
        // human_data_->setJoint("right_elbow", elbow_joint);
        // human_data_->getJoint("left_elbow", elbow_joint);
        // elbow_joint.state.angles[2] += 90.0;
        // human_data_->setJoint("left_elbow", elbow_joint);

        hrii::ergonomics::Joint shoulder_joint;
        human_data_->getJoint("right_shoulder", shoulder_joint);
        shoulder_joint.state.angles[0] += 90.0;
        human_data_->setJoint("right_shoulder", shoulder_joint);

        human_data_->getJoint("left_shoulder", shoulder_joint);
        shoulder_joint.state.angles[0] -= 90.0;
        human_data_->setJoint("left_shoulder", shoulder_joint);
    }
    // std::cout << "DEBUG: updateJointAngles completed" << std::endl;
}

void XSensClient::updateFingerJointAngles()
{
    // std::cout << "DEBUG: updateFingerJointAngles starting" << std::endl;
    if (!has_finger_data_)
    {
        std::cerr << "No finger data available" << std::endl;
        return;
    }
    
    auto finger_quats = extractFingerQuaternions();
    m_leftFingerQuats = finger_quats.first;
    m_rightFingerQuats = finger_quats.second;
    m_hasStoredFingerQuats = true;
    
    if (m_leftFingerQuats.empty() && m_rightFingerQuats.empty())
    {
        std::cerr << "No finger quaternions available" << std::endl;
        return;
    }
    
    bool allIdentity = true;
    for (const auto& q : m_leftFingerQuats) {
        if (!q.isApprox(Eigen::Quaternionf::Identity())) {
            allIdentity = false;
            break;
        }
    }
    
    for (const auto& q : m_rightFingerQuats) {
        if (!q.isApprox(Eigen::Quaternionf::Identity())) {
            allIdentity = false;
            break;
        }
    }
    
    if (allIdentity) {
        std::cerr << "All finger quaternions are identity, likely missing data" << std::endl;
        return;
    }
    
    auto joint_angles_datagram = parser_manager_.getJointAnglesDatagram();
    if (joint_angles_datagram == NULL)
    {
        joint_angles_datagram = joint_angles_datagram_ptr_;
        if (joint_angles_datagram == NULL)
        {
            std::cerr << "Joint angles datagram is NULL" << std::endl;
            return;
        }
    }
    
    // std::cout << "DEBUG: Calling calculateFingerJointAngles with " 
            //   << m_leftFingerQuats.size() << " left quats and " 
            //   << m_rightFingerQuats.size() << " right quats" << std::endl;
    
    joint_angles_datagram->calculateFingerJointAngles(m_leftFingerQuats, m_rightFingerQuats, prop_count_);
    
    if (!joint_angles_datagram->hasFingerData())
    {
        std::cerr << "No finger joint angles were calculated" << std::endl;
        return;
    }
    
    auto finger_joint_angles = joint_angles_datagram->getFingerJointAngles();
    if (xsensVerboseLoggingEnabled()) {
        std::cout << "Updating " << finger_joint_angles.size() << " finger joint angles" << std::endl;
    }
    
    int success_count = 0;
    int fail_count = 0;
    
    for (const auto& finger_joint : finger_joint_angles)
    {
        Eigen::Vector3d angles;
        angles[0] = finger_joint.rotation[0];
        angles[1] = finger_joint.rotation[1];
        angles[2] = finger_joint.rotation[2];
        
        // std::cout << "DEBUG: Joint " << finger_joint.jointName 
                //   << " angles: [" << angles[0] << ", " << angles[1] << ", " << angles[2] << "]" << std::endl;
        
        if (human_data_->setJointAngles(finger_joint.jointName, angles))
        {
            success_count++;
        }
        else
        {
            fail_count++;
            std::cerr << "Failed to set angles for joint " << finger_joint.jointName << std::endl;
        }
    }
    
    // std::cout << "DEBUG: Successfully updated " << success_count << " finger joints, failed for " 
            //   << fail_count << " joints" << std::endl;
}

void XSensClient::updateLinkPoses()
{
    // std::cout << "DEBUG: updateLinkPoses starting" << std::endl;
    auto quaternion_datagram = parser_manager_.getQuaternionDatagram();
    if(quaternion_datagram != NULL)
    {
        MvnModelNames xsens_model_names;
        
        for (int link_cnt = 0; link_cnt < link_name_list_.size(); link_cnt++)
        {
            auto link_pose = quaternion_datagram->getItem(link_cnt+1);
            Eigen::Vector3d link_pos;
            link_pos << link_pose.sensorPos[0], link_pose.sensorPos[1], link_pose.sensorPos[2];
            Eigen::Quaterniond link_orient(link_pose.quatRotation[0], link_pose.quatRotation[1], 
                                          link_pose.quatRotation[2], link_pose.quatRotation[3]);
            human_data_->setLinkPose(link_name_list_[link_cnt], link_pos, link_orient);
        }
        
        if (has_finger_data_)
        {
            // std::cout << "DEBUG: Extracting finger quaternions in updateLinkPoses" << std::endl;
            auto finger_quats = extractFingerQuaternions();
            m_leftFingerQuats = finger_quats.first;
            m_rightFingerQuats = finger_quats.second;
            m_hasStoredFingerQuats = true;
            
            for (const auto& finger_link_name : finger_link_name_list_)
            {
                int segment_id = xsens_model_names.getFingerSegmentId(finger_link_name, prop_count_);
                if (segment_id != -1)
                {
                    try {
                        auto link_pose = quaternion_datagram->getItem(segment_id);
                        Eigen::Vector3d link_pos;
                        link_pos << link_pose.sensorPos[0], link_pose.sensorPos[1], link_pose.sensorPos[2];
                        Eigen::Quaterniond link_orient(link_pose.quatRotation[0], link_pose.quatRotation[1], 
                                                      link_pose.quatRotation[2], link_pose.quatRotation[3]);
                        human_data_->setLinkPose(finger_link_name, link_pos, link_orient);
                    }
                    catch (const std::exception& e) {
                        std::cerr << "Error getting pose for finger segment " << finger_link_name 
                                  << " (ID: " << segment_id << "): " << e.what() << std::endl;
                    }
                }
            }
        }
    }
    // std::cout << "DEBUG: updateLinkPoses completed" << std::endl;
}

void XSensClient::rotateLink(const std::string& link_name, const Eigen::Quaterniond& quat)
{
    // std::cout << "DEBUG: rotateLink " << link_name << std::endl;
    hrii::ergonomics::Link link_to_rotate;
    human_data_->getLink(link_name, link_to_rotate);
    link_to_rotate.state.orientation = link_to_rotate.state.orientation * quat;
    human_data_->setLink(link_name, link_to_rotate);
}

void XSensClient::rotateJoint(const std::string& joint_name, const Eigen::Quaterniond& quat)
{
    // std::cout << "DEBUG: rotateJoint " << joint_name << std::endl;
    hrii::ergonomics::Joint joint_to_rotate;
    if (human_data_->getJoint(joint_name, joint_to_rotate))
    {
        Eigen::Vector3d angle_vec = joint_to_rotate.state.angles;
        
        Eigen::Quaterniond joint_quat = Eigen::AngleAxisd(angle_vec[0], Eigen::Vector3d::UnitX()) *
                                        Eigen::AngleAxisd(angle_vec[1], Eigen::Vector3d::UnitY()) *
                                        Eigen::AngleAxisd(angle_vec[2], Eigen::Vector3d::UnitZ());
        
        Eigen::Quaterniond rotated_quat = quat * joint_quat;
        
        Eigen::Vector3d rotated_angles = rotated_quat.toRotationMatrix().eulerAngles(0, 1, 2);
        
        joint_to_rotate.state.angles = rotated_angles;
        
        human_data_->setJoint(joint_name, joint_to_rotate);
    }
    else
    {
        std::cerr << "Joint " << joint_name << " not found" << std::endl;
    }
}

void XSensClient::updateLinkLinearTwists()
{
    // std::cout << "DEBUG: updateLinkLinearTwists starting" << std::endl;
    auto linear_segment_kinematics_datagram = parser_manager_.getLinearSegmentKinematicsDatagram();

    if (linear_segment_kinematics_datagram != NULL)
    {
        MvnModelNames xsens_model_names;
        
        for (int link_cnt = 0; link_cnt < link_name_list_.size(); link_cnt++)
        {
            auto link_linear_kinematics = linear_segment_kinematics_datagram->getItem(link_cnt+1);

            hrii::ergonomics::Link link;
            if (!human_data_->getLink(link_name_list_[link_cnt], link))
            {
                std::cerr << "Link " << link_name_list_[link_cnt] << " not found" << std::endl;
            }
            else
            {
                link.state.velocity.linear << link_linear_kinematics.velocity[0], 
                                              link_linear_kinematics.velocity[1], 
                                              link_linear_kinematics.velocity[2];
                link.state.acceleration.linear <<   link_linear_kinematics.acceleration[0], 
                                                    link_linear_kinematics.acceleration[1], 
                                                    link_linear_kinematics.acceleration[2];
        
                human_data_->setLinkState(link_name_list_[link_cnt], link.state);
            }
        }
        
        if (has_finger_data_)
        {
            for (const auto& finger_link_name : finger_link_name_list_)
            {
                int segment_id = xsens_model_names.getFingerSegmentId(finger_link_name, prop_count_);
                if (segment_id != -1)
                {
                    try {
                        auto link_linear_kinematics = linear_segment_kinematics_datagram->getItem(segment_id);
                        
                        hrii::ergonomics::Link link;
                        if (!human_data_->getLink(finger_link_name, link))
                        {
                            std::cerr << "Finger link " << finger_link_name << " not found" << std::endl;
                        }
                        else
                        {
                            link.state.velocity.linear << link_linear_kinematics.velocity[0], 
                                                        link_linear_kinematics.velocity[1], 
                                                        link_linear_kinematics.velocity[2];
                            link.state.acceleration.linear << link_linear_kinematics.acceleration[0], 
                                                            link_linear_kinematics.acceleration[1], 
                                                            link_linear_kinematics.acceleration[2];
                    
                            human_data_->setLinkState(finger_link_name, link.state);
                        }
                    }
                    catch (const std::exception& e) {
                        std::cerr << "Error getting linear kinematics for finger segment " << finger_link_name 
                                  << " (ID: " << segment_id << "): " << e.what() << std::endl;
                    }
                }
            }
        }
    }
    // std::cout << "DEBUG: updateLinkLinearTwists completed" << std::endl;
}

void XSensClient::updateLinkAngularTwists()
{
    // std::cout << "DEBUG: updateLinkAngularTwists starting" << std::endl;
    auto angular_segment_kinematics_datagram = parser_manager_.getAngularSegmentKinematicsDatagram();

    if (angular_segment_kinematics_datagram != NULL)
    {
        MvnModelNames xsens_model_names;
        
        for (int link_cnt = 0; link_cnt < link_name_list_.size(); link_cnt++)
        {
            auto link_angular_kinematics = angular_segment_kinematics_datagram->getItem(link_cnt+1);

            hrii::ergonomics::Link link;
            if (!human_data_->getLink(link_name_list_[link_cnt], link))
            {
                std::cerr << "Link " << link_name_list_[link_cnt] << " not found" << std::endl;
            }
            else
            {
                link.state.velocity.angular <<  link_angular_kinematics.angularVeloc[0]*M_PI/180, 
                                                link_angular_kinematics.angularVeloc[1]*M_PI/180, 
                                                link_angular_kinematics.angularVeloc[2]*M_PI/180;
                link.state.acceleration.angular <<  link_angular_kinematics.angularAccel[0]*M_PI/180, 
                                                    link_angular_kinematics.angularAccel[1]*M_PI/180, 
                                                    link_angular_kinematics.angularAccel[2]*M_PI/180;

                human_data_->setLinkState(link_name_list_[link_cnt], link.state);
            }
        }
        
        if (has_finger_data_)
        {
            for (const auto& finger_link_name : finger_link_name_list_)
            {
                int segment_id = xsens_model_names.getFingerSegmentId(finger_link_name, prop_count_);
                if (segment_id != -1)
                {
                    try {
                        auto link_angular_kinematics = angular_segment_kinematics_datagram->getItem(segment_id);
                        
                        hrii::ergonomics::Link link;
                        if (!human_data_->getLink(finger_link_name, link))
                        {
                            std::cerr << "Finger link " << finger_link_name << " not found" << std::endl;
                        }
                        else
                        {
                            link.state.velocity.angular <<  link_angular_kinematics.angularVeloc[0]*M_PI/180, 
                                                            link_angular_kinematics.angularVeloc[1]*M_PI/180, 
                                                            link_angular_kinematics.angularVeloc[2]*M_PI/180;
                            link.state.acceleration.angular <<  link_angular_kinematics.angularAccel[0]*M_PI/180, 
                                                                link_angular_kinematics.angularAccel[1]*M_PI/180, 
                                                                link_angular_kinematics.angularAccel[2]*M_PI/180;

                            human_data_->setLinkState(finger_link_name, link.state);
                        }
                    }
                    catch (const std::exception& e) {
                        std::cerr << "Error getting angular kinematics for finger segment " << finger_link_name 
                                  << " (ID: " << segment_id << "): " << e.what() << std::endl;
                    }
                }
            }
        }
    }
    // std::cout << "DEBUG: updateLinkAngularTwists completed" << std::endl;
}

void XSensClient::updateCOM()
{
    // std::cout << "DEBUG: updateCOM starting" << std::endl;
    auto com_datagram = parser_manager_.getCenterOfMassDatagram();

    if (com_datagram != NULL)
    {
        Eigen::Vector3d com;
        auto com_data = com_datagram->getData();
        com << com_data[0], com_data[1], com_data[2];
        human_data_->setCOM(com);
    }
    // std::cout << "DEBUG: updateCOM completed" << std::endl;
}

Eigen::Vector3d XSensClient::jointAngleToEigenVector3d(const JointAngle& joint_angle, 
                                                        const double& x_axis, 
                                                        const double& y_axis, 
                                                        const double& z_axis)
{
    Eigen::Vector3d joint_angle_eigen_vec;
    joint_angle_eigen_vec[0] = x_axis * joint_angle.rotation[0];
    joint_angle_eigen_vec[1] = z_axis * joint_angle.rotation[2];
    joint_angle_eigen_vec[2] = y_axis * joint_angle.rotation[1];
    return joint_angle_eigen_vec;
}

hrii::ergonomics::HumanDataHandler::Ptr XSensClient::getHumanData()
{
    return human_data_;
}

bool XSensClient::copyFrame(xsens::core::XsensFrame& frame, std::uint64_t& frame_sequence) const
{
    if (!human_data_)
    {
        return false;
    }

    frame = xsens::core::XsensFrame{};
    frame_sequence = frame_sequence_.load();
    frame.center_of_mass = human_data_->getCOM();

    for (const auto& joint_entry : human_data_->getJoints())
    {
        xsens::core::JointSample joint_sample;
        joint_sample.name = joint_entry.first;
        joint_sample.angles = joint_entry.second.state.angles;
        frame.joints.push_back(joint_sample);
    }

    for (const auto& link_entry : human_data_->getLinks())
    {
        xsens::core::LinkSample link_sample;
        link_sample.name = link_entry.first;
        link_sample.position = link_entry.second.state.position;
        link_sample.orientation = link_entry.second.state.orientation;
        link_sample.linear_velocity = link_entry.second.state.velocity.linear;
        link_sample.angular_velocity = link_entry.second.state.velocity.angular;
        link_sample.linear_acceleration = link_entry.second.state.acceleration.linear;
        link_sample.angular_acceleration = link_entry.second.state.acceleration.angular;
        frame.links.push_back(link_sample);
    }

    return true;
}

std::pair<std::vector<Eigen::Quaternionf>, std::vector<Eigen::Quaternionf>> XSensClient::extractFingerQuaternions()
{
    // std::cout << "DEBUG: extractFingerQuaternions starting" << std::endl;
    auto quaternion_datagram = parser_manager_.getQuaternionDatagram();
    std::vector<Eigen::Quaternionf> leftFingerQuats;
    std::vector<Eigen::Quaternionf> rightFingerQuats;
    
    if (quaternion_datagram == NULL || !has_finger_data_)
    {
        std::cerr << "No quaternion datagram or finger data flag not set" << std::endl;
        return std::make_pair(leftFingerQuats, rightFingerQuats);
    }
    
    int left_finger_base = 24 + prop_count_;
    int right_finger_base = left_finger_base + 20;

    // std::cout << "Left finger base index: " << left_finger_base << ", Right finger base index: " << right_finger_base << std::endl;
    
    leftFingerQuats.resize(20, Eigen::Quaternionf::Identity());
    rightFingerQuats.resize(20, Eigen::Quaternionf::Identity());
    
    auto quaternion_data = quaternion_datagram->getData();
    // std::cout << "DEBUG: Processing " << quaternion_data.size() << " quaternions" << std::endl;
    
    // std::cout << "DEBUG: All segment IDs in quaternion data: ";
    for (const auto& quat_kinematics : quaternion_data) {
        if (xsensVerboseLoggingEnabled()) {
            std::cout << quat_kinematics.segmentId << " ";
        }
    }
    if (xsensVerboseLoggingEnabled()) {
        std::cout << std::endl;
    }
    
    int left_finger_count = 0;
    int right_finger_count = 0;
    
    for (const auto& quat_kinematics : quaternion_data)
    {
        int segmentId = quat_kinematics.segmentId;
        
        if (segmentId >= left_finger_base && segmentId < left_finger_base + 20)
        {
            int finger_index = segmentId - left_finger_base;
            
            if (finger_index >= 0 && finger_index < 20)
            {
                leftFingerQuats[finger_index] = Eigen::Quaternionf(
                    quat_kinematics.quatRotation[0],
                    quat_kinematics.quatRotation[1],
                    quat_kinematics.quatRotation[2],
                    quat_kinematics.quatRotation[3]
                );
                left_finger_count++;
                // std::cout << "DEBUG: Stored left finger quat at index " << finger_index << " from segment ID " << segmentId << std::endl;
            }
        }
        else if (segmentId >= right_finger_base && segmentId < right_finger_base + 20)
        {
            int finger_index = segmentId - right_finger_base;
            
            if (finger_index >= 0 && finger_index < 20)
            {
                rightFingerQuats[finger_index] = Eigen::Quaternionf(
                    quat_kinematics.quatRotation[0],
                    quat_kinematics.quatRotation[1],
                    quat_kinematics.quatRotation[2],
                    quat_kinematics.quatRotation[3]
                );
                right_finger_count++;
                // std::cout << "DEBUG: Stored right finger quat at index " << finger_index << " from segment ID " << segmentId << std::endl;
            }
        }
    }
    
    // std::cout << "DEBUG: Extracted " << left_finger_count << " left finger quaternions and " 
            //   << right_finger_count << " right finger quaternions" << std::endl;
    
    return std::make_pair(leftFingerQuats, rightFingerQuats);
}

std::string XSensClient::getURDFString()
{
    return "";
}
