#ifndef _MVN_MODEL_H_
#define _MVN_MODEL_H_

#include <vector>
#include <string>
#include <tuple>

class MvnModelNames
{
public:
    const std::vector<std::string> links = {
        "base_link",      // 0 (Not used in XSens)
        "pelvis",         // 1
        "l5",             // 2
        "l3",             // 3
        "t12",            // 4
        "t8",             // 5
        "neck",           // 6
        "head",           // 7
        "right_shoulder", // 8
        "right_upper_arm",// 9
        "right_forearm",  // 10
        "right_hand",     // 11
        "left_shoulder",  // 12
        "left_upper_arm", // 13
        "left_forearm",   // 14
        "left_hand",      // 15
        "right_upper_leg",// 16
        "right_lower_leg",// 17
        "right_foot",     // 18
        "right_toe",      // 19
        "left_upper_leg", // 20
        "left_lower_leg", // 21
        "left_foot",      // 22
        "left_toe",       // 23
        "generic_link"    // 24 (placeholder for props)
    };

    const std::vector<std::string> joints = {
        "l5_s1",
        "l4_l3",
        "l1_t12",
        "t9_t8",
        "t1_c7",
        "c1_head",
        "right_c7_shoulder",
        "right_shoulder",
        "right_elbow",
        "right_wrist",
        "left_c7_shoulder",
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "right_hip",
        "right_knee",
        "right_ankle",
        "right_ballfoot",
        "left_hip",
        "left_knee",
        "left_ankle",
        "left_ballfoot",
        "t8_head_NA",
        "t8_left_upper_arm_NA",
        "t8_right_upper_arm_NA",
        "pelvis_t8_NA",
        "pelvis_pelvis_NA",
        "pelvis_t8_v2_NA"
    };

    // Define prop segments (24-27)
    const std::vector<std::string> prop_links = {
        "prop1",
        "prop2",
        "prop3",
        "prop4"
    };

    // Left hand fingers (starting at index 28 by default, but adjusts based on prop count)
    const std::vector<std::string> left_finger_links = {
        "left_carpus",               // 0
        "left_first_metacarpal",     // 1
        "left_first_proximal",       // 2
        "left_first_distal",         // 3
        "left_second_metacarpal",    // 4
        "left_second_proximal",      // 5
        "left_second_middle",        // 6
        "left_second_distal",        // 7
        "left_third_metacarpal",     // 8
        "left_third_proximal",       // 9
        "left_third_middle",         // 10
        "left_third_distal",         // 11
        "left_fourth_metacarpal",    // 12
        "left_fourth_proximal",      // 13
        "left_fourth_middle",        // 14
        "left_fourth_distal",        // 15
        "left_fifth_metacarpal",     // 16
        "left_fifth_proximal",       // 17
        "left_fifth_middle",         // 18
        "left_fifth_distal"          // 19
    };

    // Right hand fingers (starting at index 48 by default, but adjusts based on prop count)
    const std::vector<std::string> right_finger_links = {
        "right_carpus",              // 0
        "right_first_metacarpal",    // 1
        "right_first_proximal",      // 2
        "right_first_distal",        // 3
        "right_second_metacarpal",   // 4
        "right_second_proximal",     // 5
        "right_second_middle",       // 6
        "right_second_distal",       // 7
        "right_third_metacarpal",    // 8
        "right_third_proximal",      // 9
        "right_third_middle",        // 10
        "right_third_distal",        // 11
        "right_fourth_metacarpal",   // 12
        "right_fourth_proximal",     // 13
        "right_fourth_middle",       // 14
        "right_fourth_distal",       // 15
        "right_fifth_metacarpal",    // 16
        "right_fifth_proximal",      // 17
        "right_fifth_middle",        // 18
        "right_fifth_distal"         // 19
    };

    // Finger joint names - left hand, 19 joints
    const std::vector<std::string> left_finger_joints = {
        "left_first_carpus_metacarpal",      // 0-carpus to first metacarpal
        "left_first_metacarpal_proximal",    // 1-first metacarpal to proximal
        "left_first_proximal_distal",        // 2-first proximal to distal
        "left_second_carpus_metacarpal",     // 3-carpus to second metacarpal
        "left_second_metacarpal_proximal",   // 4-second metacarpal to proximal
        "left_second_proximal_middle",       // 5-second proximal to middle
        "left_second_middle_distal",         // 6-second middle to distal
        "left_third_carpus_metacarpal",      // 7-carpus to third metacarpal
        "left_third_metacarpal_proximal",    // 8-third metacarpal to proximal
        "left_third_proximal_middle",        // 9-third proximal to middle
        "left_third_middle_distal",          // 10-third middle to distal
        "left_fourth_carpus_metacarpal",     // 11-carpus to fourth metacarpal
        "left_fourth_metacarpal_proximal",   // 12-fourth metacarpal to proximal
        "left_fourth_proximal_middle",       // 13-fourth proximal to middle
        "left_fourth_middle_distal",         // 14-fourth middle to distal
        "left_fifth_carpus_metacarpal",      // 15-carpus to fifth metacarpal
        "left_fifth_metacarpal_proximal",    // 16-fifth metacarpal to proximal
        "left_fifth_proximal_middle",        // 17-fifth proximal to middle
        "left_fifth_middle_distal"           // 18-fifth middle to distal
    };

    // Finger joint names - right hand, 19 joints
    const std::vector<std::string> right_finger_joints = {
        "right_first_carpus_metacarpal",     // 0-carpus to first metacarpal
        "right_first_metacarpal_proximal",   // 1-first metacarpal to proximal
        "right_first_proximal_distal",       // 2-first proximal to distal
        "right_second_carpus_metacarpal",    // 3-carpus to second metacarpal
        "right_second_metacarpal_proximal",  // 4-second metacarpal to proximal
        "right_second_proximal_middle",      // 5-second proximal to middle
        "right_second_middle_distal",        // 6-second middle to distal
        "right_third_carpus_metacarpal",     // 7-carpus to third metacarpal
        "right_third_metacarpal_proximal",   // 8-third metacarpal to proximal
        "right_third_proximal_middle",       // 9-third proximal to middle
        "right_third_middle_distal",         // 10-third middle to distal
        "right_fourth_carpus_metacarpal",    // 11-carpus to fourth metacarpal
        "right_fourth_metacarpal_proximal",  // 12-fourth metacarpal to proximal
        "right_fourth_proximal_middle",      // 13-fourth proximal to middle
        "right_fourth_middle_distal",        // 14-fourth middle to distal
        "right_fifth_carpus_metacarpal",     // 15-carpus to fifth metacarpal
        "right_fifth_metacarpal_proximal",   // 16-fifth metacarpal to proximal
        "right_fifth_proximal_middle",       // 17-fifth proximal to middle
        "right_fifth_middle_distal"          // 18-fifth middle to distal
    };

    /**
     * Get the segment ID for a finger segment name, adjusting for prop count
     * 
     * @param segment_name The name of the finger segment
     * @param prop_count Number of props used (0-4)
     * @return The segment ID
     */
    int getFingerSegmentId(const std::string& segment_name, int prop_count = 0) const {
        // Start index for left hand fingers depends on prop count
        int left_start = 24 + prop_count;
        
        // Check if it's a left finger segment
        for (size_t i = 0; i < left_finger_links.size(); ++i) {
            if (segment_name == left_finger_links[i]) {
                return left_start + i;
            }
        }
        
        // Start index for right hand fingers is 20 segments after left start
        int right_start = left_start + 20;
        
        // Check if it's a right finger segment
        for (size_t i = 0; i < right_finger_links.size(); ++i) {
            if (segment_name == right_finger_links[i]) {
                return right_start + i;
            }
        }
        
        // Not a finger segment
        return -1;
    }
    
    /**
     * Get the segment name for a given ID, accounting for prop count
     * 
     * @param segment_id The segment ID
     * @param prop_count Number of props used (0-4)
     * @return The segment name
     */
    std::string getSegmentNameFromId(int segment_id, int prop_count = 0) const {
        // Regular body segments (0-23)
        if (0 <= segment_id && segment_id < 24) {
            if (segment_id < links.size()) {
                return links[segment_id];
            }
            return "unknown_segment_" + std::to_string(segment_id);
        }
        
        // Props (24 to 24+prop_count-1)
        if (24 <= segment_id && segment_id < 24 + prop_count) {
            int prop_index = segment_id - 24;
            return "prop" + std::to_string(prop_index + 1);
        }
        
        // Left hand fingers
        int left_start = 24 + prop_count;
        if (left_start <= segment_id && segment_id < left_start + 20) {
            int finger_index = segment_id - left_start;
            if (finger_index < left_finger_links.size()) {
                return left_finger_links[finger_index];
            }
            return "unknown_left_finger_" + std::to_string(segment_id);
        }
        
        // Right hand fingers
        int right_start = left_start + 20;
        if (right_start <= segment_id && segment_id < right_start + 20) {
            int finger_index = segment_id - right_start;
            if (finger_index < right_finger_links.size()) {
                return right_finger_links[finger_index];
            }
            return "unknown_right_finger_" + std::to_string(segment_id);
        }
        
        return "unknown_segment_" + std::to_string(segment_id);
    }
    
    /**
     * Return parent-child segment ID relationships for finger joints,
     * adjusting for prop count
     * 
     * @param prop_count Number of props used (0-4)
     * @return List of (parent_id, child_id) pairs
     */
    std::vector<std::pair<int, int>> getFingerHierarchy(int prop_count = 0) const {
        // Calculate base indices for finger segments
        int left_start = 24 + prop_count;
        int right_start = left_start + 20;
        
        int left_hand = 15;  // Left hand segment ID
        int right_hand = 11; // Right hand segment ID
        
        // This matches the hierarchy in the reference code's stream_hierarchy_map
        std::vector<std::pair<int, int>> finger_hierarchy = {
            // Left hand fingers
            {left_hand, left_start},  // left_hand to left_carpus
            
            // Left thumb
            {left_start, left_start + 1},  // left_carpus to left_first_metacarpal
            {left_start + 1, left_start + 2},  // left_first_metacarpal to left_first_proximal
            {left_start + 2, left_start + 3},  // left_first_proximal to left_first_distal
            
            // Left index finger
            {left_start, left_start + 4},  // left_carpus to left_second_metacarpal
            {left_start + 4, left_start + 5},  // left_second_metacarpal to left_second_proximal
            {left_start + 5, left_start + 6},  // left_second_proximal to left_second_middle
            {left_start + 6, left_start + 7},  // left_second_middle to left_second_distal
            
            // Left middle finger
            {left_start, left_start + 8},  // left_carpus to left_third_metacarpal
            {left_start + 8, left_start + 9},  // left_third_metacarpal to left_third_proximal
            {left_start + 9, left_start + 10},  // left_third_proximal to left_third_middle
            {left_start + 10, left_start + 11},  // left_third_middle to left_third_distal
            
            // Left ring finger
            {left_start, left_start + 12},  // left_carpus to left_fourth_metacarpal
            {left_start + 12, left_start + 13},  // left_fourth_metacarpal to left_fourth_proximal
            {left_start + 13, left_start + 14},  // left_fourth_proximal to left_fourth_middle
            {left_start + 14, left_start + 15},  // left_fourth_middle to left_fourth_distal
            
            // Left pinky finger
            {left_start, left_start + 16},  // left_carpus to left_fifth_metacarpal
            {left_start + 16, left_start + 17},  // left_fifth_metacarpal to left_fifth_proximal
            {left_start + 17, left_start + 18},  // left_fifth_proximal to left_fifth_middle
            {left_start + 18, left_start + 19},  // left_fifth_middle to left_fifth_distal
            
            // Right hand fingers
            {right_hand, right_start},  // right_hand to right_carpus
            
            // Right thumb
            {right_start, right_start + 1},  // right_carpus to right_first_metacarpal
            {right_start + 1, right_start + 2},  // right_first_metacarpal to right_first_proximal
            {right_start + 2, right_start + 3},  // right_first_proximal to right_first_distal
            
            // Right index finger
            {right_start, right_start + 4},  // right_carpus to right_second_metacarpal
            {right_start + 4, right_start + 5},  // right_second_metacarpal to right_second_proximal
            {right_start + 5, right_start + 6},  // right_second_proximal to right_second_middle
            {right_start + 6, right_start + 7},  // right_second_middle to right_second_distal
            
            // Right middle finger
            {right_start, right_start + 8},  // right_carpus to right_third_metacarpal
            {right_start + 8, right_start + 9},  // right_third_metacarpal to right_third_proximal
            {right_start + 9, right_start + 10},  // right_third_proximal to right_third_middle
            {right_start + 10, right_start + 11},  // right_third_middle to right_third_distal
            
            // Right ring finger
            {right_start, right_start + 12},  // right_carpus to right_fourth_metacarpal
            {right_start + 12, right_start + 13},  // right_fourth_metacarpal to right_fourth_proximal
            {right_start + 13, right_start + 14},  // right_fourth_proximal to right_fourth_middle
            {right_start + 14, right_start + 15},  // right_fourth_middle to right_fourth_distal
            
            // Right pinky finger
            {right_start, right_start + 16},  // right_carpus to right_fifth_metacarpal
            {right_start + 16, right_start + 17},  // right_fifth_metacarpal to right_fifth_proximal
            {right_start + 17, right_start + 18},  // right_fifth_proximal to right_fifth_middle
            {right_start + 18, right_start + 19},  // right_fifth_middle to right_fifth_distal
        };
        
        return finger_hierarchy;
    }
};

#endif //_MVN_MODEL_H_