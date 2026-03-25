#include <gtest/gtest.h>

#include "xsens_apps/LinkStateFrameFilter.h"

TEST(LinkStateFrameFilterTest, RemovesHandAndFingerLinksButKeepsBodyLinks)
{
  xsens::core::XsensFrame frame;

  xsens::core::LinkSample pelvis;
  pelvis.name = "pelvis";
  frame.links.push_back(pelvis);

  xsens::core::LinkSample left_hand;
  left_hand.name = "left_hand";
  frame.links.push_back(left_hand);

  xsens::core::LinkSample right_hand;
  right_hand.name = "right_hand";
  frame.links.push_back(right_hand);

  xsens::core::LinkSample left_carpus;
  left_carpus.name = "left_carpus";
  frame.links.push_back(left_carpus);

  xsens::core::LinkSample left_thumb;
  left_thumb.name = "left_first_distal";
  frame.links.push_back(left_thumb);

  xsens::core::LinkSample right_finger;
  right_finger.name = "right_third_middle";
  frame.links.push_back(right_finger);

  xsens::core::LinkSample right_upper_arm;
  right_upper_arm.name = "right_upper_arm";
  frame.links.push_back(right_upper_arm);

  xsens::core::LinkSample prop2;
  prop2.name = "prop2";
  frame.links.push_back(prop2);

  xsens::apps::filterOutHandLinks(frame);

  ASSERT_EQ(frame.links.size(), 4U);
  EXPECT_EQ(frame.links[0].name, "pelvis");
  EXPECT_EQ(frame.links[1].name, "left_hand");
  EXPECT_EQ(frame.links[2].name, "right_hand");
  EXPECT_EQ(frame.links[3].name, "right_upper_arm");
}
