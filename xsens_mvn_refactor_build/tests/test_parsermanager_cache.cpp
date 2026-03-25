#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "xsens_mvn_sdk/parsermanager.h"

namespace
{
void appendUint32(std::vector<char>& buffer, uint32_t value)
{
  buffer.push_back(static_cast<char>((value >> 24) & 0xFF));
  buffer.push_back(static_cast<char>((value >> 16) & 0xFF));
  buffer.push_back(static_cast<char>((value >> 8) & 0xFF));
  buffer.push_back(static_cast<char>(value & 0xFF));
}

void appendFloat(std::vector<char>& buffer, float value)
{
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  appendUint32(buffer, bits);
}

std::vector<char> makeHeader(const std::string& message_type, uint8_t count)
{
  std::vector<char> buffer;
  buffer.insert(buffer.end(), message_type.begin(), message_type.end());
  appendUint32(buffer, 1U);
  buffer.push_back(0x80);
  buffer.push_back(count);
  appendUint32(buffer, 0);
  buffer.push_back(0);
  buffer.insert(buffer.end(), 7, '\0');
  return buffer;
}

std::vector<char> makeQuaternionDatagram()
{
  auto buffer = makeHeader("MXTP02", 1);
  appendUint32(buffer, 1U);
  appendFloat(buffer, 1.0f);
  appendFloat(buffer, 2.0f);
  appendFloat(buffer, 3.0f);
  appendFloat(buffer, 0.0f);
  appendFloat(buffer, 0.0f);
  appendFloat(buffer, 0.0f);
  appendFloat(buffer, 1.0f);
  return buffer;
}

std::vector<char> makeJointAnglesDatagram()
{
  auto buffer = makeHeader("MXTP20", 1);
  appendUint32(buffer, 1U << 8);
  appendUint32(buffer, 2U << 8);
  appendFloat(buffer, 10.0f);
  appendFloat(buffer, 20.0f);
  appendFloat(buffer, 30.0f);
  return buffer;
}

std::vector<char> makeComDatagram()
{
  auto buffer = makeHeader("MXTP24", 1);
  appendFloat(buffer, 0.1f);
  appendFloat(buffer, 0.2f);
  appendFloat(buffer, 0.3f);
  return buffer;
}
}  // namespace

TEST(ParserManagerCacheTest, KeepsQuaternionAcrossOtherTypes)
{
  ParserManager parser;
  parser.readDatagram(makeQuaternionDatagram().data());
  ASSERT_NE(parser.getQuaternionDatagram(), nullptr);

  parser.readDatagram(makeComDatagram().data());
  ASSERT_NE(parser.getCenterOfMassDatagram(), nullptr);
  ASSERT_NE(parser.getQuaternionDatagram(), nullptr);
  EXPECT_EQ(parser.getQuaternionDatagram()->getData().size(), 1U);
}

TEST(ParserManagerCacheTest, KeepsJointAnglesAcrossOtherTypes)
{
  ParserManager parser;
  parser.readDatagram(makeJointAnglesDatagram().data());
  ASSERT_NE(parser.getJointAnglesDatagram(), nullptr);

  parser.readDatagram(makeQuaternionDatagram().data());
  ASSERT_NE(parser.getQuaternionDatagram(), nullptr);
  ASSERT_NE(parser.getJointAnglesDatagram(), nullptr);
  EXPECT_FLOAT_EQ(parser.getJointAnglesDatagram()->getItem(1, 2).rotation[0], 10.0f);
}
