#include "xsens_apps/LinkStateFrameFilter.h"

#include <algorithm>
#include <array>

namespace xsens
{
namespace apps
{
namespace
{
bool startsWith(const std::string& value, const std::string& prefix)
{
  return value.rfind(prefix, 0) == 0;
}
}  // namespace

bool isHandRelatedLinkName(const std::string& link_name)
{
  if (link_name == "left_carpus" || link_name == "right_carpus")
  {
    return true;
  }

  if (startsWith(link_name, "prop"))
  {
    return true;
  }

  static const std::array<const char*, 10> prefixes = {
    "left_first_",
    "left_second_",
    "left_third_",
    "left_fourth_",
    "left_fifth_",
    "right_first_",
    "right_second_",
    "right_third_",
    "right_fourth_",
    "right_fifth_",
  };

  for (const char* prefix : prefixes)
  {
    if (startsWith(link_name, prefix))
    {
      return true;
    }
  }

  return false;
}

void filterOutHandLinks(xsens::core::XsensFrame& frame)
{
  frame.links.erase(
    std::remove_if(
      frame.links.begin(),
      frame.links.end(),
      [](const xsens::core::LinkSample& link_sample) {
        return isHandRelatedLinkName(link_sample.name);
      }),
    frame.links.end());
}
}  // namespace apps
}  // namespace xsens
