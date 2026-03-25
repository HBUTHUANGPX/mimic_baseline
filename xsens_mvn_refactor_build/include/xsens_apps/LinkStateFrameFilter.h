#ifndef XSENS_APPS_LINK_STATE_FRAME_FILTER_H
#define XSENS_APPS_LINK_STATE_FRAME_FILTER_H

#include "xsens_core/XsensFrame.h"

namespace xsens
{
namespace apps
{
void filterOutHandLinks(xsens::core::XsensFrame& frame);
bool isHandRelatedLinkName(const std::string& link_name);
}  // namespace apps
}  // namespace xsens

#endif
