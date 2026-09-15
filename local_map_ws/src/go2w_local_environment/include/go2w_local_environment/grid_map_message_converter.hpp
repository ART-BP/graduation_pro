#pragma once

#include <string>
#include <vector>

#include <grid_map_core/GridMap.hpp>
#include <grid_map_msgs/GridMap.h>

namespace go2w_local_environment {

// Minimal GridMap-to-ROS conversion used by this package. Keeping the
// conversion local avoids pulling in grid_map_ros and its OpenCV dependency
// while preserving the standard grid_map_msgs interface for RViz.
void toGridMapMessage(
    const grid_map::GridMap& map,
    const std::vector<std::string>& layers,
    grid_map_msgs::GridMap& message);

}  // namespace go2w_local_environment
