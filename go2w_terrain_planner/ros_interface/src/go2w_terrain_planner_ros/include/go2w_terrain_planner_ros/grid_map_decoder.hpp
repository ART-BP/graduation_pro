#pragma once

#include <grid_map_msgs/GridMap.h>

#include <string>

#include "go2w_terrain_planner_ros/policy_observation_builder.hpp"

namespace go2w_terrain_planner_ros {

// Decodes the GridMap circular buffer and resamples its odom-aligned layers
// into the actor convention: row=x forward, column=y left, both increasing
// from -5 m to +5 m in the robot frame.
class GridMapDecoder {
 public:
  bool decodeRobotCentric(
      const grid_map_msgs::GridMap& message,
      const Pose2d& robot_pose,
      FourChannelMap& output,
      std::string& error) const;
};

}  // namespace go2w_terrain_planner_ros
