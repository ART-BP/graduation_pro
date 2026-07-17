#include <memory>

#include <pcl/console/print.h>
#include <ros/ros.h>
#include <rog_map/rog_map.h>

int main(int argc, char** argv) {
  ros::init(argc, argv, "rog_map_node");
  ros::NodeHandle private_node("~");

  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

  auto map = std::make_shared<rog_map::ROGMap>(private_node);
  ROS_INFO("ROG-Map node started.");

  ros::AsyncSpinner spinner(0);
  spinner.start();
  ros::waitForShutdown();

  return 0;
}
