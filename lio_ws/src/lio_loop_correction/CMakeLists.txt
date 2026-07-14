cmake_minimum_required(VERSION 3.0.2)
project(lio_loop_correction)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  pcl_conversions
  pcl_ros
  roscpp
  sensor_msgs
  std_msgs
  tf
  visualization_msgs
)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED COMPONENTS common filters registration)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_executable(lio_loop_correction_node src/lio_loop_correction_node.cpp)
target_link_libraries(lio_loop_correction_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)
