cmake_minimum_required(VERSION 3.0.2)
project(hesai_fastlio_adapter)

add_compile_options(-std=c++14 -O3)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS roscpp sensor_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(hesai_fastlio_adapter_node
  src/hesai_fastlio_adapter_node.cpp
)

target_link_libraries(hesai_fastlio_adapter_node
  ${catkin_LIBRARIES}
)
