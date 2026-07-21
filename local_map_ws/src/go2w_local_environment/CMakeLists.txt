cmake_minimum_required(VERSION 3.0.2)
project(go2w_local_environment)

add_compile_options(-std=c++14 -Wall -Wextra -Wpedantic)

find_package(catkin REQUIRED COMPONENTS
  grid_map_core
  grid_map_msgs
  grid_map_ros
  geometry_msgs
  message_filters
  nav_msgs
  pcl_conversions
  roscpp
  sensor_msgs
)

find_package(PCL REQUIRED COMPONENTS common)
find_package(Threads REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES elevation_projector
  CATKIN_DEPENDS
    grid_map_core
    grid_map_msgs
    grid_map_ros
    geometry_msgs
    message_filters
    nav_msgs
    pcl_conversions
    roscpp
    sensor_msgs
  DEPENDS PCL
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_library(elevation_projector
  src/elevation_projector.cpp
)

add_executable(local_environment_node
  src/local_environment_node.cpp
)
add_dependencies(local_environment_node
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(local_environment_node
  elevation_projector
  ${grid_map_ros_LIBRARIES}
  ${roscpp_LIBRARIES}
  ${PCL_LIBRARIES}
  Threads::Threads
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest(test_elevation_projector
    test/test_elevation_projector.cpp
  )
  if(TARGET test_elevation_projector)
    target_link_libraries(test_elevation_projector
      elevation_projector
      ${catkin_LIBRARIES}
    )
  endif()
endif()

install(TARGETS elevation_projector local_environment_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(DIRECTORY config launch rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
