cmake_minimum_required(VERSION 3.0.2)
project(go2w_local_environment)

add_compile_options(-std=c++14 -Wall -Wextra -Wpedantic)

find_package(catkin REQUIRED COMPONENTS
  grid_map_core
  grid_map_msgs
  grid_map_ros
  nav_msgs
  pcl_conversions
  pcl_ros
  rog_map
  roscpp
  sensor_msgs
)

find_package(PCL REQUIRED COMPONENTS common)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES dual_layer_projector
  CATKIN_DEPENDS
    grid_map_core
    grid_map_msgs
    grid_map_ros
    nav_msgs
    pcl_conversions
    pcl_ros
    rog_map
    roscpp
    sensor_msgs
  DEPENDS PCL
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_library(dual_layer_projector
  src/dual_layer_projector.cpp
)

add_executable(local_environment_node
  src/local_environment_node.cpp
)
add_dependencies(local_environment_node
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(local_environment_node
  dual_layer_projector
  ${grid_map_ros_LIBRARIES}
  ${roscpp_LIBRARIES}
  ${PCL_LIBRARIES}
)

add_executable(rog_map_node
  src/rog_map_node.cpp
)
add_dependencies(rog_map_node
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(rog_map_node
  ${rog_map_LIBRARIES}
  ${roscpp_LIBRARIES}
  ${PCL_LIBRARIES}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest(test_dual_layer_projector
    test/test_dual_layer_projector.cpp
  )
  if(TARGET test_dual_layer_projector)
    target_link_libraries(test_dual_layer_projector
      dual_layer_projector
      ${catkin_LIBRARIES}
    )
  endif()
endif()

install(TARGETS dual_layer_projector local_environment_node rog_map_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
