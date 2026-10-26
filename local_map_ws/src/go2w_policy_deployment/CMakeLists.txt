cmake_minimum_required(VERSION 3.0.2)
project(go2w_policy_deployment)

add_compile_options(-std=c++17 -Wall -Wextra -Wpedantic)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  grid_map_core
  grid_map_msgs
  message_filters
  nav_msgs
  roscpp
  std_msgs
  std_srvs
  tf2
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES go2w_observation_builder
  CATKIN_DEPENDS
    geometry_msgs
    grid_map_core
    grid_map_msgs
    message_filters
    nav_msgs
    roscpp
    std_msgs
    std_srvs
    tf2
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(go2w_observation_builder
  src/observation_builder.cpp
)
target_link_libraries(go2w_observation_builder
  ${catkin_LIBRARIES}
)

# ONNX Runtime is intentionally a system/SDK dependency rather than vendored
# into the repository.  Point ONNXRUNTIME_ROOT at the extracted release SDK on
# the robot, or install headers and libonnxruntime in standard system paths.
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to an ONNX Runtime C/C++ SDK")
find_path(ONNXRUNTIME_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  HINTS
    ${ONNXRUNTIME_ROOT}/include
    ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session
)
find_library(ONNXRUNTIME_LIBRARY
  NAMES onnxruntime
  HINTS ${ONNXRUNTIME_ROOT}/lib ${ONNXRUNTIME_ROOT}/lib64
)

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
  add_library(go2w_onnx_policy
    src/onnx_policy.cpp
  )
  target_include_directories(go2w_onnx_policy SYSTEM PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIR}
  )
  target_link_libraries(go2w_onnx_policy
    ${ONNXRUNTIME_LIBRARY}
  )

  add_executable(go2w_policy_node
    src/policy_node.cpp
  )
  add_dependencies(go2w_policy_node
    ${catkin_EXPORTED_TARGETS}
  )
  target_include_directories(go2w_policy_node SYSTEM PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIR}
  )
  target_link_libraries(go2w_policy_node
    go2w_observation_builder
    go2w_onnx_policy
    ${catkin_LIBRARIES}
  )

  install(TARGETS
    go2w_observation_builder
    go2w_onnx_policy
    go2w_policy_node
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )
else()
  message(WARNING
    "ONNX Runtime C++ SDK was not found. go2w_policy_node will not be built. "
    "Set -DONNXRUNTIME_ROOT=/absolute/path/to/onnxruntime and rebuild."
  )
  install(TARGETS go2w_observation_builder
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  )
endif()

if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest(test_observation_builder
    test/test_observation_builder.cpp
  )
  if(TARGET test_observation_builder)
    target_link_libraries(test_observation_builder
      go2w_observation_builder
      ${catkin_LIBRARIES}
    )
  endif()
endif()

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
install(DIRECTORY config launch models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(FILES README.md
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(PROGRAMS scripts/export_policy.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
