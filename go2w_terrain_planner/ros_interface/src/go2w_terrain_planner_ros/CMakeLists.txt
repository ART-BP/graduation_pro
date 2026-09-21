cmake_minimum_required(VERSION 3.10)
project(go2w_terrain_planner_ros LANGUAGES CXX)

add_compile_options(-Wall -Wextra -Wpedantic)

find_package(catkin REQUIRED COMPONENTS
  diagnostic_msgs
  geometry_msgs
  grid_map_msgs
  message_filters
  nav_msgs
  roscpp
  std_msgs
  visualization_msgs
)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  PATHS /usr/include/aarch64-linux-gnu /usr/include/x86_64-linux-gnu /usr/include)
find_path(CUDA_INCLUDE_DIR cuda_runtime_api.h
  PATHS /usr/local/cuda/include /usr/include)
find_library(NVINFER_LIBRARY nvinfer
  PATHS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/lib)
find_library(NVINFER_PLUGIN_LIBRARY nvinfer_plugin
  PATHS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/lib)
find_library(NVONNXPARSER_LIBRARY nvonnxparser
  PATHS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/lib)
find_library(CUDART_LIBRARY cudart
  PATHS
    /usr/local/cuda/lib64
    /usr/local/cuda/targets/aarch64-linux/lib
    /usr/local/cuda/targets/x86_64-linux/lib
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT CUDA_INCLUDE_DIR OR
   NOT NVINFER_LIBRARY OR NOT NVINFER_PLUGIN_LIBRARY OR
   NOT NVONNXPARSER_LIBRARY OR NOT CUDART_LIBRARY)
  message(FATAL_ERROR
    "TensorRT/CUDA development files were not found. Install libnvinfer-dev, "
    "libnvonnxparsers-dev, and the CUDA toolkit.")
endif()

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES go2w_policy_core go2w_tensorrt
  CATKIN_DEPENDS
    diagnostic_msgs geometry_msgs grid_map_msgs message_filters nav_msgs
    roscpp std_msgs visualization_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIR}
)

add_library(go2w_policy_core
  src/policy_observation_builder.cpp
  src/grid_map_decoder.cpp
)
target_compile_features(go2w_policy_core PUBLIC cxx_std_14)
target_link_libraries(go2w_policy_core ${catkin_LIBRARIES})

add_library(go2w_tensorrt src/tensorrt_engine.cpp)
target_compile_features(go2w_tensorrt PUBLIC cxx_std_14)
target_link_libraries(go2w_tensorrt
  ${NVINFER_LIBRARY}
  ${NVINFER_PLUGIN_LIBRARY}
  ${NVONNXPARSER_LIBRARY}
  ${CUDART_LIBRARY}
)

add_executable(go2w_policy_inference_node src/inference_node.cpp)
target_compile_features(go2w_policy_inference_node PUBLIC cxx_std_14)
target_link_libraries(go2w_policy_inference_node
  go2w_policy_core
  go2w_tensorrt
  ${catkin_LIBRARIES}
)

install(TARGETS go2w_policy_core go2w_tensorrt go2w_policy_inference_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest(test_policy_observation_builder
    test/test_policy_observation_builder.cpp)
  if(TARGET test_policy_observation_builder)
    target_compile_features(test_policy_observation_builder PUBLIC cxx_std_14)
    target_link_libraries(test_policy_observation_builder go2w_policy_core)
  endif()
endif()
