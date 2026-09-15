#include "go2w_local_environment/grid_map_message_converter.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

#include <std_msgs/Float32MultiArray.h>

namespace go2w_local_environment {
namespace {

void copyMatrix(
    const grid_map::Matrix& matrix,
    std_msgs::Float32MultiArray& message) {
  message.layout.data_offset = 0U;
  message.layout.dim.resize(2U);
  message.layout.dim[0].label =
      grid_map::Matrix::IsRowMajor ? "row_index" : "column_index";
  message.layout.dim[0].size =
      static_cast<std::uint32_t>(matrix.outerSize());
  message.layout.dim[0].stride =
      static_cast<std::uint32_t>(matrix.size());
  message.layout.dim[1].label =
      grid_map::Matrix::IsRowMajor ? "column_index" : "row_index";
  message.layout.dim[1].size =
      static_cast<std::uint32_t>(matrix.innerSize());
  message.layout.dim[1].stride =
      static_cast<std::uint32_t>(matrix.innerSize());
  message.data.assign(matrix.data(), matrix.data() + matrix.size());
}

std::uint16_t checkedStartIndex(const int index) {
  if (index < 0 ||
      index > static_cast<int>(
          std::numeric_limits<std::uint16_t>::max())) {
    throw std::overflow_error(
        "Grid-map circular-buffer index does not fit uint16");
  }
  return static_cast<std::uint16_t>(index);
}

}  // namespace

void toGridMapMessage(
    const grid_map::GridMap& map,
    const std::vector<std::string>& layers,
    grid_map_msgs::GridMap& message) {
  message.info.header.stamp.fromNSec(map.getTimestamp());
  message.info.header.frame_id = map.getFrameId();
  message.info.resolution = map.getResolution();
  message.info.length_x = map.getLength().x();
  message.info.length_y = map.getLength().y();
  message.info.pose.position.x = map.getPosition().x();
  message.info.pose.position.y = map.getPosition().y();
  message.info.pose.position.z = 0.0;
  message.info.pose.orientation.x = 0.0;
  message.info.pose.orientation.y = 0.0;
  message.info.pose.orientation.z = 0.0;
  message.info.pose.orientation.w = 1.0;

  message.layers = layers;
  message.basic_layers = map.getBasicLayers();
  message.data.clear();
  message.data.reserve(layers.size());
  for (const std::string& layer : layers) {
    std_msgs::Float32MultiArray data;
    copyMatrix(map.get(layer), data);
    message.data.push_back(std::move(data));
  }

  message.outer_start_index =
      checkedStartIndex(map.getStartIndex()(0));
  message.inner_start_index =
      checkedStartIndex(map.getStartIndex()(1));
}

}  // namespace go2w_local_environment
