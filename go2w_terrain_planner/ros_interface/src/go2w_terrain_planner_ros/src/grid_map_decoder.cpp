#include "go2w_terrain_planner_ros/grid_map_decoder.hpp"

#include <std_msgs/Float32MultiArray.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <unordered_map>

namespace go2w_terrain_planner_ros {
namespace {

float clamp(const float value, const float lower, const float upper) {
  return std::max(lower, std::min(value, upper));
}

float quaternionYaw(const geometry_msgs::Quaternion& quaternion) {
  const double numerator = 2.0 *
      (quaternion.w * quaternion.z + quaternion.x * quaternion.y);
  const double denominator = 1.0 - 2.0 *
      (quaternion.y * quaternion.y + quaternion.z * quaternion.z);
  return static_cast<float>(std::atan2(numerator, denominator));
}

class LayerView {
 public:
  bool initialize(
      const std_msgs::Float32MultiArray* message,
      const int outer_start,
      const int inner_start,
      std::string& error) {
    message_ = message;
    outer_start_ = outer_start;
    inner_start_ = inner_start;
    if (message_ == nullptr || message_->layout.dim.size() != 2U) {
      error = "GridMap layer must have a two-dimensional layout";
      return false;
    }
    const auto& first = message_->layout.dim[0];
    const auto& second = message_->layout.dim[1];
    if (first.label == "row_index" && second.label == "column_index") {
      row_major_ = true;
      rows_ = static_cast<int>(first.size);
      columns_ = static_cast<int>(second.size);
    } else if (first.label == "column_index" && second.label == "row_index") {
      row_major_ = false;
      columns_ = static_cast<int>(first.size);
      rows_ = static_cast<int>(second.size);
    } else {
      error = "Unsupported GridMap layer layout labels: " + first.label +
              ", " + second.label;
      return false;
    }
    const std::size_t required = static_cast<std::size_t>(rows_) * columns_;
    if (rows_ <= 0 || columns_ <= 0 ||
        message_->layout.data_offset + required > message_->data.size()) {
      error = "GridMap layer dimensions do not match its data array";
      return false;
    }
    if (outer_start_ < 0 || outer_start_ >= rows_ ||
        inner_start_ < 0 || inner_start_ >= columns_) {
      error = "GridMap circular-buffer start index is out of range";
      return false;
    }
    return true;
  }

  int rows() const { return rows_; }
  int columns() const { return columns_; }

  float value(const int logical_row, const int logical_column) const {
    if (logical_row < 0 || logical_row >= rows_ ||
        logical_column < 0 || logical_column >= columns_) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    const int buffer_row = (logical_row + outer_start_) % rows_;
    const int buffer_column = (logical_column + inner_start_) % columns_;
    std::size_t linear = 0U;
    if (row_major_) {
      linear = static_cast<std::size_t>(buffer_row) * columns_ + buffer_column;
    } else {
      linear = static_cast<std::size_t>(buffer_column) * rows_ + buffer_row;
    }
    return message_->data[message_->layout.data_offset + linear];
  }

 private:
  const std_msgs::Float32MultiArray* message_{nullptr};
  int rows_{0};
  int columns_{0};
  int outer_start_{0};
  int inner_start_{0};
  bool row_major_{false};
};

const std_msgs::Float32MultiArray* findLayer(
    const grid_map_msgs::GridMap& message,
    const std::string& name) {
  for (std::size_t index = 0; index < message.layers.size(); ++index) {
    if (message.layers[index] == name && index < message.data.size()) {
      return &message.data[index];
    }
  }
  return nullptr;
}

}  // namespace

bool GridMapDecoder::decodeRobotCentric(
    const grid_map_msgs::GridMap& message,
    const Pose2d& robot_pose,
    FourChannelMap& output,
    std::string& error) const {
  constexpr double kExpectedResolution = 0.05;
  constexpr double kResolutionTolerance = 1.0e-5;
  if (std::abs(message.info.resolution - kExpectedResolution) >
      kResolutionTolerance) {
    std::ostringstream stream;
    stream << "Expected a 0.05 m GridMap, got " << message.info.resolution;
    error = stream.str();
    return false;
  }
  if (message.info.length_x + 1.0e-6 < 10.0 ||
      message.info.length_y + 1.0e-6 < 10.0) {
    error = "GridMap must cover at least 10 m x 10 m";
    return false;
  }

  const auto* ground_message = findLayer(message, "ground_height");
  const auto* range_message = findLayer(message, "height_range");
  const auto* observed_message = findLayer(message, "observed_mask");
  if (ground_message == nullptr || range_message == nullptr ||
      observed_message == nullptr) {
    error = "GridMap is missing ground_height, height_range, or observed_mask";
    return false;
  }

  LayerView ground;
  LayerView height_range;
  LayerView observed;
  const int outer_start = static_cast<int>(message.outer_start_index);
  const int inner_start = static_cast<int>(message.inner_start_index);
  if (!ground.initialize(ground_message, outer_start, inner_start, error) ||
      !height_range.initialize(range_message, outer_start, inner_start, error) ||
      !observed.initialize(observed_message, outer_start, inner_start, error)) {
    return false;
  }
  if (ground.rows() != height_range.rows() ||
      ground.columns() != height_range.columns() ||
      ground.rows() != observed.rows() ||
      ground.columns() != observed.columns()) {
    error = "GridMap layers have inconsistent dimensions";
    return false;
  }

  const float resolution = static_cast<float>(message.info.resolution);
  const float map_center_x =
      static_cast<float>(message.info.pose.position.x);
  const float map_center_y =
      static_cast<float>(message.info.pose.position.y);
  const float map_yaw = quaternionYaw(message.info.pose.orientation);
  const float map_cosine = std::cos(map_yaw);
  const float map_sine = std::sin(map_yaw);
  const float robot_cosine = std::cos(robot_pose.yaw);
  const float robot_sine = std::sin(robot_pose.yaw);
  const float map_half_x =
      0.5F * static_cast<float>(message.info.length_x);
  const float map_half_y =
      0.5F * static_cast<float>(message.info.length_y);
  const float target_half_extent = 5.0F;

  auto validAt = [&](const int row, const int column) {
    return std::isfinite(ground.value(row, column)) &&
           std::isfinite(height_range.value(row, column));
  };
  auto normalizedGroundAt = [&](const int row, const int column) {
    return clamp(ground.value(row, column), -1.0F, 1.0F);
  };

  output = FourChannelMap();
  for (int target_row = 0; target_row < kMapSize; ++target_row) {
    const float local_x = -target_half_extent +
        (static_cast<float>(target_row) + 0.5F) * kExpectedResolution;
    for (int target_column = 0; target_column < kMapSize;
         ++target_column) {
      const float local_y = -target_half_extent +
          (static_cast<float>(target_column) + 0.5F) * kExpectedResolution;
      const float world_x = robot_pose.x + robot_cosine * local_x -
                            robot_sine * local_y;
      const float world_y = robot_pose.y + robot_sine * local_x +
                            robot_cosine * local_y;
      const float delta_x = world_x - map_center_x;
      const float delta_y = world_y - map_center_y;
      const float map_x = map_cosine * delta_x + map_sine * delta_y;
      const float map_y = -map_sine * delta_x + map_cosine * delta_y;
      const float source_row =
          (map_half_x - 0.5F * resolution - map_x) / resolution;
      const float source_column =
          (map_half_y - 0.5F * resolution - map_y) / resolution;
      const int row0 = static_cast<int>(std::floor(source_row));
      const int column0 = static_cast<int>(std::floor(source_column));
      const float row_fraction = source_row - static_cast<float>(row0);
      const float column_fraction =
          source_column - static_cast<float>(column0);
      float weighted_ground = 0.0F;
      float validity_weight = 0.0F;
      for (int row_offset = 0; row_offset <= 1; ++row_offset) {
        const int sample_row = row0 + row_offset;
        const float row_weight = row_offset == 0 ? 1.0F - row_fraction
                                                  : row_fraction;
        for (int column_offset = 0; column_offset <= 1; ++column_offset) {
          const int sample_column = column0 + column_offset;
          if (!validAt(sample_row, sample_column)) {
            continue;
          }
          const float column_weight = column_offset == 0
              ? 1.0F - column_fraction : column_fraction;
          const float weight = row_weight * column_weight;
          weighted_ground +=
              weight * normalizedGroundAt(sample_row, sample_column);
          validity_weight += weight;
        }
      }

      const int nearest_row = static_cast<int>(std::round(source_row));
      const int nearest_column = static_cast<int>(std::round(source_column));
      const std::size_t output_index =
          static_cast<std::size_t>(target_row) * kMapSize + target_column;
      const float observed_value = observed.value(nearest_row, nearest_column);
      output.channel[2][output_index] = std::isfinite(observed_value)
          ? clamp(observed_value, 0.0F, 1.0F) : 0.0F;
      if (!validAt(nearest_row, nearest_column)) {
        continue;
      }
      if (validity_weight > 1.0e-6F) {
        output.channel[0][output_index] =
            weighted_ground / validity_weight;
      } else {
        output.channel[0][output_index] =
            normalizedGroundAt(nearest_row, nearest_column);
      }
      output.channel[1][output_index] = clamp(
          height_range.value(nearest_row, nearest_column), 0.0F, 3.0F) /
          3.0F;
      output.channel[3][output_index] = 1.0F;
    }
  }
  return true;
}

}  // namespace go2w_terrain_planner_ros
