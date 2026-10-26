#include "go2w_policy_deployment/observation_builder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace go2w_policy_deployment {
namespace {

constexpr int kRawMapChannels = 4;
constexpr double kPi = 3.14159265358979323846;

double wrapAngle(const double value) {
  return std::atan2(std::sin(value), std::cos(value));
}

float clampFloat(const double value, const double minimum, const double maximum) {
  return static_cast<float>(std::max(minimum, std::min(maximum, value)));
}

bool nearlyEqual(const double first, const double second) {
  return std::abs(first - second) <= 1.0e-9;
}

void requireFinite(const float value, const char* name) {
  if (!std::isfinite(value)) {
    throw std::runtime_error(std::string(name) + " contains NaN or Inf");
  }
}

}  // namespace

PolicyMap::PolicyMap(const int map_size)
    : size(map_size),
      data(static_cast<std::size_t>(kRawMapChannels) * map_size * map_size, 0.0F) {
  if (map_size <= 1) {
    throw std::invalid_argument("PolicyMap size must be greater than one");
  }
}

float& PolicyMap::at(const int channel, const int row, const int column) {
  return data.at(
      (static_cast<std::size_t>(channel) * size + row) * size + column);
}

float PolicyMap::at(const int channel, const int row, const int column) const {
  return data.at(
      (static_cast<std::size_t>(channel) * size + row) * size + column);
}

std::size_t PolicyMap::cells() const {
  return static_cast<std::size_t>(size) * size;
}

ObservationBuilder::ObservationBuilder(ObservationConfig config)
    : config_(std::move(config)) {
  if (config_.map_size <= 1 || config_.map_channels != 7 ||
      config_.observation_fusion_length < 2 ||
      config_.command_history_length <= 0 ||
      config_.motion_history_length != config_.command_history_length ||
      config_.map_extent_m <= 0.0 || config_.ground_height_scale_m <= 0.0 ||
      config_.height_range_scale_m <= 0.0 ||
      config_.maximum_ground_deviation <= 0.0 ||
      config_.fusion_span_percentile < 0.0 ||
      config_.fusion_span_percentile > 1.0 ||
      config_.goal_maximum_distance_m <= 0.0 ||
      config_.linear_velocity_scale_mps <= 0.0 ||
      config_.angular_velocity_scale_radps <= 0.0) {
    throw std::invalid_argument("invalid policy observation configuration");
  }
}

void ObservationBuilder::reset() {
  observation_history_.clear();
  pose_history_.clear();
  command_history_.clear();
  previous_fused_ = MapFrame{};
  has_previous_fused_ = false;
}

void ObservationBuilder::recordCommand(
    const double linear_mps, const double angular_radps) {
  if (!std::isfinite(linear_mps) || !std::isfinite(angular_radps)) {
    throw std::invalid_argument("command history only accepts finite values");
  }
  if (command_history_.empty()) {
    command_history_.assign(
        static_cast<std::size_t>(config_.command_history_length),
        std::array<double, 2>{{0.0, 0.0}});
  }
  command_history_.pop_front();
  command_history_.push_back({{linear_mps, angular_radps}});
}

PolicyMap ObservationBuilder::alignFrame(
    const MapFrame& source,
    const Pose2d& target_pose,
    const double target_ground_reference_z) const {
  if (source.map.size != config_.map_size) {
    throw std::invalid_argument("map history size does not match configuration");
  }

  const int size = config_.map_size;
  const double half_extent = 0.5 * config_.map_extent_m;
  const double grid_step = config_.map_extent_m / static_cast<double>(size - 1);
  const double target_cosine = std::cos(target_pose.yaw);
  const double target_sine = std::sin(target_pose.yaw);
  const double source_cosine = std::cos(source.pose.yaw);
  const double source_sine = std::sin(source.pose.yaw);
  const float vertical_offset = static_cast<float>(
      (source.ground_reference_z - target_ground_reference_z) /
      config_.ground_height_scale_m);
  const bool horizontal_identity =
      nearlyEqual(source.pose.x, target_pose.x) &&
      nearlyEqual(source.pose.y, target_pose.y) &&
      std::abs(wrapAngle(source.pose.yaw - target_pose.yaw)) <= 1.0e-9;

  PolicyMap result(size);
  if (horizontal_identity) {
    result = source.map;
    if (vertical_offset != 0.0F) {
      for (int row = 0; row < size; ++row) {
        for (int column = 0; column < size; ++column) {
          if (result.at(3, row, column) > 0.5F) {
            result.at(0, row, column) = clampFloat(
                result.at(0, row, column) + vertical_offset, -1.0, 1.0);
          }
        }
      }
    }
    return result;
  }

  for (int row = 0; row < size; ++row) {
    const double target_x = -half_extent + row * grid_step;
    for (int column = 0; column < size; ++column) {
      const double target_y = -half_extent + column * grid_step;
      const double world_x =
          target_pose.x + target_cosine * target_x - target_sine * target_y;
      const double world_y =
          target_pose.y + target_sine * target_x + target_cosine * target_y;
      const double delta_x = world_x - source.pose.x;
      const double delta_y = world_y - source.pose.y;
      const double source_x = source_cosine * delta_x + source_sine * delta_y;
      const double source_y = -source_sine * delta_x + source_cosine * delta_y;

      if (source_x < -half_extent || source_x > half_extent ||
          source_y < -half_extent || source_y > half_extent) {
        continue;
      }

      const double source_row =
          (source_x + half_extent) / config_.map_extent_m * (size - 1);
      const double source_column =
          (source_y + half_extent) / config_.map_extent_m * (size - 1);
      const int row0 = std::max(0, std::min(size - 1,
          static_cast<int>(std::floor(source_row))));
      const int column0 = std::max(0, std::min(size - 1,
          static_cast<int>(std::floor(source_column))));
      const int row1 = std::min(size - 1, row0 + 1);
      const int column1 = std::min(size - 1, column0 + 1);
      const double row_fraction = source_row - row0;
      const double column_fraction = source_column - column0;
      const std::array<int, 4> rows{{row0, row0, row1, row1}};
      const std::array<int, 4> columns{{column0, column1, column0, column1}};
      const std::array<double, 4> weights{{
          (1.0 - row_fraction) * (1.0 - column_fraction),
          (1.0 - row_fraction) * column_fraction,
          row_fraction * (1.0 - column_fraction),
          row_fraction * column_fraction,
      }};

      double weighted_ground = 0.0;
      double valid_weight = 0.0;
      for (std::size_t sample = 0; sample < weights.size(); ++sample) {
        const double valid = source.map.at(3, rows[sample], columns[sample]);
        weighted_ground += weights[sample] * valid *
                           source.map.at(0, rows[sample], columns[sample]);
        valid_weight += weights[sample] * valid;
      }
      if (valid_weight > 1.0e-6) {
        result.at(0, row, column) = static_cast<float>(weighted_ground / valid_weight);
      }

      const int nearest_row = std::max(0, std::min(
          size - 1, static_cast<int>(std::floor(source_row + 0.5))));
      const int nearest_column = std::max(0, std::min(
          size - 1, static_cast<int>(std::floor(source_column + 0.5))));
      result.at(2, row, column) = source.map.at(2, nearest_row, nearest_column);
      result.at(3, row, column) = source.map.at(3, nearest_row, nearest_column);
      if (result.at(3, row, column) > 0.5F) {
        result.at(0, row, column) = clampFloat(
            result.at(0, row, column) + vertical_offset, -1.0, 1.0);
        result.at(1, row, column) =
            source.map.at(1, nearest_row, nearest_column);
      }
    }
  }
  return result;
}

PolicyMap ObservationBuilder::fuseHistory(
    const std::vector<PolicyMap>& aligned) const {
  if (aligned.empty()) {
    throw std::invalid_argument("cannot fuse an empty map history");
  }
  const int size = config_.map_size;
  PolicyMap fused(size);
  std::vector<float> ground_samples;
  std::vector<float> range_samples;
  ground_samples.reserve(aligned.size());
  range_samples.reserve(aligned.size());

  for (int row = 0; row < size; ++row) {
    for (int column = 0; column < size; ++column) {
      int selected_anchor = static_cast<int>(aligned.size()) - 1;
      int selected_score = -1;
      float observed = 0.0F;
      for (std::size_t candidate = 0; candidate < aligned.size(); ++candidate) {
        observed = std::max(observed, aligned[candidate].at(2, row, column));
        int count = 0;
        if (aligned[candidate].at(3, row, column) > 0.5F) {
          const float anchor = aligned[candidate].at(0, row, column);
          for (const PolicyMap& map : aligned) {
            if (map.at(3, row, column) > 0.5F &&
                std::abs(map.at(0, row, column) - anchor) <=
                    config_.maximum_ground_deviation) {
              ++count;
            }
          }
        }
        const int score = count * (static_cast<int>(aligned.size()) + 1) +
                          static_cast<int>(candidate);
        if (score > selected_score) {
          selected_score = score;
          selected_anchor = static_cast<int>(candidate);
        }
      }

      ground_samples.clear();
      range_samples.clear();
      const float selected_ground =
          aligned[static_cast<std::size_t>(selected_anchor)].at(0, row, column);
      for (const PolicyMap& map : aligned) {
        if (map.at(3, row, column) > 0.5F &&
            std::abs(map.at(0, row, column) - selected_ground) <=
                config_.maximum_ground_deviation) {
          ground_samples.push_back(map.at(0, row, column));
          range_samples.push_back(map.at(1, row, column));
        }
      }
      fused.at(2, row, column) = clampFloat(observed, 0.0, 1.0);
      if (ground_samples.empty()) {
        continue;
      }
      std::sort(ground_samples.begin(), ground_samples.end());
      std::sort(range_samples.begin(), range_samples.end());
      const std::size_t ground_rank = (ground_samples.size() - 1U) / 2U;
      const std::size_t range_rank = static_cast<std::size_t>(std::ceil(
          config_.fusion_span_percentile * (range_samples.size() - 1U)));
      fused.at(0, row, column) = ground_samples[ground_rank];
      fused.at(1, row, column) = range_samples[range_rank];
      fused.at(3, row, column) = 1.0F;
    }
  }
  return fused;
}

PolicyMap ObservationBuilder::buildCompactMap(
    const PolicyMap& fused,
    const PolicyMap& previous_aligned,
    const std::vector<PolicyMap>& aligned_observations) const {
  const int size = config_.map_size;
  const std::size_t cells = fused.cells();
  PolicyMap compact(size);
  // PolicyMap owns four channels; allocate the full seven-channel tensor here.
  compact.data.assign(static_cast<std::size_t>(config_.map_channels) * cells, 0.0F);
  for (int channel = 0; channel < kRawMapChannels; ++channel) {
    for (std::size_t cell = 0; cell < cells; ++cell) {
      compact.data[static_cast<std::size_t>(channel) * cells + cell] =
          fused.data[static_cast<std::size_t>(channel) * cells + cell];
    }
  }

  double recency_weight_sum = 0.0;
  std::vector<double> recency_weights(aligned_observations.size(), 1.0);
  for (std::size_t index = 0; index < aligned_observations.size(); ++index) {
    const double fraction = aligned_observations.size() == 1U
        ? 1.0
        : static_cast<double>(index) /
              static_cast<double>(aligned_observations.size() - 1U);
    recency_weights[index] = 0.25 + 0.75 * fraction;
    recency_weight_sum += recency_weights[index];
  }

  for (int row = 0; row < size; ++row) {
    for (int column = 0; column < size; ++column) {
      const std::size_t cell = static_cast<std::size_t>(row) * size + column;
      const bool current_valid = fused.at(3, row, column) > 0.5F;
      const bool previous_valid = previous_aligned.at(3, row, column) > 0.5F;
      float geometric_change = 0.0F;
      if (current_valid && previous_valid) {
        const float ground_change = std::min(
            1.0F, 0.5F * std::abs(
                fused.at(0, row, column) - previous_aligned.at(0, row, column)));
        const float range_change = std::min(
            1.0F, std::abs(
                fused.at(1, row, column) - previous_aligned.at(1, row, column)));
        geometric_change = std::max(ground_change, range_change);
      }
      const float mask_change = std::max(
          std::abs(fused.at(2, row, column) -
                   previous_aligned.at(2, row, column)),
          std::abs(fused.at(3, row, column) -
                   previous_aligned.at(3, row, column)));
      compact.data[4U * cells + cell] =
          clampFloat(std::max(geometric_change, mask_change), 0.0, 1.0);

      int latest_observed = -1;
      double confidence = 0.0;
      for (std::size_t index = 0; index < aligned_observations.size(); ++index) {
        const float observed = clampFloat(
            aligned_observations[index].at(2, row, column), 0.0, 1.0);
        if (observed > 0.5F) {
          latest_observed = static_cast<int>(index);
        }
        confidence += observed * recency_weights[index];
      }
      const double denominator = static_cast<double>(
          aligned_observations.size() - 1U);
      const double age = denominator > 0.0
          ? (static_cast<double>(aligned_observations.size() - 1U) -
             latest_observed) / denominator
          : 0.0;
      compact.data[5U * cells + cell] = clampFloat(age, 0.0, 1.0);
      compact.data[6U * cells + cell] = clampFloat(
          confidence / recency_weight_sum, 0.0, 1.0);
    }
  }
  return compact;
}

void ObservationBuilder::pushPose(const Pose2d& pose) {
  const std::size_t required =
      static_cast<std::size_t>(config_.motion_history_length + 1);
  if (pose_history_.empty()) {
    pose_history_.assign(required, pose);
    return;
  }
  pose_history_.pop_front();
  pose_history_.push_back(pose);
}

ObservationResult ObservationBuilder::build(
    const PolicyMap& current_map,
    const Pose2d& current_pose,
    const double current_ground_reference_z,
    const double goal_x,
    const double goal_y,
    const double measured_linear_velocity_mps,
    const double measured_angular_velocity_radps) {
  if (current_map.size != config_.map_size ||
      current_map.data.size() !=
          static_cast<std::size_t>(kRawMapChannels) * current_map.cells()) {
    throw std::invalid_argument("current policy map has the wrong shape");
  }
  for (const float value : current_map.data) {
    requireFinite(value, "current policy map");
  }
  const std::array<double, 8> scalars{{
      current_pose.x, current_pose.y, current_pose.yaw,
      current_ground_reference_z, goal_x, goal_y,
      measured_linear_velocity_mps, measured_angular_velocity_radps,
  }};
  if (!std::all_of(scalars.begin(), scalars.end(),
                   [](const double value) { return std::isfinite(value); })) {
    throw std::invalid_argument("policy observation inputs must be finite");
  }

  const MapFrame current_frame{
      current_map, current_pose, current_ground_reference_z};
  if (observation_history_.empty()) {
    observation_history_.assign(
        static_cast<std::size_t>(config_.observation_fusion_length),
        current_frame);
  } else {
    observation_history_.pop_front();
    observation_history_.push_back(current_frame);
  }
  pushPose(current_pose);
  if (command_history_.empty()) {
    command_history_.assign(
        static_cast<std::size_t>(config_.command_history_length),
        std::array<double, 2>{{0.0, 0.0}});
  }

  std::vector<PolicyMap> aligned;
  aligned.reserve(observation_history_.size());
  for (const MapFrame& frame : observation_history_) {
    aligned.push_back(alignFrame(
        frame, current_pose, current_ground_reference_z));
  }
  PolicyMap fused = fuseHistory(aligned);
  PolicyMap previous_aligned = has_previous_fused_
      ? alignFrame(previous_fused_, current_pose, current_ground_reference_z)
      : fused;
  PolicyMap compact = buildCompactMap(fused, previous_aligned, aligned);

  ObservationResult result;
  result.values.reserve(expectedDimension());
  result.values.insert(
      result.values.end(), compact.data.begin(), compact.data.end());

  const double goal_delta_x = goal_x - current_pose.x;
  const double goal_delta_y = goal_y - current_pose.y;
  const double goal_distance = std::hypot(goal_delta_x, goal_delta_y);
  const double goal_bearing = wrapAngle(
      std::atan2(goal_delta_y, goal_delta_x) - current_pose.yaw);
  result.values.push_back(clampFloat(
      goal_distance / config_.goal_maximum_distance_m, 0.0, 1.0));
  result.values.push_back(static_cast<float>(std::sin(goal_bearing)));
  result.values.push_back(static_cast<float>(std::cos(goal_bearing)));
  result.values.push_back(static_cast<float>(
      measured_linear_velocity_mps / config_.linear_velocity_scale_mps));
  result.values.push_back(static_cast<float>(
      measured_angular_velocity_radps / config_.angular_velocity_scale_radps));

  for (const auto& command : command_history_) {
    result.values.push_back(static_cast<float>(
        command[0] / config_.linear_velocity_scale_mps));
    result.values.push_back(static_cast<float>(
        command[1] / config_.angular_velocity_scale_radps));
  }
  const double half_extent = 0.5 * config_.map_extent_m;
  for (std::size_t index = 1; index < pose_history_.size(); ++index) {
    const Pose2d& previous = pose_history_[index - 1U];
    const Pose2d& current = pose_history_[index];
    const double world_x = current.x - previous.x;
    const double world_y = current.y - previous.y;
    const double cosine = std::cos(previous.yaw);
    const double sine = std::sin(previous.yaw);
    result.values.push_back(static_cast<float>(
        (cosine * world_x + sine * world_y) / half_extent));
    result.values.push_back(static_cast<float>(
        (-sine * world_x + cosine * world_y) / half_extent));
    result.values.push_back(static_cast<float>(
        wrapAngle(current.yaw - previous.yaw) / kPi));
  }

  if (result.values.size() != expectedDimension()) {
    throw std::runtime_error("assembled policy observation has the wrong dimension");
  }
  for (const float value : result.values) {
    requireFinite(value, "assembled policy observation");
  }

  double observed_sum = 0.0;
  double valid_sum = 0.0;
  for (int row = 0; row < config_.map_size; ++row) {
    for (int column = 0; column < config_.map_size; ++column) {
      observed_sum += fused.at(2, row, column);
      valid_sum += fused.at(3, row, column);
    }
  }
  const double cell_count = static_cast<double>(fused.cells());
  result.observed_ratio = observed_sum / cell_count;
  result.height_valid_ratio = valid_sum / cell_count;

  previous_fused_ = MapFrame{
      std::move(fused), current_pose, current_ground_reference_z};
  has_previous_fused_ = true;
  return result;
}

std::size_t ObservationBuilder::expectedDimension() const {
  return static_cast<std::size_t>(config_.map_channels) * config_.map_size *
             config_.map_size +
         3U + 2U +
         2U * static_cast<std::size_t>(config_.command_history_length) +
         3U * static_cast<std::size_t>(config_.motion_history_length);
}

const ObservationConfig& ObservationBuilder::config() const {
  return config_;
}

}  // namespace go2w_policy_deployment
