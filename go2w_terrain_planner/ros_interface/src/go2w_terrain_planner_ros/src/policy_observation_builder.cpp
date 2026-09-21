#include "go2w_terrain_planner_ros/policy_observation_builder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace go2w_terrain_planner_ros {
namespace {

constexpr float kPi = 3.14159265358979323846F;

std::size_t flatIndex(const int row, const int column) {
  return static_cast<std::size_t>(row) * kMapSize + column;
}

float clamp(const float value, const float lower, const float upper) {
  return std::max(lower, std::min(value, upper));
}

bool samePose(const Pose2d& first, const Pose2d& second) {
  constexpr float kTolerance = 1.0e-7F;
  return std::abs(first.x - second.x) <= kTolerance &&
         std::abs(first.y - second.y) <= kTolerance &&
         std::abs(wrapAngle(first.yaw - second.yaw)) <= kTolerance;
}

void requireFinite(const std::vector<float>& values, const char* name) {
  for (const float value : values) {
    if (!std::isfinite(value)) {
      throw std::runtime_error(std::string(name) + " contains NaN or Inf");
    }
  }
}

}  // namespace

FourChannelMap::FourChannelMap() {
  for (auto& values : channel) {
    values.assign(kCellCount, 0.0F);
  }
}

float wrapAngle(const float angle) {
  return std::atan2(std::sin(angle), std::cos(angle));
}

PolicyObservationBuilder::PolicyObservationBuilder(
    const ObservationBuilderConfig& config)
    : config_(config) {
  if (config_.map_extent_m <= 0.0F ||
      config_.maximum_ground_deviation <= 0.0F ||
      config_.fusion_span_percentile < 0.0F ||
      config_.fusion_span_percentile > 1.0F ||
      config_.maximum_goal_distance_m <= 0.0F ||
      config_.velocity_scale[0] <= 0.0F ||
      config_.velocity_scale[1] <= 0.0F) {
    throw std::invalid_argument("Invalid policy observation builder configuration");
  }
}

void PolicyObservationBuilder::clear() {
  initialized_ = false;
  sensor_history_.clear();
  policy_history_.clear();
  command_history_.clear();
}

FourChannelMap PolicyObservationBuilder::warpTo(
    const StampedMap& source,
    const Pose2d& target_pose,
    const float target_ground_reference_z) const {
  FourChannelMap result;
  const float vertical_offset =
      source.ground_reference_z - target_ground_reference_z;

  if (samePose(source.pose, target_pose)) {
    result = source.map;
    if (vertical_offset != 0.0F) {
      for (std::size_t index = 0; index < kCellCount; ++index) {
        if (result.channel[3][index] > 0.5F) {
          result.channel[0][index] = clamp(
              result.channel[0][index] + vertical_offset, -1.0F, 1.0F);
        }
      }
    }
    return result;
  }

  const float target_cosine = std::cos(target_pose.yaw);
  const float target_sine = std::sin(target_pose.yaw);
  const float source_cosine = std::cos(source.pose.yaw);
  const float source_sine = std::sin(source.pose.yaw);
  const float step = config_.map_extent_m / static_cast<float>(kMapSize - 1);
  const float half_extent = 0.5F * config_.map_extent_m;

  for (int row = 0; row < kMapSize; ++row) {
    const float target_x = -half_extent + static_cast<float>(row) * step;
    for (int column = 0; column < kMapSize; ++column) {
      const float target_y =
          -half_extent + static_cast<float>(column) * step;
      const float world_x = target_pose.x + target_cosine * target_x -
                            target_sine * target_y;
      const float world_y = target_pose.y + target_sine * target_x +
                            target_cosine * target_y;
      const float delta_x = world_x - source.pose.x;
      const float delta_y = world_y - source.pose.y;
      const float source_x = source_cosine * delta_x + source_sine * delta_y;
      const float source_y = -source_sine * delta_x + source_cosine * delta_y;
      const float source_row =
          (source_x + half_extent) / config_.map_extent_m *
          static_cast<float>(kMapSize - 1);
      const float source_column =
          (source_y + half_extent) / config_.map_extent_m *
          static_cast<float>(kMapSize - 1);
      const std::size_t output_index = flatIndex(row, column);

      // Match torch.grid_sample for the validity-weighted bilinear ground
      // channel. Samples outside the map use zero padding.
      const int row0 = static_cast<int>(std::floor(source_row));
      const int column0 = static_cast<int>(std::floor(source_column));
      const float row_fraction = source_row - static_cast<float>(row0);
      const float column_fraction =
          source_column - static_cast<float>(column0);
      float weighted_ground = 0.0F;
      float warped_weight = 0.0F;
      for (int row_offset = 0; row_offset <= 1; ++row_offset) {
        const int sample_row = row0 + row_offset;
        const float row_weight = row_offset == 0 ? 1.0F - row_fraction
                                                  : row_fraction;
        if (sample_row < 0 || sample_row >= kMapSize) {
          continue;
        }
        for (int column_offset = 0; column_offset <= 1; ++column_offset) {
          const int sample_column = column0 + column_offset;
          if (sample_column < 0 || sample_column >= kMapSize) {
            continue;
          }
          const float column_weight =
              column_offset == 0 ? 1.0F - column_fraction : column_fraction;
          const float weight = row_weight * column_weight;
          const std::size_t sample_index =
              flatIndex(sample_row, sample_column);
          const float validity = source.map.channel[3][sample_index];
          weighted_ground +=
              weight * source.map.channel[0][sample_index] * validity;
          warped_weight += weight * validity;
        }
      }
      if (warped_weight > 1.0e-6F) {
        result.channel[0][output_index] =
            weighted_ground / std::max(warped_weight, 1.0e-6F);
      }

      // Height range and both masks use nearest-neighbour sampling.
      const int nearest_row = static_cast<int>(std::round(source_row));
      const int nearest_column = static_cast<int>(std::round(source_column));
      if (nearest_row >= 0 && nearest_row < kMapSize &&
          nearest_column >= 0 && nearest_column < kMapSize) {
        const std::size_t sample_index = flatIndex(nearest_row, nearest_column);
        result.channel[1][output_index] = source.map.channel[1][sample_index];
        result.channel[2][output_index] = source.map.channel[2][sample_index];
        result.channel[3][output_index] = source.map.channel[3][sample_index];
      }
      if (result.channel[3][output_index] <= 0.5F) {
        result.channel[1][output_index] = 0.0F;
      } else {
        result.channel[0][output_index] = clamp(
            result.channel[0][output_index] + vertical_offset, -1.0F, 1.0F);
      }
    }
  }
  return result;
}

FourChannelMap PolicyObservationBuilder::fuse(
    const std::vector<FourChannelMap>& aligned_history) const {
  if (aligned_history.size() != kObservationFusionLength) {
    throw std::invalid_argument("Aligned observation history must contain 12 maps");
  }
  FourChannelMap output;
  std::array<float, kObservationFusionLength> ground_samples{};
  std::array<float, kObservationFusionLength> range_samples{};

  for (std::size_t cell = 0; cell < kCellCount; ++cell) {
    int selected_anchor = 0;
    int best_score = std::numeric_limits<int>::min();
    float observed = 0.0F;
    for (int anchor = 0; anchor < kObservationFusionLength; ++anchor) {
      int cluster_count = 0;
      if (aligned_history[anchor].channel[3][cell] > 0.5F) {
        const float anchor_ground = aligned_history[anchor].channel[0][cell];
        for (int index = 0; index < kObservationFusionLength; ++index) {
          if (aligned_history[index].channel[3][cell] > 0.5F &&
              std::abs(aligned_history[index].channel[0][cell] -
                       anchor_ground) <= config_.maximum_ground_deviation) {
            ++cluster_count;
          }
        }
      }
      const int score = cluster_count * (kObservationFusionLength + 1) + anchor;
      if (score > best_score) {
        best_score = score;
        selected_anchor = anchor;
      }
      observed = std::max(observed,
                          aligned_history[anchor].channel[2][cell]);
    }

    const float selected_ground =
        aligned_history[selected_anchor].channel[0][cell];
    int count = 0;
    for (int index = 0; index < kObservationFusionLength; ++index) {
      if (aligned_history[index].channel[3][cell] > 0.5F &&
          std::abs(aligned_history[index].channel[0][cell] -
                   selected_ground) <= config_.maximum_ground_deviation) {
        ground_samples[count] = aligned_history[index].channel[0][cell];
        range_samples[count] = aligned_history[index].channel[1][cell];
        ++count;
      }
    }
    output.channel[2][cell] = clamp(observed, 0.0F, 1.0F);
    if (count == 0) {
      continue;
    }
    std::sort(ground_samples.begin(), ground_samples.begin() + count);
    std::sort(range_samples.begin(), range_samples.begin() + count);
    const int ground_rank = (count - 1) / 2;
    const int range_rank = static_cast<int>(
        std::ceil(config_.fusion_span_percentile * (count - 1)));
    output.channel[0][cell] = ground_samples[ground_rank];
    output.channel[1][cell] = range_samples[range_rank];
    output.channel[3][cell] = 1.0F;
  }
  return output;
}

std::vector<float> PolicyObservationBuilder::compactMap(
    const FourChannelMap& current,
    const FourChannelMap& previous,
    const std::vector<FourChannelMap>& aligned_observations) const {
  if (aligned_observations.size() != kObservationFusionLength) {
    throw std::invalid_argument("Aligned observation history must contain 12 maps");
  }
  std::vector<float> output(
      static_cast<std::size_t>(kActorMapChannels) * kCellCount, 0.0F);
  for (int channel = 0; channel < kMapChannels; ++channel) {
    std::copy(current.channel[channel].begin(),
              current.channel[channel].end(),
              output.begin() + static_cast<std::size_t>(channel) * kCellCount);
  }

  float confidence_denominator = 0.0F;
  std::array<float, kObservationFusionLength> recency_weights{};
  for (int index = 0; index < kObservationFusionLength; ++index) {
    recency_weights[index] = 0.25F +
        0.75F * static_cast<float>(index) /
            static_cast<float>(kObservationFusionLength - 1);
    confidence_denominator += recency_weights[index];
  }

  for (std::size_t cell = 0; cell < kCellCount; ++cell) {
    const bool current_valid = current.channel[3][cell] > 0.5F;
    const bool previous_valid = previous.channel[3][cell] > 0.5F;
    float geometric_change = 0.0F;
    if (current_valid && previous_valid) {
      const float ground_change = std::min(
          0.5F * std::abs(current.channel[0][cell] -
                          previous.channel[0][cell]),
          1.0F);
      const float range_change = std::min(
          std::abs(current.channel[1][cell] - previous.channel[1][cell]),
          1.0F);
      geometric_change = std::max(ground_change, range_change);
    }
    const float mask_change = std::max(
        std::abs(current.channel[2][cell] - previous.channel[2][cell]),
        std::abs(current.channel[3][cell] - previous.channel[3][cell]));
    output[4U * kCellCount + cell] =
        clamp(std::max(geometric_change, mask_change), 0.0F, 1.0F);

    int latest_observed = -1;
    float confidence_sum = 0.0F;
    for (int index = 0; index < kObservationFusionLength; ++index) {
      const float observed = clamp(
          aligned_observations[index].channel[2][cell], 0.0F, 1.0F);
      if (observed > 0.5F) {
        latest_observed = index;
      }
      confidence_sum += observed * recency_weights[index];
    }
    const float age = clamp(
        static_cast<float>(kObservationFusionLength - 1 - latest_observed) /
            static_cast<float>(kObservationFusionLength - 1),
        0.0F, 1.0F);
    output[5U * kCellCount + cell] = age;
    output[6U * kCellCount + cell] =
        clamp(confidence_sum / confidence_denominator, 0.0F, 1.0F);
  }
  requireFinite(output, "compact map");
  return output;
}

std::vector<std::array<float, 3>>
PolicyObservationBuilder::motionHistory() const {
  if (policy_history_.size() != kMotionHistoryLength + 1) {
    throw std::runtime_error("Policy pose history is not initialized");
  }
  std::vector<std::array<float, 3>> output;
  output.reserve(kMotionHistoryLength);
  for (int index = 0; index < kMotionHistoryLength; ++index) {
    const Pose2d& previous = policy_history_[index].pose;
    const Pose2d& current = policy_history_[index + 1].pose;
    const float world_dx = current.x - previous.x;
    const float world_dy = current.y - previous.y;
    const float cosine = std::cos(previous.yaw);
    const float sine = std::sin(previous.yaw);
    output.push_back({{
        cosine * world_dx + sine * world_dy,
        -sine * world_dx + cosine * world_dy,
        wrapAngle(current.yaw - previous.yaw),
    }});
  }
  return output;
}

std::vector<float> PolicyObservationBuilder::assemble(
    const std::vector<float>& compact_map,
    const Pose2d& pose,
    const std::array<float, 2>& goal_world_xy,
    const std::array<float, 2>& current_velocity) const {
  if (compact_map.size() !=
      static_cast<std::size_t>(kActorMapChannels) * kCellCount ||
      command_history_.size() != kMotionHistoryLength) {
    throw std::runtime_error("Policy history has an invalid shape");
  }
  const float goal_dx = goal_world_xy[0] - pose.x;
  const float goal_dy = goal_world_xy[1] - pose.y;
  const float goal_distance = std::hypot(goal_dx, goal_dy);
  const float goal_bearing =
      wrapAngle(std::atan2(goal_dy, goal_dx) - pose.yaw);

  std::vector<float> output;
  output.reserve(kPolicyObservationDimension);
  output.insert(output.end(), compact_map.begin(), compact_map.end());
  output.push_back(clamp(
      goal_distance / config_.maximum_goal_distance_m, 0.0F, 1.0F));
  output.push_back(std::sin(goal_bearing));
  output.push_back(std::cos(goal_bearing));
  output.push_back(current_velocity[0] / config_.velocity_scale[0]);
  output.push_back(current_velocity[1] / config_.velocity_scale[1]);
  for (const auto& command : command_history_) {
    output.push_back(command[0] / config_.velocity_scale[0]);
    output.push_back(command[1] / config_.velocity_scale[1]);
  }
  for (const auto& motion : motionHistory()) {
    output.push_back(motion[0] / (0.5F * config_.map_extent_m));
    output.push_back(motion[1] / (0.5F * config_.map_extent_m));
    output.push_back(motion[2] / kPi);
  }
  if (output.size() != kPolicyObservationDimension) {
    throw std::runtime_error("Assembled policy observation dimension is invalid");
  }
  requireFinite(output, "policy observation");
  return output;
}

std::vector<float> PolicyObservationBuilder::update(
    const FourChannelMap& raw_map,
    const Pose2d& pose,
    const float ground_reference_z,
    const std::array<float, 2>& last_command,
    const std::array<float, 2>& current_velocity,
    const std::array<float, 2>& goal_world_xy,
    ObservationStatistics* statistics) {
  StampedMap raw_stamped{raw_map, pose, ground_reference_z};
  FourChannelMap current_fused;
  FourChannelMap previous_aligned;
  std::vector<FourChannelMap> aligned_observations;
  aligned_observations.reserve(kObservationFusionLength);

  if (!initialized_) {
    sensor_history_.assign(kObservationFusionLength, raw_stamped);
    aligned_observations.assign(kObservationFusionLength, raw_map);
    current_fused = fuse(aligned_observations);
    const StampedMap fused_stamped{current_fused, pose, ground_reference_z};
    policy_history_.assign(kMotionHistoryLength + 1, fused_stamped);
    command_history_.assign(kMotionHistoryLength, {{0.0F, 0.0F}});
    previous_aligned = current_fused;
    initialized_ = true;
  } else {
    previous_aligned = warpTo(
        policy_history_.back(), pose, ground_reference_z);
    sensor_history_.pop_front();
    sensor_history_.push_back(raw_stamped);
    for (const StampedMap& observation : sensor_history_) {
      aligned_observations.push_back(
          warpTo(observation, pose, ground_reference_z));
    }
    current_fused = fuse(aligned_observations);
    policy_history_.pop_front();
    policy_history_.push_back({current_fused, pose, ground_reference_z});
    command_history_.pop_front();
    command_history_.push_back(last_command);
  }

  const std::vector<float> compact = compactMap(
      current_fused, previous_aligned, aligned_observations);
  if (statistics != nullptr) {
    float observed_sum = 0.0F;
    float valid_sum = 0.0F;
    for (std::size_t cell = 0; cell < kCellCount; ++cell) {
      observed_sum += current_fused.channel[2][cell];
      valid_sum += current_fused.channel[3][cell];
    }
    statistics->observed_ratio = observed_sum / static_cast<float>(kCellCount);
    statistics->height_valid_ratio = valid_sum / static_cast<float>(kCellCount);
    statistics->goal_distance_m =
        std::hypot(goal_world_xy[0] - pose.x, goal_world_xy[1] - pose.y);
  }
  return assemble(compact, pose, goal_world_xy, current_velocity);
}

}  // namespace go2w_terrain_planner_ros
