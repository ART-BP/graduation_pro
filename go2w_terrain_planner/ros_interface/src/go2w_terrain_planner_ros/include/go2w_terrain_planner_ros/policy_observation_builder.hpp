#pragma once

#include <array>
#include <cstddef>
#include <deque>
#include <string>
#include <vector>

namespace go2w_terrain_planner_ros {

constexpr int kMapSize = 200;
constexpr int kMapChannels = 4;
constexpr int kActorMapChannels = 7;
constexpr int kObservationFusionLength = 12;
constexpr int kMotionHistoryLength = 8;
constexpr int kPolicyObservationDimension = 280045;
constexpr std::size_t kCellCount =
    static_cast<std::size_t>(kMapSize) * kMapSize;

struct Pose2d {
  float x{0.0F};
  float y{0.0F};
  float yaw{0.0F};
};

struct FourChannelMap {
  std::array<std::vector<float>, kMapChannels> channel;

  FourChannelMap();
};

struct ObservationStatistics {
  float observed_ratio{0.0F};
  float height_valid_ratio{0.0F};
  float goal_distance_m{0.0F};
};

struct ObservationBuilderConfig {
  float map_extent_m{10.0F};
  float maximum_ground_deviation{0.06F};
  float fusion_span_percentile{0.75F};
  float maximum_goal_distance_m{10.0F};
  std::array<float, 2> velocity_scale{{1.2F, 1.0F}};
};

class PolicyObservationBuilder {
 public:
  explicit PolicyObservationBuilder(
      const ObservationBuilderConfig& config = ObservationBuilderConfig());

  void clear();
  bool initialized() const { return initialized_; }

  // raw_map is already robot-centric and normalized with channel order
  // [ground, height_range, observed, height_valid]. last_command is the
  // physical [v,w] command issued at the previous decision step.
  std::vector<float> update(
      const FourChannelMap& raw_map,
      const Pose2d& pose,
      float ground_reference_z,
      const std::array<float, 2>& last_command,
      const std::array<float, 2>& current_velocity,
      const std::array<float, 2>& goal_world_xy,
      ObservationStatistics* statistics = nullptr);

 private:
  struct StampedMap {
    FourChannelMap map;
    Pose2d pose;
    float ground_reference_z{0.0F};
  };

  ObservationBuilderConfig config_;
  bool initialized_{false};
  std::deque<StampedMap> sensor_history_;
  std::deque<StampedMap> policy_history_;
  std::deque<std::array<float, 2>> command_history_;

  FourChannelMap warpTo(
      const StampedMap& source,
      const Pose2d& target_pose,
      float target_ground_reference_z) const;
  FourChannelMap fuse(
      const std::vector<FourChannelMap>& aligned_history) const;
  std::vector<float> compactMap(
      const FourChannelMap& current,
      const FourChannelMap& previous,
      const std::vector<FourChannelMap>& aligned_observations) const;
  std::vector<std::array<float, 3>> motionHistory() const;
  std::vector<float> assemble(
      const std::vector<float>& compact_map,
      const Pose2d& pose,
      const std::array<float, 2>& goal_world_xy,
      const std::array<float, 2>& current_velocity) const;
};

float wrapAngle(float angle);

}  // namespace go2w_terrain_planner_ros
