#pragma once

#include <array>
#include <cstddef>
#include <deque>
#include <vector>

namespace go2w_policy_deployment {

struct Pose2d {
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
};

// Channel-major map: [ground, height_range, observed, height_valid].
struct PolicyMap {
  PolicyMap() = default;
  explicit PolicyMap(int size);

  float& at(int channel, int row, int column);
  float at(int channel, int row, int column) const;
  std::size_t cells() const;

  int size{0};
  std::vector<float> data;
};

struct ObservationConfig {
  int map_size{200};
  int map_channels{7};
  int observation_fusion_length{12};
  int command_history_length{8};
  int motion_history_length{8};
  double map_extent_m{10.0};
  double ground_height_scale_m{1.0};
  double height_range_scale_m{3.0};
  double maximum_ground_deviation{0.06};
  double fusion_span_percentile{0.75};
  double goal_maximum_distance_m{10.0};
  double linear_velocity_scale_mps{1.2};
  double angular_velocity_scale_radps{1.0};
};

struct ObservationResult {
  std::vector<float> values;
  double observed_ratio{0.0};
  double height_valid_ratio{0.0};
};

class ObservationBuilder {
 public:
  explicit ObservationBuilder(ObservationConfig config = ObservationConfig{});

  void reset();

  // Add the newest physical command after inference. It will appear as the
  // latest entry in the next observation, matching the training-time order.
  void recordCommand(double linear_mps, double angular_radps);

  ObservationResult build(
      const PolicyMap& current_map,
      const Pose2d& current_pose,
      double current_ground_reference_z,
      double goal_x,
      double goal_y,
      double measured_linear_velocity_mps,
      double measured_angular_velocity_radps);

  std::size_t expectedDimension() const;
  const ObservationConfig& config() const;

 private:
  struct MapFrame {
    PolicyMap map;
    Pose2d pose;
    double ground_reference_z{0.0};
  };

  PolicyMap alignFrame(
      const MapFrame& source,
      const Pose2d& target_pose,
      double target_ground_reference_z) const;
  PolicyMap fuseHistory(const std::vector<PolicyMap>& aligned) const;
  PolicyMap buildCompactMap(
      const PolicyMap& fused,
      const PolicyMap& previous_aligned,
      const std::vector<PolicyMap>& aligned_observations) const;
  void pushPose(const Pose2d& pose);

  ObservationConfig config_;
  std::deque<MapFrame> observation_history_;
  std::deque<Pose2d> pose_history_;
  std::deque<std::array<double, 2>> command_history_;
  MapFrame previous_fused_;
  bool has_previous_fused_{false};
};

}  // namespace go2w_policy_deployment
