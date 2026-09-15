#pragma once

#include <cstddef>
#include <deque>
#include <vector>

namespace go2w_local_environment {

struct ElevationProjectionParameters {
  double vertical_min_offset{-1.5};
  double vertical_max_offset{2.8};
  double ground_percentile{0.10};
  double span_lower_percentile{0.05};
  double span_upper_percentile{0.95};
  std::size_t minimum_points_per_cell{1};
};

struct ElevationCell {
  float ground_height;
  float height_range;
  float minimum_height;
  float maximum_height;
  std::size_t point_count{0};
  bool observed{false};
};

struct ElevationFusionParameters {
  // A single sparse return is provisional. It becomes a usable height after
  // repeated, geometrically consistent observations.
  std::size_t minimum_confirming_frames{2};
  std::size_t reliable_frame_minimum_points{2};
  float maximum_ground_deviation{0.12F};

  // Fuse per-frame vertical spans with an upper quantile. This preserves
  // repeated discontinuities while rejecting an isolated extreme frame.
  double span_percentile{0.75};

  // Repeated flat observations at the same supporting height remove old
  // high-span measurements, which clears moved objects without treating a
  // 2-D ray traversal as proof of free ground.
  float dynamic_span_threshold{0.15F};
  float flat_span_threshold{0.05F};
  std::size_t flat_clear_confirmation_frames{2};
};

struct ElevationHoleFillParameters {
  int radius_cells{2};
  std::size_t minimum_neighbors{3};
  float maximum_ground_height_difference{0.08F};
};

struct ElevationNeighbor {
  int x_offset{0};
  int y_offset{0};
  float ground_height{0.0F};
  float height_range{0.0F};
};

struct ElevationHoleFillResult {
  float ground_height{0.0F};
  float height_range{0.0F};
  bool filled{false};
};

struct GroundReferenceParameters {
  std::size_t minimum_samples{5};
  float maximum_nominal_difference{0.30F};
};

// Estimates the supporting-ground height from nearby absolute odom-frame
// measurements. The nominal IMU-to-ground estimate is used when too few
// measurements are available and also rejects unrelated height surfaces.
class GroundReferenceEstimator {
 public:
  explicit GroundReferenceEstimator(
      const GroundReferenceParameters& parameters =
          GroundReferenceParameters());

  float estimate(
      std::vector<float>& ground_heights,
      float nominal_reference_height) const;

  const GroundReferenceParameters& parameters() const;

 private:
  GroundReferenceParameters parameters_;
};

class ElevationProjector {
 public:
  explicit ElevationProjector(
      const ElevationProjectionParameters& parameters);

  // Removes non-finite samples, sorts the input buffer, and extracts a low
  // ground percentile together with a robust vertical span.
  ElevationCell project(std::vector<float>& heights) const;

  const ElevationProjectionParameters& parameters() const;

 private:
  ElevationProjectionParameters parameters_;
};

// Produces an edge-aware estimate for a small isolated elevation hole. The
// caller owns neighborhood lookup, which keeps this class independent of the
// GridMap circular-buffer layout. Results are intended for output only and
// must not be inserted into the raw measurement history.
class ElevationHoleFiller {
 public:
  explicit ElevationHoleFiller(
      const ElevationHoleFillParameters& parameters =
          ElevationHoleFillParameters());

  ElevationHoleFillResult fill(
      const std::vector<ElevationNeighbor>& neighbors) const;

  // Reuses caller-owned scratch storage to avoid one heap allocation per
  // candidate cell during a full-map hole-fill pass.
  ElevationHoleFillResult fill(
      const std::vector<ElevationNeighbor>& neighbors,
      std::vector<float>& ground_height_scratch) const;

  const ElevationHoleFillParameters& parameters() const;

 private:
  ElevationHoleFillParameters parameters_;
};

// Bounded per-cell history used by the rolling map. Each LiDAR frame adds at
// most one measurement to a cell, so memory and update time stay bounded.
class IncrementalElevationCell {
 public:
  void add(
      const ElevationCell& measurement,
      double stamp,
      std::size_t maximum_history_length,
      const ElevationFusionParameters& parameters);

  bool removeOlderThan(double oldest_allowed_stamp);

  ElevationCell fused(
      const ElevationFusionParameters& parameters) const;

  void clear();

  bool empty() const;

  std::size_t size() const;

 private:
  struct Measurement {
    double stamp;
    float ground_height;
    float minimum_height;
    float maximum_height;
    float height_range;
    std::size_t point_count;
  };

  std::deque<Measurement> measurements_;
  std::size_t consecutive_flat_observations_{0U};
};

}  // namespace go2w_local_environment
