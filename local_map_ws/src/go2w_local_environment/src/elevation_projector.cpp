#include "go2w_local_environment/elevation_projector.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace go2w_local_environment {
namespace {

float quietNaN() {
  return std::numeric_limits<float>::quiet_NaN();
}

}  // namespace

// 地面高度估计器类，用于从附近的绝对里程计帧测量中估计支撑地面的高度。名义的IMU到地面的估计在可用测量过少时使用，并且还会拒绝不相关的高度表面。
GroundReferenceEstimator::GroundReferenceEstimator(
    const GroundReferenceParameters& parameters)
    : parameters_(parameters) {
  if (parameters_.minimum_samples == 0U) {
    throw std::invalid_argument(
        "ground-reference minimum_samples must be positive");
  }
  if (parameters_.maximum_nominal_difference <= 0.0F) {
    throw std::invalid_argument(
        "ground-reference maximum nominal difference must be positive");
  }
}

// 估计地面高度。
// 从地面高度中采用中位数的方式计算出地面高度
float GroundReferenceEstimator::estimate(
    std::vector<float>& ground_heights,
    const float nominal_reference_height) const {
  if (!std::isfinite(nominal_reference_height)) {
    throw std::invalid_argument(
        "nominal ground-reference height must be finite");
  }

  ground_heights.erase(
      std::remove_if(
          ground_heights.begin(),
          ground_heights.end(),
          [this, nominal_reference_height](const float height) {
            return !std::isfinite(height) ||
                   std::abs(height - nominal_reference_height) >
                       parameters_.maximum_nominal_difference;
          }),
      ground_heights.end());
  if (ground_heights.size() < parameters_.minimum_samples) {
    return nominal_reference_height;
  }

  const std::size_t middle = ground_heights.size() / 2U;
  std::nth_element(
      ground_heights.begin(),
      ground_heights.begin() + static_cast<std::ptrdiff_t>(middle),
      ground_heights.end());
  return ground_heights[middle];
}

const GroundReferenceParameters&
GroundReferenceEstimator::parameters() const {
  return parameters_;
}

ElevationProjector::ElevationProjector(
    const ElevationProjectionParameters& parameters)
    : parameters_(parameters) {
  if (parameters_.vertical_max_offset <=
      parameters_.vertical_min_offset) {
    throw std::invalid_argument(
        "vertical_max_offset must be greater than vertical_min_offset");
  }
  if (parameters_.ground_percentile < 0.0 ||
      parameters_.ground_percentile > 1.0) {
    throw std::invalid_argument("ground_percentile must be in [0, 1]");
  }
  if (parameters_.span_lower_percentile < 0.0 ||
      parameters_.span_lower_percentile > 1.0 ||
      parameters_.span_upper_percentile < 0.0 ||
      parameters_.span_upper_percentile > 1.0 ||
      parameters_.span_lower_percentile >
          parameters_.span_upper_percentile) {
    throw std::invalid_argument(
        "span percentiles must be ordered and in [0, 1]");
  }
  if (parameters_.minimum_points_per_cell == 0U) {
    throw std::invalid_argument(
        "minimum_points_per_cell must be at least one");
  }
}

// 获取地面高度中的低百分位数，并使用稳健上下分位数计算垂向跨度。
ElevationCell ElevationProjector::project(
    std::vector<float>& heights) const {
  ElevationCell result;
  result.ground_height = quietNaN();
  result.height_range = quietNaN();
  result.minimum_height = quietNaN();
  result.maximum_height = quietNaN();

  heights.erase(
      std::remove_if(
          heights.begin(), heights.end(),
          [](const float height) { return !std::isfinite(height); }),
      heights.end());
  result.point_count = heights.size();
  if (heights.size() < parameters_.minimum_points_per_cell) {
    return result;
  }

  std::sort(heights.begin(), heights.end());
  const std::size_t last_index = heights.size() - 1U;
  const std::size_t lower_index = static_cast<std::size_t>(
      std::floor(
          parameters_.span_lower_percentile *
          static_cast<double>(last_index)));
  const std::size_t upper_index = static_cast<std::size_t>(
      std::ceil(
          parameters_.span_upper_percentile *
          static_cast<double>(last_index)));
  result.minimum_height = heights[lower_index];
  result.maximum_height = heights[upper_index];
  result.height_range = result.maximum_height - result.minimum_height;

  const std::size_t percentile_index = static_cast<std::size_t>(
      std::floor(
          parameters_.ground_percentile *
          static_cast<double>(last_index)));
  result.ground_height = heights[percentile_index];
  result.observed = true;
  return result;
}

const ElevationProjectionParameters&
ElevationProjector::parameters() const {
  return parameters_;
}

ElevationHoleFiller::ElevationHoleFiller(
    const ElevationHoleFillParameters& parameters)
    : parameters_(parameters) {
  if (parameters_.radius_cells <= 0) {
    throw std::invalid_argument("hole-fill radius_cells must be positive");
  }
  if (parameters_.minimum_neighbors == 0U) {
    throw std::invalid_argument(
        "hole-fill minimum_neighbors must be positive");
  }
  if (parameters_.maximum_ground_height_difference < 0.0F) {
    throw std::invalid_argument(
        "hole-fill maximum height difference must not be negative");
  }
}

ElevationHoleFillResult ElevationHoleFiller::fill(
    const std::vector<ElevationNeighbor>& neighbors) const {
  std::vector<float> ground_height_scratch;
  return fill(neighbors, ground_height_scratch);
}

ElevationHoleFillResult ElevationHoleFiller::fill(
    const std::vector<ElevationNeighbor>& neighbors,
    std::vector<float>& ground_heights) const {
  ElevationHoleFillResult result;
  ground_heights.clear();
  if (ground_heights.capacity() < neighbors.size()) {
    ground_heights.reserve(neighbors.size());
  }

  float maximum_neighbor_range = 0.0F;
  bool has_negative_x = false;
  bool has_positive_x = false;
  bool has_negative_y = false;
  bool has_positive_y = false;
  for (const ElevationNeighbor& neighbor : neighbors) {
    if (!std::isfinite(neighbor.ground_height) ||
        !std::isfinite(neighbor.height_range)) {
      continue;
    }
    ground_heights.push_back(neighbor.ground_height);
    maximum_neighbor_range = std::max(
        maximum_neighbor_range, neighbor.height_range);
    has_negative_x = has_negative_x || neighbor.x_offset < 0;
    has_positive_x = has_positive_x || neighbor.x_offset > 0;
    has_negative_y = has_negative_y || neighbor.y_offset < 0;
    has_positive_y = has_positive_y || neighbor.y_offset > 0;
  }

  if (ground_heights.size() < parameters_.minimum_neighbors) {
    return result;
  }

  // Require support from opposite sides on at least one axis. This fills
  // holes between measurements without extrapolating a surface into a large
  // unknown region on only one side of the scan.
  if (!((has_negative_x && has_positive_x) ||
        (has_negative_y && has_positive_y))) {
    return result;
  }

  const auto minimum_and_maximum = std::minmax_element(
      ground_heights.begin(), ground_heights.end());
  const float ground_spread =
      *minimum_and_maximum.second - *minimum_and_maximum.first;
  if (ground_spread >
      parameters_.maximum_ground_height_difference) {
    return result;
  }

  std::sort(ground_heights.begin(), ground_heights.end());
  const std::size_t middle = ground_heights.size() / 2U;
  if (ground_heights.size() % 2U == 0U) {
    result.ground_height =
        0.5F * (ground_heights[middle - 1U] +
                ground_heights[middle]);
  } else {
    result.ground_height = ground_heights[middle];
  }
  result.height_range = std::max(maximum_neighbor_range, ground_spread);
  result.filled = true;
  return result;
}

const ElevationHoleFillParameters&
ElevationHoleFiller::parameters() const {
  return parameters_;
}

void IncrementalElevationCell::add(
    const ElevationCell& measurement,
    const double stamp,
    const std::size_t maximum_history_length,
    const ElevationFusionParameters& parameters) {
  if (!measurement.observed || maximum_history_length == 0U) {
    return;
  }
  if (!std::isfinite(stamp) ||
      !std::isfinite(measurement.ground_height) ||
      !std::isfinite(measurement.minimum_height) ||
      !std::isfinite(measurement.maximum_height) ||
      !std::isfinite(measurement.height_range)) {
    return;
  }

  const ElevationCell previous = fused(parameters);
  const bool confirms_flat_ground =
      previous.observed &&
      previous.height_range >= parameters.dynamic_span_threshold &&
      measurement.height_range <= parameters.flat_span_threshold &&
      std::abs(
          measurement.ground_height - previous.ground_height) <=
          parameters.maximum_ground_deviation;
  if (confirms_flat_ground) {
    ++consecutive_flat_observations_;
    if (consecutive_flat_observations_ >=
        parameters.flat_clear_confirmation_frames) {
      measurements_.erase(
          std::remove_if(
              measurements_.begin(),
              measurements_.end(),
              [&measurement, &parameters](
                  const Measurement& retained) {
                return
                    retained.height_range >
                        parameters.flat_span_threshold ||
                    std::abs(
                        retained.ground_height -
                        measurement.ground_height) >
                        parameters.maximum_ground_deviation;
              }),
          measurements_.end());
      consecutive_flat_observations_ = 0U;
    }
  } else {
    consecutive_flat_observations_ = 0U;
  }

  measurements_.push_back({
      stamp,
      measurement.ground_height,
      measurement.minimum_height,
      measurement.maximum_height,
      measurement.height_range,
      measurement.point_count,
  });
  while (measurements_.size() > maximum_history_length) {
    measurements_.pop_front();
  }
}

bool IncrementalElevationCell::removeOlderThan(
    const double oldest_allowed_stamp) {
  bool changed = false;
  while (!measurements_.empty() &&
         measurements_.front().stamp < oldest_allowed_stamp) {
    measurements_.pop_front();
    changed = true;
  }
  if (changed) {
    consecutive_flat_observations_ = 0U;
  }
  return changed;
}

ElevationCell IncrementalElevationCell::fused(
    const ElevationFusionParameters& parameters) const {
  ElevationCell result;
  result.ground_height = quietNaN();
  result.height_range = quietNaN();
  result.minimum_height = quietNaN();
  result.maximum_height = quietNaN();
  if (measurements_.empty()) {
    return result;
  }

  // Select the densest temporally consistent ground-height cluster. In a tie,
  // prefer the newest candidate so a changed surface does not remain anchored
  // to an older measurement.
  std::vector<const Measurement*> inliers;
  for (auto anchor = measurements_.rbegin();
       anchor != measurements_.rend(); ++anchor) {
    std::vector<const Measurement*> candidate;
    for (const Measurement& measurement : measurements_) {
      if (std::abs(
              measurement.ground_height - anchor->ground_height) <=
          parameters.maximum_ground_deviation) {
        candidate.push_back(&measurement);
      }
    }
    if (candidate.size() > inliers.size()) {
      inliers = std::move(candidate);
    }
  }

  bool contains_reliable_frame = false;
  for (const Measurement* measurement : inliers) {
    contains_reliable_frame =
        contains_reliable_frame ||
        measurement->point_count >=
            parameters.reliable_frame_minimum_points;
  }
  if (!contains_reliable_frame &&
      inliers.size() < parameters.minimum_confirming_frames) {
    return result;
  }

  std::vector<float> ground_heights;
  std::vector<float> vertical_spans;
  ground_heights.reserve(inliers.size());
  vertical_spans.reserve(inliers.size());
  for (const Measurement* measurement : inliers) {
    ground_heights.push_back(measurement->ground_height);
    vertical_spans.push_back(measurement->height_range);
    result.point_count += measurement->point_count;
  }
  std::sort(ground_heights.begin(), ground_heights.end());
  const std::size_t middle = ground_heights.size() / 2U;
  if (ground_heights.size() % 2U == 0U) {
    result.ground_height =
        0.5F * (ground_heights[middle - 1U] +
                ground_heights[middle]);
  } else {
    result.ground_height = ground_heights[middle];
  }

  std::sort(vertical_spans.begin(), vertical_spans.end());
  const std::size_t span_index = static_cast<std::size_t>(
      std::ceil(
          parameters.span_percentile *
          static_cast<double>(vertical_spans.size() - 1U)));
  result.height_range = vertical_spans[span_index];
  result.minimum_height = result.ground_height;
  result.maximum_height =
      result.ground_height + result.height_range;
  result.observed = true;
  return result;
}

void IncrementalElevationCell::clear() {
  measurements_.clear();
  consecutive_flat_observations_ = 0U;
}

bool IncrementalElevationCell::empty() const {
  return measurements_.empty();
}

std::size_t IncrementalElevationCell::size() const {
  return measurements_.size();
}

}  // namespace go2w_local_environment
