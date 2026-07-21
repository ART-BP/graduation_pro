#include "go2w_local_environment/elevation_projector.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace go2w_local_environment {
namespace {

float quietNaN() {
  return std::numeric_limits<float>::quiet_NaN();
}

}  // namespace

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
  if (parameters_.minimum_points_per_cell == 0U) {
    throw std::invalid_argument(
        "minimum_points_per_cell must be at least one");
  }
}

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

  const auto minimum_and_maximum =
      std::minmax_element(heights.begin(), heights.end());
  result.minimum_height = *minimum_and_maximum.first;
  result.maximum_height = *minimum_and_maximum.second;
  result.height_range = result.maximum_height - result.minimum_height;

  const std::size_t percentile_index = static_cast<std::size_t>(
      std::floor(
          parameters_.ground_percentile *
          static_cast<double>(heights.size() - 1U)));
  std::nth_element(
      heights.begin(),
      heights.begin() + static_cast<std::ptrdiff_t>(percentile_index),
      heights.end());
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
    const std::size_t maximum_history_length) {
  if (!measurement.observed || maximum_history_length == 0U) {
    return;
  }
  if (!std::isfinite(stamp) ||
      !std::isfinite(measurement.ground_height) ||
      !std::isfinite(measurement.minimum_height) ||
      !std::isfinite(measurement.maximum_height)) {
    return;
  }
  measurements_.push_back({
      stamp,
      measurement.ground_height,
      measurement.minimum_height,
      measurement.maximum_height,
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
  return changed;
}

ElevationCell IncrementalElevationCell::fused() const {
  ElevationCell result;
  result.ground_height = quietNaN();
  result.height_range = quietNaN();
  result.minimum_height = quietNaN();
  result.maximum_height = quietNaN();
  if (measurements_.empty()) {
    return result;
  }

  double ground_height_sum = 0.0;
  float minimum_height = std::numeric_limits<float>::infinity();
  float maximum_height = -std::numeric_limits<float>::infinity();
  for (const Measurement& measurement : measurements_) {
    ground_height_sum += measurement.ground_height;
    minimum_height = std::min(
        minimum_height, measurement.minimum_height);
    maximum_height = std::max(
        maximum_height, measurement.maximum_height);
    result.point_count += measurement.point_count;
  }

  result.ground_height = static_cast<float>(
      ground_height_sum /
      static_cast<double>(measurements_.size()));
  result.minimum_height = minimum_height;
  result.maximum_height = maximum_height;
  result.height_range = maximum_height - minimum_height;
  result.observed = true;
  return result;
}

void IncrementalElevationCell::clear() {
  measurements_.clear();
}

bool IncrementalElevationCell::empty() const {
  return measurements_.empty();
}

std::size_t IncrementalElevationCell::size() const {
  return measurements_.size();
}

}  // namespace go2w_local_environment
