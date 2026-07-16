#include "go2w_local_environment/dual_layer_projector.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace go2w_local_environment {
namespace {

double quietNaN() {
  return std::numeric_limits<double>::quiet_NaN();
}

}  // namespace

DualLayerProjector::DualLayerProjector(
    const ProjectionParameters& parameters)
    : parameters_(parameters) {
  if (parameters_.voxel_resolution <= 0.0) {
    throw std::invalid_argument("voxel_resolution must be positive");
  }
  if (parameters_.vertical_max_offset <= parameters_.vertical_min_offset) {
    throw std::invalid_argument(
        "vertical_max_offset must be greater than vertical_min_offset");
  }
  if (parameters_.ground_search_max_offset <
      parameters_.ground_search_min_offset) {
    throw std::invalid_argument(
        "ground search maximum must not be below its minimum");
  }
  if (parameters_.nominal_ground_offset <
          parameters_.ground_search_min_offset ||
      parameters_.nominal_ground_offset >
          parameters_.ground_search_max_offset) {
    throw std::invalid_argument(
        "nominal_ground_offset must be inside the ground search range");
  }
  if (parameters_.minimum_clearance <= 0.0) {
    throw std::invalid_argument("minimum_clearance must be positive");
  }
  if (parameters_.maximum_unknown_fraction < 0.0 ||
      parameters_.maximum_unknown_fraction > 1.0) {
    throw std::invalid_argument(
        "maximum_unknown_fraction must be in [0, 1]");
  }
}

ColumnProjection DualLayerProjector::project(
    const double robot_z,
    const std::vector<float>& occupied_z,
    const std::vector<float>& unknown_z,
    const bool unknown_available) const {
  ColumnProjection result;
  result.ground_height = quietNaN();
  result.ceiling_height = quietNaN();
  result.clearance = quietNaN();
  result.unknown_fraction = quietNaN();
  result.occupied_count = occupied_z.size();

  const double minimum_height =
      robot_z + parameters_.vertical_min_offset;
  const double maximum_height =
      robot_z + parameters_.vertical_max_offset;
  const int bin_count = std::max(
      1,
      static_cast<int>(std::ceil(
          (maximum_height - minimum_height) /
          parameters_.voxel_resolution)));

  std::vector<VoxelState> states(
      static_cast<std::size_t>(bin_count),
      unknown_available ? VoxelState::kFree : VoxelState::kUnknown);

  if (unknown_available) {
    for (const float height : unknown_z) {
      const int bin = heightToBin(height, minimum_height, bin_count);
      if (bin >= 0) {
        states[static_cast<std::size_t>(bin)] = VoxelState::kUnknown;
      }
    }
  }

  for (const float height : occupied_z) {
    const int bin = heightToBin(height, minimum_height, bin_count);
    if (bin >= 0) {
      states[static_cast<std::size_t>(bin)] = VoxelState::kOccupied;
    }
  }

  const double ground_minimum =
      robot_z + parameters_.ground_search_min_offset;
  const double ground_maximum =
      robot_z + parameters_.ground_search_max_offset;
  const double expected_ground_height =
      robot_z + parameters_.nominal_ground_offset;

  int ground_bin = -1;
  double ground_center = quietNaN();
  double best_ground_error = std::numeric_limits<double>::infinity();
  for (const float height : occupied_z) {
    if (height < ground_minimum || height > ground_maximum) {
      continue;
    }
    const double error =
        std::abs(static_cast<double>(height) - expected_ground_height);
    if (error < best_ground_error) {
      best_ground_error = error;
      ground_center = height;
    }
  }

  if (!std::isfinite(ground_center)) {
    return result;
  }
  ground_bin = heightToBin(ground_center, minimum_height, bin_count);
  if (ground_bin < 0) {
    return result;
  }

  result.ground_observed = true;
  result.ground_height =
      ground_center +
      0.5 * parameters_.voxel_resolution;

  double ceiling_center = std::numeric_limits<double>::infinity();
  const double distinct_voxel_threshold =
      ground_center + 0.5 * parameters_.voxel_resolution;
  for (const float height : occupied_z) {
    if (height > distinct_voxel_threshold &&
        height < ceiling_center &&
        height <= maximum_height) {
      ceiling_center = height;
    }
  }

  if (std::isfinite(ceiling_center)) {
    result.ceiling_observed = true;
    result.ceiling_height =
        ceiling_center -
        0.5 * parameters_.voxel_resolution;
    result.clearance =
        std::max(0.0, result.ceiling_height - result.ground_height);
  } else {
    result.clearance =
        std::max(0.0, maximum_height - result.ground_height);
  }

  if (unknown_available) {
    const double required_top =
        result.ground_height + parameters_.minimum_clearance;
    int relevant_count = 0;
    int unknown_count = 0;

    for (int bin = 0; bin < bin_count; ++bin) {
      const double voxel_bottom =
          binCenter(bin, minimum_height) -
          0.5 * parameters_.voxel_resolution;
      const double voxel_top =
          voxel_bottom + parameters_.voxel_resolution;
      if (voxel_top <= result.ground_height) {
        continue;
      }
      if (voxel_bottom >= required_top) {
        break;
      }
      ++relevant_count;
      if (states[static_cast<std::size_t>(bin)] ==
          VoxelState::kUnknown) {
        ++unknown_count;
      }
    }

    result.unknown_fraction =
        relevant_count > 0
            ? static_cast<double>(unknown_count) /
                  static_cast<double>(relevant_count)
            : 0.0;
    result.observed =
        result.unknown_fraction <=
        parameters_.maximum_unknown_fraction;
  } else {
    // Occupied-only input is a development fallback. It can establish
    // geometry but cannot distinguish free voxels from unknown voxels.
    result.observed = true;
  }

  result.blocked =
      result.ceiling_observed &&
      result.clearance < parameters_.minimum_clearance;
  if (parameters_.unknown_is_blocked && unknown_available &&
      !result.observed) {
    result.blocked = true;
  }

  return result;
}

const ProjectionParameters& DualLayerProjector::parameters() const {
  return parameters_;
}

int DualLayerProjector::heightToBin(
    const double height,
    const double minimum_height,
    const int bin_count) const {
  const int bin = static_cast<int>(
      std::floor(
          (height - minimum_height) /
          parameters_.voxel_resolution +
          1e-6));
  if (bin < 0 || bin >= bin_count) {
    return -1;
  }
  return bin;
}

double DualLayerProjector::binCenter(
    const int bin,
    const double minimum_height) const {
  return minimum_height +
         (static_cast<double>(bin) + 0.5) *
             parameters_.voxel_resolution;
}

}  // namespace go2w_local_environment
