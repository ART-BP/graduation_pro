#pragma once

#include <cstddef>
#include <vector>

namespace go2w_local_environment {

struct ProjectionParameters {
  double voxel_resolution{0.1};
  double vertical_min_offset{-1.5};
  double vertical_max_offset{2.0};
  double ground_search_min_offset{-1.2};
  double ground_search_max_offset{0.30};
  double nominal_ground_offset{-0.35};
  double minimum_clearance{0.65};
  double maximum_unknown_fraction{0.20};
  bool unknown_is_blocked{false};
};

struct ColumnProjection {
  double ground_height;
  double ceiling_height;
  double clearance;
  double unknown_fraction;
  std::size_t occupied_count{0};
  bool ground_observed{false};
  bool ceiling_observed{false};
  bool observed{false};
  bool blocked{false};
};

class DualLayerProjector {
 public:
  explicit DualLayerProjector(const ProjectionParameters& parameters);

  ColumnProjection project(
      double robot_z,
      const std::vector<float>& occupied_z,
      const std::vector<float>& unknown_z,
      bool unknown_available) const;

  const ProjectionParameters& parameters() const;

 private:
  enum class VoxelState : unsigned char {
    kUnknown = 0,
    kFree = 1,
    kOccupied = 2,
  };

  int heightToBin(double height, double minimum_height, int bin_count) const;
  double binCenter(int bin, double minimum_height) const;

  ProjectionParameters parameters_;
};

}  // namespace go2w_local_environment
