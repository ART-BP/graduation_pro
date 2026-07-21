#include "go2w_local_environment/elevation_projector.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <grid_map_core/BufferRegion.hpp>
#include <grid_map_core/GridMap.hpp>
#include <gtest/gtest.h>

namespace go2w_local_environment {
namespace {

ElevationProjector makeProjector(
    const double percentile = 0.20,
    const std::size_t minimum_points = 1U) {
  ElevationProjectionParameters parameters;
  parameters.ground_percentile = percentile;
  parameters.minimum_points_per_cell = minimum_points;
  return ElevationProjector(parameters);
}

TEST(GroundReferenceEstimatorTest, UsesMedianNearbyGround) {
  GroundReferenceParameters parameters;
  parameters.minimum_samples = 3U;
  parameters.maximum_nominal_difference = 0.30F;
  const GroundReferenceEstimator estimator(parameters);
  std::vector<float> heights{-0.34F, -0.36F, -0.35F, 0.80F};

  const float result = estimator.estimate(heights, -0.35F);

  EXPECT_FLOAT_EQ(result, -0.35F);
}

TEST(GroundReferenceEstimatorTest, FallsBackToNominalReference) {
  GroundReferenceParameters parameters;
  parameters.minimum_samples = 3U;
  const GroundReferenceEstimator estimator(parameters);
  std::vector<float> heights{-0.34F, 0.80F};

  EXPECT_FLOAT_EQ(estimator.estimate(heights, -0.35F), -0.35F);
}

TEST(ElevationProjectorTest, ProjectsSingleGroundReturn) {
  const ElevationProjector projector = makeProjector();
  std::vector<float> heights{0.12F};

  const ElevationCell result = projector.project(heights);

  EXPECT_TRUE(result.observed);
  EXPECT_EQ(result.point_count, 1U);
  EXPECT_FLOAT_EQ(result.ground_height, 0.12F);
  EXPECT_FLOAT_EQ(result.height_range, 0.0F);
  EXPECT_FLOAT_EQ(result.minimum_height, 0.12F);
  EXPECT_FLOAT_EQ(result.maximum_height, 0.12F);
}

TEST(ElevationProjectorTest, UsesConfiguredLowPercentile) {
  const ElevationProjector projector = makeProjector(0.25);
  std::vector<float> heights{1.0F, 0.10F, 0.20F, 0.30F, 0.40F};

  const ElevationCell result = projector.project(heights);

  EXPECT_TRUE(result.observed);
  EXPECT_FLOAT_EQ(result.ground_height, 0.20F);
  EXPECT_FLOAT_EQ(result.height_range, 0.90F);
}

TEST(ElevationProjectorTest, LeavesSparseCellUnknown) {
  const ElevationProjector projector = makeProjector(0.20, 3U);
  std::vector<float> heights{0.10F, 0.11F};

  const ElevationCell result = projector.project(heights);

  EXPECT_FALSE(result.observed);
  EXPECT_EQ(result.point_count, 2U);
  EXPECT_TRUE(std::isnan(result.ground_height));
  EXPECT_TRUE(std::isnan(result.height_range));
  EXPECT_TRUE(std::isnan(result.minimum_height));
  EXPECT_TRUE(std::isnan(result.maximum_height));
}

TEST(ElevationProjectorTest, IgnoresNonFiniteSamples) {
  const ElevationProjector projector = makeProjector(0.0, 2U);
  std::vector<float> heights{
      std::numeric_limits<float>::quiet_NaN(),
      0.40F,
      std::numeric_limits<float>::infinity(),
      0.10F};

  const ElevationCell result = projector.project(heights);

  EXPECT_TRUE(result.observed);
  EXPECT_EQ(result.point_count, 2U);
  EXPECT_FLOAT_EQ(result.ground_height, 0.10F);
  EXPECT_FLOAT_EQ(result.height_range, 0.30F);
}

TEST(ElevationProjectorTest, RejectsInvalidParameters) {
  ElevationProjectionParameters parameters;
  parameters.ground_percentile = 1.1;
  EXPECT_THROW(ElevationProjector{parameters}, std::invalid_argument);

  parameters.ground_percentile = 0.2;
  parameters.minimum_points_per_cell = 0U;
  EXPECT_THROW(ElevationProjector{parameters}, std::invalid_argument);
}

TEST(ElevationHoleFillerTest, FillsSmallFlatHoleFromOppositeSides) {
  ElevationHoleFillParameters parameters;
  parameters.radius_cells = 2;
  parameters.minimum_neighbors = 3U;
  parameters.maximum_ground_height_difference = 0.08F;
  const ElevationHoleFiller filler(parameters);
  const std::vector<ElevationNeighbor> neighbors{
      {-1, 0, 0.10F, 0.02F},
      {1, 0, 0.12F, 0.03F},
      {0, 1, 0.11F, 0.01F},
  };

  const ElevationHoleFillResult result = filler.fill(neighbors);

  EXPECT_TRUE(result.filled);
  EXPECT_FLOAT_EQ(result.ground_height, 0.11F);
  EXPECT_FLOAT_EQ(result.height_range, 0.03F);
}

TEST(ElevationHoleFillerTest, RejectsHeightDiscontinuity) {
  ElevationHoleFillParameters parameters;
  parameters.maximum_ground_height_difference = 0.08F;
  const ElevationHoleFiller filler(parameters);
  const std::vector<ElevationNeighbor> neighbors{
      {-1, 0, 0.10F, 0.02F},
      {1, 0, 0.30F, 0.02F},
      {0, 1, 0.11F, 0.02F},
  };

  EXPECT_FALSE(filler.fill(neighbors).filled);
}

TEST(ElevationHoleFillerTest, RejectsOneSidedExtrapolation) {
  const ElevationHoleFiller filler;
  const std::vector<ElevationNeighbor> neighbors{
      {1, 0, 0.10F, 0.02F},
      {1, 1, 0.11F, 0.02F},
      {2, 1, 0.12F, 0.02F},
  };

  EXPECT_FALSE(filler.fill(neighbors).filled);
}

TEST(ElevationHoleFillerTest, PreservesConservativeHeightRange) {
  const ElevationHoleFiller filler;
  const std::vector<ElevationNeighbor> neighbors{
      {-1, 0, 0.10F, 0.01F},
      {1, 0, 0.16F, 0.02F},
      {0, 1, 0.12F, 0.03F},
      {0, -1, 0.14F, 0.02F},
  };

  const ElevationHoleFillResult result = filler.fill(neighbors);

  ASSERT_TRUE(result.filled);
  EXPECT_FLOAT_EQ(result.ground_height, 0.13F);
  EXPECT_NEAR(result.height_range, 0.06F, 1e-6F);
}

TEST(ElevationHoleFillerTest, SupportsReusableScratchStorage) {
  const ElevationHoleFiller filler;
  const std::vector<ElevationNeighbor> neighbors{
      {-1, 0, 0.10F, 0.01F},
      {1, 0, 0.12F, 0.02F},
      {0, 1, 0.11F, 0.01F},
  };
  std::vector<float> scratch;
  scratch.reserve(24U);
  const std::size_t original_capacity = scratch.capacity();

  const ElevationHoleFillResult first = filler.fill(neighbors, scratch);
  const ElevationHoleFillResult second = filler.fill(neighbors, scratch);

  EXPECT_TRUE(first.filled);
  EXPECT_TRUE(second.filled);
  EXPECT_EQ(scratch.capacity(), original_capacity);
  EXPECT_FLOAT_EQ(first.ground_height, second.ground_height);
  EXPECT_FLOAT_EQ(first.height_range, second.height_range);
}

TEST(IncrementalElevationCellTest, FusesBoundedFrameHistory) {
  IncrementalElevationCell history;

  ElevationCell first;
  first.observed = true;
  first.ground_height = 0.10F;
  first.minimum_height = 0.08F;
  first.maximum_height = 0.30F;
  first.point_count = 4U;
  history.add(first, 1.0, 2U);

  ElevationCell second = first;
  second.ground_height = 0.14F;
  second.minimum_height = 0.09F;
  second.maximum_height = 0.80F;
  second.point_count = 5U;
  history.add(second, 2.0, 2U);

  const ElevationCell fused = history.fused();
  EXPECT_TRUE(fused.observed);
  EXPECT_EQ(history.size(), 2U);
  EXPECT_EQ(fused.point_count, 9U);
  EXPECT_FLOAT_EQ(fused.ground_height, 0.12F);
  EXPECT_FLOAT_EQ(fused.minimum_height, 0.08F);
  EXPECT_FLOAT_EQ(fused.maximum_height, 0.80F);
  EXPECT_FLOAT_EQ(fused.height_range, 0.72F);

  ElevationCell third = first;
  third.ground_height = 0.20F;
  third.minimum_height = 0.18F;
  third.maximum_height = 0.40F;
  history.add(third, 3.0, 2U);

  const ElevationCell bounded = history.fused();
  EXPECT_EQ(history.size(), 2U);
  EXPECT_FLOAT_EQ(bounded.ground_height, 0.17F);
  EXPECT_FLOAT_EQ(bounded.minimum_height, 0.09F);
  EXPECT_FLOAT_EQ(bounded.maximum_height, 0.80F);
}

TEST(IncrementalElevationCellTest, ExpiresOldMeasurements) {
  IncrementalElevationCell history;
  ElevationCell measurement;
  measurement.observed = true;
  measurement.ground_height = 0.10F;
  measurement.minimum_height = 0.08F;
  measurement.maximum_height = 0.30F;
  measurement.point_count = 4U;
  history.add(measurement, 1.0, 5U);
  history.add(measurement, 2.0, 5U);

  EXPECT_TRUE(history.removeOlderThan(1.5));
  EXPECT_EQ(history.size(), 1U);
  EXPECT_TRUE(history.removeOlderThan(2.5));
  EXPECT_TRUE(history.empty());
  EXPECT_FALSE(history.fused().observed);
}

TEST(RollingGridMapTest, PreservesOverlappingPhysicalCellHistory) {
  grid_map::GridMap map({"height"});
  map.setGeometry(
      grid_map::Length(2.0, 2.0),
      1.0,
      grid_map::Position::Zero());
  map["height"].setConstant(
      std::numeric_limits<float>::quiet_NaN());

  const grid_map::Size size = map.getSize();
  std::vector<int> histories(
      static_cast<std::size_t>(size(0) * size(1)), 0);
  const auto linearIndex = [&size](const grid_map::Index& index) {
    return static_cast<std::size_t>(index(0) * size(1) + index(1));
  };

  const grid_map::Position retained_position(0.5, 0.5);
  grid_map::Index retained_index;
  ASSERT_TRUE(map.getIndex(retained_position, retained_index));
  map.at("height", retained_index) = 42.0F;
  histories[linearIndex(retained_index)] = 7;

  std::vector<grid_map::BufferRegion> new_regions;
  ASSERT_TRUE(map.move(grid_map::Position(1.0, 0.0), new_regions));
  for (const grid_map::BufferRegion& region : new_regions) {
    const grid_map::Index start = region.getStartIndex();
    const grid_map::Size region_size = region.getSize();
    for (int row = start(0);
         row < start(0) + region_size(0); ++row) {
      for (int column = start(1);
           column < start(1) + region_size(1); ++column) {
        histories[linearIndex(grid_map::Index(row, column))] = 0;
      }
    }
  }

  grid_map::Index moved_retained_index;
  ASSERT_TRUE(map.getIndex(retained_position, moved_retained_index));
  EXPECT_TRUE(
      (moved_retained_index.array() == retained_index.array()).all());
  EXPECT_FLOAT_EQ(map.at("height", moved_retained_index), 42.0F);
  EXPECT_EQ(histories[linearIndex(moved_retained_index)], 7);

  grid_map::Index new_edge_index;
  ASSERT_TRUE(map.getIndex(
      grid_map::Position(1.5, 0.5), new_edge_index));
  EXPECT_TRUE(std::isnan(map.at("height", new_edge_index)));
  EXPECT_EQ(histories[linearIndex(new_edge_index)], 0);
}

}  // namespace
}  // namespace go2w_local_environment

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
