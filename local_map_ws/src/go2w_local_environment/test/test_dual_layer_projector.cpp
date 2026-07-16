#include "go2w_local_environment/dual_layer_projector.hpp"

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

namespace go2w_local_environment {
namespace {

DualLayerProjector makeProjector() {
  ProjectionParameters parameters;
  parameters.voxel_resolution = 0.1;
  parameters.vertical_min_offset = -1.0;
  parameters.vertical_max_offset = 2.0;
  parameters.ground_search_min_offset = -0.8;
  parameters.ground_search_max_offset = 0.3;
  parameters.minimum_clearance = 0.65;
  parameters.maximum_unknown_fraction = 0.2;
  return DualLayerProjector(parameters);
}

TEST(DualLayerProjectorTest, ProjectsFlatGround) {
  const DualLayerProjector projector = makeProjector();
  const ColumnProjection result = projector.project(
      0.4,
      std::vector<float>{0.0F},
      std::vector<float>{},
      true);

  EXPECT_TRUE(result.ground_observed);
  EXPECT_FALSE(result.ceiling_observed);
  EXPECT_TRUE(result.observed);
  EXPECT_FALSE(result.blocked);
  EXPECT_NEAR(result.ground_height, 0.05, 1e-6);
  EXPECT_GT(result.clearance, 1.0);
}

TEST(DualLayerProjectorTest, DetectsLowCeiling) {
  const DualLayerProjector projector = makeProjector();
  const ColumnProjection result = projector.project(
      0.4,
      std::vector<float>{0.0F, 0.6F},
      std::vector<float>{},
      true);

  EXPECT_TRUE(result.ground_observed);
  EXPECT_TRUE(result.ceiling_observed);
  EXPECT_TRUE(result.blocked);
  EXPECT_NEAR(result.clearance, 0.5, 1e-6);
}

TEST(DualLayerProjectorTest, MarksUnknownBodyVolumeUnobserved) {
  ProjectionParameters parameters = makeProjector().parameters();
  parameters.unknown_is_blocked = true;
  const DualLayerProjector projector(parameters);

  const ColumnProjection result = projector.project(
      0.4,
      std::vector<float>{0.0F},
      std::vector<float>{0.2F, 0.3F, 0.4F, 0.5F},
      true);

  EXPECT_TRUE(result.ground_observed);
  EXPECT_FALSE(result.observed);
  EXPECT_TRUE(result.blocked);
  EXPECT_GT(result.unknown_fraction, 0.2);
}

TEST(DualLayerProjectorTest, LeavesMissingGroundInvalid) {
  const DualLayerProjector projector = makeProjector();
  const ColumnProjection result = projector.project(
      0.4,
      std::vector<float>{1.2F},
      std::vector<float>{},
      true);

  EXPECT_FALSE(result.ground_observed);
  EXPECT_FALSE(result.observed);
  EXPECT_TRUE(std::isnan(result.ground_height));
  EXPECT_TRUE(std::isnan(result.clearance));
}

}  // namespace
}  // namespace go2w_local_environment

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
