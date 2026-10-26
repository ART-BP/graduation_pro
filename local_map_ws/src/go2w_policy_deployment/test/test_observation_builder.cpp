#include "go2w_policy_deployment/observation_builder.hpp"

#include <cmath>

#include <gtest/gtest.h>

namespace go2w_policy_deployment {
namespace {

PolicyMap uniformMap(
    const int size,
    const float ground,
    const float height_range,
    const float observed,
    const float valid) {
  PolicyMap map(size);
  for (int row = 0; row < size; ++row) {
    for (int column = 0; column < size; ++column) {
      map.at(0, row, column) = ground;
      map.at(1, row, column) = height_range;
      map.at(2, row, column) = observed;
      map.at(3, row, column) = valid;
    }
  }
  return map;
}

ObservationConfig smallConfig() {
  ObservationConfig config;
  config.map_size = 4;
  config.observation_fusion_length = 4;
  config.command_history_length = 2;
  config.motion_history_length = 2;
  return config;
}

TEST(ObservationBuilderTest, DefaultLayoutMatchesExportedModel) {
  ObservationBuilder builder;
  EXPECT_EQ(280045U, builder.expectedDimension());
}

TEST(ObservationBuilderTest, BuildsFiniteChannelMajorObservation) {
  const ObservationConfig config = smallConfig();
  ObservationBuilder builder(config);
  const PolicyMap map = uniformMap(4, 0.5F, 0.25F, 1.0F, 1.0F);
  const ObservationResult result = builder.build(
      map, Pose2d{}, 0.0, 5.0, 0.0, 0.6, -0.5);

  ASSERT_EQ(builder.expectedDimension(), result.values.size());
  const std::size_t cells = 16U;
  EXPECT_FLOAT_EQ(0.5F, result.values[0U * cells]);
  EXPECT_FLOAT_EQ(0.25F, result.values[1U * cells]);
  EXPECT_FLOAT_EQ(1.0F, result.values[2U * cells]);
  EXPECT_FLOAT_EQ(1.0F, result.values[3U * cells]);
  EXPECT_FLOAT_EQ(0.0F, result.values[4U * cells]);
  EXPECT_FLOAT_EQ(0.0F, result.values[5U * cells]);
  EXPECT_FLOAT_EQ(1.0F, result.values[6U * cells]);
  EXPECT_DOUBLE_EQ(1.0, result.observed_ratio);
  EXPECT_DOUBLE_EQ(1.0, result.height_valid_ratio);

  const std::size_t auxiliary = 7U * cells;
  EXPECT_FLOAT_EQ(0.5F, result.values[auxiliary + 0U]);
  EXPECT_NEAR(0.0F, result.values[auxiliary + 1U], 1.0e-6F);
  EXPECT_FLOAT_EQ(1.0F, result.values[auxiliary + 2U]);
  EXPECT_FLOAT_EQ(0.5F, result.values[auxiliary + 3U]);
  EXPECT_FLOAT_EQ(-0.5F, result.values[auxiliary + 4U]);
  for (const float value : result.values) {
    EXPECT_TRUE(std::isfinite(value));
  }
}

TEST(ObservationBuilderTest, AdvancesCommandAndBodyMotionHistory) {
  const ObservationConfig config = smallConfig();
  ObservationBuilder builder(config);
  const PolicyMap map = uniformMap(4, 0.0F, 0.0F, 1.0F, 1.0F);
  builder.build(map, Pose2d{}, 0.0, 1.0, 0.0, 0.0, 0.0);
  builder.recordCommand(0.6, 0.5);

  Pose2d moved;
  moved.x = 1.0;
  const ObservationResult result = builder.build(
      map, moved, 0.0, 2.0, 0.0, 0.0, 0.0);
  const std::size_t auxiliary = 7U * 16U;
  const std::size_t command_start = auxiliary + 5U;
  EXPECT_FLOAT_EQ(0.0F, result.values[command_start + 0U]);
  EXPECT_FLOAT_EQ(0.0F, result.values[command_start + 1U]);
  EXPECT_FLOAT_EQ(0.5F, result.values[command_start + 2U]);
  EXPECT_FLOAT_EQ(0.5F, result.values[command_start + 3U]);

  const std::size_t motion_start = command_start + 4U;
  EXPECT_FLOAT_EQ(0.0F, result.values[motion_start + 0U]);
  EXPECT_FLOAT_EQ(0.0F, result.values[motion_start + 1U]);
  EXPECT_FLOAT_EQ(0.0F, result.values[motion_start + 2U]);
  EXPECT_FLOAT_EQ(0.2F, result.values[motion_start + 3U]);
  EXPECT_FLOAT_EQ(0.0F, result.values[motion_start + 4U]);
  EXPECT_FLOAT_EQ(0.0F, result.values[motion_start + 5U]);
}

TEST(ObservationBuilderTest, MakesUnknownCellsExplicit) {
  const ObservationConfig config = smallConfig();
  ObservationBuilder builder(config);
  const PolicyMap map = uniformMap(4, 0.0F, 0.0F, 0.0F, 0.0F);
  const ObservationResult result = builder.build(
      map, Pose2d{}, 0.0, 0.0, 0.0, 0.0, 0.0);
  const std::size_t cells = 16U;
  EXPECT_FLOAT_EQ(1.0F, result.values[5U * cells]);
  EXPECT_FLOAT_EQ(0.0F, result.values[6U * cells]);
  EXPECT_DOUBLE_EQ(0.0, result.observed_ratio);
  EXPECT_DOUBLE_EQ(0.0, result.height_valid_ratio);
}

}  // namespace
}  // namespace go2w_policy_deployment

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
