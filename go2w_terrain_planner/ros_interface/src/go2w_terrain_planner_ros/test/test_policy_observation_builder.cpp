#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "go2w_terrain_planner_ros/grid_map_decoder.hpp"
#include "go2w_terrain_planner_ros/policy_observation_builder.hpp"

namespace go2w_terrain_planner_ros {
namespace {

FourChannelMap constantMap() {
  FourChannelMap map;
  std::fill(map.channel[0].begin(), map.channel[0].end(), 0.2F);
  std::fill(map.channel[1].begin(), map.channel[1].end(), 0.1F);
  std::fill(map.channel[2].begin(), map.channel[2].end(), 1.0F);
  std::fill(map.channel[3].begin(), map.channel[3].end(), 1.0F);
  return map;
}

std_msgs::Float32MultiArray makeColumnMajorLayer() {
  std_msgs::Float32MultiArray layer;
  layer.layout.dim.resize(2);
  layer.layout.dim[0].label = "column_index";
  layer.layout.dim[0].size = kMapSize;
  layer.layout.dim[0].stride = kCellCount;
  layer.layout.dim[1].label = "row_index";
  layer.layout.dim[1].size = kMapSize;
  layer.layout.dim[1].stride = kMapSize;
  layer.data.assign(kCellCount, 0.0F);
  return layer;
}

void setCircularValue(
    std_msgs::Float32MultiArray& layer,
    const int logical_row,
    const int logical_column,
    const int row_start,
    const int column_start,
    const float value) {
  const int buffer_row = (logical_row + row_start) % kMapSize;
  const int buffer_column = (logical_column + column_start) % kMapSize;
  layer.data[static_cast<std::size_t>(buffer_column) * kMapSize +
             buffer_row] = value;
}

TEST(PolicyObservationBuilder, ResetMatchesTrainingLayout) {
  PolicyObservationBuilder builder;
  ObservationStatistics statistics;
  const std::vector<float> observation = builder.update(
      constantMap(), Pose2d(), 0.0F, {{0.0F, 0.0F}}, {{0.0F, 0.0F}},
      {{1.0F, 0.0F}}, &statistics);

  ASSERT_EQ(kPolicyObservationDimension, observation.size());
  EXPECT_FLOAT_EQ(0.2F, observation[0]);
  EXPECT_FLOAT_EQ(0.1F, observation[kCellCount]);
  EXPECT_FLOAT_EQ(1.0F, observation[2U * kCellCount]);
  EXPECT_FLOAT_EQ(1.0F, observation[3U * kCellCount]);
  EXPECT_FLOAT_EQ(0.0F, observation[4U * kCellCount]);
  EXPECT_FLOAT_EQ(0.0F, observation[5U * kCellCount]);
  EXPECT_FLOAT_EQ(1.0F, observation[6U * kCellCount]);
  EXPECT_FLOAT_EQ(0.1F, observation[7U * kCellCount]);
  EXPECT_FLOAT_EQ(0.0F, observation[7U * kCellCount + 1U]);
  EXPECT_FLOAT_EQ(1.0F, observation[7U * kCellCount + 2U]);
  EXPECT_FLOAT_EQ(1.0F, statistics.observed_ratio);
  EXPECT_FLOAT_EQ(1.0F, statistics.height_valid_ratio);
  EXPECT_FLOAT_EQ(1.0F, statistics.goal_distance_m);
}

TEST(PolicyObservationBuilder, StoresPhysicalCommandAndBodyMotionHistory) {
  PolicyObservationBuilder builder;
  builder.update(
      constantMap(), Pose2d(), 0.0F, {{0.0F, 0.0F}}, {{0.0F, 0.0F}},
      {{1.0F, 0.0F}});
  Pose2d moved;
  moved.x = 0.1F;
  const std::vector<float> observation = builder.update(
      constantMap(), moved, 0.0F, {{0.6F, -0.5F}}, {{0.2F, 0.1F}},
      {{1.0F, 0.0F}});

  constexpr std::size_t kAuxiliaryStart = 7U * kCellCount;
  constexpr std::size_t kCommandStart = kAuxiliaryStart + 5U;
  constexpr std::size_t kMotionStart = kCommandStart + 16U;
  EXPECT_FLOAT_EQ(0.5F, observation[kCommandStart + 14U]);
  EXPECT_FLOAT_EQ(-0.5F, observation[kCommandStart + 15U]);
  EXPECT_NEAR(0.02F, observation[kMotionStart + 21U], 1.0e-6F);
  EXPECT_FLOAT_EQ(0.0F, observation[kMotionStart + 22U]);
  EXPECT_FLOAT_EQ(0.0F, observation[kMotionStart + 23U]);
}

TEST(GridMapDecoder, RestoresCircularBufferAndRobotAxisDirection) {
  grid_map_msgs::GridMap message;
  message.info.resolution = 0.05;
  message.info.length_x = 10.0;
  message.info.length_y = 10.0;
  message.info.pose.orientation.w = 1.0;
  message.outer_start_index = 7;
  message.inner_start_index = 11;
  message.layers = {"ground_height", "height_range", "observed_mask"};
  message.data = {
      makeColumnMajorLayer(), makeColumnMajorLayer(), makeColumnMajorLayer()};
  for (int row = 0; row < kMapSize; ++row) {
    const float world_x = 4.975F - 0.05F * static_cast<float>(row);
    for (int column = 0; column < kMapSize; ++column) {
      setCircularValue(message.data[0], row, column, 7, 11,
                       0.1F * world_x);
      setCircularValue(message.data[1], row, column, 7, 11, 0.3F);
      setCircularValue(message.data[2], row, column, 7, 11, 1.0F);
    }
  }

  GridMapDecoder decoder;
  FourChannelMap decoded;
  std::string error;
  ASSERT_TRUE(decoder.decodeRobotCentric(
      message, Pose2d(), decoded, error)) << error;
  const std::size_t back_right = 0U;
  const std::size_t front_left = kCellCount - 1U;
  EXPECT_NEAR(-0.4975F, decoded.channel[0][back_right], 1.0e-5F);
  EXPECT_NEAR(0.4975F, decoded.channel[0][front_left], 1.0e-5F);
  EXPECT_NEAR(0.1F, decoded.channel[1][back_right], 1.0e-6F);
  EXPECT_FLOAT_EQ(1.0F, decoded.channel[2][front_left]);
  EXPECT_FLOAT_EQ(1.0F, decoded.channel[3][front_left]);
}

}  // namespace
}  // namespace go2w_terrain_planner_ros

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
