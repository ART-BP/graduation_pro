#include <cmath>
#include <cstdint>

#include <gtest/gtest.h>

#include <rog_map/observation_model.hpp>

namespace rog_map {
namespace {

float logit(const float probability) {
    return std::log(probability / (1.0F - probability));
}

TEST(ObservationModelTest, CapsCorrelatedHitEvidence) {
    const float updated = applyCappedLogOddsEvidence(
            0.0F,
            100U,
            0U,
            logit(0.9F),
            logit(0.45F),
            logit(0.12F),
            logit(0.98F),
            1,
            1);

    EXPECT_NEAR(updated, logit(0.9F), 1e-6F);
}

TEST(ObservationModelTest, EndpointHitWinsOverRayMiss) {
    const float updated = applyCappedLogOddsEvidence(
            0.0F,
            1U,
            10U,
            logit(0.9F),
            logit(0.45F),
            logit(0.12F),
            logit(0.98F),
            1,
            1);

    EXPECT_NEAR(updated, logit(0.9F), 1e-6F);
}

TEST(ObservationModelTest, RequiresRepeatedMissesToConfirmFreeSpace) {
    const float miss = logit(0.45F);
    const float free_threshold = logit(0.40F);
    float value = 0.0F;

    for (int frame = 0; frame < 2; ++frame) {
        value = applyCappedLogOddsEvidence(
                value, 0U, 1U, logit(0.9F), miss,
                logit(0.12F), logit(0.98F), 1, 1);
    }
    EXPECT_GE(value, free_threshold);

    value = applyCappedLogOddsEvidence(
            value, 0U, 1U, logit(0.9F), miss,
            logit(0.12F), logit(0.98F), 1, 1);
    EXPECT_LT(value, free_threshold);
}

TEST(ObservationModelTest, ClearsSingleHitAfterThreeFreeFrames) {
    const float occupied_threshold = logit(0.85F);
    float value = applyCappedLogOddsEvidence(
            0.0F, 1U, 0U, logit(0.9F), logit(0.45F),
            logit(0.12F), logit(0.98F), 1, 1);
    ASSERT_GE(value, occupied_threshold);

    for (int frame = 0; frame < 2; ++frame) {
        value = applyCappedLogOddsEvidence(
                value, 0U, 1U, logit(0.9F), logit(0.45F),
                logit(0.12F), logit(0.98F), 1, 1);
    }
    EXPECT_GE(value, occupied_threshold);

    value = applyCappedLogOddsEvidence(
            value, 0U, 1U, logit(0.9F), logit(0.45F),
            logit(0.12F), logit(0.98F), 1, 1);
    EXPECT_LT(value, occupied_threshold);
}

TEST(ObservationModelTest, ClampsAtConfiguredLimits) {
    EXPECT_FLOAT_EQ(
            applyCappedLogOddsEvidence(
                    logit(0.98F), 1U, 0U, logit(0.9F), logit(0.45F),
                    logit(0.12F), logit(0.98F), 1, 1),
            logit(0.98F));
    EXPECT_FLOAT_EQ(
            applyCappedLogOddsEvidence(
                    logit(0.12F), 0U, 1U, logit(0.9F), logit(0.45F),
                    logit(0.12F), logit(0.98F), 1, 1),
            logit(0.12F));
}

}  // namespace
}  // namespace rog_map

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
