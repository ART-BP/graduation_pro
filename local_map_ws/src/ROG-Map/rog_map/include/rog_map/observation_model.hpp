#pragma once

#include <algorithm>
#include <cstdint>

namespace rog_map {

inline float applyCappedLogOddsEvidence(
        const float current,
        const std::uint16_t hit_count,
        const std::uint16_t miss_count,
        const float log_odds_hit,
        const float log_odds_miss,
        const float log_odds_min,
        const float log_odds_max,
        const int max_hit_updates,
        const int max_miss_updates) {
    // Endpoint evidence wins when a voxel is both crossed and hit in one batch.
    // Counts are capped so correlated points from one scan cannot immediately
    // saturate the persistent occupancy probability.
    if (hit_count > 0U) {
        const int evidence_count = std::min<int>(hit_count, max_hit_updates);
        return std::min(
                log_odds_max,
                current + log_odds_hit * static_cast<float>(evidence_count));
    }

    const int evidence_count = std::min<int>(miss_count, max_miss_updates);
    return std::max(
            log_odds_min,
            current + log_odds_miss * static_cast<float>(evidence_count));
}

}  // namespace rog_map
