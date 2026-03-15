#ifndef RR_NORMALIZATION_STATS_H
#define RR_NORMALIZATION_STATS_H

/**
 * RR Feature Normalization Statistics
 * 
 * These values are computed from the TRAINING set and must be used
 * for both training and test/inference to avoid data leakage.
 * 
 * Normalization formula:
 *   rr_normalized[i] = (rr_raw[i] - RR_MEAN[i]) / RR_STD[i]
 *                    = (rr_raw[i] - RR_MEAN[i]) * RR_STD_INV[i]
 * 
 * Input layout (184 features):
 *   [0-179]:   ECG beat samples (already z-scored per beat in CSV)
 *   [180-183]: RR interval features (need normalization)
 */

namespace hls4csnn1d_cblk_sd {

// RR feature indices
static const int RR_FEATURE_START = 180;
static const int RR_FEATURE_COUNT = 4;

// Training set mean for each RR feature
static const float RR_MEAN[4] = {
    0.7758959532f,   // RR_prev
    0.7736818790f,   // RR_next
    1.0521744490f,   // RR_ratio
    0.0022126297f    // RR_diff
};

// Training set std for each RR feature
static const float RR_STD[4] = {
    0.2283842415f,   // RR_prev
    0.2244229764f,   // RR_next
    0.4578237832f,   // RR_ratio
    0.2252828181f    // RR_diff
};

// Precomputed 1/std for faster computation (multiply instead of divide)
static const float RR_STD_INV[4] = {
    4.3785858154f,   // 1/RR_prev_std
    4.4558715820f,   // 1/RR_next_std
    2.1842465401f,   // 1/RR_ratio_std
    4.4388647079f    // 1/RR_diff_std
};

} // namespace hls4csnn1d_cblk_sd

#endif // RR_NORMALIZATION_STATS_H
