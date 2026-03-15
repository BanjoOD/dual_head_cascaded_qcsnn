#ifndef QCSNET24_CBLK1_BATCH_NORM_BN_H
#define QCSNET24_CBLK1_BATCH_NORM_BN_H

#include <hls_stream.h>
#include <ap_int.h>
#include "../constants24_sd.h"

namespace hls4csnn1d_cblk_sd {

const int qcsnet24_cblk1_batch_norm_C = 16;

const ap_int8_c qcsnet24_cblk1_batch_norm_weight[16] = {
  109, 110, 115, 114, 127, 109, 114, 116, 108, 117, 100, 111, 119, 114, 100, 115
};

const ap_int<32> qcsnet24_cblk1_batch_norm_bias[16] = {
  114, -138, -168, -203, 454, -174, -224, 20, -334, 20, -330, 8, 10, -105, -256, 1
};

const ap_int<32> qcsnet24_cblk1_batch_norm_scale_multiplier[16] = {
  1215296329, 1215296329, 1215296329, 1215296329, 1215296329, 1215296329, 1215296329, 1215296329, 1215296329, 1215296329, 1215296329, 1215296329, 1215296329, 1215296329, 1215296329, 1215296329
};

const int qcsnet24_cblk1_batch_norm_right_shift[16] = {
  37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37
};

} // namespace hls4csnn1d_cblk_sd
#endif // QCSNET24_CBLK1_BATCH_NORM_BN_H
