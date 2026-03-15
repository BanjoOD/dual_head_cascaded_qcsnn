#ifndef QCSNET24_CBLK2_BATCH_NORM_BN_H
#define QCSNET24_CBLK2_BATCH_NORM_BN_H

#include <hls_stream.h>
#include <ap_int.h>
#include "../constants24_sd.h"

namespace hls4csnn1d_cblk_sd {

const int qcsnet24_cblk2_batch_norm_C = 16;

const ap_int8_c qcsnet24_cblk2_batch_norm_weight[16] = {
  127, 112, 109, 112, 110, 117, 106, 108, 110, 100, 106, 102, 104, 115, 111, 119
};

const ap_int<32> qcsnet24_cblk2_batch_norm_bias[16] = {
  1206, -470, 683, -785, -149, 538, -659, 882, -876, -1191, -776, -342, -21, -420, -374, 159
};

const ap_int<32> qcsnet24_cblk2_batch_norm_scale_multiplier[16] = {
  1302622504, 1302622504, 1302622504, 1302622504, 1302622504, 1302622504, 1302622504, 1302622504, 1302622504, 1302622504, 1302622504, 1302622504, 1302622504, 1302622504, 1302622504, 1302622504
};

const int qcsnet24_cblk2_batch_norm_right_shift[16] = {
  37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37
};

} // namespace hls4csnn1d_cblk_sd
#endif // QCSNET24_CBLK2_BATCH_NORM_BN_H
