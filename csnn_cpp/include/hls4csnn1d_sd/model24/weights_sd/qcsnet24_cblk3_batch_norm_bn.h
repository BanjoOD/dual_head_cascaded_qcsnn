#ifndef QCSNET24_CBLK3_BATCH_NORM_BN_H
#define QCSNET24_CBLK3_BATCH_NORM_BN_H

#include <hls_stream.h>
#include <ap_int.h>
#include "../constants24_sd.h"

namespace hls4csnn1d_cblk_sd {

const int qcsnet24_cblk3_batch_norm_C = 24;

const ap_int8_c qcsnet24_cblk3_batch_norm_weight[24] = {
  103, 103, 109, 102, 122, 101, 108, 109, 100, 109, 106, 105, 110, 107, 112, 106,
    117, 126, 105, 103, 118, 100, 127, 106
};

const ap_int<32> qcsnet24_cblk3_batch_norm_bias[24] = {
  -1466, -1698, -1187, 218, 2029, -2039, -2110, -1469, -2309, -327, -1727, -1036, -335, -2, -1568, 233,
    -554, -733, -1103, -2268, 208, -1724, -150, -2306
};

const ap_int<32> qcsnet24_cblk3_batch_norm_scale_multiplier[24] = {
  1517596888, 1517596888, 1517596888, 1517596888, 1517596888, 1517596888, 1517596888, 1517596888, 1517596888, 1517596888, 1517596888, 1517596888, 1517596888, 1517596888, 1517596888, 1517596888,
    1517596888, 1517596888, 1517596888, 1517596888, 1517596888, 1517596888, 1517596888, 1517596888
};

const int qcsnet24_cblk3_batch_norm_right_shift[24] = {
  37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
    37, 37, 37, 37, 37, 37, 37, 37
};

} // namespace hls4csnn1d_cblk_sd
#endif // QCSNET24_CBLK3_BATCH_NORM_BN_H
