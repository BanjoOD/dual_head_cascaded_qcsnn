#ifndef QPARAMS_QCSNET24_CBLK1_INPUT_H
#define QPARAMS_QCSNET24_CBLK1_INPUT_H

#include <ap_int.h>

#include "../constants24_sd.h"

namespace hls4csnn1d_cblk_sd {

// Activation quantization parameters (optional for kernels)
const int   qcsnet24_cblk1_input_bit_width = 8;
const float qcsnet24_cblk1_input_scale     = 0.03954027593;  // kept for reference only
const ap_int<16> qcsnet24_cblk1_input_act_scale_int = 162;
const int   qcsnet24_cblk1_input_zero_point= 0;

} // namespace
#endif // QPARAMS_QCSNET24_CBLK1_INPUT_H
