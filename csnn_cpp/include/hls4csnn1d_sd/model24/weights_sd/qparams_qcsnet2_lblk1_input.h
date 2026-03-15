#ifndef QPARAMS_QCSNET2_LBLK1_INPUT_H
#define QPARAMS_QCSNET2_LBLK1_INPUT_H

#include <ap_int.h>

#include "../constants24_sd.h"

namespace hls4csnn1d_cblk_sd {

// Activation quantization parameters (optional for kernels)
const int   qcsnet2_lblk1_input_bit_width = 8;
// const float qcsnet2_lblk1_input_scale     = 0.03321001679;  // kept for reference only
const ap_int<16> qcsnet2_lblk1_input_act_scale_int = 136;
const int   qcsnet2_lblk1_input_zero_point= 0;

} // namespace
#endif // QPARAMS_QCSNET2_LBLK1_INPUT_H
