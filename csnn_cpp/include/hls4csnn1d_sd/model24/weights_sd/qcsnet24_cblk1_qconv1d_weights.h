#ifndef QCSNET24_CBLK1_QCONV1D_WEIGHTS_H
#define QCSNET24_CBLK1_QCONV1D_WEIGHTS_H

#include <ap_int.h>

#include "../constants24_sd.h"

namespace hls4csnn1d_cblk_sd {

const int qcsnet24_cblk1_qconv1d_OUT_CH = 16;
const int qcsnet24_cblk1_qconv1d_IN_CH  = 1;
const int qcsnet24_cblk1_qconv1d_KERNEL_SIZE = 3;
const int qcsnet24_cblk1_qconv1d_STRIDE = 1;

const ap_int<8> qcsnet24_cblk1_qconv1d_input_zero_point = 0;

const ap_int<32> qcsnet24_cblk1_qconv1d_scale_multiplier[16] = {
  1608539567, 1608539567, 1608539567, 1608539567, 1608539567, 1608539567, 1608539567, 1608539567, 1608539567, 1608539567, 1608539567, 1608539567, 1608539567, 1608539567, 1608539567, 1608539567
};

const int qcsnet24_cblk1_qconv1d_right_shift[16] = {
  38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38
};

const acc32_t qcsnet24_cblk1_qconv1d_bias[16] = {
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

const acc32_t qcsnet24_cblk1_qconv1d_weight_sum[16] = {
  127, 60, 85, 22, 223, 84, -100, 16, -160, -188, 45, 48, 158, -66, 154, 48
};

const ap_int<8> qcsnet24_cblk1_qconv1d_weights[16][1][3] = {
{
  { 85, 76, -34 }
},
{
  { 94, -33, -1 }
},
{
  { -63, 54, 94 }
},
{
  { -83, 86, 19 }
},
{
  { 108, 46, 69 }
},
{
  { -4, 83, 5 }
},
{
  { -57, 15, -58 }
},
{
  { -16, -41, 73 }
},
{
  { -79, -50, -31 }
},
{
  { -55, -6, -127 }
},
{
  { 82, -100, 63 }
},
{
  { 13, -33, 68 }
},
{
  { 27, 100, 31 }
},
{
  { -41, 17, -42 }
},
{
  { 27, 77, 50 }
},
{
  { -44, 65, 27 }
}
};

} // namespace
#endif // QCSNET24_CBLK1_QCONV1D_WEIGHTS_H
