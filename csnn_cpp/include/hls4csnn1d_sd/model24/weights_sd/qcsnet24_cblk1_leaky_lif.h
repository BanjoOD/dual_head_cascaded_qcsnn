#ifndef QCSNET24_CBLK1_LEAKY_LIF_H
#define QCSNET24_CBLK1_LEAKY_LIF_H

#include <hls_stream.h>
#include <ap_int.h>
#include "../constants24_sd.h"

namespace hls4csnn1d_cblk_sd {

enum { qcsnet24_cblk1_leaky_FRAC_BITS = 12 };
const ap_int<16> qcsnet24_cblk1_leaky_beta_int   = 1212;
const ap_int<16> qcsnet24_cblk1_leaky_theta_int  = 1743;
const ap_int<16> qcsnet24_cblk1_leaky_scale_int  = 161;

} // namespace hls4csnn1d_cblk_sd
#endif // QCSNET24_CBLK1_LEAKY_LIF_H
