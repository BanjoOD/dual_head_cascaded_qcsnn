#ifndef QCSNET24_CBLK3_LEAKY_LIF_H
#define QCSNET24_CBLK3_LEAKY_LIF_H

#include <hls_stream.h>
#include <ap_int.h>
#include "../constants24_sd.h"

namespace hls4csnn1d_cblk_sd {

enum { qcsnet24_cblk3_leaky_FRAC_BITS = 12 };
const ap_int<16> qcsnet24_cblk3_leaky_beta_int   = 4097;
const ap_int<16> qcsnet24_cblk3_leaky_theta_int  = 525;
const ap_int<16> qcsnet24_cblk3_leaky_scale_int  = 20;

} // namespace hls4csnn1d_cblk_sd
#endif // QCSNET24_CBLK3_LEAKY_LIF_H
