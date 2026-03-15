#ifndef QCSNET24_CBLK2_LEAKY_LIF_H
#define QCSNET24_CBLK2_LEAKY_LIF_H

#include <hls_stream.h>
#include <ap_int.h>
#include "../constants24_sd.h"

namespace hls4csnn1d_cblk_sd {

enum { qcsnet24_cblk2_leaky_FRAC_BITS = 12 };
const ap_int<16> qcsnet24_cblk2_leaky_beta_int   = 2962;
const ap_int<16> qcsnet24_cblk2_leaky_theta_int  = 1055;
const ap_int<16> qcsnet24_cblk2_leaky_scale_int  = 27;

} // namespace hls4csnn1d_cblk_sd
#endif // QCSNET24_CBLK2_LEAKY_LIF_H
