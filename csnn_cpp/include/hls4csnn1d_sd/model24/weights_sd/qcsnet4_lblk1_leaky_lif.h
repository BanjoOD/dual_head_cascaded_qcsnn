#ifndef QCSNET4_LBLK1_LEAKY_LIF_H
#define QCSNET4_LBLK1_LEAKY_LIF_H

#include <hls_stream.h>
#include <ap_int.h>
#include "../constants24_sd.h"

namespace hls4csnn1d_cblk_sd {

enum { qcsnet4_lblk1_leaky_FRAC_BITS = 12 };
const ap_int<16> qcsnet4_lblk1_leaky_beta_int   = 3865;
const ap_int<16> qcsnet4_lblk1_leaky_theta_int  = 673;
const ap_int<16> qcsnet4_lblk1_leaky_scale_int  = 21;

} // namespace hls4csnn1d_cblk_sd
#endif // QCSNET4_LBLK1_LEAKY_LIF_H
