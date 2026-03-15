#ifndef LIF_PARAMS_RUNTIME_H
#define LIF_PARAMS_RUNTIME_H

#include <ap_fixed.h>
#include "../constants_sd.h"
// typedef ap_fixed<8, 4, AP_RND, AP_SAT> ap_fixed_c;

// Will be defined by the testbench (for C/RTL simulation only)
extern ap_fixed_c g_cblk1_beta,  g_cblk1_theta;   // LIF in conv block 1
extern ap_fixed_c g_cblk2_beta,  g_cblk2_theta;   // LIF in conv block 2
extern ap_fixed_c g_lblk1_beta,  g_lblk1_theta;   // LIF in linear block

#endif
