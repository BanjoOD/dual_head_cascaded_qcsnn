#ifndef BN_STRUCT_SD_H
#define BN_STRUCT_SD_H

#include "../constants24_sd.h"
 
namespace hls4csnn1d_cblk_sd {

typedef ap_fixed<8, 4, AP_RND, AP_SAT> bn_stat_t;
// typedef ap_fixed<32, 26, AP_RND, AP_SAT> bn_stat_t; 


template<int CH>
struct BNParams {
    ap_fixed_c gamma[CH];
    ap_fixed_c beta[CH];
    bn_stat_t mean[CH];
    bn_stat_t var[CH];
};

}
#endif
