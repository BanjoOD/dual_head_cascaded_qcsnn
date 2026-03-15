#ifndef TOP_CLASS_CBLK1_TB_H
#define TOP_CLASS_CBLK1_TB_H


#include "../../include/hls4csnn1d_bm/constants.h"
extern "C" {
    void topFunctionCblk1(hls::stream<axi_fixed_t> &dataStream, hls::stream<axi_fixed_t> &labelStream, const ap_fixed_c *weight_ddr);
}


#endif

