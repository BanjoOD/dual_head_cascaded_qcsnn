#ifndef TOP_CLASS_TB_H
#define TOP_CLASS_TB_H


#include "../include/hls4csnn1d_bm/constants.h"
extern "C" {
    void topFunction(hls::stream<axi_fixed_t> &dataStream, hls::stream<axi_fixed_t> &labelStream);
}


#endif



