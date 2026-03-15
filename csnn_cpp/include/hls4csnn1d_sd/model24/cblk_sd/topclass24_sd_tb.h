#ifndef TOPCLASS24_SD_TB_H
#define TOPCLASS24_SD_TB_H

#include <hls_stream.h>
#include "../constants24_sd.h"

// Top-level DUT
extern "C" {
void topFunction(hls::stream<axi_fixed_t>& dmaInStream,
                 hls::stream<axi_fixed_t>& dmaOut2Stream,
                 hls::stream<axi_fixed_t>& dmaOut4Stream);
}

#endif // TOPCLASS24_SD_TB_H
