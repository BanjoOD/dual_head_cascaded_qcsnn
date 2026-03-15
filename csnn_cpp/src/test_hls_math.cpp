// #ifndef _AP_UNUSED_PARAM
// #define _AP_UNUSED_PARAM(x) (void)(x)
// #endif

// #include <hls_math.h>
// #include <iostream>

// #include "../include/hls4csnn1d/layer1d.h"

// /**
//  * Use the hls4csnn1d namespace for brevity.
//  */
// using namespace hls4csnn1d;

// int main() {

//     WeightsContainer container;
//     std::cout << "WeightsContainer successfully created!" << std::endl;
//     // float x = 4.0;
//     // std::cout << "Square root of " << x << " is " << hls::sqrt(x) << std::endl;
//     return 0;
// }


// bit_accurate_sim.cpp - Run this BEFORE HLS!
#include <iostream>
#include <cmath>
#include "ap_fixed.h"  // Use same Xilinx types but run on CPU

// CRITICAL: Match Brevitas quantization EXACTLY
template<int BITS, int INT_BITS>
class BrevitasToHLS {
    // Brevitas uses scale + zero_point quantization
    // HLS uses fixed-point with implicit scaling
    
    typedef ap_fixed<BITS, INT_BITS> fixed_t;
    
    float scale;
    int zero_point;
    int quant_min, quant_max;
    
public:
    BrevitasToHLS(float brevitas_scale, int brevitas_zero_point) {
        scale = brevitas_scale;
        zero_point = brevitas_zero_point;
        quant_min = -(1 << (BITS-1));
        quant_max = (1 << (BITS-1)) - 1;
    }
    
    fixed_t quantize_like_brevitas(float value) {
        // Exactly match Brevitas quantization scheme
        int q = round(value / scale) + zero_point;
        q = std::max(quant_min, std::min(quant_max, q));
        
        // Convert to ap_fixed
        float dequant = (q - zero_point) * scale;
        return fixed_t(dequant);
    }
};

