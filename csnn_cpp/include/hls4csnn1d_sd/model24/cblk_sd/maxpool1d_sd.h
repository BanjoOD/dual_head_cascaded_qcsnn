#ifndef MAXPOOL1D_SD_H
#define MAXPOOL1D_SD_H

#include <hls_stream.h>
#include <ap_fixed.h>
#include "../constants24_sd.h"

namespace hls4csnn1d_cblk_sd {

template<int POOL_SIZE, int STRIDE, int CH, int IN_LEN>
class MaxPool1D_SD {
public:
    void forward(hls::stream<ap_int8_c>& in,
                 hls::stream<ap_int8_c>& out) {
        
        const int OUT_LEN = (IN_LEN - POOL_SIZE) / STRIDE + 1;
        
        // Buffer to store one channel's worth of data
        ap_int8_c buffer[IN_LEN];
        #pragma HLS ARRAY_PARTITION variable=buffer complete
        
        CH_LOOP:
        for (int c = 0; c < CH; ++c) {
            
            // First, read all input data for this channel into buffer
            READ_CHANNEL:
            for (int i = 0; i < IN_LEN; ++i) {
                #pragma HLS PIPELINE II=1
                buffer[i] = in.read();
            }
            
            // Now perform pooling from the buffer
            POOL_LOOP:
            for (int out_pos = 0; out_pos < OUT_LEN; ++out_pos) {
                #pragma HLS PIPELINE II=1
                
                // Calculate window start position
                int window_start = out_pos * STRIDE;
                
                // Find maximum in the pooling window
                ap_int8_c max_val = buffer[window_start];
                
                WINDOW_LOOP:
                for (int w = 1; w < POOL_SIZE; ++w) {
                    ap_int8_c val = buffer[window_start + w];
                    if (val > max_val) {
                        max_val = val;
                    }
                }
                
                out.write(max_val);
            }
        }
    }
};

} // namespace hls4csnn1d_cblk_sd
#endif // MAXPOOL1D_SD_H


// #ifndef MAXPOOL1D_SD_H
// #define MAXPOOL1D_SD_H

// #include <hls_stream.h>
// #include <ap_fixed.h>
// #include "../constants_sd.h"

// namespace hls4csnn1d_cblk_sd {

// /**
//  * Channel‑major MaxPool1D
//  *   POOL_SIZE   – size of pooling window
//  *   STRIDE      – stride between windows
//  *   CH          – number of channels
//  *
//  * Input order:  ch0 all time‑steps, then ch1, …
//  * Output order: same (channel‑major)
//  */
// template<int POOL_SIZE, int STRIDE, int CH, int IN_LEN>
// class MaxPool1D_SD {
// public:
//     void forward(hls::stream<ap_int8_c>& in,
//                  hls::stream<ap_int8_c>& out) {     // per‑channel length before pool
    
//         const int OUT_LEN = (IN_LEN - POOL_SIZE) / STRIDE + 1;

//         CH_LOOP:
//         for (int c = 0; c < CH; ++c) {
//         #pragma HLS PIPELINE II=1
//             int in_idx = 0;

//             OUT_POS:
//             for (int o = 0; o < OUT_LEN; ++o) {
//                 // read first element of the window
//                 ap_int8_c max_v = in.read();
//                 // rest of the window
//                 for (int p = 1; p < POOL_SIZE; ++p) {
//                     ap_int8_c v = in.read();
//                     if (v > max_v) max_v = v;
//                 }
//                 // skip stride‑pool_size samples
//                 for (int s = 0; s < STRIDE - POOL_SIZE; ++s) {
//                     in.read();
//                 }
//                 // emit pooled value
//                 out.write(max_v);
//                 in_idx += STRIDE;
//             }

//             // If IN_LEN is not exactly OUT_LEN*STRIDE, drop any tail
//             for (int t = in_idx; t < IN_LEN; ++t) in.read();
//         }
//     }




// };

// } // namespace hls4csnn1d_cblk_sd
// #endif // MAXPOOL1D_SD_H
