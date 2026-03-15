#ifndef LINEAR1D_SD_H
#define LINEAR1D_SD_H

#include <hls_stream.h>
#include <ap_int.h>
#include "../constants24_sd.h"
#ifndef __SYNTHESIS__
#include <iostream>
#endif

namespace hls4csnn1d_cblk_sd {

/**
 * Fully-connected layer with quantized integer operations
 * Rewritten to be more resource-efficient:
 *  - Fold over INPUT_SIZE (sequential accumulation)
 *  - Parallelize over OUTPUT_SIZE (cheap dimension)
 */
template<int INPUT_SIZE, int OUTPUT_SIZE,
         bool USE_BIAS = false,
         bool USE_ASYMMETRIC = false>
class Linear1D_SD {
public:
    Linear1D_SD() {}

    void forward(hls::stream<ap_int8_c>& in_stream,
                 hls::stream<ap_int8_c>& out_stream,
                 const ap_int8_c weights[OUTPUT_SIZE][INPUT_SIZE],
                 const ap_int<32> scale_multiplier[OUTPUT_SIZE],
                 const int right_shift[OUTPUT_SIZE],
                 const acc32_t bias[OUTPUT_SIZE],
                 const ap_int<8> input_zero_point,
                 const acc32_t weight_sum[OUTPUT_SIZE]) {
        // Partition only along OUTPUT_SIZE so we can read weights for all outputs in parallel.
        // Do NOT partition along INPUT_SIZE to avoid massive replication.
    #pragma HLS ARRAY_PARTITION variable=weights complete dim=1
    #pragma HLS ARRAY_PARTITION variable=scale_multiplier complete dim=1
    #pragma HLS ARRAY_PARTITION variable=right_shift complete dim=1

    #pragma HLS INTERFACE mode=s_axilite port=scale_multiplier
    #pragma HLS INTERFACE mode=s_axilite port=right_shift
    #pragma HLS INTERFACE mode=s_axilite port=bias
    #pragma HLS INTERFACE mode=s_axilite port=input_zero_point
    #pragma HLS INTERFACE mode=s_axilite port=weight_sum

        // 1) Read input vector
        ap_int8_c in_vec[INPUT_SIZE];
    #pragma HLS ARRAY_PARTITION variable=in_vec complete dim=1

        READ_IN:
        for (int i = 0; i < INPUT_SIZE; ++i) {
        #pragma HLS PIPELINE II=1
            in_vec[i] = in_stream.read();
        }

        // 2) Accumulators for all outputs
        acc32_t acc[OUTPUT_SIZE];
    #pragma HLS ARRAY_PARTITION variable=acc complete dim=1

        INIT_ACC:
        for (int o = 0; o < OUTPUT_SIZE; ++o) {
        #pragma HLS UNROLL
            acc[o] = 0;
        }

        // 3) Core MAC loop:
        //    For each input element, update all outputs in parallel.
        DOT_I:
        for (int i = 0; i < INPUT_SIZE; ++i) {
        #pragma HLS PIPELINE II=1
            ap_int8_c x = in_vec[i];

            DOT_O:
            for (int o = 0; o < OUTPUT_SIZE; ++o) {
            #pragma HLS UNROLL
                acc[o] += (acc32_t)x * (acc32_t)weights[o][i];
            }
        }

#ifndef __SYNTHESIS__
        int lin_sat_hi = 0, lin_sat_lo = 0;
#endif

        // 4) Apply bias, asymmetric correction, requantize and write outputs
        OUT_LOOP:
        for (int o = 0; o < OUTPUT_SIZE; ++o) {
        #pragma HLS PIPELINE II=1
            acc32_t tmp = acc[o];

            if (USE_BIAS) {
                tmp += bias[o];
            }

            if (USE_ASYMMETRIC) {
                tmp -= (acc32_t)input_zero_point * weight_sum[o];
            }

            ap_int8_c y_q = requantize(tmp, scale_multiplier[o], right_shift[o]);

        #ifndef __SYNTHESIS__
            if (y_q == ap_int8_c(127))  ++lin_sat_hi;
            if (y_q == ap_int8_c(-128)) ++lin_sat_lo;
        #endif

            out_stream.write(y_q);
        }

    #ifndef __SYNTHESIS__
        static int lin_once = 0;
        if (lin_once++ < 1) {
            std::cout << "[Linear1D_SD] sat_hi=" << lin_sat_hi
                      << " sat_lo=" << lin_sat_lo << "\n";
        }
    #endif
    }
};

} // namespace hls4csnn1d_cblk_sd
#endif // LINEAR1D_SD_H






// #ifndef LINEAR1D_SD_H
// #define LINEAR1D_SD_H

// #include <hls_stream.h>
// #include <ap_int.h>
// #include "../constants4_sd.h"
// #ifndef __SYNTHESIS__
// #include <iostream>
// #endif

// namespace hls4csnn1d_cblk_sd {


// /**
//  * Fully-connected layer with quantized integer operations
//  * Template parameters control optional bias and asymmetric quantization
//  */
// template<int INPUT_SIZE, int OUTPUT_SIZE,
//          bool USE_BIAS = false,
//          bool USE_ASYMMETRIC = false>
// class Linear1D_SD {
// public:
//     Linear1D_SD() {}

//     void forward(hls::stream<ap_int8_c>& in_stream,
//                  hls::stream<ap_int8_c>& out_stream,
//                  const ap_int8_c weights[OUTPUT_SIZE][INPUT_SIZE],
//                  const ap_int<32> scale_multiplier[OUTPUT_SIZE],
//                  const int right_shift[OUTPUT_SIZE],
//                  const acc32_t bias[OUTPUT_SIZE],
//                  const ap_int<8> input_zero_point,
//                  const acc32_t weight_sum[OUTPUT_SIZE]) {
//     #pragma HLS ARRAY_PARTITION variable=weights complete dim=1
//     #pragma HLS ARRAY_PARTITION variable=weights complete dim=2
//     #pragma HLS ARRAY_PARTITION variable=scale_multiplier complete dim=1
//     #pragma HLS ARRAY_PARTITION variable=right_shift complete dim=1
//     #pragma HLS INTERFACE mode=s_axilite port=scale_multiplier
//     #pragma HLS INTERFACE mode=s_axilite port=right_shift
//     #pragma HLS INTERFACE mode=s_axilite port=bias
//     #pragma HLS INTERFACE mode=s_axilite port=input_zero_point
//     #pragma HLS INTERFACE mode=s_axilite port=weight_sum

//         // 1) Read input vector
//         ap_int8_c in_vec[INPUT_SIZE];
//     #pragma HLS ARRAY_PARTITION variable=in_vec complete

//         READ_IN:
//         for (int i = 0; i < INPUT_SIZE; ++i) {
//         #pragma HLS PIPELINE II=1
//             in_vec[i] = in_stream.read();
//         }

// #ifndef __SYNTHESIS__
//         int lin_sat_hi = 0, lin_sat_lo = 0;
// #endif

//         // 2) Compute output: MAC + requantization
//         DOT_OUT:
//         for (int o = 0; o < OUTPUT_SIZE; ++o) {
//         #pragma HLS PIPELINE II=1
//             acc32_t acc = 0;

//             // Integer MAC (int8 × int8 → int32)
//             MUL_ADD:
//             for (int i = 0; i < INPUT_SIZE; ++i) {
//             #pragma HLS UNROLL
//                 acc += (acc32_t)weights[o][i] * (acc32_t)in_vec[i];
//             }

//             // Optional: Add bias (already in accumulator domain)
//             if (USE_BIAS) {
//                 acc += bias[o];
//             }

//             // Optional: Asymmetric quantization correction
//             if (USE_ASYMMETRIC) {
//                 acc -= (acc32_t)input_zero_point * weight_sum[o];
//             }

//             // Requantize to next layer's int8 scale
//             ap_int8_c y_q = requantize(acc, scale_multiplier[o], right_shift[o]);

// #ifndef __SYNTHESIS__
//             if (y_q == ap_int8_c(127))  ++lin_sat_hi;
//             if (y_q == ap_int8_c(-128)) ++lin_sat_lo;
// #endif
//             out_stream.write(y_q);
//         }

// #ifndef __SYNTHESIS__
//         static int lin_once = 0;
//         if (lin_once++ < 1) {
//             std::cout << "[Linear1D_SD] sat_hi=" << lin_sat_hi
//                       << " sat_lo=" << lin_sat_lo << "\n";
//         }
// #endif
//     }
// };

// } // namespace hls4csnn1d_cblk_sd
// #endif // LINEAR1D_SD_H 
