#ifndef BATCHNORM1D_SD_H
#define BATCHNORM1D_SD_H

#include <hls_stream.h>
#include <ap_int.h>
#include "../constants24_sd.h"

namespace hls4csnn1d_cblk_sd {

// In batchnorm1d_sd.h

// Saturating cast to int8
static inline ap_int8_c bn_sat_clip_int8(ap_int<32> v) {
    if (v >  127) return ap_int8_c( 127);
    if (v < -128) return ap_int8_c(-128);
    return ap_int8_c(v);
}

// True round-to-nearest-even (banker’s) using absolute value.
// Works identically for positive/negative products.
static inline ap_int8_c requantize_rne_abs(acc32_t acc, ap_int<32> mul, int rshift) {
    ap_int<64> prod = (ap_int<64>)acc * (ap_int<64>)mul;

    if (rshift <= 0) {
        return bn_sat_clip_int8((ap_int<32>)prod);
    }

    // Work on absolute value; remember sign
    ap_int<64> ax;
    if (prod < 0) {
        ax = -prod;
    } else {
        ax = prod;
    }
    

    const ap_uint<64> mask = (ap_uint<64>(1) << rshift) - 1;   // low bits
    const ap_uint<64> half = ap_uint<64>(1) << (rshift - 1);   // 0.5 ulp

    ap_int<64> base = ax >> rshift;                            // floor(|prod| / 2^r)
    ap_uint<64> rem  = (ap_uint<64>)ax & mask;                 // remainder

    // Nearest, ties-to-even on |prod|
    bool bump = false;
    if (rem > half) {
        bump = true;
    } else if (rem == half) {
        bump = ((base & 1) != 0); // bump only if base is odd
    }

    if (bump) base += 1;

    ap_int<64> signed_val;
    if (prod < 0) {
        signed_val = -base;
    } else {
        signed_val = base;
    }

    return bn_sat_clip_int8((ap_int<32>)signed_val);
}


/**
 * BatchNorm1D as QuantScaleBias: y = x * weight + bias
 * Matches Brevitas BatchNorm1dToQuantScaleBias behavior
 * 
 * This is NOT traditional batch normalization with running mean/var.
 * It's the fused scale+bias operation after BN folding.
 */
template<int CH, int LEN,
         bool USE_BIAS = true>
class BatchNorm1D_SD {
public:
    BatchNorm1D_SD() {}

    void forward(hls::stream<ap_int8_c>& in_stream,
                 hls::stream<ap_int8_c>& out_stream,
                 const ap_int8_c weight[CH],
                 const ap_int<32> bias[CH],
                 const ap_int<32> scale_multiplier[CH],
                 const int right_shift[CH]) {
    #pragma HLS ARRAY_PARTITION variable=weight complete dim=1
    #pragma HLS ARRAY_PARTITION variable=bias complete dim=1
    #pragma HLS ARRAY_PARTITION variable=scale_multiplier complete dim=1
    #pragma HLS ARRAY_PARTITION variable=right_shift complete dim=1
    #pragma HLS INTERFACE mode=s_axilite port=weight
    #pragma HLS INTERFACE mode=s_axilite port=bias
    #pragma HLS INTERFACE mode=s_axilite port=scale_multiplier
    #pragma HLS INTERFACE mode=s_axilite port=right_shift

        // Process in channel-first order: [CH][LEN]
        CHANNEL_LOOP:
        for (int c = 0; c < CH; ++c) {
            FEATURE_LOOP:
            for (int i = 0; i < LEN; ++i) {
            #pragma HLS PIPELINE II=1
                ap_int8_c x = in_stream.read();
                
                // Compute: y = x * weight[c] + bias[c]
                // All in int8/int32 domain
                acc32_t acc = (acc32_t)x * (acc32_t)weight[c];
                
                if (USE_BIAS) {
                    acc += (acc32_t)bias[c];
                }
                
                // Requantize to next layer's scale
                ap_int8_c y = requantize_rne_abs(acc, scale_multiplier[c], right_shift[c]);
                
                out_stream.write(y);
            }
        }
    }
};

} // namespace hls4csnn1d_cblk_sd
#endif // BATCHNORM1D_SD_H




// #ifndef BATCHNORM1D_SD_H
// #define BATCHNORM1D_SD_H

// #include <hls_stream.h>
// #include <ap_fixed.h>
// #include "../constants_sd.h"   // keep if you define global constants/types there

// namespace hls4csnn1d_cblk_sd {

// // QFX rounding/overflow: convergent + saturate
// // If ap_fixed_c is already defined elsewhere, ensure it matches AP_CONV.
// // Otherwise, uncomment the typedef below and use it consistently.
// // using ap_fixed_c = ap_fixed<8, 4, AP_CONV, AP_SAT>;
// // In your BN header (channel-major BN using α/η)
// typedef ap_fixed<24, 12, AP_RND_CONV, AP_SAT> bn_mul_t;  // wider mul/add

// template<int CH, int BN_LEN>
// class BatchNorm1D_SD {
// public:
//     BatchNorm1D_SD() = default;

//     // forward(): channel-major order, processes CH × BN_LEN scalars
//     // alpha/eta are the precomputed affine BN coefficients (exported by your Python script)
//     void forward(
//         hls::stream<ap_fixed_c>& in,
//         hls::stream<ap_fixed_c>& out,
//         const ap_fixed_c alpha[CH],
//         const ap_fixed_c eta[CH]) 
//     {
//     #pragma HLS ARRAY_PARTITION variable=alpha complete dim=1
//     #pragma HLS ARRAY_PARTITION variable=eta   complete dim=1

// #ifndef __SYNTHESIS__
//         int bn_sat_hi = 0, bn_sat_lo = 0;
// #endif

//         CHANNEL_LOOP:
//         for (int c = 0; c < CH; ++c) {
//         #pragma HLS PIPELINE II=1
//         FEATURE_LOOP:
//             for (int i = 0; i < BN_LEN; ++i) {
//             #pragma HLS PIPELINE II=1
//                 ap_fixed_c x = in.read();
//                 // QFX BN: y = alpha[c] * x + eta[c]
//                 // Casts on ap_fixed_c enforce convergent rounding + saturation.
//               //   ap_fixed_c y = (ap_fixed_c)((ap_fixed_c)(alpha[c] * x) + eta[c]);
//               //   out.write(y);
//               // widen the product and sum, then cast back once
//                 bn_mul_t y_wide = bn_mul_t(alpha[c]) * bn_mul_t(x) + bn_mul_t(eta[c]);
//                 ap_fixed_c y = ap_fixed_c(y_wide);
// #ifndef __SYNTHESIS__
//                 if (y == ap_fixed_c(1.984375)) ++bn_sat_hi;
//                 if (y == ap_fixed_c(-2.0))   ++bn_sat_lo;
// #endif
//                 out.write(y);
//             }
//         }
// #ifndef __SYNTHESIS__
//             static int bn_once = 0;
//             if (bn_once++ < 1) {
//                 std::cout << "[BN1] sat_hi=" << bn_sat_hi
//                     << " sat_lo=" << bn_sat_lo << "\n";
//             }
// #endif
//     }
// };

// } // namespace hls4csnn1d_cblk_sd
// #endif // BATCHNORM1D_SD_H



// #ifndef BATCHNORM1D_SD_H
// #define BATCHNORM1D_SD_H

// #include <hls_stream.h>
// #include <ap_fixed.h>
// #include "../constants_sd.h"   
// #include "bn_struct_sd.h"                 // ap_fixed_c, etc.

// namespace hls4csnn1d_cblk_sd {

// // ──────────────────────────────────────────────────────────────
// //  Fixed‑point sqrt (Newton–Raphson, 4 iters) for BN
// // ──────────────────────────────────────────────────────────────
// static ap_fixed_c bn_sqrt_fixed(ap_fixed_c x) {
//     #pragma HLS INLINE
//     const ap_fixed_c eps = ap_fixed_c(0.0625);
//     if (x <= eps) return eps;
//     ap_fixed_c a = x;
//     for (int i = 0; i < 4; ++i) {
//         #pragma HLS UNROLL
//         a = ap_fixed_c(0.5) * (a + x / a);
//     }
//     return (a < eps) ? eps : a;
// }



// // ──────────────────────────────────────────────────────────────
// //  Channel‑major BatchNorm layer
// // ──────────────────────────────────────────────────────────────
// template<int CH, int BN_LEN>
// class BatchNorm1D_SD {
// public:
//     BatchNorm1D_SD() {
//         #pragma HLS ARRAY_PARTITION variable=scale complete dim=1
//         #pragma HLS ARRAY_PARTITION variable=shift complete dim=1
//         epsilon = ap_fixed_c(1e-5);
//     }

//     //------------------------------------------------------------------
//     // forward(): process CH×BN_LEN scalars, channel‑major order
//     //------------------------------------------------------------------
// //     void forward(
// //     hls::stream<ap_fixed_c>& in,
// //     hls::stream<ap_fixed_c>& out,
// //     const BNParams<CH>& bn) { // <-- reference to weights struct
// //     #pragma HLS ARRAY_PARTITION variable=bn.gamma complete dim=1
// //     #pragma HLS ARRAY_PARTITION variable=bn.beta  complete dim=1
// //     #pragma HLS ARRAY_PARTITION variable=bn.mean  complete dim=1
// //     #pragma HLS ARRAY_PARTITION variable=bn.var   complete dim=1
   
// //     // ========== DEBUG ADDITION START ==========
// // #ifndef __SYNTHESIS__
// //     static int debug_count = 0;
// //     static bool params_printed = false;
    
// //     if (!params_printed) {
// //         params_printed = true;
// //         std::cout << "\n=== BatchNorm Debug ===\n";
// //         std::cout << "Epsilon: " << epsilon << "\n";
// //         for (int c = 0; c < 3 && c < CH; ++c) {
// //             std::cout << "Channel " << c << ": gamma=" << bn.gamma[c] 
// //                       << ", beta=" << bn.beta[c] 
// //                       << ", mean=" << bn.mean[c] 
// //                       << ", var=" << bn.var[c] << "\n";
// //         }
// //     }
// // #endif
// //     // ========== DEBUG ADDITION END ==========
    
// //     // Precompute scale/shift for each channel
// //     for (int c = 0; c < CH; ++c) {
// //         // BatchNorm1D_SD::forward  (right before inv_std is computed)
// //         const ap_fixed_c VAR_FLOOR = ap_fixed_c(0.0625);   // 1-LSB in Q4.4
// //         ap_fixed_c var_eff = (bn.var[c] < VAR_FLOOR) ? VAR_FLOOR : bn.var[c];

// //         #pragma HLS UNROLL
// //         ap_fixed_c inv_std = ap_fixed_c(1.0) / 
// //                     bn_sqrt_fixed(var_eff);  // Hardcode epsilon here
// //         // ap_fixed_c inv_std = ap_fixed_c(1.0) /
// //         //                     bn_sqrt_fixed(bn.var[c] + epsilon);
// //         scale[c] = bn.gamma[c] * inv_std;
// //         shift[c] = bn.beta[c]  - bn.mean[c] * scale[c];
        
// //         // ========== DEBUG ADDITION START ==========
// // // #ifndef __SYNTHESIS__
// // //         if (c < 3) {
// // //             std::cout << "Channel " << c << " scale=" << scale[c] 
// // //                         << ", shift=" << shift[c] 
// // //                         << ", inv_std=" << inv_std << "\n";
// // //         }
        
// // // #endif
// //         // ========== DEBUG ADDITION END ==========
// //     }

// // CHANNEL_LOOP:
// //     for (int c = 0; c < CH; ++c) {
// //     #pragma HLS PIPELINE II=1
// //     FEATURE_LOOP:
// //         for (int i = 0; i < BN_LEN; ++i) {
// //             #pragma HLS PIPELINE II=1
// //             ap_fixed_c x = in.read();

// // #ifndef __SYNTHESIS__
// //     if  (CH == 16) {          // Conv-1 → BN-1
// //         sum_conv1 [c] += double(x);
// //         sum2_conv1[c] += double(x) * double(x);
// //         cnt_conv1 [c] += 1;
// //     } else if (CH == 24) {   // Conv-2 → BN-2
// //         sum_conv2 [c] += double(x);
// //         sum2_conv2[c] += double(x) * double(x);
// //         cnt_conv2 [c] += 1;
// //     }
// // #endif
// //             ap_fixed_c y = scale[c] * x + shift[c];
// //             out.write(y);
            
// //             // ========== DEBUG ADDITION START ==========
// // #ifndef __SYNTHESIS__
// //             if (debug_count < 20) {
// //                 std::cout << "BN input[" << debug_count << "] (ch=" << c 
// //                           << "): " << x << " -> output: " << y << "\n";
// //                 debug_count++;
// //             }
// // #endif
// //             // ========== DEBUG ADDITION END ==========
// //         }
// //     }
// // }


//     void forward(
//         hls::stream<ap_fixed_c>& in,
//         hls::stream<ap_fixed_c>& out,
//         const BNParams<CH>& bn) { // <-- reference to weights struct
//         #pragma HLS ARRAY_PARTITION variable=bn.gamma complete dim=1
//         #pragma HLS ARRAY_PARTITION variable=bn.beta  complete dim=1
//         #pragma HLS ARRAY_PARTITION variable=bn.mean  complete dim=1
//         #pragma HLS ARRAY_PARTITION variable=bn.var   complete dim=1
       
//         // Precompute scale/shift for each channel
//         for (int c = 0; c < CH; ++c) {
//             #pragma HLS UNROLL
//             ap_fixed_c inv_std = ap_fixed_c(1.0) /
//                                 bn_sqrt_fixed(bn.var[c] + epsilon);
//             scale[c] = bn.gamma[c] * inv_std;
//             shift[c] = bn.beta[c]  - bn.mean[c] * scale[c];
//         }

//     CHANNEL_LOOP:
//         for (int c = 0; c < CH; ++c) {
//         #pragma HLS PIPELINE II=1
//         FEATURE_LOOP:
//             for (int i = 0; i < BN_LEN; ++i) {
//                 #pragma HLS PIPELINE II=1
//                 ap_fixed_c x = in.read();
//                 out.write(scale[c] * x + shift[c]);
//             }
//         }
//     }
// private:
//     ap_fixed_c scale[CH], shift[CH];
//     ap_fixed_c epsilon;

// };

// } // namespace hls4csnn1d_cblk_sd
// #endif // BATCHNORM1D_SD_H