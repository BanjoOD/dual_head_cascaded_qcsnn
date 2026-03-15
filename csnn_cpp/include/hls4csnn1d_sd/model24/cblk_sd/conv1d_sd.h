#ifndef CONV1D_SD_H
#define CONV1D_SD_H

#include "hls_stream.h"
#include "ap_int.h"
#include "../constants24_sd.h"

namespace hls4csnn1d_cblk_sd {
    

template<int IN_CH, int OUT_CH,
         int KERNEL_SIZE, int STRIDE,
         int IN_LEN,
         bool USE_BIAS = false,
         bool USE_ASYMMETRIC = false>
class Conv1D_SD {
public:
    Conv1D_SD() {}

    void forward(hls::stream<ap_int8_c>& in_stream,
                 hls::stream<ap_int8_c>& out_stream,
                 const ap_int8_c weights[OUT_CH][IN_CH][KERNEL_SIZE],
                 const ap_int<32> scale_multiplier[OUT_CH],
                 const int right_shift[OUT_CH],
                 const acc32_t bias[OUT_CH],
                 const ap_int<8> input_zero_point,
                 const acc32_t weight_sum[OUT_CH]) {
    #pragma HLS ARRAY_PARTITION variable=weights complete dim=1
    #pragma HLS ARRAY_PARTITION variable=weights complete dim=2
    #pragma HLS ARRAY_PARTITION variable=scale_multiplier complete dim=1
    #pragma HLS ARRAY_PARTITION variable=right_shift complete dim=1

    #pragma HLS INTERFACE mode=s_axilite port=scale_multiplier
    #pragma HLS INTERFACE mode=s_axilite port=right_shift
    #pragma HLS INTERFACE mode=s_axilite port=bias
    #pragma HLS INTERFACE mode=s_axilite port=input_zero_point
    #pragma HLS INTERFACE mode=s_axilite port=weight_sum
    
        const int OUT_LEN = (IN_LEN - KERNEL_SIZE) / STRIDE + 1;

        // Channel-first input buffer
        ap_int8_c input_buffer[IN_CH][IN_LEN];
    #pragma HLS ARRAY_PARTITION variable=input_buffer complete dim=1

        // Read input
        for (int ic = 0; ic < IN_CH; ++ic) {
            for (int i = 0; i < IN_LEN; ++i) {
            #pragma HLS PIPELINE II=1
                input_buffer[ic][i] = in_stream.read();
            }
        }

#ifndef __SYNTHESIS__
        int conv_sat_hi = 0, conv_sat_lo = 0;
#endif

        // Convolution computation
        for (int oc = 0; oc < OUT_CH; ++oc) {
            for (int w = 0; w < OUT_LEN; ++w) {
            #pragma HLS PIPELINE II=1
                acc32_t acc = 0;

                // Integer MACs (int8×int8→int32)
                for (int ic = 0; ic < IN_CH; ++ic) {
                #pragma HLS UNROLL
                    for (int k = 0; k < KERNEL_SIZE; ++k) {
                    #pragma HLS UNROLL
                        const int t = w * STRIDE + k;
                        acc += (acc32_t)input_buffer[ic][t] * (acc32_t)weights[oc][ic][k];
                    }
                }

                // Optional: Add bias (already in accumulator domain)
                // Compile-time conditional - no runtime overhead
                if (USE_BIAS) {
                    acc += bias[oc];
                }

                // Optional: Asymmetric quantization correction
                // Compile-time conditional - no runtime overhead
                if (USE_ASYMMETRIC) {
                    acc -= (acc32_t)input_zero_point * weight_sum[oc];
                }

                // Requantize to next layer's int8 scale
                ap_int8_c y_q = requantize(acc, scale_multiplier[oc], right_shift[oc]);
#ifndef __SYNTHESIS__
                if (y_q == ap_int8_c(127))  ++conv_sat_hi;
                if (y_q == ap_int8_c(-128)) ++conv_sat_lo;
#endif
                out_stream.write(y_q);
            }
        }

#ifndef __SYNTHESIS__
        static int conv_once = 0;
        if (conv_once++ < 1) {
            std::cout << "[Conv1D_SD] sat_hi=" << conv_sat_hi
                      << " sat_lo=" << conv_sat_lo << "\n";
        }
#endif
    }
};

} // namespace hls4csnn1d_cblk_sd
#endif // CONV1D_SD_H



// /*  ────────────────────────────────────────────────────────────
//     conv1d_sd_folded.h — streaming 1-D convolution, Channel-First
//     For *folded* conv: BN baked into weights + bias (per out-ch).

//     Uses:
//       • ap_fixed_c from constants_sd.h (e.g., ap_fixed<8,2,AP_RND_CONV,AP_SAT>)
//       • widened accumulator type 'acc_t' (template param)
//       • exact OUT_LEN windows (no tail)
//       • CHANNEL-FIRST layout (PyTorch-compatible)

//     Interface:
//       forward(in_stream, out_stream,
//               weights[out_ch][in_ch][K], bias[out_ch])

//     Notes:
//       - Bias is REQUIRED here (folding guarantees a bias exists).
//       - No BN multiply/add inside the kernel (already folded).
//     ──────────────────────────────────────────────────────────── */
// #ifndef CONV1D_SD_FOLDED_H
// #define CONV1D_SD_FOLDED_H

// #include <hls_stream.h>
// #include <ap_fixed.h>
// #include "../constants_sd.h"

// namespace hls4csnn1d_cblk_sd {

// template<int IN_CH, int OUT_CH,
//          int KERNEL_SIZE, int STRIDE,
//          int IN_LEN,
//          typename acc_t>
// class CONV1D_SD_FOLDED {
// public:
//     CONV1D_SD_FOLDED() = default;

//     // Input format:  [ch0_t0, ch0_t1, ..., ch0_t(N-1), ch1_t0, ..., ch(IN-1)_t(N-1)]
//     // Output format: [ch0_w0, ch0_w1, ..., ch0_w(M-1), ch1_w0, ..., ch(OUT-1)_w(M-1)]
//     void forward(
//         hls::stream<ap_fixed_c>& in_stream,
//         hls::stream<ap_fixed_c>& out_stream,
//         const ap_fixed_c weights[OUT_CH][IN_CH][KERNEL_SIZE],
//         const ap_fixed_c bias[OUT_CH])                 // NEW: folded bias
//     {
//     #pragma HLS ARRAY_PARTITION variable=weights complete dim=1
//     #pragma HLS ARRAY_PARTITION variable=weights complete dim=2
//     #pragma HLS ARRAY_PARTITION variable=bias    complete dim=1

//         // Input buffer: store full sequence for all input channels (channel-first)
//         ap_fixed_c input_buffer[IN_CH][IN_LEN];
//     #pragma HLS ARRAY_PARTITION variable=input_buffer complete dim=1

//         // Output length (exact windows)
//         const int OUT_LEN = (IN_LEN - KERNEL_SIZE) / STRIDE + 1;

//         // 1) Read entire input (channel-first)
//         READ_INPUT:
//         for (int ic = 0; ic < IN_CH; ++ic) {
//         #pragma HLS PIPELINE II=1
//             for (int i = 0; i < IN_LEN; ++i) {
//             #pragma HLS PIPELINE II=1
//                 input_buffer[ic][i] = in_stream.read();
//             }
//         }

// #ifndef __SYNTHESIS__
//         int conv_sat_hi = 0, conv_sat_lo = 0;   // track saturation once per run
// #endif

//         // 2) Convolution per output channel
//         OUTPUT_CH:
//         for (int oc = 0; oc < OUT_CH; ++oc) {
//         #pragma HLS PIPELINE II=1

//             WINDOW_LOOP:
//             for (int w = 0; w < OUT_LEN; ++w) {
//             #pragma HLS PIPELINE II=1
//                 acc_t acc = 0;

//                 // MAC over input channels and kernel taps
//                 INPUT_CH:
//                 for (int ic = 0; ic < IN_CH; ++ic) {
//                 #pragma HLS UNROLL
//                     KERNEL_LOOP:
//                     for (int k = 0; k < KERNEL_SIZE; ++k) {
//                     #pragma HLS UNROLL
//                         const int t = w * STRIDE + k;
//                         acc += (acc_t)weights[oc][ic][k] * (acc_t)input_buffer[ic][t];
//                     }
//                 }

//                 // Add folded bias (per output channel)
//                 acc += (acc_t)bias[oc];

//                 // Cast/saturate to stream type once
//                 ap_fixed_c y = (ap_fixed_c)acc;
// #ifndef __SYNTHESIS__
//                 if (y == ap_fixed_c( 1.984375)) ++conv_sat_hi; // +2 - one LSB
//                 if (y == ap_fixed_c(-2.0     )) ++conv_sat_lo; // -2
// #endif
//                 out_stream.write(y);
//             }
//         }

// #ifndef __SYNTHESIS__
//         static int conv_once = 0;
//         if (conv_once++ < 1) {
//             std::cout << "[Conv1(FOLDED)] sat_hi=" << conv_sat_hi
//                       << " sat_lo=" << conv_sat_lo << "\n";
//         }
// #endif
//     }
// };

// } // namespace hls4csnn1d_cblk_sd
// #endif // CONV1D_SD_FOLDED_H



// /*  ────────────────────────────────────────────────────────────
//     conv1d_sd.h   —   streaming 1-D convolution, Channel-First
//     Uses:
//       • ap_fixed<8,4,AP_RND_CONV,AP_SAT> for activations/weights
//       • widened accumulator for MACs
//       • exact OUT_LEN windows (no extra tail window)
//       • CHANNEL-FIRST layout (PyTorch-compatible)
//     ──────────────────────────────────────────────────────────── */
// #ifndef CONV1D_SD_H
// #define CONV1D_SD_H

// #include <hls_stream.h>
// #include <ap_fixed.h>
// #include "../constants_sd.h"

// namespace hls4csnn1d_cblk_sd {

// // QFX: convergent (nearest-even) + saturate
// // typedef ap_fixed<8, 4, AP_RND_CONV, AP_SAT> ap_fixed_c;
// // Wider accumulator (tune if you have tighter bounds)
// // typedef ap_fixed<32, 16, AP_RND_CONV, AP_SAT> acc_t;

// template<int IN_CH,  int OUT_CH,
//          int KERNEL_SIZE, int STRIDE,
//          int IN_LEN,
//          typename acc_t>
// class Conv1D_SD {
// public:
//     Conv1D_SD() = default;

//     // forward(): reads IN_CH×IN_LEN scalars, writes OUT_CH×OUT_LEN scalars (CHANNEL-FIRST)
//     // Input format:  [ch0_t0, ch0_t1, ..., ch0_t(N-1), ch1_t0, ch1_t1, ..., ch1_t(N-1), ...]
//     // Output format: [ch0_w0, ch0_w1, ..., ch0_w(M-1), ch1_w0, ch1_w1, ..., ch1_w(M-1), ...]
//     void forward(
//         hls::stream<ap_fixed_c>& in_stream,
//         hls::stream<ap_fixed_c>& out_stream,
//         const ap_fixed_c weights[OUT_CH][IN_CH][KERNEL_SIZE]  /*, optional bias[] */) {
//     #pragma HLS ARRAY_PARTITION variable=weights complete dim=1
//     #pragma HLS ARRAY_PARTITION variable=weights complete dim=2

//         // Shift registers per input channel
//         ap_fixed_c shift_reg[IN_CH][KERNEL_SIZE];
//     #pragma HLS ARRAY_PARTITION variable=shift_reg complete dim=1
//     #pragma HLS ARRAY_PARTITION variable=shift_reg complete dim=2

//         // Input buffer to store all input channels' data
//         ap_fixed_c input_buffer[IN_CH][IN_LEN];
//     #pragma HLS ARRAY_PARTITION variable=input_buffer complete dim=1

//         // Expected number of windows
//         const int OUT_LEN = (IN_LEN - KERNEL_SIZE) / STRIDE + 1;

//         // 1) Read all input data in channel-first order
//         READ_INPUT:
//         for (int ic = 0; ic < IN_CH; ++ic) {
//         #pragma HLS PIPELINE II=1
//             for (int i = 0; i < IN_LEN; ++i) {
//             #pragma HLS PIPELINE II=1
//                 input_buffer[ic][i] = in_stream.read();
//             }
//         }
// #ifndef __SYNTHESIS__
//         int conv_sat_hi = 0, conv_sat_lo = 0;   // reset per row/sequence
// #endif

//         // 2) Process each output channel
//         OUTPUT_CH:
//         for (int oc = 0; oc < OUT_CH; ++oc) {
//         #pragma HLS PIPELINE II=1
            
//             // Process each output window for this output channel
//             WINDOW_LOOP:
//             for (int w = 0; w < OUT_LEN; ++w) {
//             #pragma HLS PIPELINE II=1
//                 acc_t acc = 0;
                
//                 // Convolution computation for this window
//                 INPUT_CH:
//                 for (int ic = 0; ic < IN_CH; ++ic) {
//                 #pragma HLS UNROLL
//                     KERNEL_LOOP:
//                     for (int k = 0; k < KERNEL_SIZE; ++k) {
//                     #pragma HLS UNROLL
//                         int time_idx = w * STRIDE + k;
//                         acc += (acc_t)weights[oc][ic][k] * (acc_t)input_buffer[ic][time_idx];
//                     }
//                 }
//                 ap_fixed_c y = ap_fixed_c(acc);      // cast to stream type once
// #ifndef __SYNTHESIS__
//                if (y == ap_fixed_c(1.984375)) ++conv_sat_hi;
//                if (y == ap_fixed_c(-2.0))   ++conv_sat_lo;
// #endif
                
//                 // Optional bias:
//                 // acc += (acc_t)bias[oc];

//                 // Write output for this window of this output channel
//                 out_stream.write((ap_fixed_c)acc);
//             }
//             // ... when you finish the row/sequence (once):
// #ifndef __SYNTHESIS__
//             static int conv_once = 0;
//             if (conv_once++ < 1) {
//                 std::cout << "[Conv1] sat_hi=" << conv_sat_hi
//                     << " sat_lo=" << conv_sat_lo << "\n";
//              }
// #endif
//         }
//     }
// };

// } // namespace hls4csnn1d_cblk_sd
// #endif // CONV1D_SD_H


// /*  ────────────────────────────────────────────────────────────
//     conv1d_sd.h   —   streaming 1-D convolution, Q4.4
//     Uses:
//       • ap_fixed<8,4,AP_RND,AP_SAT> for all data
//       • bias term (folded from BatchNorm)
//       • SHIFT (compile-time const) to undo exporter’s /2^SHIFT
//     ──────────────────────────────────────────────────────────── */
// #ifndef CONV1D_SD_H
// #define CONV1D_SD_H

// #include <hls_stream.h>
// #include <ap_fixed.h>
// #include "../constants_sd.h"

// namespace hls4csnn1d_cblk_sd {

// typedef ap_fixed<8, 4, AP_RND, AP_SAT> ap_fixed_c;

// template<int IN_CH,  int OUT_CH,
//          int KERNEL_SIZE, int STRIDE,
//          int IN_LEN>
// class Conv1D_SD {
// public:
//     Conv1D_SD() = default;

//     // ─────────────────────────────────────────────────────────
//     // forward(): reads IN_LEN×IN_CH scalars, writes
//     //            OUT_CH×OUT_LEN scalars (channel-major)
//     // ─────────────────────────────────────────────────────────
// //     void forward(
// //         hls::stream<ap_fixed_c>&  in_stream,
// //         hls::stream<ap_fixed_c>&  out_stream,
// //         const ap_fixed_c          weights[OUT_CH][IN_CH][KERNEL_SIZE],
// //         const ap_fixed_c          bias   [OUT_CH],
// //         const int                 SHIFT)                           // NEW
// //     {
// // #pragma HLS ARRAY_PARTITION variable=weights complete dim=1
// // #pragma HLS ARRAY_PARTITION variable=weights complete dim=2

// // // #ifndef __SYNTHESIS__
// // //     static int debug_count = 0;
// // //     static bool first_output = true;
// // //     // ========== DEBUG ADDITION START ==========
// // //     static int input_count = 0;
// // //     static bool debug_weights_printed = false;
    
// // //     // Print first few weights to verify they're loaded
// // //     if (!debug_weights_printed) {
// // //         debug_weights_printed = true;
// // //         std::cout << "\n=== Conv1D Weights Debug ===\n";
// // //         for (int oc = 0; oc < 3 && oc < OUT_CH; ++oc) {
// // //             std::cout << "Output channel " << oc << " weights: ";
// // //             for (int k = 0; k < KERNEL_SIZE; ++k) {
// // //                 std::cout << weights[oc][0][k] << " ";
// // //             }
// // //             std::cout << "\n";
// // //         }
// // //     }
// // //     // ========== DEBUG ADDITION END ==========
// // // #endif

// //         // ─── local buffers ───
// //         ap_fixed_c shift_reg[IN_CH][KERNEL_SIZE];
// // #pragma HLS ARRAY_PARTITION variable=shift_reg complete dim=2

// //         const int OUT_LEN = (IN_LEN - KERNEL_SIZE) / STRIDE + 1;

// //         static ap_fixed_c linebuf[OUT_CH];
// // #pragma HLS ARRAY_PARTITION variable=linebuf complete dim=1

// //         // ─── streaming convolution ───
// //         int feat_idx = 0;
// //         const int TOTAL_ITER = IN_LEN + STRIDE - 1;

// //     MAIN_LOOP:
// //         for (int i = 0; i < TOTAL_ITER; ++i) {
// // #pragma HLS PIPELINE II=1

// //             // 1) push new sample into each channel's shift register
// //         READ_CH:
// //             for (int ic = 0; ic < IN_CH; ++ic) {
// // #pragma HLS UNROLL
// //                 for (int k = 0; k < KERNEL_SIZE - 1; ++k)
// // #pragma HLS UNROLL
// //                     shift_reg[ic][k] = shift_reg[ic][k + 1];

// //                 ap_fixed_c v = (i < IN_LEN) ? in_stream.read() : ap_fixed_c(0);

// // /* ───── DEBUG: print first input row once (sim-only) ───── */
// // // #if !defined(__SYNTHESIS__) && defined(PRINT_FIRST_CONV_ROW)
// // //                 {
// // //                     static bool row_done = false;
// // //                     static int  cnt      = 0;

// // //                     if (!row_done) {
// // //                         std::cout << v.to_double();
// // //                         ++cnt;

// // //                         if (cnt == IN_LEN) {            // end of row
// // //                             std::cout << std::endl;
// // //                             row_done = true;
// // //                         } else {
// // //                             std::cout << ' ';
// // //                         }
// // //                     }
// // //                 }
// // // #endif
// // /* ───────────────────────────────────────────────────────── */


// //                 shift_reg[ic][KERNEL_SIZE - 1] = v;

// // //             // ========== DEBUG ADDITION START ==========
// // // #ifndef __SYNTHESIS__
// // //             // Debug first few input values
// // //             if (input_count < 10 && ic == 0) {
// // //                 std::cout << "Conv1D input[" << input_count << "]: " << v << std::endl;
// // //                 input_count++;
// // //             }
// // // #endif
// // //             // ========== DEBUG ADDITION END ==========

// //             }

// //             // 2) when window ready & stride boundary hit, compute OUT_CH dots
// //             if (i >= KERNEL_SIZE - 1 && ((i - (KERNEL_SIZE - 1)) % STRIDE == 0)) {

// //                 // 2a) output previous results
// //                 if (feat_idx > 0) {
// //                 WRITE_OC:
// //                     for (int oc = 0; oc < OUT_CH; ++oc) {
// // #pragma HLS UNROLL
// //                         out_stream.write(linebuf[oc]);

// // //                     // ========== DEBUG ADDITION START ==========
// // // #ifndef __SYNTHESIS__
// // //                     // Debug first few outputs
// // //                     if (debug_count < 20) {
// // //                         std::cout << "Conv1D output[" << debug_count 
// // //                                   << "] (ch=" << oc << "): " << linebuf[oc] << std::endl;
// // //                         debug_count++;
// // //                     }
// // // #endif
// // //                     // ========== DEBUG ADDITION END ==========

// //                     }
// //                 }

// //                 // 2b) compute new convolution outputs
// //             DOT_OC:
// //                 for (int oc = 0; oc < OUT_CH; ++oc) {
// // #pragma HLS UNROLL
// //                     ap_fixed_c acc = 0;
// //                 DOT_IC:
// //                     for (int ic = 0; ic < IN_CH; ++ic) {
// // #pragma HLS UNROLL
// //                     DOT_K:
// //                         for (int k = 0; k < KERNEL_SIZE; ++k) {
// // #pragma HLS UNROLL
// //                             acc += weights[oc][ic][k] * shift_reg[ic][k];
// //                         }
// //                     }
// //                     acc += bias[oc];        // add folded bias
// //                     acc <<= SHIFT;          // restore scale (2^SHIFT)
// //                     linebuf[oc] = acc;
// //                 }
// //                 ++feat_idx;
// //             }
// //         }

// //         // ─── flush last set of outputs ───
// //     FLUSH_OC:
// //         for (int oc = 0; oc < OUT_CH; ++oc) {
// // #pragma HLS UNROLL
// //             out_stream.write(linebuf[oc]);

// // //         // ========== DEBUG ADDITION START ==========
// // // #ifndef __SYNTHESIS__
// // //         // Debug last outputs
// // //         if (debug_count < 30) {
// // //             std::cout << "Conv1D final output[" << debug_count 
// // //                       << "] (ch=" << oc << "): " << linebuf[oc] << std::endl;
// // //             debug_count++;
// // //         }
// // // #endif
// // //         // ========== DEBUG ADDITION END ==========

// //         }
// //     }
// // };

// // }  // namespace hls4csnn1d_cblk_sd
// // #endif /* CONV1D_SD_H */





//     void forward(
//         hls::stream<ap_fixed_c>& in_stream,
//         hls::stream<ap_fixed_c>& out_stream,
//         const ap_fixed_c weights[OUT_CH][IN_CH][KERNEL_SIZE]) {
        
//     #pragma HLS ARRAY_PARTITION variable=weights complete dim=1
//     #pragma HLS ARRAY_PARTITION variable=weights complete dim=2
// #ifndef __SYNTHESIS__
//     static int debug_count = 0;
//     static bool first_output = true;
// #endif
//     // for (int oc=0; oc < OUT_CH; ++oc){
//     //     for (int ic=0; ic < IN_CH; ++ic){
//     //         for (int ks=0; ks < KERNEL_SIZE; ++ks){
//     //              // Print the two logits
//     //             std::cout << "W[" << oc << "][" << ic << "][" << ks << "] = " << weights[oc][ic][ks] << '\n';

//     //         }
    
//     //     }
//     // }
//         //------------------------------------------------------------------
//         // local buffers
//         //------------------------------------------------------------------
//         ap_fixed_c shift_reg[IN_CH][KERNEL_SIZE];
//     #pragma HLS ARRAY_PARTITION variable=shift_reg complete dim=2

//         const int OUT_LEN = (IN_LEN - KERNEL_SIZE) / STRIDE + 1; // stride‑aligned features

//         static ap_fixed_c linebuf[OUT_CH];
//     #pragma HLS ARRAY_PARTITION variable=linebuf complete dim=1

//         //------------------------------------------------------------------
//         // convolution streaming loop (feature‑major)
//         //------------------------------------------------------------------
//         int feature_idx = 0;
//         const int TOTAL_ITER = IN_LEN + KERNEL_SIZE - 1;

//         MAIN_LOOP:
//         for (int i = 0; i < TOTAL_ITER; ++i) {
//         #pragma HLS PIPELINE II=1
//             // 1) push new sample into each channel’s shift register
//             READ_CH:
//             for (int ic = 0; ic < IN_CH; ++ic) {
//             #pragma HLS UNROLL
//                 for (int k = 0; k < KERNEL_SIZE-1; ++k) {
//                 #pragma HLS UNROLL
//                     shift_reg[ic][k] = shift_reg[ic][k+1];
//                 }
//                 ap_fixed_c v = (i < IN_LEN) ? in_stream.read() : ap_fixed_c(0);
//                 shift_reg[ic][KERNEL_SIZE-1] = v;
//             }

//             // 2) when window ready & stride boundary hit, compute OUT_CH dots
//             if (i >= KERNEL_SIZE-1 && ((i - (KERNEL_SIZE-1)) % STRIDE == 0)) {
//                 // 2a) output previous linebuf in channel‑major order
//                 if (feature_idx > 0) {
//                     WRITE_OC:
//                     for (int oc = 0; oc < OUT_CH; ++oc) {
//                     #pragma HLS UNROLL
//                         out_stream.write(linebuf[oc]);
//                     }
//                 }

//                 // 2b) fill linebuf with new convolution results
//                 DOT_OC:
//                 for (int oc = 0; oc < OUT_CH; ++oc) {
//                 #pragma HLS UNROLL
//                     ap_fixed_c acc = 0;
//                     DOT_IC:
//                     for (int ic = 0; ic < IN_CH; ++ic) {
//                     #pragma HLS UNROLL
//                         for (int k = 0; k < KERNEL_SIZE; ++k) {
//                         #pragma HLS UNROLL
//                             acc += weights[oc][ic][k] * shift_reg[ic][k];
//                         }
//                     }
//                     linebuf[oc] = acc;
//                 }
//                 ++feature_idx;
//             }
//         }

//         //------------------------------------------------------------------
//         // flush last stride’s outputs
//         //------------------------------------------------------------------
//         FLUSH_OC:
//         for (int oc = 0; oc < OUT_CH; ++oc) {
//         #pragma HLS UNROLL
//             out_stream.write(linebuf[oc]);
//         }
//     }

// };
// }
// #endif // CONV1D_SD_H
