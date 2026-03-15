// // main.cpp — layer-by-layer testbench for csnet2 (int8, integer-only pipeline)

// #include <iostream>
// #include <vector>
// #include <string>
// #include <algorithm>

// #include <hls_stream.h>
// #include "ap_int.h"
// #include "../../include/hls4csnn1d_sd/filereader.h"

// // ===== Model constants & typedefs (project-local) =====
// #include "../../include/hls4csnn1d_sd/constants_sd.h"                      // defines ap_int8_c, acc32_t, FIXED_LENGTH1, array180_t, etc.

// // ===== Model input quantizer =====
// #include "../../include/hls4csnn1d_sd/weights_sd/qparams_qcsnet2_cblk1_input.h"               // qcsnet2_input_scale, qcsnet2_input_zero_point

// // ===== Layer parameter headers (>>> ADJUST IF NEEDED) =====
// #include "../../include/hls4csnn1d_sd/weights_sd/qcsnet2_cblk1_qconv1d_weights.h"
// #include "../../include/hls4csnn1d_sd/weights_sd/qcsnet2_cblk1_batch_norm_bn.h"
// #include "../../include/hls4csnn1d_sd/weights_sd/qcsnet2_cblk1_leaky_lif.h"

// #include "../../include/hls4csnn1d_sd/weights_sd/qcsnet2_cblk2_qconv1d_weights.h"
// #include "../../include/hls4csnn1d_sd/weights_sd/qcsnet2_cblk2_batch_norm_bn.h"
// #include "../../include/hls4csnn1d_sd/weights_sd/qcsnet2_cblk2_leaky_lif.h"

// #include "../../include/hls4csnn1d_sd/weights_sd/qcsnet2_lblk1_qlinear_weights.h"
// #include "../../include/hls4csnn1d_sd/weights_sd/qcsnet2_lblk1_leaky_lif.h"
// // #include "../../include/hls4csnn1d_sd/weights_sd/qparams_qcsnet2_cblk2_qconv1d_input.h"


// // ===== Kernel templates =====
// #include "../../include/hls4csnn1d_sd/cblk_sd/conv1d_sd.h"
// #include "../../include/hls4csnn1d_sd/cblk_sd/batchnorm1d_sd.h"         // **expects Int32 bias**: const ap_int<32> bias[CH]
// #include "../../include/hls4csnn1d_sd/cblk_sd/linear1d_sd.h"
// #include "../../include/hls4csnn1d_sd/cblk_sd/lif1d_sd.h"
// #include "../../include/hls4csnn1d_sd/cblk_sd/maxpool1d_sd.h"

// #include "../../include/hls4csnn1d_sd/cblk_sd/lif1d_integer.h"


// using namespace hls4csnn1d_cblk_sd;

// // ---------------- helpers ----------------

// template<typename T>
// static inline void clone_and_capture(hls::stream<T>& in,
//                                      hls::stream<T>& out,
//                                      std::vector<int>& cap,
//                                      int count) {
//     cap.clear(); cap.reserve(count);
//     for (int i = 0; i < count; ++i) {
//         T v = in.read();
//         out.write(v);
//         cap.push_back((int)v);
//     }
// }

// static inline void append_featmap_to_file(const std::string& fname,
//                                           const std::vector<int>& v,
//                                           int CH, int LEN,
//                                           int sample_idx) {
//     std::ofstream fp(fname, std::ios::app);
//     fp << "=== sample " << sample_idx << " ===\n";
//     for (int c = 0; c < CH; ++c) {
//         fp << "ch " << c << ": ";
//         for (int t = 0; t < LEN; ++t) {
//             fp << v[c * LEN + t] << (t + 1 < LEN ? ' ' : '\n');
//         }
//     }
//     fp.flush();
// }

// static inline void reset_output_files() {
//     std::ofstream("../../../../compare_outputs/conv1_out.txt",  std::ios::trunc).close();
//     std::ofstream("../../../../compare_outputs/bn1_out.txt",    std::ios::trunc).close();
//     std::ofstream("../../../../compare_outputs/lif1_out.txt",   std::ios::trunc).close();
//     std::ofstream("../../../../compare_outputs/mp1_out.txt",    std::ios::trunc).close();

//     std::ofstream("../../../../compare_outputs/conv2_in.txt",   std::ios::trunc).close();

//     // New files for Block 2 + Head
//     std::ofstream("../../../../compare_outputs/conv2_out.txt",  std::ios::trunc).close();
//     std::ofstream("../../../../compare_outputs/bn2_out.txt",    std::ios::trunc).close();
//     std::ofstream("../../../../compare_outputs/lif2_out.txt",   std::ios::trunc).close();
//     std::ofstream("../../../../compare_outputs/mp2_out.txt",    std::ios::trunc).close();

//     std::ofstream("../../../../compare_outputs/lin_out.txt",    std::ios::trunc).close();
//     std::ofstream("../../../../compare_outputs/lif_head_out.txt", std::ios::trunc).close();
    


// }


// // Map {0,1} spikes to {0, on_level} expected by the next layer’s input quantization
// static inline void scale_spike_stream(hls::stream<ap_int8_c>& in,
//                                       hls::stream<ap_int8_c>& out,
//                                       int CH, int LEN,
//                                       ap_int<8> on_level) {
//     for (int c = 0; c < CH; ++c) {
//         for (int t = 0; t < LEN; ++t) {
//         #pragma HLS PIPELINE II=1
//             ap_int8_c v = in.read();
//             out.write(v ? on_level : ap_int8_c(0));
//         }
//     }
// }



// // ---------------- main ----------------

// int main(int argc, char** argv) {
//     // 1) Load rows
//     std::string folderPath =
//         "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_singles/large";
//     if (argc >= 2) folderPath = argv[1];

//     FileReader reader;
//     reader.loadData(folderPath);

//     // 2) Stream rows (array180_t per sample)
//     hls::stream<array180_t> dataStream;
//     hls::stream<array2848_t> outstream_unused; // not used here
//     reader.streamData(dataStream);
//     std::cout << "Finished stream data.\n";

//     const int NUM_SAMPLES = dataStream.size();
//     if (NUM_SAMPLES <= 0) {
//         std::cout << "No rows loaded — aborting.\n";
//         return 0;
//     }
//     std::cout << "Loaded " << NUM_SAMPLES << " ECG rows\n";

//     // 3) Shapes for block 1
//     constexpr int IN_CH   = 1;
//     constexpr int OUT_CH  = 16;
//     constexpr int K       = 3;
//     constexpr int STRIDE  = 1;
//     constexpr int IN_LEN  = CONV_IN_LENGTH1;                // usually 180

//     constexpr int LEN_CONV1 = (IN_LEN - K) / STRIDE + 1;    // typically 178
//     constexpr int LEN_POOL1 = LEN_CONV1 / 2;                // typically 89

//     // ---- Block 2 + Head shapes ----
//     constexpr int IN_CH2     = 16;     // from Block1 OUT_CH
//     constexpr int OUT_CH2    = 24;
//     constexpr int K2         = 3;
//     constexpr int STRIDE2    = 1;

//     constexpr int LEN_CONV2  = (LEN_POOL1 - K2) / STRIDE2 + 1;   // e.g., 87
//     constexpr int LEN_POOL2  = LEN_CONV2 / 2;                    // e.g., 43

//     constexpr int LIN_IN     = OUT_CH2 * LEN_POOL2;              // e.g., 24*43 = 1032
//     constexpr int LIN_OUT    = 2;

//     static const ap_int<8> ON_LVL_CONV2 = (ap_int<8>)126;  // computed from your scale

//     // If Conv2 input zero_point != 0, add it here; in most signed setups it's 0.
//     // from qcsnet2_cblk2_qconv1d_input_zero_point in the weights header
//     const ap_int<8> Z_IN_CONV2 = qcsnet2_cblk2_qconv1d_input_zero_point;




//     // 4) Instantiate block-1 kernels
//     Conv1D_SD<IN_CH, OUT_CH, K, STRIDE, IN_LEN,
//               /*USE_BIAS*/ false,
//               /*USE_ASYMMETRIC*/ true> conv1;

//     BatchNorm1D_SD<OUT_CH, LEN_CONV1, /*USE_BIAS*/ true>     bn1;

    

//     // LIF1D_SD<OUT_CH, LEN_CONV1,
//     //              qcsnet2_cblk1_leaky_FRAC_BITS,
//     //              RESET_SUBTRACT>                             lif1;

//     LIF1D_SD_Integer<OUT_CH, LEN_CONV1, 
//                     qcsnet2_cblk1_leaky_FRAC_BITS,
//                     RESET_SUBTRACT_INT>                     lif1;
//     MaxPool1D_SD<2, 2, OUT_CH, LEN_CONV1>     mp1;

//     // ---- Block 2 kernels ----
//     Conv1D_SD<IN_CH2, OUT_CH2, K2, STRIDE2, LEN_POOL1,
//             /*USE_BIAS*/ false,
//             /*USE_ASYMMETRIC*/ true>                   conv2;

//     BatchNorm1D_SD<OUT_CH2, LEN_CONV2, /*USE_BIAS*/ true> bn2;

//     // LIF1D_SD<OUT_CH2, LEN_CONV2,
//     //         qcsnet2_cblk2_leaky_FRAC_BITS,
//     //         RESET_SUBTRACT>                             lif2;
//     LIF1D_SD_Integer<OUT_CH2, LEN_CONV2, 
//                     qcsnet2_cblk2_leaky_FRAC_BITS,
//                     RESET_SUBTRACT_INT>                     lif2;

//     MaxPool1D_SD<2, 2, OUT_CH2, LEN_CONV2>               mp2;

//     // ---- Linear head ----
//     Linear1D_SD<LIN_IN, LIN_OUT,
//                 /*USE_BIAS*/ false,
//                 /*USE_ASYMMETRIC*/ true>                 lin;

//     // Head LIF (scalar beta/theta, FRAC_BITS from header)
//     // LIF1D_SD<LIN_OUT, 1,
//     //         qcsnet2_lblk1_leaky_FRAC_BITS,
//     //         RESET_SUBTRACT>                             lif_head;

//     LIF1D_SD_Integer<LIN_OUT, 1, 
//                     qcsnet2_lblk1_leaky_FRAC_BITS,
//                     RESET_SUBTRACT_INT>                     lif_head;




//     reset_output_files();

//     // 5) Process each sample through the first block, write layer outputs to files
//     for (int i = 0; i < NUM_SAMPLES; ++i) {
//         lif1.reset();
//         // pop one quantized sample from FileReader
//         array180_t sample = dataStream.read();

//         // streams through the block
//         hls::stream<ap_int8_c> s0, s1, s2, s3, s4;

//         // input stream
//         for (int t = 0; t < IN_LEN; ++t) {
//         #pragma HLS PIPELINE II=1
//             s0.write(sample[t]);
//         }

//         // ---- Conv1 ----
//         conv1.forward(
//             s0, s1,
//             qcsnet2_cblk1_qconv1d_weights,
//             qcsnet2_cblk1_qconv1d_scale_multiplier,
//             qcsnet2_cblk1_qconv1d_right_shift,
//             qcsnet2_cblk1_qconv1d_bias,       // zeros if bias disabled at export
//             qcsnet2_cblk1_qconv1d_input_zero_point,         // from qparams header
//             qcsnet2_cblk1_qconv1d_weight_sum
//         );
//         std::vector<int> v_conv1; 
//         clone_and_capture(s1, s2, v_conv1, OUT_CH * LEN_CONV1);
//         append_featmap_to_file("../../../../compare_outputs/conv1_out.txt", v_conv1, OUT_CH, LEN_CONV1, i + 1);

//         // ---- BN1 (QuantScaleBias, Int32 bias) ----
//         bn1.forward(
//             s2, s3,
//             qcsnet2_cblk1_batch_norm_weight,           // ap_int8_c [OUT_CH]
//             qcsnet2_cblk1_batch_norm_bias,             // ap_int<32> [OUT_CH]
//             qcsnet2_cblk1_batch_norm_scale_multiplier, // ap_int<32> [OUT_CH]
//             qcsnet2_cblk1_batch_norm_right_shift       // int [OUT_CH]
//         );
//         std::vector<int> v_bn1; 
//         clone_and_capture(s3, s4, v_bn1, OUT_CH * LEN_CONV1);
//         append_featmap_to_file("../../../../compare_outputs/bn1_out.txt", v_bn1, OUT_CH, LEN_CONV1, i + 1);

//         // ---- LIF1 (scalar beta/theta) ----
//         lif1.reset();
//         lif1.forward(
//             s4, s1,                                   // reuse s1 for next stage
//             qcsnet2_cblk1_leaky_beta_int,
//             qcsnet2_cblk1_leaky_theta_int,
//             qcsnet2_cblk1_leaky_scale_int
//         );

//         // To (with float values):
//         // lif1.forward(
//         //     s4, s1,
//         //     0.2962f,  // beta
//         //     0.3086f,  // theta
//         //     0.0071f   // scale (from BatchNorm output scale)
//         // );
//         std::vector<int> v_lif1; 
//         clone_and_capture(s1, s2, v_lif1, OUT_CH * LEN_CONV1);
//         append_featmap_to_file("../../../../compare_outputs/lif1_out.txt", v_lif1, OUT_CH, LEN_CONV1, i + 1);

//         // ---- MaxPool1d(2,2) ----
    
//         hls::stream<ap_int8_c> s_pool_tmp;
//         mp1.forward(s2, s_pool_tmp);
//         std::vector<int> v_pool1; 
//         clone_and_capture(s_pool_tmp, s3, v_pool1, OUT_CH * LEN_POOL1);
//         append_featmap_to_file("../../../../compare_outputs/mp1_out.txt", v_pool1, OUT_CH, LEN_POOL1, i + 1);
    

//         // // NEW: re-quantize spikes {0,1} → {0, ON_LVL_CONV2} for Conv2 input
//         // hls::stream<ap_int8_c> s3_scaled_for_conv2;
//         // scale_spike_stream(s3, s3_scaled_for_conv2, OUT_CH /*16*/, LEN_POOL1 /*89*/, ON_LVL_CONV2);


//         // Re-quantize spikes for Conv2: y = v ? (Z_IN + ON_LVL_CONV2) : Z_IN
//         // Scale MP1 spikes → Conv2 input integers
//         hls::stream<ap_int8_c> s3_scaled_for_conv2;
//         for (int c = 0; c < OUT_CH; ++c) {
//             for (int t = 0; t < LEN_POOL1; ++t) {
//             #pragma HLS PIPELINE II=1
//                 ap_int8_c v = s3.read();  // v ∈ {0,1}
//                 ap_int8_c y = v ? (ap_int8_c)(Z_IN_CONV2 + ON_LVL_CONV2) : Z_IN_CONV2;
//                 s3_scaled_for_conv2.write(y);
//             }
//         }

//         // Tap EXACT Conv2 input to file (no counters, just i+1)
        
//         hls::stream<ap_int8_c> s3_for_conv2;
//         std::vector<int> v_conv2_in; 
//         v_conv2_in.reserve(OUT_CH * LEN_POOL1);

//         // clone the whole tensor (channel-major) into a buffer and a pass-through stream
//         clone_and_capture(s3_scaled_for_conv2, s3_for_conv2,
//                         v_conv2_in, OUT_CH * LEN_POOL1);

//         // append to conv2_in.txt using your existing helper (i+1 is the sample index)
//         append_featmap_to_file("../../../../compare_outputs/conv2_in.txt",
//                             v_conv2_in, OUT_CH, LEN_POOL1, i + 1);


//                 // =========================
//             // Block 2: Conv2 → BN2 → LIF2 → MP2
//             // =========================

//             // Streams for block 2 and head
//             hls::stream<ap_int8_c> s5, s6, s7, s8, s9, s10, s11;

//             // ---- Conv2 ----  (input: s3 from MP1 capture)
//             conv2.forward(
//                 s3_for_conv2, s4,
//                 qcsnet2_cblk2_qconv1d_weights,
//                 qcsnet2_cblk2_qconv1d_scale_multiplier,
//                 qcsnet2_cblk2_qconv1d_right_shift,
//                 qcsnet2_cblk2_qconv1d_bias,                // zeros if bias disabled at export
//                 qcsnet2_cblk2_qconv1d_input_zero_point,
//                 qcsnet2_cblk2_qconv1d_weight_sum
//             );
//             std::vector<int> v_conv2;
//             clone_and_capture(s4, s5, v_conv2, OUT_CH2 * LEN_CONV2);
//             append_featmap_to_file("../../../../compare_outputs/conv2_out.txt",
//                                 v_conv2, OUT_CH2, LEN_CONV2, i + 1);

//             // ---- BN2 ----
//             bn2.forward(
//                 s5, s6,
//                 qcsnet2_cblk2_batch_norm_weight,
//                 qcsnet2_cblk2_batch_norm_bias,               // Int32
//                 qcsnet2_cblk2_batch_norm_scale_multiplier,
//                 qcsnet2_cblk2_batch_norm_right_shift
//             );
//             std::vector<int> v_bn2;
//             clone_and_capture(s6, s7, v_bn2, OUT_CH2 * LEN_CONV2);
//             append_featmap_to_file("../../../../compare_outputs/bn2_out.txt",
//                                 v_bn2, OUT_CH2, LEN_CONV2, i + 1);

//             // ---- LIF2 ----
//             lif2.reset();
//             lif2.forward(
//                 s7, s8,
//                 qcsnet2_cblk2_leaky_beta_int,
//                 qcsnet2_cblk2_leaky_theta_int,
//                 qcsnet2_cblk2_leaky_scale_int
//             );
//             std::vector<int> v_lif2;
//             clone_and_capture(s8, s9, v_lif2, OUT_CH2 * LEN_CONV2);
//             append_featmap_to_file("../../../../compare_outputs/lif2_out.txt",
//                                 v_lif2, OUT_CH2, LEN_CONV2, i + 1);

//             // ---- MP2(2,2) ----
//             mp2.forward(s9, s10);
//             std::vector<int> v_mp2;
//             clone_and_capture(s10, s11, v_mp2, OUT_CH2 * LEN_POOL2);
//             append_featmap_to_file("../../../../compare_outputs/mp2_out.txt",
//                                 v_mp2, OUT_CH2, LEN_POOL2, i + 1);

//             // =========================
//             // Head: Flatten → Linear
//             // =========================
//             // Flatten is implicit: channel-major stream from MP2 already matches Linear’s input order.
//             lin.forward(
//                 s11, s0,  // reuse s0 as output
//                 qcsnet2_lblk1_qlinear_weights,
//                 qcsnet2_lblk1_qlinear_scale_multiplier,
//                 qcsnet2_lblk1_qlinear_right_shift,
//                 qcsnet2_lblk1_qlinear_bias,                 // zeros if bias disabled
//                 qcsnet2_lblk1_qlinear_input_zero_point,
//                 qcsnet2_lblk1_qlinear_weight_sum
//             );
//             // Dump linear output as (C=LIN_OUT, L=1)
//             std::vector<int> v_lin;
//             clone_and_capture(s0, s1, v_lin, LIN_OUT * 1);
//             append_featmap_to_file("../../../../compare_outputs/lin_out.txt",
//                                 v_lin, LIN_OUT, /*LEN=*/1, i + 1);

//             // ---- LIF (head) ----
//             // Input is the linear output we just forwarded into s1
//             lif_head.reset();
//             lif_head.forward(
//                 s1, s2,
//                 qcsnet2_lblk1_leaky_beta_int,
//                 qcsnet2_lblk1_leaky_theta_int,
//                 qcsnet2_lblk1_leaky_scale_int
//             );
//             std::vector<int> v_lif_head;
//             clone_and_capture(s2, s3, v_lif_head, LIN_OUT * 1);
//             append_featmap_to_file("../../../../compare_outputs/lif_head_out.txt",
//                                 v_lif_head, LIN_OUT, /*LEN=*/1, i + 1);



//     }

//     std::cout << "First block complete. Outputs written to:\n"
//               << "  ../../../../compare_outputs/conv1_out.txt\n"
//               << "  ../../../../compare_outputs/bn1_out.txt\n"
//               << "  ../../../../compare_outputs/lif1_out.txt\n"
//               << "  ../../../../compare_outputs/mp1_out.txt\n"
//               << "Second block complete. Outputs written to:\n"
//               << "  ../../../../compare_outputs/conv2_out.txt\n"
//               << "  ../../../../compare_outputs/bn2_out.txt\n"
//               << "  ../../../../compare_outputs/lif2_out.txt\n"
//               << "  ../../../../compare_outputs/mp2_out.txt\n"
//               << "Third block complete. Outputs written to:\n"
//               << "  ../../../../compare_outputs/lin_out.txt\n"
//               << "  ../../../../compare_outputs/lif_head_out.txt\n";
//     return 0;
// }

// // int main(int argc, char** argv) {
// //     // 1) Load ECG rows from folder (adjust path or pass as argv[1])
// //     std::string folderPath =
// //         "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_singles/large";
// //     if (argc >= 2) folderPath = argv[1];

// //     FileReader reader;
// //     reader.loadData(folderPath);

// //     // 2) FileReader -> dataStream (each element is array180_t)
// //     hls::stream<array180_t> dataStream;
// //     hls::stream<array2848_t> outstream_unused; // not used here
// //     reader.streamData(dataStream);
// //     std::cout << "Finished stream data.\n";

// //     const int NUM_SAMPLES = dataStream.size();
// //     if (NUM_SAMPLES <= 0) {
// //         std::cout << "No rows loaded — aborting.\n";
// //         return 0;
// //     }
// //     std::cout << "Loaded " << NUM_SAMPLES << " ECG rows\n";

// //     // 3) Instantiate ONLY Conv1D
// //     constexpr int IN_CH   = 1;
// //     constexpr int OUT_CH  = 16;
// //     constexpr int K       = 3;
// //     constexpr int STRIDE  = 1;
// //     constexpr int IN_LEN  = CONV_IN_LENGTH1;                 // typically 180
// //     constexpr int OUT_LEN = (IN_LEN - K) / STRIDE + 1;       // typically 178

// //     Conv1D_SD<IN_CH, OUT_CH, K, STRIDE, IN_LEN,
// //               /*USE_BIAS*/ false,
// //               /*USE_ASYMMETRIC*/ true> conv1;

// //     // 4) For each sample from dataStream: run Conv1D and print outputs
// //     for (int i = 0; i < NUM_SAMPLES; ++i) {
// //         // Pop one sample directly from dataStream (already int8-quantized)
// //         array180_t sample = dataStream.read();

// //         // Build the int8 input stream expected by Conv1D
// //         hls::stream<ap_int8_c> s_in, s_out;
// //         for (int t = 0; t < IN_LEN; ++t) {
// //         #pragma HLS PIPELINE II=1
// //             s_in.write(sample[t]);
// //         }

// //         // Run Conv1D
// //         conv1.forward(
// //             s_in, s_out,
// //             qcsnet2_cblk1_qconv1d_weights,
// //             qcsnet2_cblk1_qconv1d_scale_multiplier,
// //             qcsnet2_cblk1_qconv1d_right_shift,
// //             qcsnet2_cblk1_qconv1d_bias,           // zeros if bias disabled at export
// //             qcsnet2_input_zero_point,             // from qparams header
// //             qcsnet2_cblk1_qconv1d_weight_sum
// //         );

// //         // Drain and print Conv1D output per channel
// //         std::vector<int> y(OUT_CH * OUT_LEN);
// //         for (int idx = 0; idx < OUT_CH * OUT_LEN; ++idx) {
// //             y[idx] = (int)s_out.read();
// //         }

// //         std::cout << "\n=== Sample " << (i + 1)
// //                   << " — Conv1 output [" << OUT_CH << " x " << OUT_LEN << "] ===\n";
// //         for (int oc = 0; oc < OUT_CH; ++oc) {
// //             std::cout << "ch " << oc << ": ";
// //             for (int t = 0; t < OUT_LEN; ++t) {
// //                 std::cout << y[oc * OUT_LEN + t]
// //                           << (t + 1 < OUT_LEN ? ' ' : '\n');
// //             }
// //         }
// //     }

// //     std::cout << "\nDone (Conv1 only).\n";
// //     return 0;
// // }

// // ---------------- Main testbench ----------------

// // int main(int argc, char** argv) {
// //     using namespace hls4csnn1d_cblk_sd;

// //     // 1) Load ECG rows from folder
// //     // std::string folderPath = "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_train/smaller";
// //     std::string folderPath ="/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_singles/large";
// //     FileReader reader;
// //     reader.loadData(folderPath);


// //     // 2) Pull rows into an internal stream using original API
// //     hls::stream<array180_t> dataStream;
// //     hls::stream<array2848_t> outstream;
// //     reader.streamData(dataStream);   // original method, unchanged

// //     std::cout << "Finished stream data.\n";

// //     const int NUM_SAMPLES_LOADED = dataStream.size();
// //     if (NUM_SAMPLES_LOADED == 0) {
// //         std::cerr << "No rows loaded — aborting test.\n";
// //         return -1;
// //     }
// //     std::cout << "Loaded " << NUM_SAMPLES_LOADED << " ECG rows\n";

// //      Conv1D_SD<1, 16, 3, 1, CONV_IN_LENGTH1>       qcsnet2_cblk1_qconv1d_int;
     
// //     for (int i=0; i < NUM_SAMPLES_LOADED; i++) {

// //         const array180_t s_in = dataStream.read();
// //         /* 1st block */
   

// //     qcsnet2_cblk1_qconv1d_int.forward(s_in, outStream, 
// //                 qcsnet2_cblk1_qconv1d_weights,
// //                 qcsnet2_cblk1_qconv1d_scale_multiplier,
// //                 qcsnet2_cblk1_qconv1d_right_shift,
// //                 qcsnet2_cblk1_qconv1d_bias,
// //                 qcsnet2_cblk1_qconv1d_input_zero_point,
// //                 qcsnet2_cblk1_qconv1d_weight_sum);

// //     }
  

    


//     // BatchNorm1D_SD<OUT_CH1, FEATURE_LENGTH1>         qcsnet2_cblk1_bn1d_int;
//     // LIF1D_SD<OUT_CH1, FEATURE_LENGTH1>           qcsnet2_cblk1_lif1d_int;
//     // MaxPool1D_SD<2, 2, OUT_CH1, FEATURE_LENGTH1>     qcsnet2_cblk1_maxpool1d;

//     // /* 2nd block */
//     // Conv1D_SD<16, 24, 3, 1, CONV_IN_LENGTH2>      qcsnet2_cblk2_qconv1d_int;
//     // BatchNorm1D_SD<OUT_CH2, FEATURE_LENGTH2>         qcsnet2_cblk2_bn1d_int;
//     // LIF1D_SD<OUT_CH2, FEATURE_LENGTH2>           qcsnet2_cblk2_lif1d_int;
//     // MaxPool1D_SD<2, 2, OUT_CH2, FEATURE_LENGTH2>     qcsnet2_cblk2_maxpool1d;

//     // /* dense head */
//     // Linear1D_SD <LINEAR_IN_SIZE, 2>                  qcsnet2_lblk1_qlinear1d_int;
//     // LIF1D_SD   <2, 1>                               qcsnet2_lblk1_lif1d_int;

//     // return 0;


// // }




// // #ifndef _AP_UNUSED_PARAM
// // #define _AP_UNUSED_PARAM(x) (void)(x)
// // #endif
// // #include <hls_stream.h>
// // #include <ap_fixed.h>
// // #include <array>
// // #include <string>
// // #include <iostream>
// // #include <sys/stat.h>
// // #include <cstdlib>
// // #include <cerrno>
// // #include <cstring>
// // #include <fstream>
// // #include <vector>
// // #include <algorithm>
// // #include <chrono>
// // #include <iomanip>
// // #include <cmath>
// // #include <set>

// // // Include your project headers
// // #include "../../include/hls4csnn1d_sd/lif1d_params_search.h"
// // #include "../../include/hls4csnn1d_sd/constants_sd.h"
// // #include "../../include/hls4csnn1d_sd/filereader.h"
// // #include "../../include/hls4csnn1d_sd/cblk_sd/conv1d_sd.h"
// // #include "../../include/hls4csnn1d_sd/cblk_sd/batchnorm1d_sd.h"
// // #include "../../include/hls4csnn1d_sd/cblk_sd/lif1d_sd.h"
// // #include "../../include/hls4csnn1d_sd/cblk_sd/maxpool1d_sd.h"
// // #include "../../include/hls4csnn1d_sd/cblk_sd/nn2_cblk1_sd.h"
// // #include "../../include/hls4csnn1d_sd/cblk_sd/modeleval2_sd.h"
// // // Include the header that contains modelEval function
// // // #include "../your_model_header.h"


// // #include <memory>
// // #include <nlohmann/json.hpp>




// // // ─────────────────────────────────────────────────────────────
// // // main.cpp  —  Multi-Objective Parametric Search (Simplified)
// // //   • Direct model evaluation without AXI conversion
// // //   • Pareto-optimal frontier for multiple objectives
// // //   • Customizable objective weights for different use cases
// // // ─────────────────────────────────────────────────────────────

// // // --- globals for LIF parameters ---
// // ap_fixed_c g_cblk1_beta  = ap_fixed_c(0.5);
// // ap_fixed_c g_cblk1_theta = ap_fixed_c(0.5);
// // ap_fixed_c g_cblk2_beta  = ap_fixed_c(0.5);
// // ap_fixed_c g_cblk2_theta = ap_fixed_c(0.5);
// // ap_fixed_c g_lblk1_beta  = ap_fixed_c(0.5);
// // ap_fixed_c g_lblk1_theta = ap_fixed_c(0.5);

// // // Structure to store search results with multi-objective support
// // struct SearchResult {
// //     // Parameters
// //     float cblk1_beta, cblk1_theta;
// //     float cblk2_beta, cblk2_theta;
// //     float lblk1_beta, lblk1_theta;
    
// //     // Raw metrics
// //     int tp, tn, fp, fn;
    
// //     // Derived metrics (objectives we want to optimize)
// //     float precision;      // Maximize (reduce FP)
// //     float recall;         // Maximize (reduce FN) 
// //     float accuracy;       // Overall correctness
// //     float f1_score;       // Harmonic mean of precision/recall
    
// //     // Error metrics
// //     float specificity;    // TN/(TN+FP) - kept for completeness
// //     float fp_rate;        // FP/(FP+TN) 
// //     float fn_rate;        // FN/(FN+TP)
// //     int total_errors;     // FP + FN - Minimize
    
// //     // Pareto dominance
// //     bool is_pareto_optimal;
// //     int dominance_count;  // How many solutions this dominates
    
// //     // Timing
// //     double execution_time_ms;
    
// //     // Custom weighted score based on use case
// //     float weighted_score;
// // };

// // // Simplified objective weights structure
// // struct ObjectiveWeights {
// //     float w_precision;      // Weight for precision (reduces FP)
// //     float w_recall;         // Weight for recall (reduces FN)  
// //     float w_error_penalty;  // Penalty for total errors (FP + FN)
    
// //     ObjectiveWeights() {
// //         // Default: Equal importance to all three objectives
// //         w_precision = 0.35f;
// //         w_recall = 0.45f;
// //         w_error_penalty = 0.20f;
// //     }
    
// //     ObjectiveWeights(float precision, float recall, float error_penalty) {
// //         w_precision = precision;
// //         w_recall = recall;
// //         w_error_penalty = error_penalty;
// //     }
// // };

// // // External function declaration - adjust based on your actual model interface
// // ModelEvaluation2 modelEval;

// // // snap float to Q4.4 (0.0625) and return ap_fixed<8,4>
// // static inline ap_fixed_c q44(float x) {
// //     const float STEP = 0.0625f;
// //     float snapped = std::round(x / STEP) * STEP;
// //     if (snapped >  7.9375f) snapped =  7.9375f;
// //     if (snapped < -8.0000f) snapped = -8.0000f;
// //     return ap_fixed_c(snapped);
// // }

// // // Calculate weighted score based on multiple objectives
// // static float calculate_weighted_score(const SearchResult& r, const ObjectiveWeights& w, int total_samples) {
// //     // Normalize error rate to [0,1] range
// //     float error_rate = (float)r.total_errors / total_samples;
    
// //     // Simple weighted score: maximize precision and recall, minimize error rate
// //     float score = w.w_precision * r.precision + 
// //                   w.w_recall * r.recall - 
// //                   w.w_error_penalty * error_rate;
    
// //     return score;
// // }

// // // Check if solution A dominates solution B (for Pareto optimality)
// // static bool dominates(const SearchResult& a, const SearchResult& b) {
// //     // A dominates B if A is at least as good in all objectives and better in at least one
// //     // For our three objectives: maximize precision, maximize recall, minimize total_errors
// //     bool at_least_as_good = (a.precision >= b.precision) && 
// //                            (a.recall >= b.recall) && 
// //                            (a.total_errors <= b.total_errors);
    
// //     bool better_in_one = (a.precision > b.precision) || 
// //                         (a.recall > b.recall) || 
// //                         (a.total_errors < b.total_errors);
    
// //     return at_least_as_good && better_in_one;
// // }

// // // Find Pareto optimal solutions
// // static void find_pareto_front(std::vector<SearchResult>& results) {
// //     int n = results.size();
    
// //     // Reset all flags
// //     for (auto& r : results) {
// //         r.is_pareto_optimal = true;
// //         r.dominance_count = 0;
// //     }
    
// //     // Check dominance relationships
// //     for (int i = 0; i < n; ++i) {
// //         for (int j = 0; j < n; ++j) {
// //             if (i != j && dominates(results[j], results[i])) {
// //                 results[i].is_pareto_optimal = false;
// //             }
// //             if (i != j && dominates(results[i], results[j])) {
// //                 results[i].dominance_count++;
// //             }
// //         }
// //     }
// // }

// // // Print result with cleaner format
// // static void print_result(const SearchResult& r, bool is_header = false, bool show_pareto = false) {
// //     if (is_header) {
// //         std::cout << "\n"
// //                   << std::setw(8) << "Params" << " | "
// //                   << std::setw(6) << "TP" << " "
// //                   << std::setw(6) << "TN" << " "
// //                   << std::setw(6) << "FP" << " "
// //                   << std::setw(6) << "FN" << " | "
// //                   << std::setw(8) << "Prec" << " "
// //                   << std::setw(8) << "Recall" << " "
// //                   << std::setw(8) << "F1" << " | "
// //                   << std::setw(10) << "TotalErr" << " "
// //                   << std::setw(8) << "Score";
// //         if (show_pareto) std::cout << " " << std::setw(7) << "Pareto";
// //         std::cout << "\n" << std::string(110, '-') << "\n";
// //     } else {
// //         // Compact parameter display
// //         std::cout << std::fixed << std::setprecision(3)
// //                   << "L" << r.cblk1_theta << ","
// //                   << r.cblk2_theta << ","
// //                   << r.lblk1_theta << " | ";
        
// //         std::cout << std::setw(6) << r.tp << " "
// //                   << std::setw(6) << r.tn << " "
// //                   << std::setw(6) << r.fp << " "
// //                   << std::setw(6) << r.fn << " | ";
        
// //         std::cout << std::fixed << std::setprecision(4)
// //                   << std::setw(8) << r.precision << " "
// //                   << std::setw(8) << r.recall << " "
// //                   << std::setw(8) << r.f1_score << " | "
// //                   << std::setw(10) << r.total_errors << " "
// //                   << std::setw(8) << r.weighted_score;
        
// //         if (show_pareto) {
// //             if (r.is_pareto_optimal) {
// //                 std::cout << " ⭐(" << r.dominance_count << ")";
// //             } else {
// //                 std::cout << "       ";
// //             }
// //         }
// //         std::cout << "\n";
// //     }
// // }

// // // Save detailed results including Pareto front
// // static void save_results_to_csv(const std::vector<SearchResult>& results, 
// //                                const std::string& filename) {
// //     std::ofstream file(filename);
// //     if (!file.is_open()) {
// //         std::cerr << "Failed to open " << filename << " for writing\n";
// //         return;
// //     }
    
// //     // Comprehensive header
// //     file << "LIF1_Beta,LIF1_Theta,LIF2_Beta,LIF2_Theta,Head_Beta,Head_Theta,"
// //          << "TP,TN,FP,FN,Precision,Recall,Specificity,F1_Score,Accuracy,"
// //          << "FP_Rate,FN_Rate,Total_Errors,Weighted_Score,Is_Pareto_Optimal,"
// //          << "Dominance_Count,Time_ms\n";
    
// //     for (const auto& r : results) {
// //         file << r.cblk1_beta << "," << r.cblk1_theta << ","
// //              << r.cblk2_beta << "," << r.cblk2_theta << ","
// //              << r.lblk1_beta << "," << r.lblk1_theta << ","
// //              << r.tp << "," << r.tn << "," << r.fp << "," << r.fn << ","
// //              << r.precision << "," << r.recall << "," << r.specificity << ","
// //              << r.f1_score << "," << r.accuracy << ","
// //              << r.fp_rate << "," << r.fn_rate << "," << r.total_errors << ","
// //              << r.weighted_score << "," << (r.is_pareto_optimal ? 1 : 0) << ","
// //              << r.dominance_count << "," << r.execution_time_ms << "\n";
// //     }
    
// //     file.close();
// //     std::cout << "\n✅ Results saved to " << filename << "\n";
// // }

// // // Data cache structure to avoid reloading
// // struct DataCache {
// //     std::vector<array180_t> samples;
// //     std::vector<ap_fixed_c> labels;
// //     int num_samples;
    
// //     void loadFromReader(FileReader& reader) {
// //         // Load samples
// //         hls::stream<array180_t> tempStream;
// //         reader.streamData(tempStream);
        
// //         num_samples = tempStream.size();
// //         samples.clear();
// //         samples.reserve(num_samples);
        
// //         while (!tempStream.empty()) {
// //             samples.push_back(tempStream.read());
// //         }
        
// //         // Load labels
// //         hls::stream<ap_fixed_c> tempLabelStream;
// //         reader.streamLabel(tempLabelStream, true);
        
// //         labels.clear();
// //         labels.reserve(num_samples);
        
// //         while (!tempLabelStream.empty()) {
// //             labels.push_back(tempLabelStream.read());
// //         }
        
// //         std::cout << "✅ Cached " << num_samples << " samples and labels in memory\n";
// //     }
    
// //     void fillStream(hls::stream<array180_t>& stream) const {
// //         for (const auto& sample : samples) {
// //             stream.write(sample);
// //         }
// //     }
    
// //     void fillLabelStream(hls::stream<ap_fixed_c>& stream) const {
// //         for (const auto& label : labels) {
// //             stream.write(label);
// //         }
// //     }
// // };

// // // ADD THIS: Configuration tracking struct
// // struct ConfigKey {
// //     float c1_beta, c1_theta, c2_beta, c2_theta, h_beta, h_theta;
    
// //     bool operator<(const ConfigKey& other) const {
// //         if (c1_beta != other.c1_beta) return c1_beta < other.c1_beta;
// //         if (c1_theta != other.c1_theta) return c1_theta < other.c1_theta;
// //         if (c2_beta != other.c2_beta) return c2_beta < other.c2_beta;
// //         if (c2_theta != other.c2_theta) return c2_theta < other.c2_theta;
// //         if (h_beta != other.h_beta) return h_beta < other.h_beta;
// //         return h_theta < other.h_theta;
// //     }
// // };

// // // ADD THIS: Analysis function (simplified version)
// // void analyzeResults(const std::vector<SearchResult>& results, 
// //                    const DataCache& cache) {
// //     std::cout << "\n\n=== DETAILED ANALYSIS ===\n";
    
// //     // Class distribution
// //     int total_positive = 0, total_negative = 0;
// //     for (const auto& label : cache.labels) {
// //         if (label == 0) total_negative++;
// //         else total_positive++;
// //     }
    
// //     std::cout << "\nDataset Statistics:\n";
// //     std::cout << "  Total samples: " << cache.num_samples << "\n";
// //     std::cout << "  Positive samples: " << total_positive 
// //               << " (" << (100.0f * total_positive / cache.num_samples) << "%)\n";
// //     std::cout << "  Negative samples: " << total_negative 
// //               << " (" << (100.0f * total_negative / cache.num_samples) << "%)\n";
// // }

// // // Run a single test configuration - Using cached data
// // static SearchResult run_single_test(const DataCache& cache,
// //                                    float c1_beta, float c1_theta,
// //                                    float c2_beta, float c2_theta,
// //                                    float h_beta, float h_theta,
// //                                    const ObjectiveWeights& weights) {
// //     auto start_time = std::chrono::high_resolution_clock::now();
    
// //     // Set global parameters
// //     g_cblk1_beta  = q44(c1_beta);
// //     g_cblk1_theta = q44(c1_theta);
// //     g_cblk2_beta  = q44(c2_beta);
// //     g_cblk2_theta = q44(c2_theta);
// //     g_lblk1_beta  = q44(h_beta);
// //     g_lblk1_theta = q44(h_theta);
    
// //     // Create streams from cached data
// //     hls::stream<array180_t> inputStream;
// //     cache.fillStream(inputStream);

// //     hls4csnn1d_cblk_sd::NeuralNetwork2_Cblk1_sd model2;
    
// //     // Run model evaluation
// //     hls::stream<array2_t> outputStream;
// //     modelEval.evaluate(model2, inputStream, outputStream);
    
// //     // Calculate metrics
// //     SearchResult result;
// //     result.cblk1_beta = (float)g_cblk1_beta;
// //     result.cblk1_theta = (float)g_cblk1_theta;
// //     result.cblk2_beta = (float)g_cblk2_beta;
// //     result.cblk2_theta = (float)g_cblk2_theta;
// //     result.lblk1_beta = (float)g_lblk1_beta;
// //     result.lblk1_theta = (float)g_lblk1_theta;
    
// //     result.tp = result.tn = result.fp = result.fn = 0;
    
// //     // Compare outputs with cached labels
// //     for (int i = 0; i < cache.num_samples; ++i) {
// //         array2_t output = outputStream.read();
// //         ap_fixed_c true_label = cache.labels[i];
        
// //         int predicted = (output[1] > output[0]) ? 1 : 0;
// //         int actual = (true_label == 0) ? 0 : 1;
        
// //         if (actual == 1 && predicted == 1) result.tp++;
// //         else if (actual == 0 && predicted == 0) result.tn++;
// //         else if (actual == 0 && predicted == 1) result.fp++;
// //         else if (actual == 1 && predicted == 0) result.fn++;
// //     }
    
// //     // Calculate all metrics
// //     int total = result.tp + result.tn + result.fp + result.fn;
// //     int total_positive = result.tp + result.fn;
// //     int total_negative = result.tn + result.fp;
    
// //     result.accuracy = (float)(result.tp + result.tn) / total;
// //     result.precision = (result.tp + result.fp > 0) ? 
// //                       (float)result.tp / (result.tp + result.fp) : 0.0f;
// //     result.recall = (total_positive > 0) ? 
// //                    (float)result.tp / total_positive : 0.0f;
// //     result.specificity = (total_negative > 0) ? 
// //                         (float)result.tn / total_negative : 0.0f;
    
// //     // Error rates
// //     result.fp_rate = (total_negative > 0) ? 
// //                     (float)result.fp / total_negative : 0.0f;
// //     result.fn_rate = (total_positive > 0) ? 
// //                     (float)result.fn / total_positive : 0.0f;
    
// //     result.total_errors = result.fp + result.fn;
    
// //     // F1 score
// //     if (result.precision + result.recall > 0) {
// //         result.f1_score = 2.0f * (result.precision * result.recall) / 
// //                          (result.precision + result.recall);
// //     } else {
// //         result.f1_score = 0.0f;
// //     }
    
// //     // Calculate weighted score
// //     result.weighted_score = calculate_weighted_score(result, weights, total);
    
// //     auto end_time = std::chrono::high_resolution_clock::now();
// //     result.execution_time_ms = std::chrono::duration<double, std::milli>(
// //         end_time - start_time).count();
    
// //     return result;
// // }

// // int main() {
// //     std::cout << "\n=== Multi-Objective LIF Parameter Search ===\n";
// //     std::cout << "Optimizing for: HIGH RECALL + HIGH PRECISION + LOW TOTAL ERRORS\n";
    
// //     // Simplified weights for your three objectives
// //     ObjectiveWeights weights(0.35f, 0.45f, 0.20f);  // Precision, Recall, Error penalty
    
// //     std::cout << "\nOptimization weights:\n"
// //               << "  Precision weight: " << weights.w_precision << " (35%)\n"
// //               << "  Recall weight: " << weights.w_recall << " (45%)\n"
// //               << "  Error penalty: " << weights.w_error_penalty << " (20%)\n"
// //               << "  Formula: Score = " << weights.w_precision << "*Precision + " 
// //               << weights.w_recall << "*Recall - " << weights.w_error_penalty << "*ErrorRate\n";
    
// //     // Load data once and cache it
// //     std::string folderPath = "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_train/smaller";
// //     FileReader reader;
// //     reader.loadData(folderPath);
    
// //     // Cache the data in memory
// //     DataCache dataCache;
// //     dataCache.loadFromReader(reader);
    
// //     std::cout << "Data loading complete. Starting parameter search...\n";
    
// //     // Configuration flags - modify these as needed
// //     const bool USE_COARSE_SEARCH = true;  // Set to false for fine search
// //     const bool ENABLE_ADAPTIVE_REFINEMENT = true;  // Refine around best results
    
// //     // Define search ranges
// //     std::vector<float> beta_coarse = { 0.375f, 0.500f, 0.625f };
// //     std::vector<float> theta_coarse = { 0.375f, 0.500f, 0.625f, 0.750f };
    
// //     std::vector<float> beta_fine = {
// //         0.3125f, 0.375f, 0.4375f, 0.500f, 0.5625f, 0.625f, 0.6875f
// //     };
// //     std::vector<float> theta_fine = {
// //         0.3125f, 0.375f, 0.4375f, 0.500f, 0.5625f, 0.625f, 0.6875f, 0.750f, 0.8125f
// //     };
    
// //     std::vector<float>& beta_search = USE_COARSE_SEARCH ? beta_coarse : beta_fine;
// //     std::vector<float>& theta_search = USE_COARSE_SEARCH ? theta_coarse : theta_fine;
    
// //     std::cout << "\nSearch configuration:\n"
// //               << "  Search type: " << (USE_COARSE_SEARCH ? "COARSE" : "FINE") << "\n"
// //               << "  Beta values: " << beta_search.size() << "\n"
// //               << "  Theta values: " << theta_search.size() << "\n"
// //               << "  Adaptive refinement: " << (ENABLE_ADAPTIVE_REFINEMENT ? "ENABLED" : "DISABLED") << "\n";
    
// //     // Store all results
// //     std::vector<SearchResult> all_results;
// //     std::vector<SearchResult> phase1_results;  // For adaptive refinement
    
// //     // Progress tracking
// //     int total_configs = beta_search.size() * theta_search.size() *
// //                        beta_search.size() * theta_search.size() *
// //                        beta_search.size() * theta_search.size();
// //     int current_config = 0;
    
// //     std::cout << "\nPhase 1: Initial search\n";
// //     std::cout << "Total configurations to test: " << total_configs << "\n\n";
    
// //     // Track best solutions for each objective
// //     SearchResult best_precision, best_recall, best_weighted, best_minimal_errors, best_f1;
// //     best_precision.precision = -1;
// //     best_recall.recall = -1;
// //     best_weighted.weighted_score = -100;
// //     best_minimal_errors.total_errors = INT_MAX;
// //     best_f1.f1_score = -1;
    
// //     print_result(best_weighted, true, false);

// //     // 3. ADD THIS: Before your search loops
// //     std::set<ConfigKey> tested_configs;
    
// //     // Main search loops
// //     for (float c1_beta : beta_search) {
// //         for (float c1_theta : theta_search) {
// //             for (float c2_beta : beta_search) {
// //                 for (float c2_theta : theta_search) {
// //                     for (float h_beta : beta_search) {
// //                         for (float h_theta : theta_search) {
// //                             current_config++;

// //                              // 4. ADD THIS: Skip duplicate configurations
// //                             ConfigKey config_key{c1_beta, c1_theta, c2_beta, c2_theta, h_beta, h_theta};
// //                             if (tested_configs.count(config_key) > 0) {
// //                                 current_config--;  // Adjust counter
// //                                 continue;
// //                             }
// //                             tested_configs.insert(config_key);
                            
// //                             // Progress indicator
// //                             if (current_config % 50 == 0 || current_config == 1) {
// //                                 std::cout << "\n[Progress: " << current_config 
// //                                          << "/" << total_configs << " ("
// //                                          << std::fixed << std::setprecision(1)
// //                                          << (100.0 * current_config / total_configs) 
// //                                          << "%)]\n";
// //                             }
                            
// //                             // Run test with cached data
// //                             SearchResult result = run_single_test(dataCache,
// //                                 c1_beta, c1_theta,
// //                                 c2_beta, c2_theta,
// //                                 h_beta, h_theta,
// //                                 weights);
                            
// //                             all_results.push_back(result);
// //                             phase1_results.push_back(result);
                            
// //                             // Check if it's a new best for any objective
// //                             bool is_new_best = false;
// //                             std::string best_type = "";
                            
// //                             if (result.precision > best_precision.precision) {
// //                                 best_precision = result;
// //                                 is_new_best = true;
// //                                 best_type += "PREC ";
// //                             }
// //                             if (result.recall > best_recall.recall) {
// //                                 best_recall = result;
// //                                 is_new_best = true;
// //                                 best_type += "REC ";
// //                             }
// //                             if (result.f1_score > best_f1.f1_score) {
// //                                 best_f1 = result;
// //                                 is_new_best = true;
// //                                 best_type += "F1 ";
// //                             }
// //                             if (result.weighted_score > best_weighted.weighted_score) {
// //                                 best_weighted = result;
// //                                 is_new_best = true;
// //                                 best_type += "SCORE ";
// //                             }
// //                             if (result.total_errors < best_minimal_errors.total_errors) {
// //                                 best_minimal_errors = result;
// //                                 is_new_best = true;
// //                                 best_type += "ERR ";
// //                             }
                            
// //                             // Print if new best or periodic update
// //                             if (is_new_best || current_config % 50 == 0) {
// //                                 print_result(result, false, false);
// //                                 if (is_new_best) {
// //                                     std::cout << "  ⭐ NEW BEST: " << best_type << "\n";
// //                                 }
// //                             }
// //                         }
// //                     }
// //                 }
// //             }
// //         }
// //     }
    
// //     // Adaptive refinement phase
// //     if (ENABLE_ADAPTIVE_REFINEMENT && USE_COARSE_SEARCH) {
// //         std::cout << "\n\n=== Phase 2: Adaptive Refinement ===\n";
// //         std::cout << "Refining around best configurations...\n";
        
// //         // Select top configurations to refine around
// //         std::vector<SearchResult> candidates_to_refine;
// //         candidates_to_refine.push_back(best_weighted);
// //         candidates_to_refine.push_back(best_minimal_errors);
// //         candidates_to_refine.push_back(best_recall);
// //         candidates_to_refine.push_back(best_precision);
        
// //         // Remove duplicates
// //         std::sort(candidates_to_refine.begin(), candidates_to_refine.end(),
// //             [](const SearchResult& a, const SearchResult& b) {
// //                 return a.weighted_score > b.weighted_score;
// //             });
// //         auto last = std::unique(candidates_to_refine.begin(), candidates_to_refine.end(),
// //             [](const SearchResult& a, const SearchResult& b) {
// //                 return a.cblk1_theta == b.cblk1_theta && 
// //                        a.cblk2_theta == b.cblk2_theta && 
// //                        a.lblk1_theta == b.lblk1_theta;
// //             });
// //         candidates_to_refine.erase(last, candidates_to_refine.end());
        
// //         int refinement_count = 0;
// //         for (const auto& candidate : candidates_to_refine) {
// //             std::cout << "\nRefining around configuration with score " 
// //                       << candidate.weighted_score << "\n";
            
// //             // Define refinement range (±1 step around best)
// //             std::vector<float> refine_values = { -0.0625f, 0.0f, 0.0625f };
            
// //             for (float d1 : refine_values) {
// //                 for (float d2 : refine_values) {
// //                     for (float d3 : refine_values) {
// //                         float new_theta1 = candidate.cblk1_theta + d1;
// //                         float new_theta2 = candidate.cblk2_theta + d2;
// //                         float new_theta3 = candidate.lblk1_theta + d3;
                        
// //                         // Keep within valid range
// //                         if (new_theta1 < 0.25f || new_theta1 > 0.875f) continue;
// //                         if (new_theta2 < 0.25f || new_theta2 > 0.875f) continue;
// //                         if (new_theta3 < 0.25f || new_theta3 > 0.875f) continue;
                        
// //                         refinement_count++;
                        
// //                         SearchResult result = run_single_test(dataCache,
// //                             candidate.cblk1_beta, new_theta1,
// //                             candidate.cblk2_beta, new_theta2,
// //                             candidate.lblk1_beta, new_theta3,
// //                             weights);
                        
// //                         all_results.push_back(result);
                        
// //                         // Update bests
// //                         if (result.weighted_score > best_weighted.weighted_score) {
// //                             best_weighted = result;
// //                             print_result(result, false, false);
// //                             std::cout << "  ⭐ NEW BEST SCORE in refinement!\n";
// //                         }
// //                         if (result.total_errors < best_minimal_errors.total_errors) {
// //                             best_minimal_errors = result;
// //                             print_result(result, false, false);
// //                             std::cout << "  ⭐ NEW MINIMAL ERRORS in refinement!\n";
// //                         }
// //                     }
// //                 }
// //             }
// //         }
// //         std::cout << "\nRefinement complete. Tested " << refinement_count 
// //                   << " additional configurations.\n";
// //     }
    
// //     // Find Pareto optimal solutions
// //     std::cout << "\n\nFinding Pareto optimal solutions...\n";
// //     find_pareto_front(all_results);
    
// //     // Extract and sort Pareto optimal solutions
// //     std::vector<SearchResult> pareto_solutions;
// //     for (const auto& r : all_results) {
// //         if (r.is_pareto_optimal) {
// //             pareto_solutions.push_back(r);
// //         }
// //     }
    
// //     // Sort Pareto solutions by dominance count
// //     std::sort(pareto_solutions.begin(), pareto_solutions.end(),
// //               [](const SearchResult& a, const SearchResult& b) {
// //                   return a.dominance_count > b.dominance_count;
// //               });
    
// //     // Display Pareto front
// //     std::cout << "\n\n=== PARETO OPTIMAL SOLUTIONS (" 
// //               << pareto_solutions.size() << " found) ===\n";
// //     print_result(pareto_solutions[0], true, true);
// //     for (size_t i = 0; i < std::min(size_t(20), pareto_solutions.size()); ++i) {
// //         print_result(pareto_solutions[i], false, true);
// //     }

    
// //     // Display best solutions for each individual objective
// //     std::cout << "\n\n=== BEST SOLUTIONS BY OBJECTIVE ===\n";
    
// //     std::cout << "\n📊 HIGHEST PRECISION (minimize false positives):\n";
// //     print_result(best_precision, true, false);
// //     print_result(best_precision, false, false);
    
// //     std::cout << "\n📊 HIGHEST RECALL (minimize false negatives):\n";
// //     print_result(best_recall, true, false);
// //     print_result(best_recall, false, false);
    
// //     std::cout << "\n📊 LOWEST TOTAL ERRORS (FP + FN):\n";
// //     print_result(best_minimal_errors, true, false);
// //     print_result(best_minimal_errors, false, false);
    
// //     std::cout << "\n📊 HIGHEST F1 SCORE (harmonic mean of precision/recall):\n";
// //     print_result(best_f1, true, false);
// //     print_result(best_f1, false, false);
    
// //     std::cout << "\n📊 BEST WEIGHTED SCORE (your combined objective):\n";
// //     print_result(best_weighted, true, false);
// //     print_result(best_weighted, false, false);
    
// //     // Save all results
// //     std::string timestamp = std::to_string(
// //         std::chrono::system_clock::now().time_since_epoch().count());
// //     std::string filename = "multi_objective_lif_search_" + timestamp + ".csv";
// //     save_results_to_csv(all_results, filename);


// //      // ADD THIS: After finding Pareto front
// //     analyzeResults(all_results, dataCache);
    
    
// //     // Generate configurations for different use cases
// //     std::cout << "\n\n=== RECOMMENDED CONFIGURATIONS ===\n";
    
// //     std::cout << "\n// For HIGH RECALL (catch most positives, minimize FN):\n"
// //               << "static const ap_fixed_c qcsnet2_cblk1_leaky_beta      = "
// //               << best_recall.cblk1_beta << ";\n"
// //               << "static const ap_fixed_c qcsnet2_cblk1_leaky_threshold = "
// //               << best_recall.cblk1_theta << ";\n"
// //               << "static const ap_fixed_c qcsnet2_cblk2_leaky_beta      = "
// //               << best_recall.cblk2_beta << ";\n"
// //               << "static const ap_fixed_c qcsnet2_cblk2_leaky_threshold = "
// //               << best_recall.cblk2_theta << ";\n"
// //               << "static const ap_fixed_c qcsnet2_lblk1_leaky_beta      = "
// //               << best_recall.lblk1_beta << ";\n"
// //               << "static const ap_fixed_c qcsnet2_lblk1_leaky_threshold = "
// //               << best_recall.lblk1_theta << ";\n";
    
// //     std::cout << "\n// For HIGH PRECISION (minimize false alarms, minimize FP):\n"
// //               << "static const ap_fixed_c qcsnet2_cblk1_leaky_beta      = "
// //               << best_precision.cblk1_beta << ";\n"
// //               << "static const ap_fixed_c qcsnet2_cblk1_leaky_threshold = "
// //               << best_precision.cblk1_theta << ";\n"
// //               << "static const ap_fixed_c qcsnet2_cblk2_leaky_beta      = "
// //               << best_precision.cblk2_beta << ";\n"
// //               << "static const ap_fixed_c qcsnet2_cblk2_leaky_threshold = "
// //               << best_precision.cblk2_theta << ";\n"
// //               << "static const ap_fixed_c qcsnet2_lblk1_leaky_beta      = "
// //               << best_precision.lblk1_beta << ";\n"
// //               << "static const ap_fixed_c qcsnet2_lblk1_leaky_threshold = "
// //               << best_precision.lblk1_theta << ";\n";
    
// //     std::cout << "\n// For BALANCED PERFORMANCE (best weighted score):\n"
// //               << "static const ap_fixed_c qcsnet2_cblk1_leaky_beta      = "
// //               << best_weighted.cblk1_beta << ";\n"
// //               << "static const ap_fixed_c qcsnet2_cblk1_leaky_threshold = "
// //               << best_weighted.cblk1_theta << ";\n"
// //               << "static const ap_fixed_c qcsnet2_cblk2_leaky_beta      = "
// //               << best_weighted.cblk2_beta << ";\n"
// //               << "static const ap_fixed_c qcsnet2_cblk2_leaky_threshold = "
// //               << best_weighted.cblk2_theta << ";\n"
// //               << "static const ap_fixed_c qcsnet2_lblk1_leaky_beta      = "
// //               << best_weighted.lblk1_beta << ";\n"
// //               << "static const ap_fixed_c qcsnet2_lblk1_leaky_threshold = "
// //               << best_weighted.lblk1_theta << ";\n";
    
// //     // Performance summary
// //     std::cout << "\n\n=== PERFORMANCE SUMMARY ===\n";
// //     std::cout << "\nBest Recall Configuration:\n"
// //               << "  Recall: " << best_recall.recall << " (FN: " << best_recall.fn << ")\n"
// //               << "  Precision: " << best_recall.precision << " (FP: " << best_recall.fp << ")\n"
// //               << "  F1: " << best_recall.f1_score << "\n";
    
// //     std::cout << "\nBest Precision Configuration:\n"
// //               << "  Precision: " << best_precision.precision << " (FP: " << best_precision.fp << ")\n"
// //               << "  Recall: " << best_precision.recall << " (FN: " << best_precision.fn << ")\n"
// //               << "  F1: " << best_precision.f1_score << "\n";
    
// //     std::cout << "\nBest Weighted Score Configuration:\n"
// //               << "  Weighted Score: " << best_weighted.weighted_score << "\n"
// //               << "  Precision: " << best_weighted.precision << " (FP: " << best_weighted.fp << ")\n"
// //               << "  Recall: " << best_weighted.recall << " (FN: " << best_weighted.fn << ")\n"
// //               << "  F1: " << best_weighted.f1_score << "\n"
// //               << "  Total Errors: " << best_weighted.total_errors << "\n";
    
// //     // Visualization suggestion
// //     std::cout << "\n\n=== VISUALIZATION SUGGESTIONS ===\n";
// //     std::cout << "To visualize the multi-objective trade-offs, plot:\n"
// //               << "1. Precision vs Recall scatter plot (with Pareto front highlighted)\n"
// //               << "2. FP vs FN scatter plot\n"
// //               << "3. 3D plot: Precision, Recall, Specificity\n"
// //               << "4. Parallel coordinates plot for all objectives\n";
    
// //     std::cout << "\nPython snippet for visualization:\n"
// //               << "```python\n"
// //               << "import pandas as pd\n"
// //               << "import matplotlib.pyplot as plt\n"
// //               << "from mpl_toolkits.mplot3d import Axes3D\n\n"
// //               << "df = pd.read_csv('" << filename << "')\n"
// //               << "pareto = df[df['Is_Pareto_Optimal'] == 1]\n\n"
// //               << "# 2D Precision-Recall trade-off\n"
// //               << "plt.figure(figsize=(10, 8))\n"
// //               << "plt.scatter(df['Recall'], df['Precision'], alpha=0.5, label='All configs')\n"
// //               << "plt.scatter(pareto['Recall'], pareto['Precision'], \n"
// //               << "           color='red', s=100, label='Pareto optimal')\n"
// //               << "plt.xlabel('Recall')\n"
// //               << "plt.ylabel('Precision')\n"
// //               << "plt.title('Precision-Recall Trade-off')\n"
// //               << "plt.legend()\n"
// //               << "plt.grid(True)\n"
// //               << "plt.show()\n"
// //               << "```\n";
    
// //     return 0;
// // }




// // // #ifndef _AP_UNUSED_PARAM
// // // #define _AP_UNUSED_PARAM(x) (void)(x)
// // // #endif

// // // #include <iostream>
// // // #include <cmath>
// // // #include <memory>
// // // #include <algorithm>
// // // #include <ap_fixed.h>
// // // #include <hls_stream.h>
// // // #include <nlohmann/json.hpp>

// // // #include "../../include/hls4csnn1d_sd/constants_sd.h"
// // // #include "../../include/hls4csnn1d_sd/filereader.h"
// // // #include "../../include/hls4csnn1d_sd/cblk_sd/conv1d_sd.h"
// // // #include "../../include/hls4csnn1d_sd/cblk_sd/batchnorm1d_sd.h"
// // // #include "../../include/hls4csnn1d_sd/cblk_sd/lif1d_sd.h"
// // // #include "../../include/hls4csnn1d_sd/cblk_sd/maxpool1d_sd.h"
// // // #include "../../include/hls4csnn1d_sd/cblk_sd/nn2_cblk1_sd.h"
// // // #include "../../include/hls4csnn1d_sd/cblk_sd/modeleval2_sd.h"


// // // /**
// // //  * Use the hls4csnn1d_bm namespace for brevity.
// // //  */
// // // using namespace hls4csnn1d_cblk_sd;

// // // /**
// // //  * Helper function to print the JSON map for debugging.
// // //  */
// // // void printJsonMap(const JsonMap& jsonMap) {
// // //     for (const auto& [key, value] : jsonMap) {
// // //         std::cout << "Key: " << key << "\n";
// // //         std::cout << "Value: " << value.dump(4) << "\n"; // Pretty print with 4-space indentation
// // //         std::cout << "---------------------------------------------\n";
// // //     }
// // // }

// // // int main() {
// // //     // Paths to input data and trained model weights/dimensions
// // //     std::string folderPath = "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_train/smaller";
  

// // //     // Instantiate FileReader to load input data
// // //     FileReader reader;

// // //     // Create and initialize the neural network

// // //     NeuralNetwork2_Cblk1_sd model2;

  
// // //     // Load the dataset
// // //     reader.loadData(folderPath);

// // //     // Streams for data, labels, and predictions
// // //     hls::stream<array180_t> dataStream;
// // //     hls::stream<array2_t> outStream;

// // //     // Stream the data into the input streams
// // //     reader.streamData(dataStream);
// // //     std::cout << "Finished stream data.\n";

// // //     const int NUM_SAMPLES_LOADED = dataStream.size();
// // //     if (NUM_SAMPLES_LOADED == 0) {
// // //         std::cerr << "No rows loaded — aborting test.\n";
// // //         return -1;
// // //     }
// // //     std::cout << "Loaded " << NUM_SAMPLES_LOADED << " ECG rows\n";


    
// // //     // Evaluate the model using the provided data and labels
// // //     ModelEvaluation2 modelEval;


// // //     modelEval.evaluate(model2, dataStream, outStream);

// // //     hls::stream<ap_fixed_c> labelStream;
// // //     reader.streamLabel(labelStream, true);

// // //         // Calculate metrics
// // //         int true_positive = 0;
// // //         int true_negative = 0;
// // //         int false_positive = 0;
// // //         int false_negative = 0;

// // //         // Add counters before the loop
// // //         int actual_0_count = 0;
// // //         int actual_1_count = 0;
// // //         int pred_0_count = 0;
// // //         int pred_1_count = 0;
// // //         for (int i = 0; i < NUM_SAMPLES_LOADED; ++i) {
// // //             array2_t result = outStream.read();
// // //             ap_fixed_c true_label = labelStream.read();
// // //             // Get predicted class (0 or 1) based on which logit is higher
// // //             int predicted = (result[1] > result[0]) ? 1 : 0;
// // //             int actual = (true_label == 0) ? 0 : 1;  // Ensure binary conversion here too
// // //             // Count actual and predicted classes
// // //             if (actual == 0) actual_0_count++; else actual_1_count++;
// // //             if (predicted == 0) pred_0_count++; else pred_1_count++;
            
// // //             // // Debug: print first few predictions vs actual
// // //             // if (i < 10) {
// // //             //     std::cout << "Row " << i << ": logits=[" << result[0] << ", " << result[1] 
// // //             //             << "], predicted=" << predicted << ", actual=" << actual 
// // //             //             << ", true_label=" << true_label << "\n";
// // //             // }

// // //             // // Get predicted class (0 or 1) based on which logit is higher
// // //             // int predicted = (result[1] > result[0]) ? 1 : 0;
// // //             // int actual = (int)true_label;
            
// // //             // Update confusion matrix
// // //             if (actual == 1 && predicted == 1) true_positive++;
// // //             else if (actual == 0 && predicted == 0) true_negative++;
// // //             else if (actual == 0 && predicted == 1) false_positive++;
// // //             else if (actual == 1 && predicted == 0) false_negative++;
// // //         }
        
// // //         // Calculate metrics
// // //         float accuracy = (float)(true_positive + true_negative) / NUM_SAMPLES_LOADED;
// // //         float precision = (true_positive + false_positive > 0) ? 
// // //                         (float)true_positive / (true_positive + false_positive) : 0.0f;
// // //         float recall = (true_positive + false_negative > 0) ? 
// // //                     (float)true_positive / (true_positive + false_negative) : 0.0f;
        
// // //         std::cout << "\n=== Classification Metrics ===\n";
// // //         std::cout << "Confusion Matrix:\n";
// // //         std::cout << "  TP: " << true_positive << ", FP: " << false_positive << "\n";
// // //         std::cout << "  FN: " << false_negative << ", TN: " << true_negative << "\n";
// // //         std::cout << "Accuracy:  " << std::fixed << std::setprecision(4) << accuracy << "\n";
// // //         std::cout << "Precision: " << std::fixed << std::setprecision(4) << precision << "\n";
// // //         std::cout << "Recall:    " << std::fixed << std::setprecision(4) << recall << "\n";
        
        
// // //         //     std::cout << "\nTest PASSED — " << NUM_SAMPLES_LOADED << " rows processed.\n";
// // //         //     return 0;
// // //         // } else {
// // //         //     std::cout << "Test FAILED (got " << labelStream.size()
// // //         //               << " rows).\n";
// // //         //     return -1;
// // //         // }
    



// // //     return 0;
// // // }







