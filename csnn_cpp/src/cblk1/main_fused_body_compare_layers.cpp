// testbench_dump_layers24.cpp
// Layer-by-layer output dumper for QCSNN24 RR→Both two-stage model
// Usage: ./testbench_dump_layers24 [output_dir] [data_folder]

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <vector>
#include "hls_stream.h"
#include "ap_int.h"

// Include layer implementations
#include "../../include/hls4csnn1d_sd/model24/cblk_sd/conv1d_sd.h"
#include "../../include/hls4csnn1d_sd/model24/cblk_sd/batchnorm1d_sd.h"
#include "../../include/hls4csnn1d_sd/model24/cblk_sd/lif1d_integer.h"
#include "../../include/hls4csnn1d_sd/model24/cblk_sd/maxpool1d_sd.h"
#include "../../include/hls4csnn1d_sd/model24/cblk_sd/linear1d_sd.h"
#include "../../include/hls4csnn1d_sd/model24/cblk_sd/quantidentity1d_sd.h"

// Include constants and weights
#include "../../include/hls4csnn1d_sd/model24/constants24_sd.h"
#include "../../include/hls4csnn1d_sd/model24/filereader24.h"
#include "../../include/hls4csnn1d_sd/model24/cblk_sd/includeheaders24_sd.h"

using namespace hls4csnn1d_cblk_sd;

/* ================================================================
 *  Constants for RR→Both architecture
 * ================================================================ */
static const int RR_DIM = 4;
static const int SIGNAL_LEN = FIXED_LENGTH1;                    // 180
static const int TRUNK_OUT = 480;                               // 24×20
static const int STAGE1_IN = TRUNK_OUT + RR_DIM;                // 484
static const int STAGE2_IN = TRUNK_OUT + RR_DIM;                // 484

/* ================================================================
 *  Helper: Write layer output to file (pass-through)
 * ================================================================ */
template<int N_CHANNELS, int LENGTH>
void tee_and_write(hls::stream<ap_int8_c>& in,
                   hls::stream<ap_int8_c>& out,
                   const std::string& filename,
                   int sample_num,
                   bool write_header,
                   bool should_write) {
    std::ofstream outfile;
    if (should_write) {
        if (write_header) {
            outfile.open(filename);
            if (outfile.is_open()) {
                outfile << "# shape: (N, " << N_CHANNELS << ", " << LENGTH << "), mode: int\n";
            }
        } else {
            outfile.open(filename, std::ios::app);
        }
        
        if (!outfile.is_open()) {
            std::cerr << "Error: Cannot open " << filename << "\n";
            should_write = false;
        } else {
            outfile << "=== sample " << sample_num << " ===\n";
        }
    }
    
    for (int c = 0; c < N_CHANNELS; ++c) {
        if (should_write) outfile << "ch " << c << ":";
        for (int l = 0; l < LENGTH; ++l) {
            ap_int8_c val = in.read();
            out.write(val);
            if (should_write) outfile << " " << int(val);
        }
        if (should_write) outfile << "\n";
    }
    
    if (should_write) outfile.close();
}

/* ================================================================
 *  Helper: Write spike output (normalize to binary for comparison)
 * ================================================================ */
template<int N_CHANNELS, int LENGTH>
void tee_and_write_spikes(hls::stream<ap_int8_c>& in,
                          hls::stream<ap_int8_c>& out,
                          const std::string& filename,
                          int sample_num,
                          bool write_header,
                          bool should_write) {
    std::ofstream outfile;
    if (should_write) {
        if (write_header) {
            outfile.open(filename);
            if (outfile.is_open()) {
                outfile << "# shape: (N, " << N_CHANNELS << ", " << LENGTH << "), mode: int\n";
            }
        } else {
            outfile.open(filename, std::ios::app);
        }
        
        if (!outfile.is_open()) {
            std::cerr << "Error: Cannot open " << filename << "\n";
            should_write = false;
        } else {
            outfile << "=== sample " << sample_num << " ===\n";
        }
    }
    
    for (int c = 0; c < N_CHANNELS; ++c) {
        if (should_write) outfile << "ch " << c << ":";
        for (int l = 0; l < LENGTH; ++l) {
            ap_int8_c val = in.read();
            out.write(val);  // Pass through original value
            
            if (should_write) {
                // Normalize to binary: >0 → 1, else → 0
                int binary_val = (val > 0) ? 1 : 0;
                outfile << " " << binary_val;
            }
        }
        if (should_write) outfile << "\n";
    }
    
    if (should_write) outfile.close();
}

/* ================================================================
 *  Helper: Simple passthrough (no write)
 * ================================================================ */
template<int SIZE>
void passthrough(hls::stream<ap_int8_c>& in, hls::stream<ap_int8_c>& out) {
    for (int i = 0; i < SIZE; ++i) {
        out.write(in.read());
    }
}

/* ================================================================
 *  Helper: Accumulate buffer to file
 * ================================================================ */
template<int N_CHANNELS, int LENGTH>
void append_buffer_to_file(const ap_int<32>* buf,
                          const std::string& filename,
                          int sample_num,
                          bool write_header) {
    std::ofstream outfile;
    if (write_header) {
        outfile.open(filename);
        if (!outfile.is_open()) {
            std::cerr << "Error: Cannot open " << filename << "\n";
            return;
        }
        outfile << "# shape: (N, " << N_CHANNELS << ", " << LENGTH << "), mode: int\n";
    } else {
        outfile.open(filename, std::ios::app);
        if (!outfile.is_open()) {
            std::cerr << "Error: Cannot open " << filename << "\n";
            return;
        }
    }
    
    outfile << "=== sample " << sample_num << " ===\n";
    
    for (int c = 0; c < N_CHANNELS; ++c) {
        outfile << "ch " << c << ":";
        for (int l = 0; l < LENGTH; ++l) {
            outfile << " " << int(buf[c * LENGTH + l]);
        }
        outfile << "\n";
    }
    
    outfile.close();
}

/* ================================================================
 *  Gate helpers (same as model)
 * ================================================================ */
static inline ap_int8_c gate_abnormal(ap_int<16> sum_norm, ap_int<16> sum_abn) {
    return (sum_abn > sum_norm) ? ap_int8_c(1) : ap_int8_c(0);
}

static inline ap_int8_c argmax4(ap_int<16> s0, ap_int<16> s1, ap_int<16> s2, ap_int<16> s3) {
    ap_int8_c best = 0;
    ap_int<16> bestv = s0;
    if (s1 > bestv) { bestv = s1; best = 1; }
    if (s2 > bestv) { bestv = s2; best = 2; }
    if (s3 > bestv) { bestv = s3; best = 3; }
    return best;
}

/* ================================================================
 *  Main Testbench
 * ================================================================ */
int main(int argc, char** argv) {
    std::cout << "\n=== QCSNN24 RR→Both Layer-by-Layer Dumper (RR Bypass Fix) ===\n\n";

    // Parse arguments
    std::string out_dir = "../../../../compare_fusedbody_outputs/";
    if (argc >= 2) out_dir = argv[1];
    if (out_dir.back() != '/') out_dir += '/';

    std::string folderPath =
        "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_intra_patient_4class_180_center90_rr_features_filtered/single_test";
    if (argc >= 3) folderPath = argv[2];

    // Load data
    FileReader reader;
    reader.loadData(folderPath);

    hls::stream<array180rr_t> dataStreamInternal;
    reader.streamData(dataStreamInternal);

#ifndef __SYNTHESIS__
    const int NUM_SAMPLES_LOADED = dataStreamInternal.size();
#else
    const int NUM_SAMPLES_LOADED = NUM_SAMPLES;
#endif

    if (NUM_SAMPLES_LOADED <= 0) {
        std::cerr << "No data loaded!\n";
        return -1;
    }

    std::cout << "Loaded " << NUM_SAMPLES_LOADED << " ECG samples\n";
    std::cout << "Output directory: " << out_dir << "\n\n";

    // Pull all samples into memory
    std::vector<array180rr_t> all_samples;
    all_samples.reserve(NUM_SAMPLES_LOADED);
    for (int i = 0; i < NUM_SAMPLES_LOADED; ++i) {
        all_samples.push_back(dataStreamInternal.read());
    }

    // Ground-truth labels (4-class)
    hls::stream<ap_int8_c> gtLabelStream;
    reader.streamLabel(gtLabelStream, /*binary=*/false);

    // Confusion matrices
    const int N2 = 2, N4 = 4;
    long cm2[N2][N2] = {{0,0},{0,0}};
    long cm4[N4][N4] = {{0}};
    long routed_count = 0, total_count = 0;

    const int LOCAL_NUM_STEPS = 10;

    // Process each sample
    for (int sample_idx = 0; sample_idx < NUM_SAMPLES_LOADED; ++sample_idx) {
        std::cout << "\n========================================\n";
        std::cout << "Processing Sample " << (sample_idx + 1) << "/" << NUM_SAMPLES_LOADED << "\n";
        std::cout << "========================================\n";

        array180rr_t sample = all_samples[sample_idx];
        bool is_first_sample = (sample_idx == 0);

        // ===== CHANGE 1: Extract signal (180) only =====
        ap_int8_c sig_buf[SIGNAL_LEN];
        for (int i = 0; i < SIGNAL_LEN; ++i) sig_buf[i] = sample[i];

        // ===== CHANGE 2: Get pre-quantized RR from reader (with correct head scales) =====
        array_rr_t rr_s1 = reader.getRRStage1(sample_idx);  // Stage 1 scale
        array_rr_t rr_s2 = reader.getRRStage2(sample_idx);  // Stage 2 scale

        // ======================== INSTANTIATE LAYERS ========================
        // Trunk layers (fresh per sample to reset LIF states)
        Conv1D_SD<1, 16, 3, 1, SIGNAL_LEN>       trunk_conv1;
        BatchNorm1D_SD<16, 178>                  trunk_bn1;
        LIF1D_SD_Integer<16, 178>                trunk_lif1;
        MaxPool1D_SD<2, 2, 16, 178>              trunk_mp1;

        QuantIdentityPerTensor_Int8<16, 89>      trunk_qi2;
        Conv1D_SD<16, 16, 3, 1, 89>              trunk_conv2;
        BatchNorm1D_SD<16, 87>                   trunk_bn2;
        LIF1D_SD_Integer<16, 87>                 trunk_lif2;
        MaxPool1D_SD<2, 2, 16, 87>               trunk_mp2;

        QuantIdentityPerTensor_Int8<16, 43>      trunk_qi3;
        Conv1D_SD<16, 24, 3, 1, 43>              trunk_conv3;
        BatchNorm1D_SD<24, 41>                   trunk_bn3;
        LIF1D_SD_Integer<24, 41>                 trunk_lif3;
        MaxPool1D_SD<2, 2, 24, 41>               trunk_mp3;

        // ===== CHANGE 3: Binary head - QuantIdentity now processes trunk only (480) =====
        QuantIdentityPerTensor_Int8<TRUNK_OUT, 1>  bin_qi;   // 480, not 484
        Linear1D_SD<STAGE1_IN, 2>                  bin_fc;   // Still 484->2
        LIF1D_SD_Integer<2, 1>                     bin_lif;

        // ===== CHANGE 4: Multi head - QuantIdentity now processes trunk only (480) =====
        QuantIdentityPerTensor_Int8<TRUNK_OUT, 1>  multi_qi1;  // 480, not 484
        Linear1D_SD<STAGE2_IN, 128>                multi_fc1;  // Still 484->128
        LIF1D_SD_Integer<128, 1>                   multi_lif1;

        QuantIdentityPerTensor_Int8<128, 1>        multi_qi2;
        Linear1D_SD<128, 4>                        multi_fc2;
        LIF1D_SD_Integer<4, 1>                     multi_lif2;

        // Cache trunk output for stage-2
        ap_int8_c body_cache[LOCAL_NUM_STEPS][TRUNK_OUT];

        // Binary head accumulators
        ap_int<16> sum_bin0 = 0, sum_bin1 = 0;

        // Ground truth
        int y_true4 = (int)gtLabelStream.read();
        int y_true2 = (y_true4 > 0) ? 1 : 0;

        // ======================== STAGE-1: TRUNK + BINARY HEAD ========================
        for (int step = 0; step < LOCAL_NUM_STEPS; ++step) {
            bool should_write = (step == 0) || (step == LOCAL_NUM_STEPS - 1);
            std::string step_suffix = "";
            if (should_write) {
                char buf[16];
                std::snprintf(buf, sizeof(buf), "_step%02d", step);
                step_suffix = buf;
            }

            // --- Input stream (signal only, 180) ---
            hls::stream<ap_int8_c> s_input("s_input");
            for (int i = 0; i < SIGNAL_LEN; ++i) {
                s_input.write(sig_buf[i]);
            }

            // Dump input
            hls::stream<ap_int8_c> s_input_pass("s_input_pass");
            if (should_write) {
                tee_and_write<1, SIGNAL_LEN>(
                    s_input, s_input_pass,
                    out_dir + "input" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                passthrough<SIGNAL_LEN>(s_input, s_input_pass);
            }

            // ---- CONV1 ----
            hls::stream<ap_int8_c> s_conv1("s_conv1");
            trunk_conv1.forward(
                s_input_pass, s_conv1,
                qcsnet24_cblk1_qconv1d_weights,
                qcsnet24_cblk1_qconv1d_scale_multiplier,
                qcsnet24_cblk1_qconv1d_right_shift,
                qcsnet24_cblk1_qconv1d_bias,
                qcsnet24_cblk1_qconv1d_input_zero_point,
                qcsnet24_cblk1_qconv1d_weight_sum);

            hls::stream<ap_int8_c> s_conv1_pass("s_conv1_pass");
            if (should_write) {
                tee_and_write<16, 178>(
                    s_conv1, s_conv1_pass,
                    out_dir + "conv1" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                passthrough<16 * 178>(s_conv1, s_conv1_pass);
            }

            // ---- BN1 ----
            hls::stream<ap_int8_c> s_bn1("s_bn1");
            trunk_bn1.forward(
                s_conv1_pass, s_bn1,
                qcsnet24_cblk1_batch_norm_weight,
                qcsnet24_cblk1_batch_norm_bias,
                qcsnet24_cblk1_batch_norm_scale_multiplier,
                qcsnet24_cblk1_batch_norm_right_shift);

            hls::stream<ap_int8_c> s_bn1_pass("s_bn1_pass");
            if (should_write) {
                tee_and_write<16, 178>(
                    s_bn1, s_bn1_pass,
                    out_dir + "bn1" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                passthrough<16 * 178>(s_bn1, s_bn1_pass);
            }

            // ---- LIF1 ----
            hls::stream<ap_int8_c> s_lif1("s_lif1");
            trunk_lif1.forward(
                s_bn1_pass, s_lif1,
                qcsnet24_cblk1_leaky_beta_int,
                qcsnet24_cblk1_leaky_theta_int,
                qcsnet24_cblk1_leaky_scale_int);

            hls::stream<ap_int8_c> s_lif1_pass("s_lif1_pass");
            if (should_write) {
                tee_and_write_spikes<16, 178>(
                    s_lif1, s_lif1_pass,
                    out_dir + "lif1" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                passthrough<16 * 178>(s_lif1, s_lif1_pass);
            }

            // ---- MP1 ----
            hls::stream<ap_int8_c> s_mp1("s_mp1");
            trunk_mp1.forward(s_lif1_pass, s_mp1);

            hls::stream<ap_int8_c> s_mp1_pass("s_mp1_pass");
            if (should_write) {
                tee_and_write<16, 89>(
                    s_mp1, s_mp1_pass,
                    out_dir + "mp1" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                passthrough<16 * 89>(s_mp1, s_mp1_pass);
            }

            // ---- QI2 ----
            hls::stream<ap_int8_c> s_qi2("s_qi2");
            trunk_qi2.forward(s_mp1_pass, s_qi2, qcsnet24_cblk2_input_act_scale_int);

            hls::stream<ap_int8_c> s_qi2_pass("s_qi2_pass");
            if (should_write) {
                tee_and_write<16, 89>(
                    s_qi2, s_qi2_pass,
                    out_dir + "qi2" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                passthrough<16 * 89>(s_qi2, s_qi2_pass);
            }

            // ---- CONV2 ----
            hls::stream<ap_int8_c> s_conv2("s_conv2");
            trunk_conv2.forward(
                s_qi2_pass, s_conv2,
                qcsnet24_cblk2_qconv1d_weights,
                qcsnet24_cblk2_qconv1d_scale_multiplier,
                qcsnet24_cblk2_qconv1d_right_shift,
                qcsnet24_cblk2_qconv1d_bias,
                qcsnet24_cblk2_qconv1d_input_zero_point,
                qcsnet24_cblk2_qconv1d_weight_sum);

            hls::stream<ap_int8_c> s_conv2_pass("s_conv2_pass");
            if (should_write) {
                tee_and_write<16, 87>(
                    s_conv2, s_conv2_pass,
                    out_dir + "conv2" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                passthrough<16 * 87>(s_conv2, s_conv2_pass);
            }

            // ---- BN2 ----
            hls::stream<ap_int8_c> s_bn2("s_bn2");
            trunk_bn2.forward(
                s_conv2_pass, s_bn2,
                qcsnet24_cblk2_batch_norm_weight,
                qcsnet24_cblk2_batch_norm_bias,
                qcsnet24_cblk2_batch_norm_scale_multiplier,
                qcsnet24_cblk2_batch_norm_right_shift);

            hls::stream<ap_int8_c> s_bn2_pass("s_bn2_pass");
            if (should_write) {
                tee_and_write<16, 87>(
                    s_bn2, s_bn2_pass,
                    out_dir + "bn2" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                passthrough<16 * 87>(s_bn2, s_bn2_pass);
            }

            // ---- LIF2 ----
            hls::stream<ap_int8_c> s_lif2("s_lif2");
            trunk_lif2.forward(
                s_bn2_pass, s_lif2,
                qcsnet24_cblk2_leaky_beta_int,
                qcsnet24_cblk2_leaky_theta_int,
                qcsnet24_cblk2_leaky_scale_int);

            hls::stream<ap_int8_c> s_lif2_pass("s_lif2_pass");
            if (should_write) {
                tee_and_write_spikes<16, 87>(
                    s_lif2, s_lif2_pass,
                    out_dir + "lif2" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                passthrough<16 * 87>(s_lif2, s_lif2_pass);
            }

            // ---- MP2 ----
            hls::stream<ap_int8_c> s_mp2("s_mp2");
            trunk_mp2.forward(s_lif2_pass, s_mp2);

            hls::stream<ap_int8_c> s_mp2_pass("s_mp2_pass");
            if (should_write) {
                tee_and_write<16, 43>(
                    s_mp2, s_mp2_pass,
                    out_dir + "mp2" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                passthrough<16 * 43>(s_mp2, s_mp2_pass);
            }

            // ---- QI3 ----
            hls::stream<ap_int8_c> s_qi3("s_qi3");
            trunk_qi3.forward(s_mp2_pass, s_qi3, qcsnet24_cblk3_input_act_scale_int);

            hls::stream<ap_int8_c> s_qi3_pass("s_qi3_pass");
            if (should_write) {
                tee_and_write<16, 43>(
                    s_qi3, s_qi3_pass,
                    out_dir + "qi3" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                passthrough<16 * 43>(s_qi3, s_qi3_pass);
            }

            // ---- CONV3 ----
            hls::stream<ap_int8_c> s_conv3("s_conv3");
            trunk_conv3.forward(
                s_qi3_pass, s_conv3,
                qcsnet24_cblk3_qconv1d_weights,
                qcsnet24_cblk3_qconv1d_scale_multiplier,
                qcsnet24_cblk3_qconv1d_right_shift,
                qcsnet24_cblk3_qconv1d_bias,
                qcsnet24_cblk3_qconv1d_input_zero_point,
                qcsnet24_cblk3_qconv1d_weight_sum);

            hls::stream<ap_int8_c> s_conv3_pass("s_conv3_pass");
            if (should_write) {
                tee_and_write<24, 41>(
                    s_conv3, s_conv3_pass,
                    out_dir + "conv3" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                passthrough<24 * 41>(s_conv3, s_conv3_pass);
            }

            // ---- BN3 ----
            hls::stream<ap_int8_c> s_bn3("s_bn3");
            trunk_bn3.forward(
                s_conv3_pass, s_bn3,
                qcsnet24_cblk3_batch_norm_weight,
                qcsnet24_cblk3_batch_norm_bias,
                qcsnet24_cblk3_batch_norm_scale_multiplier,
                qcsnet24_cblk3_batch_norm_right_shift);

            hls::stream<ap_int8_c> s_bn3_pass("s_bn3_pass");
            if (should_write) {
                tee_and_write<24, 41>(
                    s_bn3, s_bn3_pass,
                    out_dir + "bn3" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                passthrough<24 * 41>(s_bn3, s_bn3_pass);
            }

            // ---- LIF3 ----
            hls::stream<ap_int8_c> s_lif3("s_lif3");
            trunk_lif3.forward(
                s_bn3_pass, s_lif3,
                qcsnet24_cblk3_leaky_beta_int,
                qcsnet24_cblk3_leaky_theta_int,
                qcsnet24_cblk3_leaky_scale_int);

            hls::stream<ap_int8_c> s_lif3_pass("s_lif3_pass");
            if (should_write) {
                tee_and_write_spikes<24, 41>(
                    s_lif3, s_lif3_pass,
                    out_dir + "lif3" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                passthrough<24 * 41>(s_lif3, s_lif3_pass);
            }

            // ---- MP3 (trunk output: 24×20 = 480) ----
            hls::stream<ap_int8_c> s_mp3("s_mp3");
            trunk_mp3.forward(s_lif3_pass, s_mp3);

            hls::stream<ap_int8_c> s_mp3_pass("s_mp3_pass");
            if (should_write) {
                tee_and_write<24, 20>(
                    s_mp3, s_mp3_pass,
                    out_dir + "mp3" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                passthrough<24 * 20>(s_mp3, s_mp3_pass);
            }

            // ===== CHANGE 5: Cache trunk output (480 only) =====
            hls::stream<ap_int8_c> s_bin_qi_in("s_bin_qi_in");
            for (int i = 0; i < TRUNK_OUT; ++i) {
                ap_int8_c v = s_mp3_pass.read();
                body_cache[step][i] = v;
                s_bin_qi_in.write(v);  // Trunk spikes (480) → QuantIdentity
            }

            // ===== CHANGE 6: QuantIdentity processes trunk only (480) =====
            hls::stream<ap_int8_c> s_bin_qi_out("s_bin_qi_out");
            bin_qi.forward(s_bin_qi_in, s_bin_qi_out, qcsnet2_lblk1_input_act_scale_int);

            // Dump bin_qi output (480)
            hls::stream<ap_int8_c> s_bin_qi_pass("s_bin_qi_pass");
            if (should_write) {
                tee_and_write<1, TRUNK_OUT>(
                    s_bin_qi_out, s_bin_qi_pass,
                    out_dir + "bin_qi" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                passthrough<TRUNK_OUT>(s_bin_qi_out, s_bin_qi_pass);
            }

            // ===== CHANGE 7: Concatenate remapped trunk (480) + pre-quantized RR (4) =====
            hls::stream<ap_int8_c> s_bin_fc_in("s_bin_fc_in");
            // First: trunk (480 remapped values)
            for (int i = 0; i < TRUNK_OUT; ++i) {
                s_bin_fc_in.write(s_bin_qi_pass.read());
            }
            // Then: RR (4 pre-quantized values, bypass QuantIdentity)
            for (int i = 0; i < RR_DIM; ++i) {
                s_bin_fc_in.write(rr_s1[i]);  // Stage 1 RR
            }

            // Dump binary head input (484)
            hls::stream<ap_int8_c> s_bin_fc_in_pass("s_bin_fc_in_pass");
            if (should_write) {
                tee_and_write<1, STAGE1_IN>(
                    s_bin_fc_in, s_bin_fc_in_pass,
                    out_dir + "bin_input" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                passthrough<STAGE1_IN>(s_bin_fc_in, s_bin_fc_in_pass);
            }

            // ---- BIN_FC ----
            hls::stream<ap_int8_c> s_bin_fc("s_bin_fc");
            bin_fc.forward(
                s_bin_fc_in_pass, s_bin_fc,
                qcsnet2_lblk1_qlinear_weights,
                qcsnet2_lblk1_qlinear_scale_multiplier,
                qcsnet2_lblk1_qlinear_right_shift,
                qcsnet2_lblk1_qlinear_bias,
                qcsnet2_lblk1_qlinear_input_zero_point,
                qcsnet2_lblk1_qlinear_weight_sum);

            hls::stream<ap_int8_c> s_bin_fc_pass("s_bin_fc_pass");
            if (should_write) {
                tee_and_write<2, 1>(
                    s_bin_fc, s_bin_fc_pass,
                    out_dir + "bin_fc" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                passthrough<2>(s_bin_fc, s_bin_fc_pass);
            }

            // ---- BIN_LIF ----
            hls::stream<ap_int8_c> s_bin_lif("s_bin_lif");
            bin_lif.forward(
                s_bin_fc_pass, s_bin_lif,
                qcsnet2_lblk1_leaky_beta_int,
                qcsnet2_lblk1_leaky_theta_int,
                qcsnet2_lblk1_leaky_scale_int);

            hls::stream<ap_int8_c> s_bin_lif_pass("s_bin_lif_pass");
            if (should_write) {
                tee_and_write_spikes<2, 1>(
                    s_bin_lif, s_bin_lif_pass,
                    out_dir + "bin_lif" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                passthrough<2>(s_bin_lif, s_bin_lif_pass);
            }

            // Accumulate binary head spikes
            ap_int8_c b0 = s_bin_lif_pass.read();
            ap_int8_c b1 = s_bin_lif_pass.read();
            sum_bin0 += (ap_int<16>)b0;
            sum_bin1 += (ap_int<16>)b1;

            if ((step % 3) == 0) {
                std::cout << "  Timestep " << step << " complete\n";
            }
        } // end STAGE-1 steps

        // Gate decision
        ap_int8_c pred2 = gate_abnormal(sum_bin0, sum_bin1);

        std::cout << "Stage-1: sum_bin=[" << int(sum_bin0) << "," << int(sum_bin1) 
                  << "] → pred2=" << int(pred2) << " (true2=" << y_true2 << ")\n";

        // ======================== STAGE-2: MULTI HEAD (if routed) ========================
        ap_int8_c pred4;
        if (pred2 == 0) {
            pred4 = 0;  // Normal
        } else {
            routed_count++;

            // Reset multi head LIFs
            multi_lif1.reset();
            multi_lif2.reset();

            ap_int<16> sum4_0 = 0, sum4_1 = 0, sum4_2 = 0, sum4_3 = 0;

            for (int step = 0; step < LOCAL_NUM_STEPS; ++step) {
                bool should_write = (step == 0) || (step == LOCAL_NUM_STEPS - 1);
                std::string step_suffix = "";
                if (should_write) {
                    char buf[16];
                    std::snprintf(buf, sizeof(buf), "_step%02d", step);
                    step_suffix = buf;
                }

                // ===== CHANGE 8: Feed cached trunk (480) to QuantIdentity =====
                hls::stream<ap_int8_c> s_multi_qi1_in("s_multi_qi1_in");
                for (int i = 0; i < TRUNK_OUT; ++i) {
                    s_multi_qi1_in.write(body_cache[step][i]);
                }

                // ===== CHANGE 9: QuantIdentity processes trunk only (480) =====
                hls::stream<ap_int8_c> s_multi_qi1_out("s_multi_qi1_out");
                multi_qi1.forward(s_multi_qi1_in, s_multi_qi1_out, qcsnet4_lblk1_input_act_scale_int);

                // Dump multi_qi1 output (480)
                hls::stream<ap_int8_c> s_multi_qi1_pass("s_multi_qi1_pass");
                if (should_write) {
                    tee_and_write<1, TRUNK_OUT>(
                        s_multi_qi1_out, s_multi_qi1_pass,
                        out_dir + "multi_qi1" + step_suffix + "_cpp.txt",
                        sample_idx + 1, is_first_sample, true);
                } else {
                    passthrough<TRUNK_OUT>(s_multi_qi1_out, s_multi_qi1_pass);
                }

                // ===== CHANGE 10: Concatenate remapped trunk (480) + pre-quantized RR (4) =====
                hls::stream<ap_int8_c> s_multi_fc1_in("s_multi_fc1_in");
                // First: trunk (480 remapped values)
                for (int i = 0; i < TRUNK_OUT; ++i) {
                    s_multi_fc1_in.write(s_multi_qi1_pass.read());
                }
                // Then: RR (4 pre-quantized values, bypass QuantIdentity)
                for (int i = 0; i < RR_DIM; ++i) {
                    s_multi_fc1_in.write(rr_s2[i]);  // Stage 2 RR
                }

                // Dump multi head input (484)
                hls::stream<ap_int8_c> s_multi_fc1_in_pass("s_multi_fc1_in_pass");
                if (should_write) {
                    tee_and_write<1, STAGE2_IN>(
                        s_multi_fc1_in, s_multi_fc1_in_pass,
                        out_dir + "multi_input" + step_suffix + "_cpp.txt",
                        sample_idx + 1, is_first_sample, true);
                } else {
                    passthrough<STAGE2_IN>(s_multi_fc1_in, s_multi_fc1_in_pass);
                }

                // ---- MULTI_FC1 ----
                hls::stream<ap_int8_c> s_multi_fc1("s_multi_fc1");
                multi_fc1.forward(
                    s_multi_fc1_in_pass, s_multi_fc1,
                    qcsnet4_lblk1_qlinear_weights,
                    qcsnet4_lblk1_qlinear_scale_multiplier,
                    qcsnet4_lblk1_qlinear_right_shift,
                    qcsnet4_lblk1_qlinear_bias,
                    qcsnet4_lblk1_qlinear_input_zero_point,
                    qcsnet4_lblk1_qlinear_weight_sum);

                hls::stream<ap_int8_c> s_multi_fc1_pass("s_multi_fc1_pass");
                if (should_write) {
                    tee_and_write<128, 1>(
                        s_multi_fc1, s_multi_fc1_pass,
                        out_dir + "multi_fc1" + step_suffix + "_cpp.txt",
                        sample_idx + 1, is_first_sample, true);
                } else {
                    passthrough<128>(s_multi_fc1, s_multi_fc1_pass);
                }

                // ---- MULTI_LIF1 ----
                hls::stream<ap_int8_c> s_multi_lif1("s_multi_lif1");
                multi_lif1.forward(
                    s_multi_fc1_pass, s_multi_lif1,
                    qcsnet4_lblk1_leaky_beta_int,
                    qcsnet4_lblk1_leaky_theta_int,
                    qcsnet4_lblk1_leaky_scale_int);

                hls::stream<ap_int8_c> s_multi_lif1_pass("s_multi_lif1_pass");
                if (should_write) {
                    tee_and_write_spikes<128, 1>(
                        s_multi_lif1, s_multi_lif1_pass,
                        out_dir + "multi_lif1" + step_suffix + "_cpp.txt",
                        sample_idx + 1, is_first_sample, true);
                } else {
                    passthrough<128>(s_multi_lif1, s_multi_lif1_pass);
                }

                // ---- MULTI_QI2 ----
                hls::stream<ap_int8_c> s_multi_qi2("s_multi_qi2");
                multi_qi2.forward(s_multi_lif1_pass, s_multi_qi2, qcsnet4_lblk2_input_act_scale_int);

                hls::stream<ap_int8_c> s_multi_qi2_pass("s_multi_qi2_pass");
                if (should_write) {
                    tee_and_write<128, 1>(
                        s_multi_qi2, s_multi_qi2_pass,
                        out_dir + "multi_qi2" + step_suffix + "_cpp.txt",
                        sample_idx + 1, is_first_sample, true);
                } else {
                    passthrough<128>(s_multi_qi2, s_multi_qi2_pass);
                }

                // ---- MULTI_FC2 ----
                hls::stream<ap_int8_c> s_multi_fc2("s_multi_fc2");
                multi_fc2.forward(
                    s_multi_qi2_pass, s_multi_fc2,
                    qcsnet4_lblk2_qlinear_weights,
                    qcsnet4_lblk2_qlinear_scale_multiplier,
                    qcsnet4_lblk2_qlinear_right_shift,
                    qcsnet4_lblk2_qlinear_bias,
                    qcsnet4_lblk2_qlinear_input_zero_point,
                    qcsnet4_lblk2_qlinear_weight_sum);

                hls::stream<ap_int8_c> s_multi_fc2_pass("s_multi_fc2_pass");
                if (should_write) {
                    tee_and_write<4, 1>(
                        s_multi_fc2, s_multi_fc2_pass,
                        out_dir + "multi_fc2" + step_suffix + "_cpp.txt",
                        sample_idx + 1, is_first_sample, true);
                } else {
                    passthrough<4>(s_multi_fc2, s_multi_fc2_pass);
                }

                // ---- MULTI_LIF2 ----
                hls::stream<ap_int8_c> s_multi_lif2("s_multi_lif2");
                multi_lif2.forward(
                    s_multi_fc2_pass, s_multi_lif2,
                    qcsnet4_lblk2_leaky_beta_int,
                    qcsnet4_lblk2_leaky_theta_int,
                    qcsnet4_lblk2_leaky_scale_int);

                hls::stream<ap_int8_c> s_multi_lif2_pass("s_multi_lif2_pass");
                if (should_write) {
                    tee_and_write_spikes<4, 1>(
                        s_multi_lif2, s_multi_lif2_pass,
                        out_dir + "multi_lif2" + step_suffix + "_cpp.txt",
                        sample_idx + 1, is_first_sample, true);
                } else {
                    passthrough<4>(s_multi_lif2, s_multi_lif2_pass);
                }

                // Accumulate multi head spikes
                ap_int8_c y0 = s_multi_lif2_pass.read();
                ap_int8_c y1 = s_multi_lif2_pass.read();
                ap_int8_c y2 = s_multi_lif2_pass.read();
                ap_int8_c y3 = s_multi_lif2_pass.read();

                sum4_0 += (ap_int<16>)y0;
                sum4_1 += (ap_int<16>)y1;
                sum4_2 += (ap_int<16>)y2;
                sum4_3 += (ap_int<16>)y3;
            } // end STAGE-2 steps

            pred4 = argmax4(sum4_0, sum4_1, sum4_2, sum4_3);

            std::cout << "Stage-2: sum4=[" << int(sum4_0) << "," << int(sum4_1) 
                      << "," << int(sum4_2) << "," << int(sum4_3) 
                      << "] → pred4=" << int(pred4) << " (true4=" << y_true4 << ")\n";
        }

        // Update confusion matrices
        total_count++;
        if (y_true2 >= 0 && y_true2 < 2 && pred2 >= 0 && pred2 < 2) {
            cm2[y_true2][pred2]++;
        }
        if (y_true4 >= 0 && y_true4 < 4 && pred4 >= 0 && pred4 < 4) {
            cm4[y_true4][pred4]++;
        }

        std::cout << "Sample " << (sample_idx + 1)
                  << "  y_true4=" << y_true4
                  << "  pred2=" << int(pred2)
                  << "  pred4=" << int(pred4) << "\n";

    } // end samples

    // ======================== PRINT METRICS ========================
    const double eps = 1e-12;

    std::cout << "\n=== Binary Confusion Matrix [true x pred] ===\n";
    std::cout << "      pred:  0      1\n";
    for (int i = 0; i < N2; ++i) {
        std::cout << "true " << i << ": ";
        for (int j = 0; j < N2; ++j) {
            std::cout << std::setw(6) << cm2[i][j] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\n=== Final 4-Class Confusion Matrix [true x pred] ===\n";
    std::cout << "      pred:  0      1      2      3\n";
    for (int i = 0; i < N4; ++i) {
        std::cout << "true " << i << ": ";
        for (int j = 0; j < N4; ++j) {
            std::cout << std::setw(6) << cm4[i][j] << " ";
        }
        std::cout << "\n";
    }

    // Stage-1 metrics
    std::cout << "\n=== Stage-1 Binary Metrics ===\n";
    std::cout << std::fixed << std::setprecision(4);
    for (int c = 0; c < 2; ++c) {
        long TP = cm2[c][c];
        long FN = cm2[c][1 - c];
        long FP = cm2[1 - c][c];
        double precision = TP / (double)(TP + FP + eps);
        double recall    = TP / (double)(TP + FN + eps);
        double f1        = (2.0 * precision * recall) / (precision + recall + eps);
        std::cout << "Class " << c << "  Prec=" << precision 
                  << "  Rec=" << recall << "  F1=" << f1 << "\n";
    }
    long correct2 = cm2[0][0] + cm2[1][1];
    long total2 = cm2[0][0] + cm2[0][1] + cm2[1][0] + cm2[1][1];
    std::cout << "Stage-1 Accuracy = " << (correct2 / (double)(total2 + eps)) << "\n";

    // Stage-2 metrics
    std::cout << "\n=== Final 4-Class Metrics ===\n";
    long total_correct = 0, total_samples = 0;
    for (int i = 0; i < N4; ++i) {
        for (int j = 0; j < N4; ++j) {
            total_samples += cm4[i][j];
            if (i == j) total_correct += cm4[i][j];
        }
    }
    for (int c = 0; c < N4; ++c) {
        long TP = cm4[c][c];
        long FN = 0, FP = 0;
        for (int i = 0; i < N4; ++i) {
            for (int j = 0; j < N4; ++j) {
                if (i == c && j != c) FN += cm4[i][j];
                else if (i != c && j == c) FP += cm4[i][j];
            }
        }
        double precision = TP / (double)(TP + FP + eps);
        double recall    = TP / (double)(TP + FN + eps);
        double f1        = (2.0 * precision * recall) / (precision + recall + eps);
        std::cout << "Class " << c << "  Prec=" << precision 
                  << "  Rec=" << recall << "  F1=" << f1 << "\n";
    }
    std::cout << "Final Accuracy = " << (total_correct / (double)(total_samples + eps)) << "\n";

    double routed_pct = 100.0 * routed_count / (double)(total_count + eps);
    std::cout << "\nRouted to Stage-2: " << routed_count << "/" << total_count 
              << " (" << routed_pct << "%)\n";

    std::cout << "\n=== Complete ===\n";
    return 0;
}