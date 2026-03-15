// testbench_dump_layers.cpp
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <vector>
#include "hls_stream.h"
#include "ap_int.h"

#include "../../include/hls4csnn1d_sd/model4/constants4_sd.h"
#include "../../include/hls4csnn1d_sd/model4/filereader4.h"
#include "../../include/hls4csnn1d_sd/model4/cblk_sd/nn4_cblk1_sd.h"

// Include individual layer headers
#include "../../include/hls4csnn1d_sd/model4/cblk_sd/conv1d_sd.h"
#include "../../include/hls4csnn1d_sd/model4/cblk_sd/batchnorm1d_sd.h"
#include "../../include/hls4csnn1d_sd/model4/cblk_sd/lif1d_integer.h"
// #include "../../include/hls4csnn1d_sd/model4/cblk_sd/lif1d_float.h"
#include "../../include/hls4csnn1d_sd/model4/cblk_sd/maxpool1d_sd.h"
#include "../../include/hls4csnn1d_sd/model4/cblk_sd/linear1d_sd.h"
// Add include
#include "../../include/hls4csnn1d_sd/model4/cblk_sd/quantidentity1d_sd.h"



using namespace hls4csnn1d_cblk_sd;

/* ================================================================
 *  Helper: Copy stream (tee)
 * ================================================================ */
template<int SIZE>
void copy_stream(hls::stream<ap_int8_c>& in,
                 hls::stream<ap_int8_c>& out1,
                 hls::stream<ap_int8_c>& out2) {
    for (int i = 0; i < SIZE; ++i) {
#pragma HLS PIPELINE II=1
        ap_int8_c val = in.read();
        out1.write(val);
        out2.write(val);
    }
}

/* ================================================================
 *  Helper: Copy to file and pass through
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
 *  Helper: Normalize quantized spikes to binary for file output
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
            out.write(val);  // Pass through original quantized value (126)
            
            if (should_write) {
                // Normalize to binary for file comparison: 126 → 1, 0 → 0
                int binary_val = (val > 0) ? 1 : 0;
                outfile << " " << binary_val;
            }
        }
        if (should_write) outfile << "\n";
    }
    
    if (should_write) outfile.close();
}

/* ================================================================
 *  Helper: Read stream and accumulate
 * ================================================================ */
template<int SIZE>
void read_and_accumulate(hls::stream<ap_int8_c>& in,
                        hls::stream<ap_int8_c>& out,
                        ap_int<32> acc_buf[SIZE]) {
    for (int i = 0; i < SIZE; ++i) {
#pragma HLS PIPELINE II=1
        ap_int8_c val = in.read();
        acc_buf[i] += (ap_int<32>)val;
        out.write(val);
    }
}

/* ================================================================
 *  Helper: Write buffer to file
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
 *  Main Testbench
 * ================================================================ */
int main(int argc, char** argv) {
    std::cout << "\n=== Layer-by-Layer Output Dumper (QCSNN-4, C++) ===\n\n";

    // -------------------------------------------------------------------------
    // 1. Resolve output directory and dataset folder
    // -------------------------------------------------------------------------
    std::string out_dir = "../../../../compare_outputs/model4/";
    if (argc >= 2) out_dir = argv[1];

    std::string folderPath =
        "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_test/smallvhls";
    if (argc >= 3) folderPath = argv[2];

    // -------------------------------------------------------------------------
    // 2. Load data
    // -------------------------------------------------------------------------
    FileReader reader;
    reader.loadData(folderPath);

    hls::stream<array180_t> dataStreamInternal;
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

    std::cout << "Loaded " << NUM_SAMPLES_LOADED << " ECG samples\n\n";

    // Pull all samples into a vector
    std::vector<array180_t> all_samples;
    all_samples.reserve(NUM_SAMPLES_LOADED);
    for (int i = 0; i < NUM_SAMPLES_LOADED; ++i) {
        all_samples.push_back(dataStreamInternal.read());
    }

    // -------------------------------------------------------------------------
    // 3. Label stream (4-class: 0=N,1=SVEB,2=VEB,3=F)
    // -------------------------------------------------------------------------
    hls::stream<ap_int8_c> labelStream;
    reader.streamLabel(labelStream, /*binary=*/false);

    // -------------------------------------------------------------------------
    // 4. Multi-class confusion matrix
    // -------------------------------------------------------------------------
    const int N_CLASSES = 4;
    long cm[N_CLASSES][N_CLASSES];
    for (int i = 0; i < N_CLASSES; ++i) {
        for (int j = 0; j < N_CLASSES; ++j) {
            cm[i][j] = 0;
        }
    }

    const int NUM_STEPS = 10;

    // -------------------------------------------------------------------------
    // 5. Process each sample
    // -------------------------------------------------------------------------
    for (int sample_idx = 0; sample_idx < NUM_SAMPLES_LOADED; ++sample_idx) {
        std::cout << "\n========================================\n";
        std::cout << "Processing Sample " << (sample_idx + 1) << "/" << NUM_SAMPLES_LOADED << "\n";
        std::cout << "========================================\n";

        array180_t sample = all_samples[sample_idx];
        bool is_first_sample = (sample_idx == 0);

        // Fresh layer objects (constructor reset -> once per sample)
        // --- Conv block 1 ---
        Conv1D_SD<1, 16, 3, 1, CONV_IN_LENGTH1>              conv1;
        BatchNorm1D_SD<OUT_CH1, FEATURE_LENGTH1>             bn1;
        LIF1D_SD_Integer<OUT_CH1, FEATURE_LENGTH1>           lif1;
        MaxPool1D_SD<2, 2, OUT_CH1, FEATURE_LENGTH1>         mp1;

        // --- Conv block 2 ---
        QuantIdentityPerTensor_Int8<OUT_CH1, CONV_IN_LENGTH2>  qi2;
        Conv1D_SD<16, 16, 3, 1, CONV_IN_LENGTH2>              conv2;
        BatchNorm1D_SD<OUT_CH2, FEATURE_LENGTH2>              bn2;
        LIF1D_SD_Integer<OUT_CH2, FEATURE_LENGTH2>            lif2;
        MaxPool1D_SD<2, 2, OUT_CH2, FEATURE_LENGTH2>          mp2;

        // --- Conv block 3 ---
        QuantIdentityPerTensor_Int8<OUT_CH2, CONV_IN_LENGTH3>  qi3;
        Conv1D_SD<16, 24, 3, 1, CONV_IN_LENGTH3>              conv3;
        BatchNorm1D_SD<OUT_CH3, FEATURE_LENGTH3>              bn3;
        LIF1D_SD_Integer<OUT_CH3, FEATURE_LENGTH3>            lif3;
        MaxPool1D_SD<2, 2, OUT_CH3, FEATURE_LENGTH3>          mp3;

        // --- Dense head ---
        QuantIdentityPerTensor_Int8<OUT_CH3, POOL3_OUT_LEN>   qi_lin1;  // 24×20 = 480
        Linear1D_SD<LINEAR_IN_SIZE1, LINEAR_OUT_SIZE1>        fc1;      // 480→128
        LIF1D_SD_Integer<LINEAR_OUT_SIZE1, 1>                 lif_lin1; // 128

        QuantIdentityPerTensor_Int8<LINEAR_OUT_SIZE1, 1>      qi_lin2;  // 128
        Linear1D_SD<LINEAR_OUT_SIZE1, 4>                      fc2;      // 128→4
        LIF1D_SD_Integer<4, 1>                                lif_head; // 4

        // --- Accumulators for spike counts over time ---
        static ap_int<32> lif1_acc[OUT_CH1 * FEATURE_LENGTH1];   // 16 × 178
        static ap_int<32> lif2_acc[OUT_CH2 * FEATURE_LENGTH2];   // 16 × 87
        static ap_int<32> lif3_acc[OUT_CH3 * FEATURE_LENGTH3];   // 24 × 41
        static ap_int<32> lif_lin1_acc[LINEAR_OUT_SIZE1 * 1];    // 128
        static ap_int<32> lif_head_acc[4];                       // 4 outputs

        for (int i = 0; i < OUT_CH1 * FEATURE_LENGTH1; ++i) lif1_acc[i] = 0;
        for (int i = 0; i < OUT_CH2 * FEATURE_LENGTH2; ++i) lif2_acc[i] = 0;
        for (int i = 0; i < OUT_CH3 * FEATURE_LENGTH3; ++i) lif3_acc[i] = 0;
        for (int i = 0; i < LINEAR_OUT_SIZE1; ++i)          lif_lin1_acc[i] = 0;
        for (int i = 0; i < 4; ++i)                         lif_head_acc[i] = 0;

        // Ground-truth label for this sample
        const int y_true = int(labelStream.read());

        // ---------------------------------------------------------------------
        // Temporal loop
        // ---------------------------------------------------------------------
        for (int step = 0; step < NUM_STEPS; ++step) {
            bool should_write = (step == 0) || (step == NUM_STEPS - 1);

            std::string step_suffix = "";
            if (should_write) {
                char buf[16];
                std::snprintf(buf, sizeof(buf), "_step%02d", step);
                step_suffix = buf;
            }

            // --- Input stream for this step ---
            hls::stream<ap_int8_c> s_input("s_input");
            for (int i = 0; i < CONV_IN_LENGTH1; ++i) {
                s_input.write(sample[i]);
            }

            // ================== QIN1 (raw input logging) =====================
            hls::stream<ap_int8_c> s_input_pass("s_input_pass");
            if (should_write) {
                tee_and_write<1, CONV_IN_LENGTH1>(
                    s_input, s_input_pass,
                    out_dir + "qin1" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                for (int i = 0; i < CONV_IN_LENGTH1; ++i)
                    s_input_pass.write(s_input.read());
            }

            // ================== Block 1: conv1 → bn1 → lif1 → mp1 ============
            // Conv1
            hls::stream<ap_int8_c> s_conv1("s_conv1");
            conv1.forward(
                s_input_pass, s_conv1,
                qcsnet4_cblk1_qconv1d_weights,
                qcsnet4_cblk1_qconv1d_scale_multiplier,
                qcsnet4_cblk1_qconv1d_right_shift,
                qcsnet4_cblk1_qconv1d_bias,
                qcsnet4_cblk1_qconv1d_input_zero_point,
                qcsnet4_cblk1_qconv1d_weight_sum
            );

            hls::stream<ap_int8_c> s_conv1_pass("s_conv1_pass");
            if (should_write) {
                tee_and_write<OUT_CH1, FEATURE_LENGTH1>(
                    s_conv1, s_conv1_pass,
                    out_dir + "conv1" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                for (int i = 0; i < OUT_CH1 * FEATURE_LENGTH1; ++i)
                    s_conv1_pass.write(s_conv1.read());
            }

            // BN1
            hls::stream<ap_int8_c> s_bn1("s_bn1");
            bn1.forward(
                s_conv1_pass, s_bn1,
                qcsnet4_cblk1_batch_norm_weight,
                qcsnet4_cblk1_batch_norm_bias,
                qcsnet4_cblk1_batch_norm_scale_multiplier,
                qcsnet4_cblk1_batch_norm_right_shift
            );

            hls::stream<ap_int8_c> s_bn1_pass("s_bn1_pass");
            if (should_write) {
                tee_and_write<OUT_CH1, FEATURE_LENGTH1>(
                    s_bn1, s_bn1_pass,
                    out_dir + "bn1" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                for (int i = 0; i < OUT_CH1 * FEATURE_LENGTH1; ++i)
                    s_bn1_pass.write(s_bn1.read());
            }

            // LIF1
            hls::stream<ap_int8_c> s_lif1("s_lif1");
            lif1.forward(
                s_bn1_pass, s_lif1,
                qcsnet4_cblk1_leaky_beta_int,
                qcsnet4_cblk1_leaky_theta_int,
                qcsnet4_cblk1_leaky_scale_int
            );

            hls::stream<ap_int8_c> s_lif1_bin("s_lif1_bin");
            if (should_write) {
                tee_and_write_spikes<OUT_CH1, FEATURE_LENGTH1>(
                    s_lif1, s_lif1_bin,
                    out_dir + "lif1" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                for (int i = 0; i < OUT_CH1 * FEATURE_LENGTH1; ++i) {
                    ap_int8_c v = s_lif1.read();
                    s_lif1_bin.write((v > 0) ? ap_int8_c(1) : ap_int8_c(0));
                }
            }

            // Accumulate LIF1 spikes
            hls::stream<ap_int8_c> s_lif1_pass("s_lif1_pass");
            read_and_accumulate<OUT_CH1 * FEATURE_LENGTH1>(s_lif1_bin, s_lif1_pass, lif1_acc);

            // MaxPool1
            hls::stream<ap_int8_c> s_mp1("s_mp1");
            mp1.forward(s_lif1_pass, s_mp1);

            hls::stream<ap_int8_c> s_mp1_pass("s_mp1_pass");
            if (should_write) {
                tee_and_write<OUT_CH1, POOL1_OUT_LEN>(
                    s_mp1, s_mp1_pass,
                    out_dir + "mp1" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                for (int i = 0; i < OUT_CH1 * POOL1_OUT_LEN; ++i)
                    s_mp1_pass.write(s_mp1.read());
            }

            // ================== Block 2: QI2 → conv2 → bn2 → lif2 → mp2 ======
            // QI2
            hls::stream<ap_int8_c> s_qi2("s_qi2");
            qi2.forward(s_mp1_pass, s_qi2, qcsnet4_cblk2_input_act_scale_int);

            hls::stream<ap_int8_c> s_qi2_pass("s_qi2_pass");
            if (should_write) {
                tee_and_write<OUT_CH1, CONV_IN_LENGTH2>(
                    s_qi2, s_qi2_pass,
                    out_dir + "qin2" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                for (int i = 0; i < OUT_CH1 * CONV_IN_LENGTH2; ++i)
                    s_qi2_pass.write(s_qi2.read());
            }

            // Conv2
            hls::stream<ap_int8_c> s_conv2("s_conv2");
            conv2.forward(
                s_qi2_pass, s_conv2,
                qcsnet4_cblk2_qconv1d_weights,
                qcsnet4_cblk2_qconv1d_scale_multiplier,
                qcsnet4_cblk2_qconv1d_right_shift,
                qcsnet4_cblk2_qconv1d_bias,
                qcsnet4_cblk2_qconv1d_input_zero_point,
                qcsnet4_cblk2_qconv1d_weight_sum
            );

            hls::stream<ap_int8_c> s_conv2_pass("s_conv2_pass");
            if (should_write) {
                tee_and_write<OUT_CH2, FEATURE_LENGTH2>(
                    s_conv2, s_conv2_pass,
                    out_dir + "conv2" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                for (int i = 0; i < OUT_CH2 * FEATURE_LENGTH2; ++i)
                    s_conv2_pass.write(s_conv2.read());
            }

            // BN2
            hls::stream<ap_int8_c> s_bn2("s_bn2");
            bn2.forward(
                s_conv2_pass, s_bn2,
                qcsnet4_cblk2_batch_norm_weight,
                qcsnet4_cblk2_batch_norm_bias,
                qcsnet4_cblk2_batch_norm_scale_multiplier,
                qcsnet4_cblk2_batch_norm_right_shift
            );

            hls::stream<ap_int8_c> s_bn2_pass("s_bn2_pass");
            if (should_write) {
                tee_and_write<OUT_CH2, FEATURE_LENGTH2>(
                    s_bn2, s_bn2_pass,
                    out_dir + "bn2" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                for (int i = 0; i < OUT_CH2 * FEATURE_LENGTH2; ++i)
                    s_bn2_pass.write(s_bn2.read());
            }

            // LIF2
            hls::stream<ap_int8_c> s_lif2("s_lif2");
            lif2.forward(
                s_bn2_pass, s_lif2,
                qcsnet4_cblk2_leaky_beta_int,
                qcsnet4_cblk2_leaky_theta_int,
                qcsnet4_cblk2_leaky_scale_int
            );

            hls::stream<ap_int8_c> s_lif2_bin("s_lif2_bin");
            if (should_write) {
                tee_and_write_spikes<OUT_CH2, FEATURE_LENGTH2>(
                    s_lif2, s_lif2_bin,
                    out_dir + "lif2" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                for (int i = 0; i < OUT_CH2 * FEATURE_LENGTH2; ++i) {
                    ap_int8_c v = s_lif2.read();
                    s_lif2_bin.write((v > 0) ? ap_int8_c(1) : ap_int8_c(0));
                }
            }

            // Accumulate LIF2 spikes
            hls::stream<ap_int8_c> s_lif2_pass("s_lif2_pass");
            read_and_accumulate<OUT_CH2 * FEATURE_LENGTH2>(s_lif2_bin, s_lif2_pass, lif2_acc);

            // MaxPool2
            hls::stream<ap_int8_c> s_mp2("s_mp2");
            mp2.forward(s_lif2_pass, s_mp2);

            hls::stream<ap_int8_c> s_mp2_pass("s_mp2_pass");
            if (should_write) {
                tee_and_write<OUT_CH2, POOL2_OUT_LEN>(
                    s_mp2, s_mp2_pass,
                    out_dir + "mp2" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                for (int i = 0; i < OUT_CH2 * POOL2_OUT_LEN; ++i)
                    s_mp2_pass.write(s_mp2.read());
            }

            // ================== Block 3: QI3 → conv3 → bn3 → lif3 → mp3 ======
            // QI3
            hls::stream<ap_int8_c> s_qi3("s_qi3");
            qi3.forward(s_mp2_pass, s_qi3, qcsnet4_cblk3_input_act_scale_int);

            hls::stream<ap_int8_c> s_qi3_pass("s_qi3_pass");
            if (should_write) {
                tee_and_write<OUT_CH2, CONV_IN_LENGTH3>(
                    s_qi3, s_qi3_pass,
                    out_dir + "qin3" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                for (int i = 0; i < OUT_CH2 * CONV_IN_LENGTH3; ++i)
                    s_qi3_pass.write(s_qi3.read());
            }

            // Conv3
            hls::stream<ap_int8_c> s_conv3("s_conv3");
            conv3.forward(
                s_qi3_pass, s_conv3,
                qcsnet4_cblk3_qconv1d_weights,
                qcsnet4_cblk3_qconv1d_scale_multiplier,
                qcsnet4_cblk3_qconv1d_right_shift,
                qcsnet4_cblk3_qconv1d_bias,
                qcsnet4_cblk3_qconv1d_input_zero_point,
                qcsnet4_cblk3_qconv1d_weight_sum
            );

            hls::stream<ap_int8_c> s_conv3_pass("s_conv3_pass");
            if (should_write) {
                tee_and_write<OUT_CH3, FEATURE_LENGTH3>(
                    s_conv3, s_conv3_pass,
                    out_dir + "conv3" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                for (int i = 0; i < OUT_CH3 * FEATURE_LENGTH3; ++i)
                    s_conv3_pass.write(s_conv3.read());
            }

            // BN3
            hls::stream<ap_int8_c> s_bn3("s_bn3");
            bn3.forward(
                s_conv3_pass, s_bn3,
                qcsnet4_cblk3_batch_norm_weight,
                qcsnet4_cblk3_batch_norm_bias,
                qcsnet4_cblk3_batch_norm_scale_multiplier,
                qcsnet4_cblk3_batch_norm_right_shift
            );

            hls::stream<ap_int8_c> s_bn3_pass("s_bn3_pass");
            if (should_write) {
                tee_and_write<OUT_CH3, FEATURE_LENGTH3>(
                    s_bn3, s_bn3_pass,
                    out_dir + "bn3" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                for (int i = 0; i < OUT_CH3 * FEATURE_LENGTH3; ++i)
                    s_bn3_pass.write(s_bn3.read());
            }

            // LIF3
            hls::stream<ap_int8_c> s_lif3("s_lif3");
            lif3.forward(
                s_bn3_pass, s_lif3,
                qcsnet4_cblk3_leaky_beta_int,
                qcsnet4_cblk3_leaky_theta_int,
                qcsnet4_cblk3_leaky_scale_int
            );

            hls::stream<ap_int8_c> s_lif3_bin("s_lif3_bin");
            if (should_write) {
                tee_and_write_spikes<OUT_CH3, FEATURE_LENGTH3>(
                    s_lif3, s_lif3_bin,
                    out_dir + "lif3" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                for (int i = 0; i < OUT_CH3 * FEATURE_LENGTH3; ++i) {
                    ap_int8_c v = s_lif3.read();
                    s_lif3_bin.write((v > 0) ? ap_int8_c(1) : ap_int8_c(0));
                }
            }

            // Accumulate LIF3 spikes
            hls::stream<ap_int8_c> s_lif3_pass("s_lif3_pass");
            read_and_accumulate<OUT_CH3 * FEATURE_LENGTH3>(s_lif3_bin, s_lif3_pass, lif3_acc);

            // MaxPool3
            hls::stream<ap_int8_c> s_mp3("s_mp3");
            mp3.forward(s_lif3_pass, s_mp3);

            hls::stream<ap_int8_c> s_mp3_pass("s_mp3_pass");
            if (should_write) {
                tee_and_write<OUT_CH3, POOL3_OUT_LEN>(
                    s_mp3, s_mp3_pass,
                    out_dir + "mp3" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                for (int i = 0; i < OUT_CH3 * POOL3_OUT_LEN; ++i)
                    s_mp3_pass.write(s_mp3.read());
            }

            // ================== Dense Head 1: QI_lin1 → FC1 → LIF_lin1 =======
            // QI before first linear (24×20 → 24×20)
            hls::stream<ap_int8_c> s_qi_lin1("s_qi_lin1");
            qi_lin1.forward(s_mp3_pass, s_qi_lin1, qcsnet4_lblk1_input_act_scale_int);

            hls::stream<ap_int8_c> s_qi_lin1_pass("s_qi_lin1_pass");
            if (should_write) {
                tee_and_write<OUT_CH3, POOL3_OUT_LEN>(
                    s_qi_lin1, s_qi_lin1_pass,
                    out_dir + "qin_lin1" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                for (int i = 0; i < OUT_CH3 * POOL3_OUT_LEN; ++i)
                    s_qi_lin1_pass.write(s_qi_lin1.read());
            }

            // FC1: 480 → 128
            hls::stream<ap_int8_c> s_lin1("s_lin1");
            fc1.forward(
                s_qi_lin1_pass, s_lin1,
                qcsnet4_lblk1_qlinear_weights,
                qcsnet4_lblk1_qlinear_scale_multiplier,
                qcsnet4_lblk1_qlinear_right_shift,
                qcsnet4_lblk1_qlinear_bias,
                qcsnet4_lblk1_qlinear_input_zero_point,
                qcsnet4_lblk1_qlinear_weight_sum
            );

            hls::stream<ap_int8_c> s_lin1_pass("s_lin1_pass");
            if (should_write) {
                tee_and_write<1, LINEAR_OUT_SIZE1>(
                    s_lin1, s_lin1_pass,
                    out_dir + "lin1" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                for (int i = 0; i < LINEAR_OUT_SIZE1; ++i)
                    s_lin1_pass.write(s_lin1.read());
            }

            // LIF_lin1: 128
            hls::stream<ap_int8_c> s_lif_lin1("s_lif_lin1");
            lif_lin1.forward(
                s_lin1_pass, s_lif_lin1,
                qcsnet4_lblk1_leaky_beta_int,
                qcsnet4_lblk1_leaky_theta_int,
                qcsnet4_lblk1_leaky_scale_int
            );

            hls::stream<ap_int8_c> s_lif_lin1_bin("s_lif_lin1_bin");
            if (should_write) {
                tee_and_write_spikes<LINEAR_OUT_SIZE1, 1>(
                    s_lif_lin1, s_lif_lin1_bin,
                    out_dir + "lif_lin1" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                for (int i = 0; i < LINEAR_OUT_SIZE1; ++i) {
                    ap_int8_c v = s_lif_lin1.read();
                    s_lif_lin1_bin.write((v > 0) ? ap_int8_c(1) : ap_int8_c(0));
                }
            }

            // Accumulate LIF_lin1 spikes
            hls::stream<ap_int8_c> s_lif_lin1_pass("s_lif_lin1_pass");
            read_and_accumulate<LINEAR_OUT_SIZE1 * 1>(s_lif_lin1_bin, s_lif_lin1_pass, lif_lin1_acc);

            // ================== Dense Head 2: QI_lin2 → FC2 → LIF_head =======
            // QI before final linear (128)
            hls::stream<ap_int8_c> s_qi_lin2("s_qi_lin2");
            qi_lin2.forward(s_lif_lin1_pass, s_qi_lin2, qcsnet4_lblk2_input_act_scale_int);

            hls::stream<ap_int8_c> s_qi_lin2_pass("s_qi_lin2_pass");
            if (should_write) {
                tee_and_write<LINEAR_OUT_SIZE1, 1>(
                    s_qi_lin2, s_qi_lin2_pass,
                    out_dir + "qin_lin2" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                for (int i = 0; i < LINEAR_OUT_SIZE1; ++i)
                    s_qi_lin2_pass.write(s_qi_lin2.read());
            }

            // FC2: 128 → 4
            hls::stream<ap_int8_c> s_lin2("s_lin2");
            fc2.forward(
                s_qi_lin2_pass, s_lin2,
                qcsnet4_lblk2_qlinear_weights,
                qcsnet4_lblk2_qlinear_scale_multiplier,
                qcsnet4_lblk2_qlinear_right_shift,
                qcsnet4_lblk2_qlinear_bias,
                qcsnet4_lblk2_qlinear_input_zero_point,
                qcsnet4_lblk2_qlinear_weight_sum
            );

            hls::stream<ap_int8_c> s_lin2_pass("s_lin2_pass");
            if (should_write) {
                tee_and_write<4, 1>(
                    s_lin2, s_lin2_pass,
                    out_dir + "lin2" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                for (int i = 0; i < 4; ++i)
                    s_lin2_pass.write(s_lin2.read());
            }

            // LIF_head: 4 outputs
            hls::stream<ap_int8_c> s_lif_head("s_lif_head");
            lif_head.forward(
                s_lin2_pass, s_lif_head,
                qcsnet4_lblk2_leaky_beta_int,
                qcsnet4_lblk2_leaky_theta_int,
                qcsnet4_lblk2_leaky_scale_int
            );

            hls::stream<ap_int8_c> s_lif_head_bin("s_lif_head_bin");
            if (should_write) {
                tee_and_write_spikes<4, 1>(
                    s_lif_head, s_lif_head_bin,
                    out_dir + "lif_head" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                for (int i = 0; i < 4; ++i) {
                    ap_int8_c v = s_lif_head.read();
                    s_lif_head_bin.write((v > 0) ? ap_int8_c(1) : ap_int8_c(0));
                }
            }

            // Accumulate head spikes
            for (int i = 0; i < 4; ++i) {
                ap_int8_c v = s_lif_head_bin.read();
                lif_head_acc[i] += (ap_int<32>)v;
            }

            if ((step & 1) == 0) {
                std::cout << "  Timestep " << step << " complete\n";
            }

        } // end NUM_STEPS temporal loop

        // ---------------------------------------------------------------------
        // Write accumulated outputs (optional: aggregated spike dumps)
        // ---------------------------------------------------------------------
        append_buffer_to_file<OUT_CH1, FEATURE_LENGTH1>(
            lif1_acc, out_dir + "lif1_accumulated_cpp.txt",
            sample_idx + 1, is_first_sample);

        append_buffer_to_file<OUT_CH2, FEATURE_LENGTH2>(
            lif2_acc, out_dir + "lif2_accumulated_cpp.txt",
            sample_idx + 1, is_first_sample);

        append_buffer_to_file<OUT_CH3, FEATURE_LENGTH3>(
            lif3_acc, out_dir + "lif3_accumulated_cpp.txt",
            sample_idx + 1, is_first_sample);

        append_buffer_to_file<LINEAR_OUT_SIZE1, 1>(
            lif_lin1_acc, out_dir + "lif_lin1_accumulated_cpp.txt",
            sample_idx + 1, is_first_sample);

        append_buffer_to_file<4, 1>(
            lif_head_acc, out_dir + "lif_head_accumulated_cpp.txt",
            sample_idx + 1, is_first_sample);

        // ---------------------------------------------------------------------
        // Per-sample prediction & confusion matrix update (multi-class)
        // ---------------------------------------------------------------------
        int pred_class = 0;
        ap_int<32> best = lif_head_acc[0];
        for (int c = 1; c < 4; ++c) {
            if (lif_head_acc[c] > best) {
                best = lif_head_acc[c];
                pred_class = c;
            }
        }

        if (y_true < 0 || y_true >= N_CLASSES) {
            std::cerr << "Warning: label out of range (" << y_true
                      << ") for sample " << (sample_idx + 1) << "\n";
        } else {
            cm[y_true][pred_class]++;
        }

        std::cout << "Sample " << (sample_idx + 1)
                  << "  pred=" << pred_class
                  << "  true=" << y_true
                  << "  acc=[" << int(lif_head_acc[0]) << ","
                               << int(lif_head_acc[1]) << ","
                               << int(lif_head_acc[2]) << ","
                               << int(lif_head_acc[3]) << "]\n";

    } // end samples loop

    // -------------------------------------------------------------------------
    // 6. Final confusion matrix and metrics
    // -------------------------------------------------------------------------
    std::cout << "\n=== Confusion Matrix [true x pred] ===\n";
    std::cout << "      pred:  0      1      2      3\n";
    for (int i = 0; i < N_CLASSES; ++i) {
        std::cout << "true " << i << ": ";
        for (int j = 0; j < N_CLASSES; ++j) {
            std::cout << std::setw(6) << cm[i][j] << " ";
        }
        std::cout << "\n";
    }

    const double eps = 1e-12;
    long total_correct = 0, total_samples = 0;
    for (int i = 0; i < N_CLASSES; ++i) {
        for (int j = 0; j < N_CLASSES; ++j) {
            total_samples += cm[i][j];
            if (i == j) total_correct += cm[i][j];
        }
    }

    static const char* CLASS_NAMES[N_CLASSES] = {
        "Normal (N)",
        "SVEB (S)",
        "VEB (V)",
        "F (F)"
    };

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n=== Per-class metrics (multi-class) ===\n";

    for (int c = 0; c < N_CLASSES; ++c) {
        long TP = cm[c][c];
        long FN = 0, FP = 0, TN = 0;

        for (int i = 0; i < N_CLASSES; ++i) {
            for (int j = 0; j < N_CLASSES; ++j) {
                long val = cm[i][j];
                if (i == c && j != c)      FN += val;  // true c, predicted others
                else if (i != c && j == c) FP += val;  // true others, predicted c
                else if (i != c && j != c) TN += val;  // all the rest
            }
        }

        double precision = TP / (double)(TP + FP + eps);
        double recall    = TP / (double)(TP + FN + eps);
        double f1        = (2.0 * precision * recall) / (precision + recall + eps);
        double accuracy  = (TP + TN) / (double)(TP + TN + FP + FN + eps);

        std::cout << "\nClass " << c << " - " << CLASS_NAMES[c] << "\n"
                  << "  Accuracy : " << accuracy  << "\n"
                  << "  Precision: " << precision << "\n"
                  << "  Recall   : " << recall    << "\n"
                  << "  F1       : " << f1        << "\n";
    }

    double acc_global = total_correct / (double)(total_samples + eps);
    std::cout << "\n=== Overall multi-class accuracy ===\n"
              << "Accuracy : " << acc_global << "\n";

    std::cout << "\n=== Complete ===\n";
    return 0;
}
