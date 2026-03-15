// testbench_dump_layers.cpp
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <vector>
#include "hls_stream.h"
#include "ap_int.h"

#include "../../include/hls4csnn1d_sd/model2/constants_sd.h"
#include "../../include/hls4csnn1d_sd/model2/filereader.h"
#include "../../include/hls4csnn1d_sd/model2/cblk_sd/nn2_cblk1_sd.h"

// Include individual layer headers
#include "../../include/hls4csnn1d_sd/model2/cblk_sd/conv1d_sd.h"
#include "../../include/hls4csnn1d_sd/model2/cblk_sd/batchnorm1d_sd.h"
#include "../../include/hls4csnn1d_sd/model2/cblk_sd/lif1d_integer.h"
// #include "../../include/hls4csnn1d_sd/model2/cblk_sd/lif1d_float.h"
#include "../../include/hls4csnn1d_sd/model2/cblk_sd/maxpool1d_sd.h"
#include "../../include/hls4csnn1d_sd/model2/cblk_sd/linear1d_sd.h"
// Add include
#include "../../include/hls4csnn1d_sd/model2/cblk_sd/quantidentity1d_sd.h"



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
    std::cout << "\n=== Layer-by-Layer Output Dumper (C++) ===\n\n";

    std::string out_dir = "../../../../compare_outputs/";
    if (argc >= 2) out_dir = argv[1];

    std::string folderPath =
        // "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_singles/large";
        "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_test/smallvhls";
    if (argc >= 3) folderPath = argv[2];

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

    // Pull all samples
    std::vector<array180_t> all_samples;
    all_samples.reserve(NUM_SAMPLES_LOADED);
    for (int i = 0; i < NUM_SAMPLES_LOADED; ++i) {
        all_samples.push_back(dataStreamInternal.read());
    }

    // === Label stream (binary: 0 stays 0; {1,2,3} -> 1) ===
    hls::stream<ap_int8_c> labelStream;
    reader.streamLabel(labelStream, /*binary=*/true);


    // === Metrics (binary) ===
    long TP0 = 0, TN0 = 0, FP0 = 0, FN0 = 0;
    long TP1 = 0, TN1 = 0, FP1 = 0, FN1 = 0;

    const int NUM_STEPS = 10;

    // Process each sample
    for (int sample_idx = 0; sample_idx < NUM_SAMPLES_LOADED; ++sample_idx) {
        std::cout << "\n========================================\n";
        std::cout << "Processing Sample " << (sample_idx + 1) << "/" << NUM_SAMPLES_LOADED << "\n";
        std::cout << "========================================\n";

        array180_t sample = all_samples[sample_idx];
        bool is_first_sample = (sample_idx == 0);

        // Fresh layer objects (constructor reset -> once per sample)
        Conv1D_SD<1, 16, 3, 1, CONV_IN_LENGTH1>                  conv1;
        BatchNorm1D_SD<OUT_CH1, FEATURE_LENGTH1>                 bn1;
        LIF1D_SD_Integer<OUT_CH1, FEATURE_LENGTH1>               lif1;
        // LIF1D_Float<OUT_CH1, FEATURE_LENGTH1>                    lif1;
        MaxPool1D_SD<2, 2, OUT_CH1, FEATURE_LENGTH1>             mp1;

        Conv1D_SD<16, 24, 3, 1, CONV_IN_LENGTH2>                 conv2;
        BatchNorm1D_SD<OUT_CH2, FEATURE_LENGTH2>                 bn2;
        LIF1D_SD_Integer<OUT_CH2, FEATURE_LENGTH2>               lif2;
        // LIF1D_Float<OUT_CH2, FEATURE_LENGTH2>                    lif2;
        MaxPool1D_SD<2, 2, OUT_CH2, FEATURE_LENGTH2>             mp2;

        // QuantIdentity blocks (before Conv2, before Linear)
        QuantIdentityPerTensor_Int8<OUT_CH1, FEATURE_LENGTH1/2>  qi2;
        QuantIdentityPerTensor_Int8<1, LINEAR_IN_SIZE>           qi_lin;

        Linear1D_SD<LINEAR_IN_SIZE, 2>                           fc;
        LIF1D_SD_Integer<2, 1>                                   lif_head;
        // LIF1D_Float<2, 1>                                        lif_head;

        // Accumulators
        static ap_int<32> lif1_acc[OUT_CH1 * FEATURE_LENGTH1];
        static ap_int<32> lif2_acc[OUT_CH2 * FEATURE_LENGTH2];
        static ap_int<32> lif_head_acc[2];

        for (int i = 0; i < OUT_CH1 * FEATURE_LENGTH1; ++i) lif1_acc[i] = 0;
        for (int i = 0; i < OUT_CH2 * FEATURE_LENGTH2; ++i) lif2_acc[i] = 0;
        lif_head_acc[0] = lif_head_acc[1] = 0;

        // Read ground-truth label for this sample
        const int y_true = int(labelStream.read());

        // Temporal loop
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

            // QIN1
            hls::stream<ap_int8_c> s_input_pass("s_input_pass");
            if (should_write) {
                tee_and_write<1, CONV_IN_LENGTH1>(
                    s_input, s_input_pass,
                    out_dir + "qin1" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                for (int i = 0; i < CONV_IN_LENGTH1; ++i) s_input_pass.write(s_input.read());
            }

            // Conv1
            hls::stream<ap_int8_c> s_conv1("s_conv1");
            conv1.forward(
                s_input_pass, s_conv1,
                qcsnet2_cblk1_qconv1d_weights,
                qcsnet2_cblk1_qconv1d_scale_multiplier,
                qcsnet2_cblk1_qconv1d_right_shift,
                qcsnet2_cblk1_qconv1d_bias,
                qcsnet2_cblk1_qconv1d_input_zero_point,
                qcsnet2_cblk1_qconv1d_weight_sum
            );

            hls::stream<ap_int8_c> s_conv1_pass("s_conv1_pass");
            if (should_write) {
                tee_and_write<OUT_CH1, FEATURE_LENGTH1>(
                    s_conv1, s_conv1_pass,
                    out_dir + "conv1" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                for (int i = 0; i < OUT_CH1 * FEATURE_LENGTH1; ++i) s_conv1_pass.write(s_conv1.read());
            }

            // BN1
            hls::stream<ap_int8_c> s_bn1("s_bn1");
            bn1.forward(
                s_conv1_pass, s_bn1,
                qcsnet2_cblk1_batch_norm_weight,
                qcsnet2_cblk1_batch_norm_bias,
                qcsnet2_cblk1_batch_norm_scale_multiplier,
                qcsnet2_cblk1_batch_norm_right_shift
            );

            hls::stream<ap_int8_c> s_bn1_pass("s_bn1_pass");
            if (should_write) {
                tee_and_write<OUT_CH1, FEATURE_LENGTH1>(
                    s_bn1, s_bn1_pass,
                    out_dir + "bn1" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                for (int i = 0; i < OUT_CH1 * FEATURE_LENGTH1; ++i) s_bn1_pass.write(s_bn1.read());
            }

            // LIF1 (emit/propagate binary spikes)
            hls::stream<ap_int8_c> s_lif1("s_lif1");
            lif1.forward(
                s_bn1_pass, s_lif1,
                qcsnet2_cblk1_leaky_beta_int,
                qcsnet2_cblk1_leaky_theta_int,
                qcsnet2_cblk1_leaky_scale_int
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
                tee_and_write<OUT_CH1, FEATURE_LENGTH1/2>(
                    s_mp1, s_mp1_pass,
                    out_dir + "mp1" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                for (int i = 0; i < OUT_CH1 * (FEATURE_LENGTH1/2); ++i) s_mp1_pass.write(s_mp1.read());
            }

            // QuantIdentity before Conv2
            hls::stream<ap_int8_c> s_qi2("s_qi2");
            qi2.forward(s_mp1_pass, s_qi2, qcsnet2_cblk2_input_act_scale_int);

            // QIN2 dump
            hls::stream<ap_int8_c> s_qin2_pass("s_qin2_pass");
            if (should_write) {
                tee_and_write<OUT_CH1, FEATURE_LENGTH1/2>(
                    s_qi2, s_qin2_pass,
                    out_dir + "qin2" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                for (int i = 0; i < OUT_CH1 * (FEATURE_LENGTH1/2); ++i) s_qin2_pass.write(s_qi2.read());
            }

            // Conv2
            hls::stream<ap_int8_c> s_conv2("s_conv2");
            conv2.forward(
                s_qin2_pass, s_conv2,
                qcsnet2_cblk2_qconv1d_weights,
                qcsnet2_cblk2_qconv1d_scale_multiplier,
                qcsnet2_cblk2_qconv1d_right_shift,
                qcsnet2_cblk2_qconv1d_bias,
                qcsnet2_cblk2_qconv1d_input_zero_point,
                qcsnet2_cblk2_qconv1d_weight_sum
            );

            hls::stream<ap_int8_c> s_conv2_pass("s_conv2_pass");
            if (should_write) {
                tee_and_write<OUT_CH2, FEATURE_LENGTH2>(
                    s_conv2, s_conv2_pass,
                    out_dir + "conv2" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                for (int i = 0; i < OUT_CH2 * FEATURE_LENGTH2; ++i) s_conv2_pass.write(s_conv2.read());
            }

            // BN2
            hls::stream<ap_int8_c> s_bn2("s_bn2");
            bn2.forward(
                s_conv2_pass, s_bn2,
                qcsnet2_cblk2_batch_norm_weight,
                qcsnet2_cblk2_batch_norm_bias,
                qcsnet2_cblk2_batch_norm_scale_multiplier,
                qcsnet2_cblk2_batch_norm_right_shift
            );

            hls::stream<ap_int8_c> s_bn2_pass("s_bn2_pass");
            if (should_write) {
                tee_and_write<OUT_CH2, FEATURE_LENGTH2>(
                    s_bn2, s_bn2_pass,
                    out_dir + "bn2" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                for (int i = 0; i < OUT_CH2 * FEATURE_LENGTH2; ++i) s_bn2_pass.write(s_bn2.read());
            }

            // LIF2
            hls::stream<ap_int8_c> s_lif2("s_lif2");
            lif2.forward(
                s_bn2_pass, s_lif2,
                qcsnet2_cblk2_leaky_beta_int,
                qcsnet2_cblk2_leaky_theta_int,
                qcsnet2_cblk2_leaky_scale_int
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
                tee_and_write<OUT_CH2, FEATURE_LENGTH2/2>(
                    s_mp2, s_mp2_pass,
                    out_dir + "mp2" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                for (int i = 0; i < OUT_CH2 * (FEATURE_LENGTH2/2); ++i) s_mp2_pass.write(s_mp2.read());
            }

            // QuantIdentity before Linear
            hls::stream<ap_int8_c> s_qin_lin("s_qin_lin");
            qi_lin.forward(s_mp2_pass, s_qin_lin, qcsnet2_lblk1_input_act_scale_int);

            // QIN_LIN dump
            hls::stream<ap_int8_c> s_qin_lin_pass("s_qin_lin_pass");
            if (should_write) {
                tee_and_write<1, LINEAR_IN_SIZE>(
                    s_qin_lin, s_qin_lin_pass,
                    out_dir + "qin_lin" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                for (int i = 0; i < LINEAR_IN_SIZE; ++i) s_qin_lin_pass.write(s_qin_lin.read());
            }

            // Linear
            hls::stream<ap_int8_c> s_lin("s_lin");
            fc.forward(
                s_qin_lin_pass, s_lin,
                qcsnet2_lblk1_qlinear_weights,
                qcsnet2_lblk1_qlinear_scale_multiplier,
                qcsnet2_lblk1_qlinear_right_shift,
                qcsnet2_lblk1_qlinear_bias,
                qcsnet2_lblk1_qlinear_input_zero_point,
                qcsnet2_lblk1_qlinear_weight_sum
            );

            hls::stream<ap_int8_c> s_lin_pass("s_lin_pass");
            if (should_write) {
                tee_and_write<2, 1>(
                    s_lin, s_lin_pass,
                    out_dir + "lin_out" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                for (int i = 0; i < 2; ++i) s_lin_pass.write(s_lin.read());
            }

            // LIF head
            hls::stream<ap_int8_c> s_lif_head("s_lif_head");
            lif_head.forward(
                s_lin_pass, s_lif_head,
                qcsnet2_lblk1_leaky_beta_int,
                qcsnet2_lblk1_leaky_theta_int,
                qcsnet2_lblk1_leaky_scale_int
            );

            hls::stream<ap_int8_c> s_lif_head_bin("s_lif_head_bin");
            if (should_write) {
                tee_and_write_spikes<2, 1>(
                    s_lif_head, s_lif_head_bin,
                    out_dir + "lif_head" + step_suffix + "_cpp.txt",
                    sample_idx + 1, is_first_sample, true);
            } else {
                for (int i = 0; i < 2; ++i) {
                    ap_int8_c v = s_lif_head.read();
                    s_lif_head_bin.write((v > 0) ? ap_int8_c(1) : ap_int8_c(0));
                }
            }

            // Accumulate head spikes (binary)
            for (int i = 0; i < 2; ++i) {
                ap_int8_c v = s_lif_head_bin.read();
                lif_head_acc[i] += (ap_int<32>)v;
            }

            if ((step & 1) == 0) {
                std::cout << "  Timestep " << step << " complete\n";
            }
        } // end NUM_STEPS

        // Write accumulated outputs (optional files)
        append_buffer_to_file<OUT_CH1, FEATURE_LENGTH1>(
            lif1_acc, out_dir + "lif1_accumulated_cpp.txt",
            sample_idx + 1, is_first_sample);

        append_buffer_to_file<OUT_CH2, FEATURE_LENGTH2>(
            lif2_acc, out_dir + "lif2_accumulated_cpp.txt",
            sample_idx + 1, is_first_sample);

        append_buffer_to_file<2, 1>(
            lif_head_acc, out_dir + "lif_head_accumulated_cpp.txt",
            sample_idx + 1, is_first_sample);

        // === PER-SAMPLE PREDICTION & METRICS UPDATE ===
        // Use accumulated counts (or average—monotonic either way)
        int pred_class = (lif_head_acc[1] > lif_head_acc[0]) ? 1 : 0;

        // If you prefer average:
        // ap_int<32> avg0 = lif_head_acc[0] / NUM_STEPS;
        // ap_int<32> avg1 = lif_head_acc[1] / NUM_STEPS;
        // int pred_class = (avg1 > avg0) ? 1 : 0;


        // Update confusion matrix
        if (pred_class == 1 && y_true == 1) {++TP1; ++TN0;}
        else if (pred_class == 0 && y_true == 0) {++TN1; ++TP0;}
        else if (pred_class == 1 && y_true == 0) {++FP1; ++FN0;}
        else if (pred_class == 0 && y_true == 1) {++FN1; ++FP0;}

        std::cout << "Sample " << (sample_idx + 1)
                << "  pred=" << pred_class
                << "  true=" << y_true
                << "  acc=[" << int(lif_head_acc[0]) << "," << int(lif_head_acc[1]) << "]\n";

    } // end samples

    // === Final metrics ===
    const double eps = 1e-12;
    const double TP1d = double(TP1), TN1d = double(TN1), FP1d = double(FP1), FN1d = double(FN1);
    const double TP0d = double(TP0), TN0d = double(TN0), FP0d = double(FP0), FN0d = double(FN0);

    const double precision1 = TP1d / (TP1d + FP1d + eps);
    const double recall1    = TP1d / (TP1d + FN1d + eps);
    const double f11        = (2.0 * precision1 * recall1) / (precision1 + recall1 + eps);
    const double accuracy1  = (TP1d + TN1d) / (TP1d + TN1d + FP1d + FN1d + eps);

    const double precision0 = TP0d / (TP0d + FP0d + eps);
    const double recall0    = TP0d / (TP0d + FN0d + eps);
    const double f10        = (2.0 * precision0 * recall0) / (precision0 + recall0 + eps);
    const double accuracy0  = (TP0d + TN0d) / (TP0d + TN0d + FP0d + FN0d + eps);

    std::cout << std::fixed << std::setprecision(4)
              << "\n=== Metrics for normal class ===\n"
              << "Accuracy : " << accuracy0  << "\n"
              << "Precision: " << precision0 << "\n"
              << "Recall   : " << recall0    << "\n"
              << "F1       : " << f10        << "\n" << "\n";

    std::cout << std::fixed << std::setprecision(4)
              << "\n=== Metrics for abnormal class ===\n"
              << "Accuracy : " << accuracy1  << "\n"
              << "Precision: " << precision1 << "\n"
              << "Recall   : " << recall1    << "\n"
              << "F1       : " << f11        << "\n";

    std::cout << "\n=== Complete ===\n";
    return 0;
}



 
// int main(int argc, char** argv) {
//     std::cout << "\n=== Layer-by-Layer Output Dumper (C++) ===\n\n";
    
//     std::string out_dir = "../../../../compare_outputs/";
//     if (argc >= 2) out_dir = argv[1];
    
//     std::string folderPath = 
//         "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_singles/large";
//     if (argc >= 3) folderPath = argv[2];
    
//     FileReader reader;
//     reader.loadData(folderPath);
    
//     hls::stream<array180_t> dataStreamInternal;
//     reader.streamData(dataStreamInternal);
    
// #ifndef __SYNTHESIS__
//     const int NUM_SAMPLES_LOADED = dataStreamInternal.size();
// #else
//     const int NUM_SAMPLES_LOADED = NUM_SAMPLES;
// #endif
    
//     if (NUM_SAMPLES_LOADED <= 0) {
//         std::cerr << "No data loaded!\n";
//         return -1;
//     }
    
//     std::cout << "Loaded " << NUM_SAMPLES_LOADED << " ECG samples\n\n";
    
//     // Store all samples
//     std::vector<array180_t> all_samples;
//     for (int i = 0; i < NUM_SAMPLES_LOADED; ++i) {
//         all_samples.push_back(dataStreamInternal.read());
//     }
    
//     const int NUM_STEPS = 10;
    
//     // Process each sample
//     for (int sample_idx = 0; sample_idx < NUM_SAMPLES_LOADED; ++sample_idx) {
//         std::cout << "\n========================================\n";
//         std::cout << "Processing Sample " << (sample_idx + 1) << "/" << NUM_SAMPLES_LOADED << "\n";
//         std::cout << "========================================\n";
        
//         array180_t sample = all_samples[sample_idx];
//         bool is_first_sample = (sample_idx == 0);
        
//         // Create fresh layer objects
//         Conv1D_SD<1, 16, 3, 1, CONV_IN_LENGTH1>                  conv1;
//         BatchNorm1D_SD<OUT_CH1, FEATURE_LENGTH1>                 bn1;
//         LIF1D_SD_Integer<OUT_CH1, FEATURE_LENGTH1>               lif1;
//         MaxPool1D_SD<2, 2, OUT_CH1, FEATURE_LENGTH1>             mp1;

//         // Instantiate QI blocks near other layer objects
//         QuantIdentityPerTensor_Int8<OUT_CH1, FEATURE_LENGTH1/2>  qi2;
        
        
//         Conv1D_SD<16, 24, 3, 1, CONV_IN_LENGTH2>                 conv2;
//         BatchNorm1D_SD<OUT_CH2, FEATURE_LENGTH2>                 bn2;
//         LIF1D_SD_Integer<OUT_CH2, FEATURE_LENGTH2>               lif2;
//         MaxPool1D_SD<2, 2, OUT_CH2, FEATURE_LENGTH2>             mp2;

//         QuantIdentityPerTensor_Int8<1, LINEAR_IN_SIZE>           qi_lin;
        
//         Linear1D_SD<LINEAR_IN_SIZE, 2>                           fc;
//         LIF1D_SD_Integer<2, 1>                                   lif_head;
        
//         // Accumulators
//         static ap_int<32> lif1_acc[OUT_CH1 * FEATURE_LENGTH1];
//         static ap_int<32> lif2_acc[OUT_CH2 * FEATURE_LENGTH2];
//         static ap_int<32> lif_head_acc[2];
        
//         for (int i = 0; i < OUT_CH1 * FEATURE_LENGTH1; ++i) lif1_acc[i] = 0;
//         for (int i = 0; i < OUT_CH2 * FEATURE_LENGTH2; ++i) lif2_acc[i] = 0;
//         lif_head_acc[0] = lif_head_acc[1] = 0;
        
//         // Temporal loop
//         for (int step = 0; step < NUM_STEPS; ++step) {
//             bool should_write = (step == 0) || (step == NUM_STEPS - 1);
            
//             std::string step_suffix = "";
//             if (should_write) {
//                 char buf[16];
//                 sprintf(buf, "_step%02d", step);
//                 step_suffix = buf;
//             }
            
//             // ========== FORWARD PASS (always execute) ==========
            
//             // Input stream
//             hls::stream<ap_int8_c> s_input("s_input");
//             for (int i = 0; i < CONV_IN_LENGTH1; ++i) {
//                 s_input.write(sample[i]);
//             }
            
//             // QIN1 - Write at BOTH step 0 and 9
//             hls::stream<ap_int8_c> s_input_pass("s_input_pass");
//             if (should_write) {
//                 tee_and_write<1, CONV_IN_LENGTH1>(s_input, s_input_pass,
//                     out_dir + "qin1" + step_suffix + "_cpp.txt",
//                     sample_idx + 1, is_first_sample, true);
//             } else {
//                 copy_stream<CONV_IN_LENGTH1>(s_input, s_input_pass, s_input_pass);
//             }
            
//             // Conv1 - Write at BOTH step 0 and 9
//             hls::stream<ap_int8_c> s_conv1("s_conv1");
//             conv1.forward(s_input_pass, s_conv1,
//                 qcsnet2_cblk1_qconv1d_weights,
//                 qcsnet2_cblk1_qconv1d_scale_multiplier,
//                 qcsnet2_cblk1_qconv1d_right_shift,
//                 qcsnet2_cblk1_qconv1d_bias,
//                 qcsnet2_cblk1_qconv1d_input_zero_point,
//                 qcsnet2_cblk1_qconv1d_weight_sum);
            
//             hls::stream<ap_int8_c> s_conv1_pass("s_conv1_pass");
//             if (should_write) {
//                 tee_and_write<OUT_CH1, FEATURE_LENGTH1>(s_conv1, s_conv1_pass,
//                     out_dir + "conv1" + step_suffix + "_cpp.txt",
//                     sample_idx + 1, is_first_sample, true);
//             } else {
//                 copy_stream<OUT_CH1 * FEATURE_LENGTH1>(s_conv1, s_conv1_pass, s_conv1_pass);
//             }
            
//             // BN1 - Write at BOTH step 0 and 9
//             hls::stream<ap_int8_c> s_bn1("s_bn1");
//             bn1.forward(s_conv1_pass, s_bn1,
//                 qcsnet2_cblk1_batch_norm_weight,
//                 qcsnet2_cblk1_batch_norm_bias,
//                 qcsnet2_cblk1_batch_norm_scale_multiplier,
//                 qcsnet2_cblk1_batch_norm_right_shift);
            
//             hls::stream<ap_int8_c> s_bn1_pass("s_bn1_pass");
//             if (should_write) {
//                 tee_and_write<OUT_CH1, FEATURE_LENGTH1>(s_bn1, s_bn1_pass,
//                     out_dir + "bn1" + step_suffix + "_cpp.txt",
//                     sample_idx + 1, is_first_sample, true);
//             } else {
//                 copy_stream<OUT_CH1 * FEATURE_LENGTH1>(s_bn1, s_bn1_pass, s_bn1_pass);
//             }
            
//             // LIF1 - Write at BOTH step 0 and 9
//             hls::stream<ap_int8_c> s_lif1("s_lif1");
//             lif1.forward(s_bn1_pass, s_lif1,
//                 qcsnet2_cblk1_leaky_beta_int,
//                 qcsnet2_cblk1_leaky_theta_int,
//                 qcsnet2_cblk1_leaky_scale_int);
            
//             // hls::stream<ap_int8_c> s_lif1_acc("s_lif1_acc");
//             // if (should_write) {
//             //     tee_and_write<OUT_CH1, FEATURE_LENGTH1>(s_lif1, s_lif1_acc,
//             //         out_dir + "lif1" + step_suffix + "_cpp.txt",
//             //         sample_idx + 1, is_first_sample, true);
//             // } else {
//             //     copy_stream<OUT_CH1 * FEATURE_LENGTH1>(s_lif1, s_lif1_acc, s_lif1_acc);
//             // }
//             // LIF1 - Write normalized binary spikes for comparison
//             hls::stream<ap_int8_c> s_lif1_acc("s_lif1_acc");
//             if (should_write) {
//                 tee_and_write_spikes<OUT_CH1, FEATURE_LENGTH1>(s_lif1, s_lif1_acc,  // ← Use spike version
//                     out_dir + "lif1" + step_suffix + "_cpp.txt",
//                     sample_idx + 1, is_first_sample, true);
//             } else {
//                 copy_stream<OUT_CH1 * FEATURE_LENGTH1>(s_lif1, s_lif1_acc, s_lif1_acc);
//             }

// // Keep using regular tee_and_write for non-LIF layers
            
//             hls::stream<ap_int8_c> s_lif1_pass("s_lif1_pass");
//             read_and_accumulate<OUT_CH1 * FEATURE_LENGTH1>(s_lif1_acc, s_lif1_pass, lif1_acc);
            
//             // MaxPool1 - Write at BOTH step 0 and 9
//             hls::stream<ap_int8_c> s_mp1("s_mp1");
//             mp1.forward(s_lif1_pass, s_mp1);
            
//             hls::stream<ap_int8_c> s_mp1_pass("s_mp1_pass");
//             if (should_write) {
//                 tee_and_write<OUT_CH1, FEATURE_LENGTH1/2>(s_mp1, s_mp1_pass,
//                     out_dir + "mp1" + step_suffix + "_cpp.txt",
//                     sample_idx + 1, is_first_sample, true);
//             } else {
//                 copy_stream<OUT_CH1 * (FEATURE_LENGTH1/2)>(s_mp1, s_mp1_pass, s_mp1_pass);
//             }

//             // --- after mp1 ---
//             hls::stream<ap_int8_c> s_qin2("s_qin2");
//             qi2.forward(
//                 s_mp1_pass, s_qin2,
//                 qcsnet2_cblk2_act_scale_int  // from your generated qparams header
//             );

//             // QIN2 - Write at BOTH step 0 and 9
//             hls::stream<ap_int8_c> s_qin2_pass("s_qin2_pass");
//             if (should_write) {
//                 tee_and_write<OUT_CH1, FEATURE_LENGTH1/2>(s_qin2, s_qin2_pass,
//                     out_dir + "qin2" + step_suffix + "_cpp.txt",
//                     sample_idx + 1, is_first_sample, true);
//             } else {
//                 copy_stream<OUT_CH1 * (FEATURE_LENGTH1/2)>(s_qin2, s_qin2_pass, s_qin2_pass);
//             }
            
//             // Conv2 - Write at BOTH step 0 and 9
//             hls::stream<ap_int8_c> s_conv2("s_conv2");
//             conv2.forward(s_qin2_pass, s_conv2,
//                 qcsnet2_cblk2_qconv1d_weights,
//                 qcsnet2_cblk2_qconv1d_scale_multiplier,
//                 qcsnet2_cblk2_qconv1d_right_shift,
//                 qcsnet2_cblk2_qconv1d_bias,
//                 qcsnet2_cblk2_qconv1d_input_zero_point,
//                 qcsnet2_cblk2_qconv1d_weight_sum);
            
//             hls::stream<ap_int8_c> s_conv2_pass("s_conv2_pass");
//             if (should_write) {
//                 tee_and_write<OUT_CH2, FEATURE_LENGTH2>(s_conv2, s_conv2_pass,
//                     out_dir + "conv2" + step_suffix + "_cpp.txt",
//                     sample_idx + 1, is_first_sample, true);
//             } else {
//                 copy_stream<OUT_CH2 * FEATURE_LENGTH2>(s_conv2, s_conv2_pass, s_conv2_pass);
//             }
            
//             // BN2 - Write at BOTH step 0 and 9
//             hls::stream<ap_int8_c> s_bn2("s_bn2");
//             bn2.forward(s_conv2_pass, s_bn2,
//                 qcsnet2_cblk2_batch_norm_weight,
//                 qcsnet2_cblk2_batch_norm_bias,
//                 qcsnet2_cblk2_batch_norm_scale_multiplier,
//                 qcsnet2_cblk2_batch_norm_right_shift);
            
//             hls::stream<ap_int8_c> s_bn2_pass("s_bn2_pass");
//             if (should_write) {
//                 tee_and_write<OUT_CH2, FEATURE_LENGTH2>(s_bn2, s_bn2_pass,
//                     out_dir + "bn2" + step_suffix + "_cpp.txt",
//                     sample_idx + 1, is_first_sample, true);
//             } else {
//                 copy_stream<OUT_CH2 * FEATURE_LENGTH2>(s_bn2, s_bn2_pass, s_bn2_pass);
//             }
            
//             // LIF2 - Write at BOTH step 0 and 9
//             hls::stream<ap_int8_c> s_lif2("s_lif2");
//             lif2.forward(s_bn2_pass, s_lif2,
//                 qcsnet2_cblk2_leaky_beta_int,
//                 qcsnet2_cblk2_leaky_theta_int,
//                 qcsnet2_cblk2_leaky_scale_int);
            
//             hls::stream<ap_int8_c> s_lif2_acc("s_lif2_acc");
//             if (should_write) {
//                 tee_and_write<OUT_CH2, FEATURE_LENGTH2>(s_lif2, s_lif2_acc,
//                     out_dir + "lif2" + step_suffix + "_cpp.txt",
//                     sample_idx + 1, is_first_sample, true);
//             } else {
//                 copy_stream<OUT_CH2 * FEATURE_LENGTH2>(s_lif2, s_lif2_acc, s_lif2_acc);
//             }
            
//             hls::stream<ap_int8_c> s_lif2_pass("s_lif2_pass");
//             read_and_accumulate<OUT_CH2 * FEATURE_LENGTH2>(s_lif2_acc, s_lif2_pass, lif2_acc);
            
//             // MaxPool2 - Write at BOTH step 0 and 9
//             hls::stream<ap_int8_c> s_mp2("s_mp2");
//             mp2.forward(s_lif2_pass, s_mp2);
            
//             hls::stream<ap_int8_c> s_mp2_pass("s_mp2_pass");
//             if (should_write) {
//                 tee_and_write<OUT_CH2, FEATURE_LENGTH2/2>(s_mp2, s_mp2_pass,
//                     out_dir + "mp2" + step_suffix + "_cpp.txt",
//                     sample_idx + 1, is_first_sample, true);
//             } else {
//                 copy_stream<OUT_CH2 * (FEATURE_LENGTH2/2)>(s_mp2, s_mp2_pass, s_mp2_pass);
//             }
            
//             // Flatten - Write at BOTH step 0 and 9
//             hls::stream<ap_int8_c> s_flat_pass("s_flat_pass");
//             if (should_write) {
//                 tee_and_write<OUT_CH2, FEATURE_LENGTH2/2>(s_mp2_pass, s_flat_pass,
//                     out_dir + "flat" + step_suffix + "_cpp.txt",
//                     sample_idx + 1, is_first_sample, true);
//             } else {
//                 copy_stream<OUT_CH2 * (FEATURE_LENGTH2/2)>(s_mp2_pass, s_flat_pass, s_flat_pass);
//             }

//             // --- before linear head ---
//             hls::stream<ap_int8_c> s_qin_lin("s_qin_lin");
//             qi_lin.forward(
//                 s_flat_pass, s_qin_lin,
//                 qcsnet2_lblk1_act_scale_int  // from qparams
//             );
            
//             // QIN_LIN - Write at BOTH step 0 and 9
//             hls::stream<ap_int8_c> s_qin_lin_pass("s_qin_lin_pass");
//             if (should_write) {
//                 tee_and_write<1, LINEAR_IN_SIZE>(s_flat_pass, s_qin_lin_pass,
//                     out_dir + "qin_lin" + step_suffix + "_cpp.txt",
//                     sample_idx + 1, is_first_sample, true);
//             } else {
//                 copy_stream<LINEAR_IN_SIZE>(s_flat_pass, s_qin_lin_pass, s_qin_lin_pass);
//             }
            
//             // Linear - Write at BOTH step 0 and 9
//             hls::stream<ap_int8_c> s_lin("s_lin");
//             fc.forward(s_qin_lin_pass, s_lin,
//                 qcsnet2_lblk1_qlinear_weights,
//                 qcsnet2_lblk1_qlinear_scale_multiplier,
//                 qcsnet2_lblk1_qlinear_right_shift,
//                 qcsnet2_lblk1_qlinear_bias,
//                 qcsnet2_lblk1_qlinear_input_zero_point,
//                 qcsnet2_lblk1_qlinear_weight_sum);
            
//             hls::stream<ap_int8_c> s_lin_pass("s_lin_pass");
//             if (should_write) {
//                 tee_and_write<2, 1>(s_lin, s_lin_pass,
//                     out_dir + "lin_out" + step_suffix + "_cpp.txt",
//                     sample_idx + 1, is_first_sample, true);
//             } else {
//                 copy_stream<2>(s_lin, s_lin_pass, s_lin_pass);
//             }
            
//             // LIF head - Write at BOTH step 0 and 9
//             hls::stream<ap_int8_c> s_lif_head("s_lif_head");
//             lif_head.forward(s_lin_pass, s_lif_head,
//                 qcsnet2_lblk1_leaky_beta_int,
//                 qcsnet2_lblk1_leaky_theta_int,
//                 qcsnet2_lblk1_leaky_scale_int);
            
//             hls::stream<ap_int8_c> s_lif_head_acc("s_lif_head_acc");
//             if (should_write) {
//                 tee_and_write<2, 1>(s_lif_head, s_lif_head_acc,
//                     out_dir + "lif_head" + step_suffix + "_cpp.txt",
//                     sample_idx + 1, is_first_sample, true);
//             } else {
//                 copy_stream<2>(s_lif_head, s_lif_head_acc, s_lif_head_acc);
//             }
            
//             // Accumulate head
//             for (int i = 0; i < 2; ++i) {
//                 ap_int8_c val = s_lif_head_acc.read();
//                 lif_head_acc[i] += (ap_int<32>)val;
//             }
            
//             if (step % 2 == 0) {
//                 std::cout << "  Timestep " << step << " complete\n";
//             }
//         }
        
//         // Write accumulated outputs
//         append_buffer_to_file<OUT_CH1, FEATURE_LENGTH1>(lif1_acc,
//             out_dir + "lif1_accumulated_cpp.txt",
//             sample_idx + 1, is_first_sample);
        
//         append_buffer_to_file<OUT_CH2, FEATURE_LENGTH2>(lif2_acc,
//             out_dir + "lif2_accumulated_cpp.txt",
//             sample_idx + 1, is_first_sample);
        
//         append_buffer_to_file<2, 1>(lif_head_acc,
//             out_dir + "lif_head_accumulated_cpp.txt",
//             sample_idx + 1, is_first_sample);
        
//         // Averaged
//         static ap_int<32> lif_head_avg[2];
//         lif_head_avg[0] = lif_head_acc[0] / NUM_STEPS;
//         lif_head_avg[1] = lif_head_acc[1] / NUM_STEPS;
        
//         append_buffer_to_file<2, 1>(lif_head_avg,
//             out_dir + "lif_head_averaged_cpp.txt",
//             sample_idx + 1, is_first_sample);
        
//         std::cout << "Sample " << (sample_idx + 1) << " accumulated: ["
//                   << int(lif_head_acc[0]) << ", " << int(lif_head_acc[1]) << "]\n";
//     }
    
//     std::cout << "\n=== Complete ===\n";
//     return 0;
// }