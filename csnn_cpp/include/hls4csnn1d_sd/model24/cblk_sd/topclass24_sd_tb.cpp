#include <iostream>
#include <iomanip>
#include <string>
#include <hls_stream.h>

#include "topclass24_sd_tb.h"
#include "../filereader24.h"
#include "../constants24_sd.h"

// ===== CHANGE 1: Updated constants =====
static const int TOTAL_INPUT = FIXED_LENGTH1 + RR_FEATURE_LENGTH + RR_FEATURE_LENGTH;  // 188
static const int WORDS_PER_ROW = (TOTAL_INPUT + 7) / 8;                                // 24

// ------------------------------------------------------------
// Pack ONE sample: signal (180) + RR_s1 (4) + RR_s2 (4) = 188 bytes
// into 24 AXI 64b words.
// ===== CHANGE 2: Updated signature and packing =====
// ------------------------------------------------------------
static void input_row_to_axi(const array180rr_t& sample,
                             const array_rr_t& rr_s1,
                             const array_rr_t& rr_s2,
                             hls::stream<axi_fixed_t>& dst)
{
    // Build a temporary buffer with all 188 bytes
    ap_int8_c buf[TOTAL_INPUT];

    // Copy signal (0-179)
    for (int i = 0; i < FIXED_LENGTH1; ++i) {
        buf[i] = sample[i];
    }

    // Copy RR_s1 (180-183)
    for (int i = 0; i < RR_FEATURE_LENGTH; ++i) {
        buf[FIXED_LENGTH1 + i] = rr_s1[i];
    }

    // Copy RR_s2 (184-187)
    for (int i = 0; i < RR_FEATURE_LENGTH; ++i) {
        buf[FIXED_LENGTH1 + RR_FEATURE_LENGTH + i] = rr_s2[i];
    }

    // Pack into AXI words
    for (int w = 0; w < WORDS_PER_ROW; ++w) {
#pragma HLS PIPELINE II=1
        axi_fixed_t word;
        word.data = 0;
        word.keep = 0;
        word.strb = 0;
        word.last = 0;

        for (int j = 0; j < 8; ++j) {
#pragma HLS UNROLL
            int idx = w * 8 + j;
            ap_uint<8> byte_val = 0;

            if (idx < TOTAL_INPUT) {
                byte_val = buf[idx].range(7, 0);
                word.keep |= (ap_uint<8>)(1u << j);
            }
            word.data.range(j * 8 + 7, j * 8) = byte_val;
        }

        word.strb = word.keep;
        if (w == WORDS_PER_ROW - 1) word.last = 1;

        dst.write(word);
    }
}

// ------------------------------------------------------------
// Read ONE AXI word that encodes ONE scalar pred:
// pred in byte0, bytes1-3 = 0, KEEP=0x0F, LAST=1.
// ------------------------------------------------------------
static ap_int8_c axi_to_pred_scalar(hls::stream<axi_fixed_t>& src)
{
#pragma HLS PIPELINE II=1
    axi_fixed_t word = src.read();

#ifndef __SYNTHESIS__
    if (word.last != 1)  std::cerr << "[WARN] pred word TLAST != 1\n";
    if (word.keep != 0x0F) std::cerr << "[WARN] pred word TKEEP != 0x0F\n";
#endif

    ap_int8_c pred;
    pred.range(7,0) = word.data.range(7,0); // byte0
    return pred;
}


static void print_cm2(long cm2[2][2])
{
    std::cout << "\n=== Binary Confusion Matrix [true x pred] ===\n";
    std::cout << "      pred:  0      1\n";
    std::cout << "true 0: " << std::setw(6) << cm2[0][0] << " " << std::setw(6) << cm2[0][1] << " \n";
    std::cout << "true 1: " << std::setw(6) << cm2[1][0] << " " << std::setw(6) << cm2[1][1] << " \n";
}

static void print_cm4(long cm4[4][4])
{
    std::cout << "\n=== Final 4-Class Confusion Matrix [true x pred] ===\n";
    std::cout << "      pred:  0      1      2      3\n";
    for (int i = 0; i < 4; ++i) {
        std::cout << "true " << i << ": ";
        for (int j = 0; j < 4; ++j) {
            std::cout << std::setw(6) << cm4[i][j] << " ";
        }
        std::cout << "\n";
    }
}

int main(int argc, char** argv)
{
    std::cout << "\n=== QCSNN24 Top-Level AXI Testbench (RR Bypass Fix) ===\n\n";

    // ------------------------------------------------------------
    // 1) Resolve dataset folder
    // ------------------------------------------------------------
    std::string folderPath =
            "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_intra_patient_4class_180_center90_rr_features_filtered/single_test";

    if (argc >= 2) {
        folderPath = argv[1];
    }

    // ------------------------------------------------------------
    // 2) Load data
    // ------------------------------------------------------------
    FileReader reader;
    reader.loadData(folderPath);

    hls::stream<array180rr_t> dataStreamInternal;
    reader.streamData(dataStreamInternal);

#ifndef __SYNTHESIS__
    const int NUM_SAMPLES_LOADED = (int)dataStreamInternal.size();
#else
    const int NUM_SAMPLES_LOADED = NUM_SAMPLES;
#endif

    if (NUM_SAMPLES_LOADED <= 0) {
        std::cerr << "No data loaded!\n";
        return -1;
    }

    std::cout << "Loaded " << NUM_SAMPLES_LOADED << " ECG samples\n";

    // Ground truth (4-class) as in native
    hls::stream<ap_int8_c> gtLabelStream;
    reader.streamLabel(gtLabelStream, /*binary=*/false);

    // ------------------------------------------------------------
    // 3) Confusion matrices (match native C++ output)
    // ------------------------------------------------------------
    long cm2[2][2] = {{0,0},{0,0}};
    long cm4[4][4] = {{0,0,0,0},
                      {0,0,0,0},
                      {0,0,0,0},
                      {0,0,0,0}};

    long routed = 0;

    // ------------------------------------------------------------
    // 4) Per-sample loop: AXI->DUT->AXI, update CMs
    // ------------------------------------------------------------
    for (int n = 0; n < NUM_SAMPLES_LOADED; ++n) {

        array180rr_t sample = dataStreamInternal.read();

        // ===== CHANGE 3: Get pre-quantized RR from reader =====
        array_rr_t rr_s1 = reader.getRRStage1(n);
        array_rr_t rr_s2 = reader.getRRStage2(n);

        // ===== CHANGE 4: Pack input with separate RR arrays =====
        hls::stream<axi_fixed_t> dmaInStream;
        input_row_to_axi(sample, rr_s1, rr_s2, dmaInStream);

        // run DUT
        hls::stream<axi_fixed_t> dmaOut2Stream;
        hls::stream<axi_fixed_t> dmaOut4Stream;
        topFunction(dmaInStream, dmaOut2Stream, dmaOut4Stream);

        // read scalar preds from AXI outputs
        ap_int8_c pred2_i8 = axi_to_pred_scalar(dmaOut2Stream); // 0/1
        ap_int8_c pred4_i8 = axi_to_pred_scalar(dmaOut4Stream); // 0..3

        int p2 = (int)pred2_i8;
        int p4 = (int)pred4_i8;

        // read ground truth
        int y_true4 = (int)gtLabelStream.read();     // 0..3
        int y_true2 = (y_true4 > 0) ? 1 : 0;         // Normal vs Abnormal

        if (p2 == 1) routed++;

        // NO "defensive bounds" — same as your native intent
        cm2[y_true2][p2] += 1;
        cm4[y_true4][p4] += 1;

#ifndef __SYNTHESIS__
        if (n < 10) {
            std::cout << "Sample " << n
                      << "  y_true4=" << y_true4
                      << "  y_true2=" << y_true2
                      << "  pred2=" << p2
                      << "  pred4=" << p4
                      << "\n";
        }
#endif
    }

    // ------------------------------------------------------------
    // 5) Print CMs (same headings/layout as native C++)
    // ------------------------------------------------------------
    print_cm2(cm2);
    print_cm4(cm4);

    const double eps = 1e-12;

    // ------------------------------------------------------------
    // 6) Stage-1 metrics (binary)
    // ------------------------------------------------------------
    std::cout << "\n=== Stage-1 Binary Metrics (Normal vs Abnormal) ===\n";

    long total2 = 0;
    long correct2 = 0;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            total2 += cm2[i][j];
            if (i == j) correct2 += cm2[i][j];
        }
    }

    for (int c = 0; c < 2; ++c) {
        long TP = cm2[c][c];
        long FP = 0, FN = 0;

        for (int i = 0; i < 2; ++i) if (i != c) FP += cm2[i][c];
        for (int j = 0; j < 2; ++j) if (j != c) FN += cm2[c][j];

        double prec = TP / (double)(TP + FP + eps);
        double rec  = TP / (double)(TP + FN + eps);
        double f1   = (2.0 * prec * rec) / (prec + rec + eps);

        std::cout << "Class " << c
                  << "  Precision=" << std::fixed << std::setprecision(4) << prec
                  << "  Recall="    << std::fixed << std::setprecision(4) << rec
                  << "  F1="        << std::fixed << std::setprecision(4) << f1
                  << "\n";
    }

    double acc2 = correct2 / (double)(total2 + eps);
    std::cout << "Stage-1 Accuracy = " << std::fixed << std::setprecision(4) << acc2 << "\n";

    // ------------------------------------------------------------
    // 7) Final metrics (4-class)
    // ------------------------------------------------------------
    std::cout << "\n=== Final 4-Class Metrics (Option-A output) ===\n";

    long total4 = 0;
    long correct4 = 0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            total4 += cm4[i][j];
            if (i == j) correct4 += cm4[i][j];
        }
    }

    for (int c = 0; c < 4; ++c) {
        long TP = cm4[c][c];
        long FP = 0, FN = 0;

        for (int i = 0; i < 4; ++i) if (i != c) FP += cm4[i][c];
        for (int j = 0; j < 4; ++j) if (j != c) FN += cm4[c][j];

        double prec = TP / (double)(TP + FP + eps);
        double rec  = TP / (double)(TP + FN + eps);
        double f1   = (2.0 * prec * rec) / (prec + rec + eps);

        std::cout << "Class " << c
                  << "  Precision=" << std::fixed << std::setprecision(4) << prec
                  << "  Recall="    << std::fixed << std::setprecision(4) << rec
                  << "  F1="        << std::fixed << std::setprecision(4) << f1
                  << "\n";
    }

    double acc4 = correct4 / (double)(total4 + eps);
    std::cout << "Overall Accuracy = " << std::fixed << std::setprecision(4) << acc4 << "\n";

    // ------------------------------------------------------------
    // 8) Routed %
    // ------------------------------------------------------------
    double routed_pct = 100.0 * (double)routed / (double)(NUM_SAMPLES_LOADED + eps);
    std::cout << "\nRouted to Stage-2 (pred2==1): "
              << routed << " / " << NUM_SAMPLES_LOADED
              << "  (" << std::fixed << std::setprecision(4) << routed_pct << "%)\n";

    return 0;
}