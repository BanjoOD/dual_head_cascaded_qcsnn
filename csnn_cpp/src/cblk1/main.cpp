// testbench_qcsnn24_optionA.cpp
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <vector>

#include "hls_stream.h"
#include "ap_int.h"

// Include individual layer headers
#include "../../include/hls4csnn1d_sd/model24/cblk_sd/conv1d_sd.h"
#include "../../include/hls4csnn1d_sd/model24/cblk_sd/batchnorm1d_sd.h"
#include "../../include/hls4csnn1d_sd/model24/cblk_sd/lif1d_integer.h"
#include "../../include/hls4csnn1d_sd/model24/cblk_sd/maxpool1d_sd.h"
#include "../../include/hls4csnn1d_sd/model24/cblk_sd/linear1d_sd.h"
#include "../../include/hls4csnn1d_sd/model24/cblk_sd/quantidentity1d_sd.h"

#include "../../include/hls4csnn1d_sd/model24/constants24_sd.h"
#include "../../include/hls4csnn1d_sd/model24/filereader24.h"

#include "../../include/hls4csnn1d_sd/model24/cblk_sd/qcsnn24_rrboth_sd.h"
#include "../../include/hls4csnn1d_sd/model24/cblk_sd/modeleval24_sd.h"


/* ================================================================
 *  Main Testbench (Two-Stage Option-A)
 * ================================================================ */

int main(int argc, char** argv) {
    std::cout << "\n=== QCSNN24 Option-A HLS Evaluation (Two-Stage, Gated, RR Bypass Fix) ===\n\n";

    // -------------------------------------------------------------------------
    // 1) Resolve dataset folder
    // -------------------------------------------------------------------------
    std::string folderPath =
        "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_intra_patient_4class_180_center90_rr_features_filtered/test";

    if (argc >= 2) {
        folderPath = argv[1];
    }

    // -------------------------------------------------------------------------
    // 2) Load data with FileReader
    // -------------------------------------------------------------------------
    FileReader reader;
    reader.loadData(folderPath);

    // Stream of full ECG records (each is array180rr_t)
    hls::stream<array180rr_t> dataStreamInternal;
    reader.streamData(dataStreamInternal);

#ifndef __SYNTHESIS__
    const int NUM_SAMPLES_LOADED = dataStreamInternal.size();
#else
    const int NUM_SAMPLES_LOADED = NUM_SAMPLES;   // macro from constants24_sd.h
#endif

    if (NUM_SAMPLES_LOADED <= 0) {
        std::cerr << "No data loaded!\n";
        return -1;
    }

    std::cout << "Loaded " << NUM_SAMPLES_LOADED << " ECG samples\n";

    // -------------------------------------------------------------------------
    // 3) Ground-truth labels (4-class: 0..3)
    // -------------------------------------------------------------------------
    hls::stream<ap_int8_c> gtLabelStream;
    reader.streamLabel(gtLabelStream, /*binary=*/false);

    // -------------------------------------------------------------------------
    // 4) Instantiate model and evaluator
    // -------------------------------------------------------------------------
    hls4csnn1d_cblk_sd::QCSNN24_RRBOTH_SD<NUM_STEPS> model24;
    ModelEvaluation evaluator24;

    // -------------------------------------------------------------------------
    // 5) Confusion matrices
    //    - Binary stage: cm2[true][pred]  (0=N,1=Abn)
    //    - Final output: cm4[true][pred]  (0..3)
    // -------------------------------------------------------------------------
    const int N2 = 2;
    const int N4 = 4;

    long cm2[N2][N2] = {{0,0},{0,0}};
    long cm4[N4][N4];

    for (int i = 0; i < N4; ++i) {
        for (int j = 0; j < N4; ++j) {
            cm4[i][j] = 0;
        }
    }

    long routed_count = 0;   // number of samples sent to stage-2
    long total_count  = 0;

    std::cout << "=== Per-sample predictions ===\n";

    // -------------------------------------------------------------------------
    // 6) Per-sample inference (Two-stage Option-A)
    // -------------------------------------------------------------------------
    for (int n = 0; n < NUM_SAMPLES_LOADED; ++n) {

        // ===== CHANGE 1: Build scalar datastream (180 signal samples only) =====
        hls::stream<ap_int8_c> datastream("datastream");
        array180rr_t sample = dataStreamInternal.read();

        // Only write signal samples (180), not RR
        for (int i = 0; i < FIXED_LENGTH1; ++i) {
            datastream.write(sample[i]);
        }

        // ===== CHANGE 2: Get pre-quantized RR from reader =====
        array_rr_t rr_s1 = reader.getRRStage1(n);
        array_rr_t rr_s2 = reader.getRRStage2(n);

        // 6.2) Outputs (1 scalar each)
        hls::stream<ap_int8_c> out2Stream("out2Stream");
        hls::stream<ap_int8_c> out4Stream("out4Stream");

        // ===== CHANGE 3: Pass RR arrays to evaluate =====
        evaluator24.evaluate(model24, datastream, rr_s1.data(), rr_s2.data(), out2Stream, out4Stream);

        // Read predictions
        ap_int8_c pred2_hw = out2Stream.read();   // 0/1
        ap_int8_c pred4_hw = out4Stream.read();   // 0..3

        int pred2 = (int)pred2_hw;
        int pred4 = (int)pred4_hw;

        // True label (0..3)
        int y_true4 = (int)gtLabelStream.read();
        if (y_true4 < 0 || y_true4 >= N4) {
            std::cerr << "Warning: label out of range (" << y_true4
                      << ") at sample " << n << "\n";
            continue;
        }

        // Binary true label
        int y_true2 = (y_true4 > 0) ? 1 : 0;

        // Update counts
        total_count++;
        if (pred2 == 1) routed_count++;

        // Update confusion matrices
        if (y_true2 >= 0 && y_true2 < 2 && pred2 >= 0 && pred2 < 2) {
            cm2[y_true2][pred2]++;
        }
        if (pred4 >= 0 && pred4 < 4) {
            cm4[y_true4][pred4]++;
        }

#ifndef __SYNTHESIS__
        if (n < 25) {
            std::cout << "Sample " << n
                      << "  y_true4=" << y_true4
                      << "  y_true2=" << y_true2
                      << "  pred2=" << pred2
                      << "  pred4=" << pred4
                      << "\n";
        }
#endif
    }

    // -------------------------------------------------------------------------
    // 7) Print confusion matrices
    // -------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // 8) Metrics helpers
    // -------------------------------------------------------------------------
    const double eps = 1e-12;

    auto print_metrics = [&](const char* title, long cm[][4], int C) {
        std::cout << "\n=== " << title << " ===\n";
        std::cout << std::fixed << std::setprecision(4);

        long total_correct = 0;
        long total_samples = 0;

        for (int i = 0; i < C; ++i) {
            for (int j = 0; j < C; ++j) {
                total_samples += cm[i][j];
                if (i == j) total_correct += cm[i][j];
            }
        }

        for (int c = 0; c < C; ++c) {
            long TP = cm[c][c];
            long FN = 0;
            long FP = 0;
            long TN = 0;

            for (int i = 0; i < C; ++i) {
                for (int j = 0; j < C; ++j) {
                    long v = cm[i][j];
                    if (i == c && j != c) FN += v;
                    else if (i != c && j == c) FP += v;
                    else if (i != c && j != c) TN += v;
                }
            }

            double precision = TP / (double)(TP + FP + eps);
            double recall    = TP / (double)(TP + FN + eps);
            double f1        = (2.0 * precision * recall) / (precision + recall + eps);

            std::cout << "Class " << c
                      << "  Precision=" << precision
                      << "  Recall=" << recall
                      << "  F1=" << f1 << "\n";
        }

        double acc = total_correct / (double)(total_samples + eps);
        std::cout << "Overall Accuracy = " << acc << "\n";
    };

    // Binary metrics (special 2-class cm)
    {
        long cm2_4x4[2][4] = {{0,0,0,0},{0,0,0,0}};
        // pack into [2][4] for reuse of helper (only first 2 columns used)
        cm2_4x4[0][0] = cm2[0][0];
        cm2_4x4[0][1] = cm2[0][1];
        cm2_4x4[1][0] = cm2[1][0];
        cm2_4x4[1][1] = cm2[1][1];

        std::cout << "\n=== Stage-1 Binary Metrics (Normal vs Abnormal) ===\n";
        std::cout << std::fixed << std::setprecision(4);

        for (int c = 0; c < 2; ++c) {
            long TP = cm2[c][c];
            long FN = cm2[c][1 - c];
            long FP = cm2[1 - c][c];
            long TN = cm2[1 - c][1 - c];

            double precision = TP / (double)(TP + FP + eps);
            double recall    = TP / (double)(TP + FN + eps);
            double f1        = (2.0 * precision * recall) / (precision + recall + eps);

            std::cout << "Class " << c
                      << "  Precision=" << precision
                      << "  Recall=" << recall
                      << "  F1=" << f1 << "\n";
        }

        long correct2 = cm2[0][0] + cm2[1][1];
        long total2   = cm2[0][0] + cm2[0][1] + cm2[1][0] + cm2[1][1];
        double acc2   = correct2 / (double)(total2 + eps);

        std::cout << "Stage-1 Accuracy = " << acc2 << "\n";
    }

    // Final 4-class metrics
    {
        long cm4_alias[4][4];
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                cm4_alias[i][j] = cm4[i][j];

        print_metrics("Final 4-Class Metrics (Option-A output)", (long (*)[4])cm4_alias, 4);
    }

    // Routed percentage
    double routed_pct = 100.0 * routed_count / (double)(total_count + eps);
    std::cout << "\nRouted to Stage-2 (pred2==1): " << routed_count
              << " / " << total_count
              << "  (" << routed_pct << "%)\n";

    std::cout << "\n=== Complete ===\n";
    return 0;
}




// // testbench_two_stage.cpp
// #include <iostream>
// #include <fstream>
// #include <string>
// #include <iomanip>
// #include <vector>
// #include "hls_stream.h"
// #include "ap_int.h"

// // Binary model (2-class) includes
// #include "../../include/hls4csnn1d_sd/model2/constants_sd.h"
// #include "../../include/hls4csnn1d_sd/model2/filereader.h"
// #include "../../include/hls4csnn1d_sd/model2/cblk_sd/nn2_cblk1_sd.h"
// #include "../../include/hls4csnn1d_sd/model2/cblk_sd/modeleval_sd.h"

// // 4-class model includes
// #include "../../include/hls4csnn1d_sd/model4/constants4_sd.h"
// #include "../../include/hls4csnn1d_sd/model4/cblk_sd/nn4_cblk1_sd.h"
// #include "../../include/hls4csnn1d_sd/model4/cblk_sd/modeleval4_sd.h"

// /* ================================================================
//  *  Main Testbench - Two-Stage Cascaded Classifier
//  * ================================================================ */

// int main(int argc, char** argv) {
//     std::cout << "\n=== TWO-STAGE CASCADED CLASSIFIER (Binary + 4-Class) ===\n\n";

//     // -------------------------------------------------------------------------
//     // 1. Load test data
//     // -------------------------------------------------------------------------
//     std::string folderPath =
//         "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_intra_patient_4class_180_center90_filtered/test";
//     if (argc >= 2) {
//         folderPath = argv[1];
//     }

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

//     std::cout << "Loaded " << NUM_SAMPLES_LOADED << " ECG samples\n";

//     // Load ground-truth labels (4-class: 0=Normal, 1=SVEB, 2=VEB, 3=F)
//     hls::stream<ap_int8_c> gtLabelStream;
//     reader.streamLabel(gtLabelStream, /*binary=*/false);

//     // -------------------------------------------------------------------------
//     // 2. Instantiate both models
//     // -------------------------------------------------------------------------
//     hls4csnn1d_cblk_sd::NeuralNetwork2_Cblk1_sd<NUM_STEPS> model2;  // Binary
//     ModelEvaluation evaluator2;

//     hls4csnn1d_cblk_sd::NeuralNetwork4_Cblk1_sd<NUM_STEPS> model4;  // 4-class
//     ModelEvaluation evaluator4;

//     // -------------------------------------------------------------------------
//     // 3. Threshold configurations
//     // -------------------------------------------------------------------------
//     // Stage 1: Binary threshold (softmax-equivalent)
//     const int BINARY_THRESHOLD_SCALED = 85;  // 0.70 threshold
//     const float BINARY_THRESHOLD = 0.70;

//     // Stage 2: 4-class logit adjustments (softmax-equivalent weighting)
//     const int LOGIT_ADJ_NORMAL = 47;   // +0.465 * 100
//     const int LOGIT_ADJ_SVEB   = -19;  // -0.185 * 100
//     const int LOGIT_ADJ_VEB    = -43;  // -0.426 * 100
//     const int LOGIT_ADJ_F      = -41;  // -0.405 * 100

//     std::cout << "Stage 1 - Binary threshold: " << BINARY_THRESHOLD << "\n";
//     std::cout << "Stage 2 - 4-class weights: Normal=0.628, SVEB=1.203, VEB=1.531, F=1.500\n\n";

//     // -------------------------------------------------------------------------
//     // 4. Confusion matrix (4-class final output)
//     // -------------------------------------------------------------------------
//     const int N_CLASSES = 4;
//     long cm[N_CLASSES][N_CLASSES];
//     for (int i = 0; i < N_CLASSES; ++i) {
//         for (int j = 0; j < N_CLASSES; ++j) {
//             cm[i][j] = 0;
//         }
//     }

//     std::cout << "=== Per-sample predictions ===\n";

//     // -------------------------------------------------------------------------
//     // 5. Two-stage inference loop
//     // -------------------------------------------------------------------------
//     for (int n = 0; n < NUM_SAMPLES_LOADED; ++n) {
//         // Flatten one ECG sample
//         hls::stream<ap_int8_c> datastream;
//         array180_t sample = dataStreamInternal.read();
//         for (int i = 0; i < FIXED_LENGTH1; ++i) {
//             datastream.write(sample[i]);
//         }

//         // ========== STAGE 1: Binary Classification ==========
//         hls::stream<ap_int8_c> outStream2;
//         evaluator2.evaluate(model2, datastream, outStream2);

//         ap_int8_c y0_binary = outStream2.read();  // Normal score
//         ap_int8_c y1_binary = outStream2.read();  // Abnormal score

//         // Apply softmax-equivalent threshold
//         ap_int<16> diff_binary = (ap_int<16>)y1_binary - (ap_int<16>)y0_binary;
//         ap_int<16> diff_binary_scaled = diff_binary * 100;

//         bool is_abnormal = (diff_binary_scaled >= BINARY_THRESHOLD_SCALED);

//         int pred_final;

//         if (!is_abnormal) {
//             // Stage 1 predicts Normal → Final prediction = 0 (Normal)
//             pred_final = 0;
//         } else {
//             // ========== STAGE 2: 4-Class Classification ==========
//             // Need to re-stream the same sample for model4
//             hls::stream<ap_int8_c> datastream4;
//             for (int i = 0; i < FIXED_LENGTH1; ++i) {
//                 datastream4.write(sample[i]);
//             }

//             hls::stream<ap_int8_c> outStream4;
//             evaluator4.evaluate(model4, datastream4, outStream4);

//             ap_int8_c y0_4class = outStream4.read();  // Normal
//             ap_int8_c y1_4class = outStream4.read();  // SVEB
//             ap_int8_c y2_4class = outStream4.read();  // VEB
//             ap_int8_c y3_4class = outStream4.read();  // F

//             // Apply logit adjustments (softmax-equivalent weighting)
//             ap_int<16> adj0 = (ap_int<16>)y0_4class * 100 + LOGIT_ADJ_NORMAL;
//             ap_int<16> adj1 = (ap_int<16>)y1_4class * 100 + LOGIT_ADJ_SVEB;
//             ap_int<16> adj2 = (ap_int<16>)y2_4class * 100 + LOGIT_ADJ_VEB;
//             ap_int<16> adj3 = (ap_int<16>)y3_4class * 100 + LOGIT_ADJ_F;

//             // Argmax over adjusted logits
//             pred_final = 0;
//             ap_int<16> best = adj0;

//             if (adj1 > best) {
//                 best = adj1;
//                 pred_final = 1;
//             }
//             if (adj2 > best) {
//                 best = adj2;
//                 pred_final = 2;
//             }
//             if (adj3 > best) {
//                 best = adj3;
//                 pred_final = 3;
//             }
//         }

//         // True label
//         int y_true = (int)gtLabelStream.read();
//         if (y_true < 0 || y_true >= N_CLASSES) {
//             std::cerr << "Warning: label out of range (" << y_true << ") at sample " << n << "\n";
//             continue;
//         }

//         cm[y_true][pred_final]++;

// #ifndef __SYNTHESIS__
//         if (n < 20) {
//             std::cout << "Sample " << n
//                       << "  y_true=" << y_true
//                       << "  stage1_abnormal=" << (is_abnormal ? "Yes" : "No")
//                       << "  pred_final=" << pred_final << "\n";
//         }
// #endif
//     }

//     // -------------------------------------------------------------------------
//     // 6. Print confusion matrix
//     // -------------------------------------------------------------------------
//     std::cout << "\n=== Confusion Matrix [true x pred] ===\n";
//     std::cout << "      pred:  0      1      2      3\n";
//     for (int i = 0; i < N_CLASSES; ++i) {
//         std::cout << "true " << i << ": ";
//         for (int j = 0; j < N_CLASSES; ++j) {
//             std::cout << std::setw(6) << cm[i][j] << " ";
//         }
//         std::cout << "\n";
//     }

//     // -------------------------------------------------------------------------
//     // 7. Calculate metrics
//     // -------------------------------------------------------------------------
//     const double eps = 1e-12;
//     long total_correct = 0;
//     long total_samples = 0;

//     for (int i = 0; i < N_CLASSES; ++i) {
//         for (int j = 0; j < N_CLASSES; ++j) {
//             total_samples += cm[i][j];
//             if (i == j) {
//                 total_correct += cm[i][j];
//             }
//         }
//     }

//     static const char* CLASS_NAMES[N_CLASSES] = {
//         "Normal (N)",
//         "SVEB (S)",
//         "VEB (V)",
//         "F (F)"
//     };

//     std::cout << std::fixed << std::setprecision(4);
//     std::cout << "\n=== Per-class metrics (Two-Stage Cascaded) ===\n";

//     for (int c = 0; c < N_CLASSES; ++c) {
//         long TP = cm[c][c];
//         long FN = 0;
//         long FP = 0;
//         long TN = 0;

//         for (int i = 0; i < N_CLASSES; ++i) {
//             for (int j = 0; j < N_CLASSES; ++j) {
//                 long val = cm[i][j];
//                 if (i == c && j != c) {
//                     FN += val;
//                 } else if (i != c && j == c) {
//                     FP += val;
//                 } else if (i != c && j != c) {
//                     TN += val;
//                 }
//             }
//         }

//         double precision = TP / (double)(TP + FP + eps);
//         double recall    = TP / (double)(TP + FN + eps);
//         double f1        = (2.0 * precision * recall) / (precision + recall + eps);
//         double accuracy  = (TP + TN) / (double)(TP + TN + FP + FN + eps);

//         std::cout << "\nClass " << c << " - " << CLASS_NAMES[c] << "\n"
//                   << "  Accuracy : " << accuracy  << "\n"
//                   << "  Precision: " << precision << "\n"
//                   << "  Recall   : " << recall    << "\n"
//                   << "  F1       : " << f1        << "\n";
//     }

//     double acc_global = total_correct / (double)(total_samples + eps);
//     std::cout << "\n=== Overall multi-class accuracy ===\n"
//               << "Accuracy : " << acc_global << "\n";

//     std::cout << "\n=== Expected Python Performance ===\n"
//               << "Two-Stage Accuracy: 94.79%\n"
//               << "Normal Prec/Rec:    97.96% / 96.64%\n"
//               << "SVEB Prec/Rec:      81.23% / 59.07%\n"
//               << "VEB Prec/Rec:       69.49% / 88.44%\n"
//               << "F Prec/Rec:         59.43% / 65.41%\n";

//     std::cout << "\n=== Complete ===\n";
//     return 0;
// }





// // testbench_dump_layers.cpp
// #include <iostream>
// #include <fstream>
// #include <string>
// #include <iomanip>
// #include <vector>
// #include "hls_stream.h"
// #include "ap_int.h"

// #include "../../include/hls4csnn1d_sd/model4/constants4_sd.h"
// #include "../../include/hls4csnn1d_sd/model4/filereader4.h"

// // Include individual layer headers
// #include "../../include/hls4csnn1d_sd/model4/cblk_sd/conv1d_sd.h"
// #include "../../include/hls4csnn1d_sd/model4/cblk_sd/batchnorm1d_sd.h"
// #include "../../include/hls4csnn1d_sd/model4/cblk_sd/lif1d_integer.h"
// #include "../../include/hls4csnn1d_sd/model4/cblk_sd/maxpool1d_sd.h"
// #include "../../include/hls4csnn1d_sd/model4/cblk_sd/linear1d_sd.h"
// #include "../../include/hls4csnn1d_sd/model4/cblk_sd/nn4_cblk1_sd.h"
// #include "../../include/hls4csnn1d_sd/model4/cblk_sd/modeleval4_sd.h"
// #include "../../include/hls4csnn1d_sd/model4/cblk_sd/quantidentity1d_sd.h"



// /* ================================================================
//  *  Main Testbench
//  * ================================================================ */

//  int main(int argc, char** argv) {
//     std::cout << "\n=== QCSNN HLS Evaluation (4-Class, ARGMAX) ===\n\n";

//     // -------------------------------------------------------------------------
//     // 1. Resolve dataset folder
//     // -------------------------------------------------------------------------
// #ifndef __SYNTHESIS__
//     std::string folderPath =
//         "/..../data/mitbih_processed_intra_patient_4class_180_center90_filtered/test";
// #else
//     std::string folderPath =
//         "/..../data/mitbih_processed_test/smallvhls";
// #endif

//     if (argc >= 2) {
//         folderPath = argv[1];
//     }

//     // -------------------------------------------------------------------------
//     // 2. Load data with FileReader
//     // -------------------------------------------------------------------------
//     FileReader reader;
//     reader.loadData(folderPath);

//     // Stream of full 1×180 ECG records
//     hls::stream<array180_t> dataStreamInternal;
//     reader.streamData(dataStreamInternal);

// #ifndef __SYNTHESIS__
//     const int NUM_SAMPLES_LOADED = dataStreamInternal.size();
// #else
//     const int NUM_SAMPLES_LOADED = NUM_SAMPLES;   // macro from constants_sd.h
// #endif

//     if (NUM_SAMPLES_LOADED <= 0) {
//         std::cerr << "No data loaded!\n";
//         return -1;
//     }

//     std::cout << "Loaded " << NUM_SAMPLES_LOADED << " ECG samples\n";

//     // -------------------------------------------------------------------------
//     // 3. Ground-truth labels: 4-class (0..3) for Normal, SVEB, VEB, F
//     // -------------------------------------------------------------------------
//     hls::stream<ap_int8_c> gtLabelStream;
//     reader.streamLabel(gtLabelStream, /*binary=*/false);

//     // -------------------------------------------------------------------------
//     // 4. Instantiate model and evaluation helper
//     // -------------------------------------------------------------------------
//     hls4csnn1d_cblk_sd::QCSNN24_SD<NUM_STEPS> model24;
//     ModelEvaluation evaluator24;

//     // -------------------------------------------------------------------------
//     // 5. Multi-class confusion matrix
//     //    cm[true][pred] for 4 classes: 0=Normal,1=SVEB,2=VEB,3=F
//     // -------------------------------------------------------------------------
//     const int N_CLASSES = 4;
//     long cm[N_CLASSES][N_CLASSES];
//     for (int i = 0; i < N_CLASSES; ++i) {
//         for (int j = 0; j < N_CLASSES; ++j) {
//             cm[i][j] = 0;
//         }
//     }

//     std::cout << "=== Per-sample predictions ===\n";

//     // -------------------------------------------------------------------------
//     // 6. Per-sample inference + confusion matrix update (NO WEIGHTS)
//     // -------------------------------------------------------------------------
//     for (int n = 0; n < NUM_SAMPLES_LOADED; ++n) {
//         // Flatten one ECG sample [array180_t → scalar stream]
//         hls::stream<ap_int8_c> datastream;
//         array180_t sample = dataStreamInternal.read();
//         for (int i = 0; i < FIXED_LENGTH1; ++i) {
//             datastream.write(sample[i]);
//         }

//         // Process through model (multi-class QCSNN)
//         hls::stream<ap_int8_c> out2Stream;
//         hls::stream<ap_int8_c> out4Stream;
//         evaluator24.evaluate(model24, datastream, out2Stream, out4Stream);

//         // Read 4 outputs from the model (e.g., averaged spike counts per class)
//         ap_int8_c y0 = outStream.read();   // class 0: Normal
//         ap_int8_c y1 = outStream.read();   // class 1: SVEB
//         ap_int8_c y2 = outStream.read();   // class 2: VEB
//         ap_int8_c y3 = outStream.read();   // class 3: F

//         // -------- Plain ARGMAX over raw model outputs (NO adjustments) --------
//         int pred_class = 0;
//         ap_int8_c best = y0;

//         if (y1 > best) { best = y1; pred_class = 1; }
//         if (y2 > best) { best = y2; pred_class = 2; }
//         if (y3 > best) { best = y3; pred_class = 3; }
//         // ---------------------------------------------------------------------

//         // True label (0..3)
//         int y_true = (int)gtLabelStream.read();
//         if (y_true < 0 || y_true >= N_CLASSES) {
//             std::cerr << "Warning: label out of range (" << y_true << ") at sample " << n << "\n";
//             continue;
//         }

//         cm[y_true][pred_class]++;

// #ifndef __SYNTHESIS__
//         if (n < 20) {
//             std::cout << "Sample " << n
//                       << "  y_true=" << y_true
//                       << "  pred=" << pred_class
//                       << "  raw=[" << (int)y0 << "," << (int)y1
//                       << "," << (int)y2 << "," << (int)y3 << "]\n";
//         }
// #endif
//     }

//     // -------------------------------------------------------------------------
//     // 7. Print confusion matrix
//     // -------------------------------------------------------------------------
//     std::cout << "\n=== Confusion Matrix [true x pred] ===\n";
//     std::cout << "      pred:  0      1      2      3\n";
//     for (int i = 0; i < N_CLASSES; ++i) {
//         std::cout << "true " << i << ": ";
//         for (int j = 0; j < N_CLASSES; ++j) {
//             std::cout << std::setw(6) << cm[i][j] << " ";
//         }
//         std::cout << "\n";
//     }

//     // -------------------------------------------------------------------------
//     // 8. Per-class metrics + overall accuracy
//     // -------------------------------------------------------------------------
//     const double eps = 1e-12;
//     long total_correct = 0;
//     long total_samples = 0;

//     for (int i = 0; i < N_CLASSES; ++i) {
//         for (int j = 0; j < N_CLASSES; ++j) {
//             total_samples += cm[i][j];
//             if (i == j) total_correct += cm[i][j];
//         }
//     }

//     static const char* CLASS_NAMES[N_CLASSES] = {
//         "Normal (N)",
//         "SVEB (S)",
//         "VEB (V)",
//         "F (F)"
//     };

//     std::cout << std::fixed << std::setprecision(4);
//     std::cout << "\n=== Per-class metrics (ARGMAX) ===\n";

//     for (int c = 0; c < N_CLASSES; ++c) {
//         long TP = cm[c][c];
//         long FN = 0;
//         long FP = 0;
//         long TN = 0;

//         for (int i = 0; i < N_CLASSES; ++i) {
//             for (int j = 0; j < N_CLASSES; ++j) {
//                 long val = cm[i][j];
//                 if (i == c && j != c) FN += val;
//                 else if (i != c && j == c) FP += val;
//                 else if (i != c && j != c) TN += val;
//             }
//         }

//         double precision = TP / (double)(TP + FP + eps);
//         double recall    = TP / (double)(TP + FN + eps);
//         double f1        = (2.0 * precision * recall) / (precision + recall + eps);
//         double accuracy  = (TP + TN) / (double)(TP + TN + FP + FN + eps);

//         std::cout << "\nClass " << c << " - " << CLASS_NAMES[c] << "\n"
//                   << "  Accuracy : " << accuracy  << "\n"
//                   << "  Precision: " << precision << "\n"
//                   << "  Recall   : " << recall    << "\n"
//                   << "  F1       : " << f1        << "\n";
//     }

//     double acc_global = total_correct / (double)(total_samples + eps);
//     std::cout << "\n=== Overall multi-class accuracy ===\n"
//               << "Accuracy : " << acc_global << "\n";

//     std::cout << "\n=== Expected Python Performance (weighted softmax) ===\n"
//               << "Accuracy:       0.9838 (98.38%)\n"
//               << "Normal Recall:  0.9893 (98.93%)\n"
//               << "SVEB Recall:    0.8887 (88.87%)\n"
//               << "VEB Recall:     0.9634 (96.34%)\n"
//               << "F Recall:       0.8742 (87.42%)\n";

//     std::cout << "\n=== Complete ===\n";
//     return 0;
// }


// int main(int argc, char** argv) {
//     std::cout << "\n=== QCSNN HLS Evaluation (4-Class + Softmax-Equivalent Weighting) ===\n\n";

//     // -------------------------------------------------------------------------
//     // 1. Resolve dataset folder
//     // -------------------------------------------------------------------------
// #ifndef __SYNTHESIS__
//     std::string folderPath =
//         "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_intra_patient_4class_180_center90_filtered/test";
// #else
//     std::string folderPath =
//         "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_test/smallvhls";
// #endif
//     if (argc >= 2) {
//         folderPath = argv[1];
//     }

//     // -------------------------------------------------------------------------
//     // 2. Load data with FileReader
//     // -------------------------------------------------------------------------
//     FileReader reader;
//     reader.loadData(folderPath);

//     // Stream of full 1×180 ECG records
//     hls::stream<array180_t> dataStreamInternal;
//     reader.streamData(dataStreamInternal);

// #ifndef __SYNTHESIS__
//     const int NUM_SAMPLES_LOADED = dataStreamInternal.size();
// #else
//     const int NUM_SAMPLES_LOADED = NUM_SAMPLES;   // macro from constants_sd.h
// #endif

//     if (NUM_SAMPLES_LOADED <= 0) {
//         std::cerr << "No data loaded!\n";
//         return -1;
//     }

//     std::cout << "Loaded " << NUM_SAMPLES_LOADED << " ECG samples\n";

//     // -------------------------------------------------------------------------
//     // 3. Ground-truth labels: 4-class (0..3) for Normal, SVEB, VEB, F
//     // -------------------------------------------------------------------------
//     hls::stream<ap_int8_c> gtLabelStream;
//     reader.streamLabel(gtLabelStream, /*binary=*/false);

//     // -------------------------------------------------------------------------
//     // 4. Instantiate model and evaluation helper
//     // -------------------------------------------------------------------------
//     hls4csnn1d_cblk_sd::NeuralNetwork4_Cblk1_sd<NUM_STEPS> model4;
//     ModelEvaluation evaluator4;

//     // -------------------------------------------------------------------------
//     // 5. Multi-class confusion matrix
//     //    cm[true][pred] for 4 classes: 0=Normal,1=SVEB,2=VEB,3=F
//     // -------------------------------------------------------------------------
//     const int N_CLASSES = 4;
//     long cm[N_CLASSES][N_CLASSES];
//     for (int i = 0; i < N_CLASSES; ++i) {
//         for (int j = 0; j < N_CLASSES; ++j) {
//             cm[i][j] = 0;
//         }
//     }

//     // -------------------------------------------------------------------------
//     // 6. SOFTMAX-EQUIVALENT LOGIT ADJUSTMENTS
//     //    Python weights: Normal=0.628, SVEB=1.203, VEB=1.531, F=1.500
//     //    Logit adjustment = -log(weight) scaled by 100
//     //    Normal: -log(0.628) = +0.465 → +47
//     //    SVEB:   -log(1.203) = -0.185 → -19
//     //    VEB:    -log(1.531) = -0.426 → -43
//     //    F:      -log(1.500) = -0.405 → -41
//     // -------------------------------------------------------------------------
//     const int LOGIT_ADJ_NORMAL = 47;   // Boost Normal (lower weight)
//     const int LOGIT_ADJ_SVEB   = -19;  // Suppress SVEB (higher weight)
//     const int LOGIT_ADJ_VEB    = -43;  // Suppress VEB (higher weight)
//     const int LOGIT_ADJ_F      = -41;  // Suppress F (higher weight)

//     std::cout << "Deployed weights (softmax-equivalent via logit adjustment):\n";
//     std::cout << "  Normal: 0.628 (adj = +" << LOGIT_ADJ_NORMAL << ")\n";
//     std::cout << "  SVEB:   1.203 (adj = " << LOGIT_ADJ_SVEB << ")\n";
//     std::cout << "  VEB:    1.531 (adj = " << LOGIT_ADJ_VEB << ")\n";
//     std::cout << "  F:      1.500 (adj = " << LOGIT_ADJ_F << ")\n\n";

//     std::cout << "=== Per-sample predictions ===\n";

//     // -------------------------------------------------------------------------
//     // 7. Per-sample inference + confusion matrix update
//     // -------------------------------------------------------------------------
//     for (int n = 0; n < NUM_SAMPLES_LOADED; ++n) {
//         // Flatten one ECG sample [array180_t → scalar stream]
//         hls::stream<ap_int8_c> datastream;
//         array180_t sample = dataStreamInternal.read();
//         for (int i = 0; i < FIXED_LENGTH1; ++i) {
//             datastream.write(sample[i]);
//         }

//         // Process through model (multi-class QCSNN)
//         hls::stream<ap_int8_c> outStream;
//         evaluator4.evaluate(model4, datastream, outStream);

//         // Read 4 outputs from the model (averaged spike counts per class)
//         ap_int8_c y0 = outStream.read();   // class 0: Normal
//         ap_int8_c y1 = outStream.read();   // class 1: SVEB
//         ap_int8_c y2 = outStream.read();   // class 2: VEB
//         ap_int8_c y3 = outStream.read();   // class 3: F

//         // ========== SOFTMAX-EQUIVALENT WEIGHTING VIA LOGIT ADJUSTMENT ==========
//         // Adjusted logit = raw_logit * 100 + adjustment
//         ap_int<16> adj0 = (ap_int<16>)y0 * 100 + LOGIT_ADJ_NORMAL;
//         ap_int<16> adj1 = (ap_int<16>)y1 * 100 + LOGIT_ADJ_SVEB;
//         ap_int<16> adj2 = (ap_int<16>)y2 * 100 + LOGIT_ADJ_VEB;
//         ap_int<16> adj3 = (ap_int<16>)y3 * 100 + LOGIT_ADJ_F;

//         // Argmax over adjusted logits
//         int pred_class = 0;
//         ap_int<16> best = adj0;

//         if (adj1 > best) {
//             best = adj1;
//             pred_class = 1;
//         }
//         if (adj2 > best) {
//             best = adj2;
//             pred_class = 2;
//         }
//         if (adj3 > best) {
//             best = adj3;
//             pred_class = 3;
//         }
//         // =======================================================================

//         // True label (0..3)
//         int y_true = (int)gtLabelStream.read();
//         if (y_true < 0 || y_true >= N_CLASSES) {
//             std::cerr << "Warning: label out of range (" << y_true << ") at sample " << n << "\n";
//             continue;
//         }

//         cm[y_true][pred_class]++;

// #ifndef __SYNTHESIS__
//         if (n < 20) {
//             std::cout << "Sample " << n
//                       << "  y_true=" << y_true
//                       << "  pred=" << pred_class
//                       << "  raw=[" << (int)y0 << "," << (int)y1 
//                       << "," << (int)y2 << "," << (int)y3
//                       << "]  adj=[" << (int)adj0 << "," << (int)adj1
//                       << "," << (int)adj2 << "," << (int)adj3 << "]\n";
//         }
// #endif
//     }

//     // -------------------------------------------------------------------------
//     // 8. Print confusion matrix
//     // -------------------------------------------------------------------------
//     std::cout << "\n=== Confusion Matrix [true x pred] ===\n";
//     std::cout << "      pred:  0      1      2      3\n";
//     for (int i = 0; i < N_CLASSES; ++i) {
//         std::cout << "true " << i << ": ";
//         for (int j = 0; j < N_CLASSES; ++j) {
//             std::cout << std::setw(6) << cm[i][j] << " ";
//         }
//         std::cout << "\n";
//     }

//     // -------------------------------------------------------------------------
//     // 9. Per-class metrics + overall accuracy
//     // -------------------------------------------------------------------------
//     const double eps = 1e-12;
//     long total_correct = 0;
//     long total_samples = 0;

//     for (int i = 0; i < N_CLASSES; ++i) {
//         for (int j = 0; j < N_CLASSES; ++j) {
//             total_samples += cm[i][j];
//             if (i == j) {
//                 total_correct += cm[i][j];
//             }
//         }
//     }

//     static const char* CLASS_NAMES[N_CLASSES] = {
//         "Normal (N)",
//         "SVEB (S)",
//         "VEB (V)",
//         "F (F)"
//     };

//     std::cout << std::fixed << std::setprecision(4);
//     std::cout << "\n=== Per-class metrics (softmax-equivalent weighting) ===\n";

//     for (int c = 0; c < N_CLASSES; ++c) {
//         long TP = cm[c][c];
//         long FN = 0;
//         long FP = 0;
//         long TN = 0;

//         for (int i = 0; i < N_CLASSES; ++i) {
//             for (int j = 0; j < N_CLASSES; ++j) {
//                 long val = cm[i][j];
//                 if (i == c && j != c) {
//                     FN += val;
//                 } else if (i != c && j == c) {
//                     FP += val;
//                 } else if (i != c && j != c) {
//                     TN += val;
//                 }
//             }
//         }

//         double precision = TP / (double)(TP + FP + eps);
//         double recall    = TP / (double)(TP + FN + eps);
//         double f1        = (2.0 * precision * recall) / (precision + recall + eps);
//         double accuracy  = (TP + TN) / (double)(TP + TN + FP + FN + eps);

//         std::cout << "\nClass " << c << " - " << CLASS_NAMES[c] << "\n"
//                   << "  Accuracy : " << accuracy  << "\n"
//                   << "  Precision: " << precision << "\n"
//                   << "  Recall   : " << recall    << "\n"
//                   << "  F1       : " << f1        << "\n";
//     }

//     double acc_global = total_correct / (double)(total_samples + eps);
//     std::cout << "\n=== Overall multi-class accuracy ===\n"
//               << "Accuracy : " << acc_global << "\n";

//     std::cout << "\n=== Expected Python Performance (weighted softmax) ===\n"
//               << "Accuracy:       0.9838 (98.38%)\n"
//               << "Normal Recall:  0.9893 (98.93%)\n"
//               << "SVEB Recall:    0.8887 (88.87%)\n"
//               << "VEB Recall:     0.9634 (96.34%)\n"
//               << "F Recall:       0.8742 (87.42%)\n";

//     std::cout << "\n=== Complete ===\n";
//     return 0;
// }

// // testbench_dump_layers.cpp for argmax only
// #include <iostream>
// #include <fstream>
// #include <string>
// #include <iomanip>
// #include <vector>
// #include "hls_stream.h"
// #include "ap_int.h"

// #include "../../include/hls4csnn1d_sd/model4/constants4_sd.h"
// #include "../../include/hls4csnn1d_sd/model4/filereader4.h"

// // Include individual layer headers
// #include "../../include/hls4csnn1d_sd/model4/cblk_sd/conv1d_sd.h"
// #include "../../include/hls4csnn1d_sd/model4/cblk_sd/batchnorm1d_sd.h"
// #include "../../include/hls4csnn1d_sd/model4/cblk_sd/lif1d_integer.h"
// #include "../../include/hls4csnn1d_sd/model4/cblk_sd/maxpool1d_sd.h"
// #include "../../include/hls4csnn1d_sd/model4/cblk_sd/linear1d_sd.h"
// #include "../../include/hls4csnn1d_sd/model4/cblk_sd/nn4_cblk1_sd.h"
// #include "../../include/hls4csnn1d_sd/model4/cblk_sd/modeleval4_sd.h"
// #include "../../include/hls4csnn1d_sd/model4/cblk_sd/quantidentity1d_sd.h"



// /* ================================================================
//  *  Main Testbench
//  * ================================================================ */

// int main(int argc, char** argv) {
//     std::cout << "\n=== QCSNN HLS Evaluation (NeuralNetwork4_Cblk1_sd + Multi-Class Metrics) ===\n\n";

//     // -------------------------------------------------------------------------
//     // 1. Resolve dataset folder
//     // -------------------------------------------------------------------------
// #ifndef __SYNTHESIS__
//     std::string folderPath =
//         "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_intra_patient_4class_180_center90_filtered/test";
        
// #else
//     std::string folderPath =
//         "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_test/smallvhls";
// #endif
//     if (argc >= 2) {
//         folderPath = argv[1];
//     }

//     // -------------------------------------------------------------------------
//     // 2. Load data with FileReader
//     // -------------------------------------------------------------------------
//     FileReader reader;
//     reader.loadData(folderPath);

//     // Stream of full 1×180 ECG records
//     hls::stream<array180_t> dataStreamInternal;
//     reader.streamData(dataStreamInternal);

// #ifndef __SYNTHESIS__
//     const int NUM_SAMPLES_LOADED = dataStreamInternal.size();
// #else
//     const int NUM_SAMPLES_LOADED = NUM_SAMPLES;   // macro from constants_sd.h
// #endif

//     if (NUM_SAMPLES_LOADED <= 0) {
//         std::cerr << "No data loaded!\n";
//         return -1;
//     }

//     std::cout << "Loaded " << NUM_SAMPLES_LOADED << " ECG samples\n";

//     // -------------------------------------------------------------------------
//     // 3. Ground-truth labels: 4-class (0..3) for Normal, SVEB, VEB, F
//     // -------------------------------------------------------------------------
//     hls::stream<ap_int8_c> gtLabelStream;
//     reader.streamLabel(gtLabelStream, /*binary=*/false);

//     // -------------------------------------------------------------------------
//     // 4. Instantiate model and evaluation helper
//     // -------------------------------------------------------------------------
//     hls4csnn1d_cblk_sd::NeuralNetwork4_Cblk1_sd<NUM_STEPS> model4;
//     ModelEvaluation evaluator4;

//     // -------------------------------------------------------------------------
//     // 5. Multi-class confusion matrix
//     //    cm[true][pred] for 4 classes: 0=Normal,1=SVEB,2=VEB,3=F
//     // -------------------------------------------------------------------------
//     const int N_CLASSES = 4;
//     long cm[N_CLASSES][N_CLASSES];
//     for (int i = 0; i < N_CLASSES; ++i) {
//         for (int j = 0; j < N_CLASSES; ++j) {
//             cm[i][j] = 0;
//         }
//     }

//     // -------------------------------------------------------------------------
//     // 6. Probability re-weighting multipliers (DEPLOYED WEIGHTS)
//     //    Python weights: Normal=0.628, SVEB=1.203, VEB=1.531, F=1.500
//     //    Integer multipliers = 10000 / weight
//     // -------------------------------------------------------------------------
//     const ap_int<16> MULT_NORMAL = 15924;  // 10000 / 0.628
//     const ap_int<16> MULT_SVEB   = 8313;   // 10000 / 1.203
//     const ap_int<16> MULT_VEB    = 6533;   // 10000 / 1.531
//     const ap_int<16> MULT_F      = 6667;   // 10000 / 1.500

//     std::cout << "\n=== Per-sample predictions (index + true + pred) ===\n";

//     // -------------------------------------------------------------------------
//     // 7. Per-sample inference + confusion matrix update
//     // -------------------------------------------------------------------------
//     for (int n = 0; n < NUM_SAMPLES_LOADED; ++n) {
//         // Flatten one ECG sample [array180_t → scalar stream]
//         hls::stream<ap_int8_c> datastream;
//         array180_t sample = dataStreamInternal.read();
//         for (int i = 0; i < FIXED_LENGTH1; ++i) {
//             datastream.write(sample[i]);
//         }

//         // Process through model (multi-class QCSNN)
//         hls::stream<ap_int8_c> outStream;
//         evaluator4.evaluate(model4, datastream, outStream);

//         // Read 4 outputs from the model (averaged spike counts per class)
//         ap_int8_c y0 = outStream.read();   // class 0: Normal
//         ap_int8_c y1 = outStream.read();   // class 1: SVEB
//         ap_int8_c y2 = outStream.read();   // class 2: VEB
//         ap_int8_c y3 = outStream.read();   // class 3: F

//         // Apply probability re-weighting (integer-only)
//         // weighted[i] = y[i] * multiplier[i]
//         // Use ap_int<24> to handle max value: 127 * 15924 = 2,022,348
//         ap_int<24> weighted0 = (ap_int<24>)y0 * (ap_int<24>)MULT_NORMAL;
//         ap_int<24> weighted1 = (ap_int<24>)y1 * (ap_int<24>)MULT_SVEB;
//         ap_int<24> weighted2 = (ap_int<24>)y2 * (ap_int<24>)MULT_VEB;
//         ap_int<24> weighted3 = (ap_int<24>)y3 * (ap_int<24>)MULT_F;

//         // Argmax over weighted scores
//         int pred_class = 0;
//         ap_int<24> best = weighted0;

//         if (weighted1 > best) {
//             best = weighted1;
//             pred_class = 1;
//         }
//         if (weighted2 > best) {
//             best = weighted2;
//             pred_class = 2;
//         }
//         if (weighted3 > best) {
//             best = weighted3;
//             pred_class = 3;
//         }

//         // True label (0..3)
//         int y_true = (int)gtLabelStream.read();
//         if (y_true < 0 || y_true >= N_CLASSES) {
//             std::cerr << "Warning: label out of range (" << y_true << ") at sample " << n << "\n";
//             continue;
//         }

//         cm[y_true][pred_class]++;

// #ifndef __SYNTHESIS__
//         if (n < 20) {  // Print first 20 samples
//             std::cout << "Sample " << n
//                     << "  y_true=" << y_true
//                     << "  pred=" << pred_class
//                     << "  raw=[" << (int)y0
//                     << ", " << (int)y1
//                     << ", " << (int)y2
//                     << ", " << (int)y3
//                     << "]  weighted=[" << (int)weighted0
//                     << ", " << (int)weighted1
//                     << ", " << (int)weighted2
//                     << ", " << (int)weighted3 << "]\n";
//         }
// #endif
//     }

//     // -------------------------------------------------------------------------
//     // 8. Print confusion matrix
//     // -------------------------------------------------------------------------
//     std::cout << "\n=== Confusion Matrix [true x pred] ===\n";
//     std::cout << "      pred:  0      1      2      3\n";
//     for (int i = 0; i < N_CLASSES; ++i) {
//         std::cout << "true " << i << ": ";
//         for (int j = 0; j < N_CLASSES; ++j) {
//             std::cout << std::setw(6) << cm[i][j] << " ";
//         }
//         std::cout << "\n";
//     }

//     // -------------------------------------------------------------------------
//     // 9. Per-class metrics + overall accuracy
//     // -------------------------------------------------------------------------
//     const double eps = 1e-12;
//     long total_correct = 0;
//     long total_samples = 0;

//     for (int i = 0; i < N_CLASSES; ++i) {
//         for (int j = 0; j < N_CLASSES; ++j) {
//             total_samples += cm[i][j];
//             if (i == j) {
//                 total_correct += cm[i][j];
//             }
//         }
//     }

//     static const char* CLASS_NAMES[N_CLASSES] = {
//         "Normal (N)",
//         "SVEB (S)",
//         "VEB (V)",
//         "F (F)"
//     };

//     std::cout << std::fixed << std::setprecision(4);
//     std::cout << "\n=== Per-class metrics (multi-class with probability re-weighting) ===\n";

//     for (int c = 0; c < N_CLASSES; ++c) {
//         long TP = cm[c][c];
//         long FN = 0;
//         long FP = 0;
//         long TN = 0;

//         for (int i = 0; i < N_CLASSES; ++i) {
//             for (int j = 0; j < N_CLASSES; ++j) {
//                 long val = cm[i][j];
//                 if (i == c && j != c) {
//                     FN += val;
//                 } else if (i != c && j == c) {
//                     FP += val;
//                 } else if (i != c && j != c) {
//                     TN += val;
//                 }
//             }
//         }

//         double precision = TP / (double)(TP + FP + eps);
//         double recall    = TP / (double)(TP + FN + eps);
//         double f1        = (2.0 * precision * recall) / (precision + recall + eps);
//         double accuracy  = (TP + TN) / (double)(TP + TN + FP + FN + eps);

//         std::cout << "\nClass " << c << " - " << CLASS_NAMES[c] << "\n"
//                   << "  Accuracy : " << accuracy  << "\n"
//                   << "  Precision: " << precision << "\n"
//                   << "  Recall   : " << recall    << "\n"
//                   << "  F1       : " << f1        << "\n";
//     }

//     double acc_global = total_correct / (double)(total_samples + eps);
//     std::cout << "\n=== Overall multi-class accuracy ===\n"
//               << "Accuracy : " << acc_global << "\n";

//     std::cout << "\n=== Complete ===\n";
//     return 0;
// }



// // testbench_dump_layers.cpp for 4-class layers
// #include <iostream>
// #include <fstream>
// #include <string>
// #include <iomanip>
// #include <vector>
// #include "hls_stream.h"
// #include "ap_int.h"

// #include "../../include/hls4csnn1d_sd/model4/constants4_sd.h"
// #include "../../include/hls4csnn1d_sd/model4/filereader4.h"

// // Include individual layer headers
// #include "../../include/hls4csnn1d_sd/model4/cblk_sd/conv1d_sd.h"
// #include "../../include/hls4csnn1d_sd/model4/cblk_sd/batchnorm1d_sd.h"
// #include "../../include/hls4csnn1d_sd/model4/cblk_sd/lif1d_integer.h"
// // #include "../../include/hls4csnn1d_sd/model4/cblk_sd/lif1d_float.h"
// #include "../../include/hls4csnn1d_sd/model4/cblk_sd/maxpool1d_sd.h"
// #include "../../include/hls4csnn1d_sd/model4/cblk_sd/linear1d_sd.h"
// #include "../../include/hls4csnn1d_sd/model4/cblk_sd/nn4_cblk1_sd.h"
// #include "../../include/hls4csnn1d_sd/model4/cblk_sd/modeleval4_sd.h"
// // Add include
// #include "../../include/hls4csnn1d_sd/model4/cblk_sd/quantidentity1d_sd.h"



// /* ================================================================
//  *  Main Testbench
//  * ================================================================ */

// int main(int argc, char** argv) {
//     std::cout << "\n=== QCSNN HLS Evaluation (NeuralNetwork4_Cblk1_sd + Multi-Class Metrics) ===\n\n";

//     // -------------------------------------------------------------------------
//     // 1. Resolve dataset folder
//     // -------------------------------------------------------------------------
// #ifndef __SYNTHESIS__
//     std::string folderPath =
//         "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_train/experiment";
// #else
//     std::string folderPath =
//         "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_test/smallvhls";
// #endif
//     if (argc >= 2) {
//         folderPath = argv[1];
//     }

//     // -------------------------------------------------------------------------
//     // 2. Load data with FileReader
//     // -------------------------------------------------------------------------
//     FileReader reader;
//     reader.loadData(folderPath);

//     // Stream of full 1×180 ECG records
//     hls::stream<array180_t> dataStreamInternal;
//     reader.streamData(dataStreamInternal);

// #ifndef __SYNTHESIS__
//     const int NUM_SAMPLES_LOADED = dataStreamInternal.size();
// #else
//     const int NUM_SAMPLES_LOADED = NUM_SAMPLES;   // macro from constants_sd.h
// #endif

//     if (NUM_SAMPLES_LOADED <= 0) {
//         std::cerr << "No data loaded!\n";
//         return -1;
//     }

//     std::cout << "Loaded " << NUM_SAMPLES_LOADED << " ECG samples\n";

//     // -------------------------------------------------------------------------
//     // 3. Ground-truth labels: 4-class (0..3) for Normal, SVEB, VEB, F
//     // -------------------------------------------------------------------------
//     hls::stream<ap_int8_c> gtLabelStream;
//     // NOTE: multi-class labels, not binary
//     reader.streamLabel(gtLabelStream, /*binary=*/false);

//     // -------------------------------------------------------------------------
//     // 4. Instantiate model and evaluation helper
//     // -------------------------------------------------------------------------
//     hls4csnn1d_cblk_sd::NeuralNetwork4_Cblk1_sd<NUM_STEPS> model4;
//     ModelEvaluation evaluator4;

//     // -------------------------------------------------------------------------
//     // 5. Multi-class confusion matrix
//     //    cm[true][pred] for 4 classes: 0=Normal,1=SVEB,2=VEB,3=F
//     // -------------------------------------------------------------------------
//     const int N_CLASSES = 4;
//     long cm[N_CLASSES][N_CLASSES];
//     for (int i = 0; i < N_CLASSES; ++i) {
//         for (int j = 0; j < N_CLASSES; ++j) {
//             cm[i][j] = 0;
//         }
//     }

//     std::cout << "\n=== Per-sample predictions (index + true + pred) ===\n";

//     // -------------------------------------------------------------------------
//     // 6. Per-sample inference + confusion matrix update
//     // -------------------------------------------------------------------------
//     for (int n = 0; n < NUM_SAMPLES_LOADED; ++n) {
//         // Flatten one ECG sample [array180_t → scalar stream]
//         hls::stream<ap_int8_c> datastream;
//         array180_t sample = dataStreamInternal.read();
//         for (int i = 0; i < FIXED_LENGTH1; ++i) {
//             datastream.write(sample[i]);
//         }

//         // Process through model (multi-class QCSNN)
//         hls::stream<ap_int8_c> outStream;
//         evaluator4.evaluate(model4, datastream, outStream);

//         // Read 4 outputs from the model (scores / spike counts per class)
//         ap_int8_c y0 = outStream.read();   // class 0: Normal
//         ap_int8_c y1 = outStream.read();   // class 1: SVEB
//         ap_int8_c y2 = outStream.read();   // class 2: VEB
//         ap_int8_c y3 = outStream.read();   // class 3: F

//         // Argmax over all 4 classes → predicted class in {0,1,2,3}
//         ap_int8_c y[4] = { y0, y1, y2, y3 };
//         int pred_class = 0;
//         ap_int8_c best = y[0];
//         for (int c = 1; c < N_CLASSES; ++c) {
//             if ((int)y[c] > (int)best) {
//                 best = y[c];
//                 pred_class = c;
//             }
//         }

//         // True label (0..3)
//         int y_true = (int)gtLabelStream.read();
//         if (y_true < 0 || y_true >= N_CLASSES) {
//             std::cerr << "Warning: label out of range (" << y_true << ") at sample " << n << "\n";
//             continue;
//         }

//         cm[y_true][pred_class]++;

// #ifndef __SYNTHESIS__
//         std::cout << "Sample " << n
//                   << "  y_true=" << y_true
//                   << "  pred=" << pred_class
//                   << "  [y0=" << (int)y0
//                   << ", y1=" << (int)y1
//                   << ", y2=" << (int)y2
//                   << ", y3=" << (int)y3 << "]\n";
// #endif
//     }

//     // -------------------------------------------------------------------------
//     // 7. Print confusion matrix
//     // -------------------------------------------------------------------------
//     std::cout << "\n=== Confusion Matrix [true x pred] ===\n";
//     std::cout << "      pred:  0      1      2      3\n";
//     for (int i = 0; i < N_CLASSES; ++i) {
//         std::cout << "true " << i << ": ";
//         for (int j = 0; j < N_CLASSES; ++j) {
//             std::cout << std::setw(6) << cm[i][j] << " ";
//         }
//         std::cout << "\n";
//     }

//     // -------------------------------------------------------------------------
//     // 8. Per-class metrics + overall accuracy
//     // -------------------------------------------------------------------------
//     const double eps = 1e-12;
//     long total_correct = 0;
//     long total_samples = 0;

//     for (int i = 0; i < N_CLASSES; ++i) {
//         for (int j = 0; j < N_CLASSES; ++j) {
//             total_samples += cm[i][j];
//             if (i == j) {
//                 total_correct += cm[i][j];
//             }
//         }
//     }

//     static const char* CLASS_NAMES[N_CLASSES] = {
//         "Normal (N)",
//         "SVEB (S)",
//         "VEB (V)",
//         "F (F)"
//     };

//     std::cout << std::fixed << std::setprecision(4);
//     std::cout << "\n=== Per-class metrics (multi-class) ===\n";

//     for (int c = 0; c < N_CLASSES; ++c) {
//         long TP = cm[c][c];
//         long FN = 0;
//         long FP = 0;
//         long TN = 0;

//         for (int i = 0; i < N_CLASSES; ++i) {
//             for (int j = 0; j < N_CLASSES; ++j) {
//                 long val = cm[i][j];
//                 if (i == c && j != c) {
//                     FN += val;  // true c, predicted other
//                 } else if (i != c && j == c) {
//                     FP += val;  // true other, predicted c
//                 } else if (i != c && j != c) {
//                     TN += val;  // all others
//                 }
//                 // i==c && j==c is TP (already accounted)
//             }
//         }

//         double precision = TP / (double)(TP + FP + eps);
//         double recall    = TP / (double)(TP + FN + eps);
//         double f1        = (2.0 * precision * recall) / (precision + recall + eps);
//         double accuracy  = (TP + TN) / (double)(TP + TN + FP + FN + eps);

//         std::cout << "\nClass " << c << " - " << CLASS_NAMES[c] << "\n"
//                   << "  Accuracy : " << accuracy  << "\n"
//                   << "  Precision: " << precision << "\n"
//                   << "  Recall   : " << recall    << "\n"
//                   << "  F1       : " << f1        << "\n";
//     }

//     double acc_global = total_correct / (double)(total_samples + eps);
//     std::cout << "\n=== Overall multi-class accuracy ===\n"
//               << "Accuracy : " << acc_global << "\n";

//     std::cout << "\n=== Complete ===\n";
//     return 0;
// }

// ##############################################################################################   /////

// // testbench_dump_layers.cpp
// #include <iostream>
// #include <fstream>
// #include <string>
// #include <iomanip>
// #include <vector>
// #include "hls_stream.h"
// #include "ap_int.h"

// #include "../../include/hls4csnn1d_sd/model2/constants_sd.h"
// #include "../../include/hls4csnn1d_sd/model2/filereader.h"
// #include "../../include/hls4csnn1d_sd/model2/cblk_sd/nn2_cblk1_sd.h"

// // Include individual layer headers
// #include "../../include/hls4csnn1d_sd/model2/cblk_sd/conv1d_sd.h"
// #include "../../include/hls4csnn1d_sd/model2/cblk_sd/batchnorm1d_sd.h"
// #include "../../include/hls4csnn1d_sd/model2/cblk_sd/lif1d_integer.h"
// #include "../../include/hls4csnn1d_sd/model2/cblk_sd/maxpool1d_sd.h"
// #include "../../include/hls4csnn1d_sd/model2/cblk_sd/linear1d_sd.h"
// #include "../../include/hls4csnn1d_sd/model2/cblk_sd/nn2_cblk1_sd.h"
// #include "../../include/hls4csnn1d_sd/model2/cblk_sd/modeleval_sd.h"
// #include "../../include/hls4csnn1d_sd/model2/cblk_sd/quantidentity1d_sd.h"


// /* ================================================================
//  *  Main Testbench
//  * ================================================================ */
// int main(int argc, char** argv) {
//     std::cout << "\n=== QCSNN HLS Evaluation (NeuralNetwork2_Cblk1_sd + Softmax Threshold) ===\n\n";

//     // -------------------------------------------------------------------------
//     // 1. Resolve dataset folder
//     // -------------------------------------------------------------------------
//     std::string folderPath =
//         // "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_intra_patient_4class_180_center90_filtered/test";
//         "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_intra_patient_4class_180_center90_filtered/test";
//     if (argc >= 2) {
//         folderPath = argv[1];
//     }

//     // -------------------------------------------------------------------------
//     // 2. Load data with FileReader
//     // -------------------------------------------------------------------------
//     FileReader reader;
//     reader.loadData(folderPath);

//     // Stream of full 1×180 records
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

//     std::cout << "Loaded " << NUM_SAMPLES_LOADED << " ECG samples\n";

//     // Ground-truth labels (binary: 0 normal, 1 abnormal)
//     hls::stream<ap_int8_c> gtLabelStream;
//     reader.streamLabel(gtLabelStream, /*binary=*/true);

//     // -------------------------------------------------------------------------
//     // 3. Instantiate model and evaluator
//     // -------------------------------------------------------------------------
//     hls4csnn1d_cblk_sd::NeuralNetwork2_Cblk1_sd<NUM_STEPS> model2;
//     ModelEvaluation evaluator;

//     // -------------------------------------------------------------------------
//     // 4. SOFTMAX-EQUIVALENT THRESHOLD = 0.60
//     //    softmax(out)[1] >= 0.70  ⟺  y1 - y0 >= log(0.60/0.40) = log(...) ≈ 0.405
//     //    Integer scaled by 100: (y1 - y0) * 100 >= 41
//     // -------------------------------------------------------------------------
//     const int LOGIT_THRESHOLD_SCALED = 40; // 0.405 * 100 ≈ 40 or 41
//     const float SOFTMAX_THRESHOLD = 60;

//     std::cout << "Deployed softmax threshold: " << SOFTMAX_THRESHOLD << " (Fold 3)\n";
//     std::cout << "Logit threshold (scaled): " << LOGIT_THRESHOLD_SCALED << "/100 = 0.40\n";

//     // -------------------------------------------------------------------------
//     // 5. Compute per-class metrics
//     // -------------------------------------------------------------------------
//     long TP0 = 0, TN0 = 0, FP0 = 0, FN0 = 0;
//     long TP1 = 0, TN1 = 0, FP1 = 0, FN1 = 0;

//     std::cout << "\n=== Per-sample predictions ===\n";

//     for (int n = 0; n < NUM_SAMPLES_LOADED; ++n) {
//         // Flatten one sample
//         hls::stream<ap_int8_c> datastream;
//         array180_t sample = dataStreamInternal.read();
//         for (int i = 0; i < FIXED_LENGTH1; ++i) {
//             datastream.write(sample[i]);
//         }
        
//         // Process through model
//         hls::stream<ap_int8_c> outStream;
//         evaluator.evaluate(model2, datastream, outStream);
        
//         // Get predictions (averaged spike counts as int8)
//         ap_int8_c y0 = outStream.read();   // Normal score
//         ap_int8_c y1 = outStream.read();   // Abnormal score

//         // ============ SOFTMAX-EQUIVALENT THRESHOLD ============
//         // Check: (y1 - y0) * 100 >= 40 or 41
//         ap_int<16> diff = (ap_int<16>)y1 - (ap_int<16>)y0;
//         ap_int<16> diff_scaled = diff * 100;
        
//         int pred_class;
//         if (diff_scaled >= LOGIT_THRESHOLD_SCALED) {
//             pred_class = 1;  // Abnormal
//         } else {
//             pred_class = 0;  // Normal
//         }
//         // ======================================================

//         int y_true = (int)gtLabelStream.read();

//         // Update confusion matrices
//         if (pred_class == 1 && y_true == 1) { ++TP1; ++TN0; }
//         else if (pred_class == 0 && y_true == 0) { ++TN1; ++TP0; }
//         else if (pred_class == 1 && y_true == 0) { ++FP1; ++FN0; }
//         else if (pred_class == 0 && y_true == 1) { ++FN1; ++FP0; }

// #ifndef __SYNTHESIS__
//         if (n < 20) {  // Print first 20 samples
//             std::cout << "Sample " << n 
//                       << "  y_true=" << y_true
//                       << "  y0=" << (int)y0 
//                       << "  y1=" << (int)y1
//                       << "  diff=" << (int)diff
//                       << "  pred=" << pred_class << "\n";
//         }
// #endif
//     }

//     // -------------------------------------------------------------------------
//     // 6. Final metrics
//     // -------------------------------------------------------------------------
//     const double eps = 1e-12;
//     const double TP1d = double(TP1), TN1d = double(TN1), FP1d = double(FP1), FN1d = double(FN1);
//     const double TP0d = double(TP0), TN0d = double(TN0), FP0d = double(FP0), FN0d = double(FN0);

//     // Abnormal class (1) as positive
//     const double precision1 = TP1d / (TP1d + FP1d + eps);
//     const double recall1    = TP1d / (TP1d + FN1d + eps);
//     const double f11        = (2.0 * precision1 * recall1) / (precision1 + recall1 + eps);
//     const double accuracy1  = (TP1d + TN1d) / (TP1d + TN1d + FP1d + FN1d + eps);

//     // Normal class (0) as positive
//     const double precision0 = TP0d / (TP0d + FP0d + eps);
//     const double recall0    = TP0d / (TP0d + FN0d + eps);
//     const double f10        = (2.0 * precision0 * recall0) / (precision0 + recall0 + eps);
//     const double accuracy0  = (TP0d + TN0d) / (TP0d + TN0d + FP0d + FN0d + eps);

//     // Global accuracy
//     const double correct = TP1d + TN1d;
//     const double total   = TP1d + TN1d + FP1d + FN1d + eps;
//     const double acc_global = correct / total;

//     std::cout << std::fixed << std::setprecision(4)
//               << "\n=== Metrics for Normal class (label 0) ===\n"
//               << "Accuracy : " << accuracy0  << "\n"
//               << "Precision: " << precision0 << "\n"
//               << "Recall   : " << recall0    << "\n"
//               << "F1       : " << f10        << "\n\n";

//     std::cout << std::fixed << std::setprecision(4)
//               << "=== Metrics for Abnormal class (label 1) ===\n"
//               << "Accuracy : " << accuracy1  << "\n"
//               << "Precision: " << precision1 << "\n"
//               << "Recall   : " << recall1    << "\n"
//               << "F1       : " << f11        << "\n\n";

//     std::cout << std::fixed << std::setprecision(4)
//               << "=== Overall binary accuracy ===\n"
//               << "Accuracy : " << acc_global << "\n";

//     std::cout << "\n=== Expected Abnormal Performance (Fold 3, threshold=0.60) ===\n"
//               << "Precision: ~0.7470\n"
//               << "Recall:    ~0.9113\n"
//               << "F1:        ~0.8210\n";

//     std::cout << "\n=== Complete ===\n";
//     return 0;
// }


// // testbench_dump_layers.cpp
// #include <iostream>
// #include <fstream>
// #include <string>
// #include <iomanip>
// #include <vector>
// #include "hls_stream.h"
// #include "ap_int.h"

// #include "../../include/hls4csnn1d_sd/model2/constants_sd.h"
// #include "../../include/hls4csnn1d_sd/model2/filereader.h"
// #include "../../include/hls4csnn1d_sd/model2/cblk_sd/nn2_cblk1_sd.h"

// // Include individual layer headers
// #include "../../include/hls4csnn1d_sd/model2/cblk_sd/conv1d_sd.h"
// #include "../../include/hls4csnn1d_sd/model2/cblk_sd/batchnorm1d_sd.h"
// #include "../../include/hls4csnn1d_sd/model2/cblk_sd/lif1d_integer.h"
// // #include "../../include/hls4csnn1d_sd/model2/cblk_sd/lif1d_float.h"
// #include "../../include/hls4csnn1d_sd/model2/cblk_sd/maxpool1d_sd.h"
// #include "../../include/hls4csnn1d_sd/model2/cblk_sd/linear1d_sd.h"
// #include "../../include/hls4csnn1d_sd/model2/cblk_sd/nn2_cblk1_sd.h"
// #include "../../include/hls4csnn1d_sd/model2/cblk_sd/modeleval_sd.h"
// // Add include
// #include "../../include/hls4csnn1d_sd/model2/cblk_sd/quantidentity1d_sd.h"


// /* ================================================================
//  *  Main Testbench
//  * ================================================================ */
// int main(int argc, char** argv) {
//     std::cout << "\n=== QCSNN HLS Evaluation (NeuralNetwork2_Cblk1_sd + ModelEvaluation) ===\n\n";

//     // -------------------------------------------------------------------------
//     // 1. Resolve dataset folder
//     // -------------------------------------------------------------------------
//     std::string folderPath =
//         "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_intra_patient_4class_180_center90_filtered/test";
//     if (argc >= 2) {
//         folderPath = argv[1];
//     }

//     // -------------------------------------------------------------------------
//     // 2. Load data with FileReader (same pattern as before)
//     // -------------------------------------------------------------------------
//     FileReader reader;
//     reader.loadData(folderPath);

//     // Stream of full 1×180 records (as in your previous main)
//     hls::stream<array180_t> dataStreamInternal;
//     reader.streamData(dataStreamInternal);

// #ifndef __SYNTHESIS__
//     const int NUM_SAMPLES_LOADED = dataStreamInternal.size();
// #else
//     const int NUM_SAMPLES_LOADED = NUM_SAMPLES;   // macro from constants_sd.h
// #endif

//     if (NUM_SAMPLES_LOADED <= 0) {
//         std::cerr << "No data loaded!\n";
//         return -1;
//     }

//     std::cout << "Loaded " << NUM_SAMPLES_LOADED << " ECG samples\n";

//     // Ground-truth labels (binary: 0 normal, 1 abnormal)
//     hls::stream<ap_int8_c> gtLabelStream;
//     reader.streamLabel(gtLabelStream, /*binary=*/true);

//     // -------------------------------------------------------------------------
//     // 3. Flatten array180_t stream → scalar ap_int8_c stream for ModelEvaluation
//     // -------------------------------------------------------------------------
//     hls4csnn1d_cblk_sd::NeuralNetwork2_Cblk1_sd<NUM_STEPS> model2;
//     ModelEvaluation evaluator;

//     // -------------------------------------------------------------------------
//     // 4. Threshold configuration (MODIFIED)
//     // -------------------------------------------------------------------------
//     const int THRESHOLD_NUM = 32;   // P(abnormal) >= 0.32
//     const int THRESHOLD_DEN = 68;   // 1.0 - 0.32 = 0.68
//     const float THRESHOLD_VALUE = (float)THRESHOLD_NUM / (THRESHOLD_NUM + THRESHOLD_DEN);

//     // -------------------------------------------------------------------------
//     // 5. Compute per-class metrics from outputs vs. ground truth
//     // -------------------------------------------------------------------------
//     long TP0 = 0, TN0 = 0, FP0 = 0, FN0 = 0;
//     long TP1 = 0, TN1 = 0, FP1 = 0, FN1 = 0;

//     std::cout << "\n=== Per-sample predictions (threshold=" << THRESHOLD_VALUE << ") ===\n";

//     for (int n = 0; n < NUM_SAMPLES_LOADED; ++n) {
//         // Flatten one sample
//         hls::stream<ap_int8_c> datastream;
//         array180_t sample = dataStreamInternal.read();
//         for (int i = 0; i < FIXED_LENGTH1; ++i) {
//             datastream.write(sample[i]);
//         }
        
//         // Process through model
//         hls::stream<ap_int8_c> outStream;
//         evaluator.evaluate(model2, datastream, outStream);
        
//         // Get predictions (averaged spike counts as int8)
//         ap_int8_c y0 = outStream.read();   // class 0 score
//         ap_int8_c y1 = outStream.read();   // class 1 score

//         // ============ MODIFIED: Apply threshold = 0.32 ============
//         // Original: int pred_class = ( (int)y1 > (int)y0 ) ? 1 : 0;
//         // New: y1/(y0+y1) >= 0.32  →  y1 * 68 >= y0 * 32
//         int pred_class;
//         if ((int)y1 * THRESHOLD_DEN >= (int)y0 * THRESHOLD_NUM) {
//             pred_class = 1;  // abnormal
//         } else {
//             pred_class = 0;  // normal
//         }
//         // ==========================================================

//         int y_true = (int)gtLabelStream.read();

//         // Update confusion matrices
//         if (pred_class == 1 && y_true == 1) { ++TP1; ++TN0; }
//         else if (pred_class == 0 && y_true == 0) { ++TN1; ++TP0; }
//         else if (pred_class == 1 && y_true == 0) { ++FP1; ++FN0; }
//         else if (pred_class == 0 && y_true == 1) { ++FN1; ++FP0; }
//     }

//     // -------------------------------------------------------------------------
//     // 6. Final metrics (normal vs abnormal), same definitions as before
//     // -------------------------------------------------------------------------
//     const double eps = 1e-12;
//     const double TP1d = double(TP1), TN1d = double(TN1), FP1d = double(FP1), FN1d = double(FN1);
//     const double TP0d = double(TP0), TN0d = double(TN0), FP0d = double(FP0), FN0d = double(FN0);

//     // Abnormal class (1) as positive
//     const double precision1 = TP1d / (TP1d + FP1d + eps);
//     const double recall1    = TP1d / (TP1d + FN1d + eps);
//     const double f11        = (2.0 * precision1 * recall1) / (precision1 + recall1 + eps);
//     const double accuracy1  = (TP1d + TN1d) / (TP1d + TN1d + FP1d + FN1d + eps);

//     // Normal class (0) as positive
//     const double precision0 = TP0d / (TP0d + FP0d + eps);
//     const double recall0    = TP0d / (TP0d + FN0d + eps);
//     const double f10        = (2.0 * precision0 * recall0) / (precision0 + recall0 + eps);
//     const double accuracy0  = (TP0d + TN0d) / (TP0d + TN0d + FP0d + FN0d + eps);

//     // Global accuracy (identical to accuracy0/accuracy1 in binary case)
//     const double correct = TP1d + TN1d;
//     const double total   = TP1d + TN1d + FP1d + FN1d + eps;
//     const double acc_global = correct / total;

//     std::cout << std::fixed << std::setprecision(4)
//               << "\n=== Metrics for normal class (label 0) ===\n"
//               << "Accuracy : " << accuracy0  << "\n"
//               << "Precision: " << precision0 << "\n"
//               << "Recall   : " << recall0    << "\n"
//               << "F1       : " << f10        << "\n\n";

//     std::cout << std::fixed << std::setprecision(4)
//               << "=== Metrics for abnormal class (label 1) ===\n"
//               << "Accuracy : " << accuracy1  << "\n"
//               << "Precision: " << precision1 << "\n"
//               << "Recall   : " << recall1    << "\n"
//               << "F1       : " << f11        << "\n\n";

//     std::cout << std::fixed << std::setprecision(4)
//               << "=== Overall binary accuracy ===\n"
//               << "Accuracy : " << acc_global << "\n";

//     std::cout << "\n=== Complete ===\n";
//     return 0;
// }

