#include <iostream>
#include <string>
#include <iomanip>

#include "hls_stream.h"
#include "../../include/hls4csnn1d_sd/constants_sd.h"       // FIXED_LENGTH1 == 180, FIXED_LENGTH6 == 2, NUM_SAMPLES, types
#include "../../include/hls4csnn1d_sd/cblk_sd/modeleval_sd.h"      // ModelEvaluation (uses NUM_SAMPLES internally)
#include "../../include/hls4csnn1d_sd/filereader.h"      // FileReader: loadData, streamData, streamLabel
#include "../../include/hls4csnn1d_sd/cblk_sd/nn2_cblk1_sd.h"        // NeuralNetwork2_Cblk1_sd<NUM_STEPS>


/* reshape array180_t rows → scalar ap_int8_c stream (for evaluator I/O) */
static void streamArrayRowsToScalars(hls::stream<array180_t>& in_rows,
                                     int num_rows_needed,
                                     hls::stream<ap_int8_c>& out_scalars) {
    for (int n = 0; n < num_rows_needed; ++n) {
        array180_t row = in_rows.read();
        for (int i = 0; i < FIXED_LENGTH1; ++i) {
#pragma HLS PIPELINE II=1
            out_scalars.write(row[i]);
        }
    }
}

/* pack 2 scalar logits/sample → array2_t (for convenient printing & metrics) */
static void streamScalarsToPairs(hls::stream<ap_int8_c>& in_scalars,
                                 int num_rows,
                                 hls::stream<array2_t>& out_pairs) {
    for (int n = 0; n < num_rows; ++n) {
        array2_t p;
        p[0] = in_scalars.read();
        p[1] = in_scalars.read();
        out_pairs.write(p);
    }
}

/* metrics struct (includes abstained for Policy E) */
struct Metrics {
    long long TP=0, TN=0, FP=0, FN=0, abstained=0;
    double accuracy=0, precision=0, recall=0, specificity=0, f1=0;
};

/* compute metrics for a given tie policy */
static Metrics compute_metrics_policy(const array2_t* logits,
                                      const ap_int8_c* labels,
                                      int N,
                                      int policy_id,
                                      int margin /*used by policy D*/) {
    Metrics m;
    int toggle = 0; // for alternating ties (policy C)

    for (int i = 0; i < N; ++i) {
        int l0 = int(logits[i][0]);
        int l1 = int(logits[i][1]);
        int y  = (labels[i] == 0) ? 0 : 1;

        int pred = 0;
        switch (policy_id) {
            case 0: // A) strict '>' (ties -> 0)
                pred = (l1 > l0) ? 1 : 0;
                break;

            case 1: // B) '>=' (ties -> 1)
                pred = (l1 >= l0) ? 1 : 0;
                break;

            case 2: // C) alternate ties 0,1,0,1,...
                if      (l1 > l0) pred = 1;
                else if (l1 < l0) pred = 0;
                else { pred = (toggle & 1); toggle ^= 1; }
                break;

            case 3: { // D) margin rule: |l1-l0| <= margin -> 0
                int d = l1 - l0;
                if      (d >  margin) pred = 1;
                else if (d < -margin) pred = 0;
                else                  pred = 0;
                break;
            }

            case 4: // E) ties ABSTAIN (exact equality)
            default:
                if      (l1 > l0) pred = 1;
                else if (l1 < l0) pred = 0;
                else { ++m.abstained; continue; }  // skip confusion-matrix update
        }

        if (y == 1 && pred == 1) ++m.TP;
        else if (y == 0 && pred == 0) ++m.TN;
        else if (y == 0 && pred == 1) ++m.FP;
        else if (y == 1 && pred == 0) ++m.FN;
    }

    // Denominator EXCLUDES abstained samples (only policy E uses it > 0)
    const double denom = double(m.TP + m.TN + m.FP + m.FN);
    m.accuracy    = (denom > 0.0) ? double(m.TP + m.TN) / denom : 0.0;
    m.precision   = (m.TP + m.FP) ? double(m.TP) / double(m.TP + m.FP) : 0.0;
    m.recall      = (m.TP + m.FN) ? double(m.TP) / double(m.TP + m.FN) : 0.0; // sensitivity
    m.specificity = (m.TN + m.FP) ? double(m.TN) / double(m.TN + m.FP) : 0.0;
    m.f1          = (m.precision + m.recall) ? (2.0 * m.precision * m.recall) / (m.precision + m.recall) : 0.0;
    return m;
}

static void print_metrics(const char* title, const Metrics& m) {
    std::cout << "\n=== " << title << " ===\n";
    std::cout << "Confusion Matrix: "
              << "TP=" << m.TP << "  FP=" << m.FP
              << "  FN=" << m.FN << "  TN=" << m.TN << "\n";
    if (m.abstained) std::cout << "Abstained : " << m.abstained << "\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Accuracy   : " << m.accuracy    << "\n";
    std::cout << "Precision  : " << m.precision   << "\n";
    std::cout << "Recall     : " << m.recall      << "\n";
    std::cout << "Specificity: " << m.specificity << "\n";
    std::cout << "F1-score   : " << m.f1          << "\n";
}

int main(int argc, char** argv) {
    // 1) Load ECG rows
    std::string folderPath =
        "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_singles/large";
    if (argc >= 2) folderPath = argv[1];

    FileReader reader;
    reader.loadData(folderPath);

    // 2) Stream rows (array180_t per sample)
    hls::stream<array180_t> dataStreamInternal;
    reader.streamData(dataStreamInternal);
    std::cout << "Finished stream data.\n";

#ifndef __SYNTHESIS__
    const int NUM_SAMPLES_LOADED = dataStreamInternal.size(); // CSIM only
#else
    const int NUM_SAMPLES_LOADED = NUM_SAMPLES;               // synthesis fallback
#endif

    if (NUM_SAMPLES_LOADED <= 0) {
        std::cerr << "No rows loaded — aborting test.\n";
        return -1;
    }
    if (NUM_SAMPLES_LOADED < NUM_SAMPLES) {
        std::cerr << "[ERROR] rows (" << NUM_SAMPLES_LOADED
                  << ") < NUM_SAMPLES (" << NUM_SAMPLES << ")\n";
        return -1;
    }
    std::cout << "Loaded " << NUM_SAMPLES_LOADED << " ECG rows\n";

    // 3) array180_t → scalar stream for evaluator
    hls::stream<ap_int8_c> dataScalarStream;
    streamArrayRowsToScalars(dataStreamInternal, NUM_SAMPLES, dataScalarStream);

    // 4) Run DUT → 2 scalars/sample
    hls4csnn1d_cblk_sd::NeuralNetwork2_Cblk1_sd<NUM_STEPS> model2;
    hls::stream<ap_int8_c> outScalarStream;
    ModelEvaluation modelEval;
    modelEval.evaluate(model2, dataScalarStream, outScalarStream);

    // 5) Repack predictions for convenience
    hls::stream<array2_t> predPairs;
    streamScalarsToPairs(outScalarStream, NUM_SAMPLES, predPairs);

    // 6) True labels (binary)
    hls::stream<ap_int8_c> trueLabelStream;
    reader.streamLabel(trueLabelStream, /*binary=*/true);

    // 7) Store logits+labels into arrays and print BEFORE comparing
    static array2_t  logits_buf[NUM_SAMPLES];
    static ap_int8_c labels_buf[NUM_SAMPLES];

    std::cout << "\n--- Per-record outputs BEFORE comparison ---\n";
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        array2_t l = predPairs.read();
        ap_int8_c y = trueLabelStream.read();
        logits_buf[i] = l;
        labels_buf[i] = y;

        std::cout << "Row " << i
                  << " : logits=[" << int(l[0]) << ", " << int(l[1]) << "]"
                  << "  label=" << int(y) << "\n";
    }

    // 8) Evaluate five tie-break policies
    const int MARGIN = 1; // ap_int8_c units for Policy D

    Metrics mA = compute_metrics_policy(logits_buf, labels_buf, NUM_SAMPLES, 0, MARGIN); // '>'  (ties->0)
    Metrics mB = compute_metrics_policy(logits_buf, labels_buf, NUM_SAMPLES, 1, MARGIN); // '>=' (ties->1)
    Metrics mC = compute_metrics_policy(logits_buf, labels_buf, NUM_SAMPLES, 2, MARGIN); // alternate ties
    Metrics mD = compute_metrics_policy(logits_buf, labels_buf, NUM_SAMPLES, 3, MARGIN); // margin rule
    Metrics mE = compute_metrics_policy(logits_buf, labels_buf, NUM_SAMPLES, 4, MARGIN); // abstain on ties

    // 9) Report
    print_metrics("Policy A: strict '>' (ties -> 0)",           mA);
    print_metrics("Policy B: '>=' (ties -> 1)",                  mB);
    print_metrics("Policy C: alternate ties 0,1,...",            mC);
    print_metrics("Policy D: margin (|l1-l0|<=1 -> 0)",          mD);
    print_metrics("Policy E: ties ABSTAIN (excluded)",           mE);

    std::cout << "\nProcessed " << NUM_SAMPLES << " rows.\n";
    return 0;
}




// #include <iostream>
// #include <string>
// #include <iomanip>

// #include "hls_stream.h"
// #include "../../include/hls4csnn1d_sd/constants_sd.h"       // FIXED_LENGTH1 == 180, FIXED_LENGTH6 == 2, NUM_SAMPLES, types
// #include "../../include/hls4csnn1d_sd/cblk_sd/modeleval_sd.h"      // ModelEvaluation (uses NUM_SAMPLES internally)
// #include "../../include/hls4csnn1d_sd/filereader.h"      // FileReader: loadData, streamData, streamLabel
// #include "../../include/hls4csnn1d_sd/cblk_sd/nn2_cblk1_sd.h"        // NeuralNetwork2_Cblk1_sd<NUM_STEPS>

// // Convert array180_t rows → scalar ap_int8_c stream (no change to user structs; just reshaping for evaluator)
// static void streamArrayRowsToScalars(hls::stream<array180_t>& in_rows,
//                                      int num_rows_needed,
//                                      hls::stream<ap_int8_c>& out_scalars) {
//     for (int n = 0; n < num_rows_needed; ++n) {
// #pragma HLS LOOP_TRIPCOUNT min=1 max=1048576
//         array180_t row = in_rows.read();
//         for (int i = 0; i < FIXED_LENGTH1; ++i) {  // 180
// #pragma HLS PIPELINE II=1
//             out_scalars.write(row[i]);
//         }
//     }
// }

// // Re-pack 2 scalar logits/sample → array2_t (for convenient postproc)
// static void streamScalarsToPairs(hls::stream<ap_int8_c>& in_scalars,
//                                  int num_rows,
//                                  hls::stream<array2_t>& out_pairs) {
//     for (int n = 0; n < num_rows; ++n) {
// #pragma HLS LOOP_TRIPCOUNT min=1 max=1048576
//         array2_t p;
//         p[0] = in_scalars.read();
//         p[1] = in_scalars.read();
//         out_pairs.write(p);
//     }
// }

// // //-----------------------------------------------------------------
// // //  MAIN — keeps original FileReader usage; no AXI anywhere
// // //-----------------------------------------------------------------
// int main(int argc, char** argv) {
//     // 1) Load ECG rows from folder
//     std::string folderPath =
//         "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_singles/large";
//     if (argc >= 2) folderPath = argv[1];

//     FileReader reader;
//     reader.loadData(folderPath);

//     // 2) Pull rows into an internal stream using original API (array180_t per sample)
//     hls::stream<array180_t> dataStreamInternal;
//     reader.streamData(dataStreamInternal);   // original method, unchanged
//     std::cout << "Finished stream data.\n";

//     const int NUM_SAMPLES_LOADED = dataStreamInternal.size(); // CSIM-only; in HLS this isn’t synthesizable
 

//     if (NUM_SAMPLES_LOADED <= 0) {
//         std::cerr << "No rows loaded — aborting test.\n";
//         return -1;
//     }
//     std::cout << "Loaded " << NUM_SAMPLES_LOADED << " ECG rows\n";

//     // IMPORTANT: ModelEvaluation uses compile-time NUM_SAMPLES internally.
//     // Ensure we have at least that many rows to avoid underflow.
//     if (NUM_SAMPLES_LOADED < NUM_SAMPLES) {
//         std::cerr << "[ERROR] Dataset rows (" << NUM_SAMPLES_LOADED
//                   << ") < configured NUM_SAMPLES (" << NUM_SAMPLES
//                   << "). Increase data or lower NUM_SAMPLES in constants_sd.h.\n";
//         return -1;
//     }

//     // 3) Reshape array180_t rows → scalar stream for evaluator (no AXI)
//     hls::stream<ap_int8_c> dataScalarStream;   // 180 scalars per sample
//     streamArrayRowsToScalars(dataStreamInternal, /*num_rows_needed=*/NUM_SAMPLES, dataScalarStream);

//     // 4) Run the DUT via ModelEvaluation (produces 2 scalars/sample)
//     hls4csnn1d_cblk_sd::NeuralNetwork2_Cblk1_sd<NUM_STEPS> model2;
//     hls::stream<ap_int8_c> outScalarStream; // 2 logits per sample

//     ModelEvaluation modelEval;
//     modelEval.evaluate(model2, dataScalarStream, outScalarStream);

//     // 5) Convert scalar logits back to array2_t for convenience
//     hls::stream<array2_t> predPairs;
//     streamScalarsToPairs(outScalarStream, /*num_rows=*/NUM_SAMPLES, predPairs);

//     // 6) True labels (binary) from FileReader
//     hls::stream<ap_int8_c> trueLabelStream;
//     reader.streamLabel(trueLabelStream, /*binary=*/true);


//     if (trueLabelStream.size() < NUM_SAMPLES) {
//         std::cerr << "[ERROR] Fewer labels than NUM_SAMPLES.\n";
//         return -1;
//     }


//     // 7) Metrics
//     long long TP = 0, TN = 0, FP = 0, FN = 0;
//     int actual_0_count = 0, actual_1_count = 0;
//     int pred_0_count   = 0, pred_1_count   = 0;

//     for (int i = 0; i < NUM_SAMPLES; ++i) {
//         array2_t logits = predPairs.read();
//         ap_int8_c y_bin = trueLabelStream.read();  // already binary from streamLabel(..., true)

//          // *** PRINT FIRST, as requested ***
//         std::cout << "Row " << i
//                   << " : logits=[" << int(logits[0]) << ", " << int(logits[1]) << "]"
//                   << "  label=" << int(y_bin) << "\n";

//         // Predicted class by comparing the two head outputs
//         int predicted = (int(logits[1]) > int(logits[0])) ? 1 : 0;

//         // Actual (binary already), keep the explicit mapping if you prefer symmetry
//         int actual = (y_bin == 0) ? 0 : 1;

//         (actual == 0) ? ++actual_0_count : ++actual_1_count;
//         (predicted == 0) ? ++pred_0_count : ++pred_1_count;

//         if (actual == 1 && predicted == 1) ++TP;
//         else if (actual == 0 && predicted == 0) ++TN;
//         else if (actual == 0 && predicted == 1) ++FP;
//         else if (actual == 1 && predicted == 0) ++FN;
//     }

//     // 8) Compute and print
//     const double denom = double(TP + TN + FP + FN);
//     double accuracy    = denom > 0.0 ? double(TP + TN) / denom : 0.0;
//     double precision   = (TP + FP) > 0 ? double(TP) / double(TP + FP) : 0.0;  // PPV
//     double recall      = (TP + FN) > 0 ? double(TP) / double(TP + FN) : 0.0;  // Sensitivity/TPR
//     double specificity = (TN + FP) > 0 ? double(TN) / double(TN + FP) : 0.0;  // Optional
//     double f1 = (precision + recall) > 0 ? 2.0 * precision * recall / (precision + recall) : 0.0;

//     std::cout << "\n=== Classification Metrics (positive=1) ===\n";
//     std::cout << "Confusion Matrix:\n";
//     std::cout << "  TP: " << TP << "  FP: " << FP << "\n";
//     std::cout << "  FN: " << FN << "  TN: " << TN << "\n";
//     std::cout << std::fixed << std::setprecision(4);
//     std::cout << "Accuracy   : " << accuracy    << "\n";
//     std::cout << "Precision  : " << precision   << "\n";
//     std::cout << "Recall     : " << recall      << "\n";
//     std::cout << "Specificity: " << specificity << "\n";
//     std::cout << "F1-score   : " << f1          << "\n";

//     std::cout << "\nProcessed " << NUM_SAMPLES << " rows.\n";
//     return 0;
// }
