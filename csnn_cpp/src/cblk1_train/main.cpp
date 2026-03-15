// #ifndef _AP_UNUSED_PARAM
// #define _AP_UNUSED_PARAM(x) (void)(x)
// #endif

// #include <iostream>
// #include <cmath>
// #include <memory>
// #include <algorithm>
// #include <ap_fixed.h>
// #include <hls_stream.h>
// #include <nlohmann/json.hpp>

// #include <iomanip>

// // load_test.cpp  ----------------------------------------------
// #include <torch/torch.h>

// // #include "../../include/hls4csnn1d_sd_train/constants_sd_train.h"
// #include "../../include/hls4csnn1d_sd_train/filereader.h"
// // #include "../../include/hls4csnn1d_sd_train/cblk_sd_train/conv1d_sd_train.h"
// // #include "../../include/hls4csnn1d_sd_train/cblk_sd_train/batchnorm1d_sd_train.h"
// // #include "../../include/hls4csnn1d_sd_train/cblk_sd_train/lif1d_sd_train.h"
// // #include "../../include/hls4csnn1d_sd_train/cblk_sd_train/maxpool1d_sd_train.h"
// // #include "../../include/hls4csnn1d_sd_train/cblk_sd_train/nn2_cblk1_sd_train.h"

// #include "../../include/hls4csnn1d_sd_train/cblk_sd_train/conv1d_weights_utils.h"
// #include "../../include/hls4csnn1d_sd_train/cblk_sd_train/conv1d_trainable.h"
// #include "../../include/hls4csnn1d_sd_train/cblk_sd_train/batchnorm1d_trainable.h"
// #include "../../include/hls4csnn1d_sd_train/cblk_sd_train/lif1d_trainable.h"
// #include "../../include/hls4csnn1d_sd_train/cblk_sd_train/block1_sd_train.h"

// #include "../../include/hls4csnn1d_sd_train/weights_sd/qcsnet2_cblk1_qconv1d_weights.h"
// #include "../../include/hls4csnn1d_sd_train/weights_sd/qcsnet2_cblk2_qconv1d_weights.h"

// #include "../../include/hls4csnn1d_sd_train/weights_sd/qcsnet2_cblk1_batch_norm_weights.h"
// #include "../../include/hls4csnn1d_sd_train/weights_sd/qcsnet2_cblk2_batch_norm_weights.h"




// /**
//  * Use the hls4csnn1d_bm namespace for brevity.
//  */
// using namespace hls4csnn1d_cblk_sd_train;

// /**
//  * Helper function to print the JSON map for debugging.
//  */
// void printJsonMap(const JsonMap& jsonMap) {
//     for (const auto& [key, value] : jsonMap) {
//         std::cout << "Key: " << key << "\n";
//         std::cout << "Value: " << value.dump(4) << "\n"; // Pretty print with 4-space indentation
//         std::cout << "---------------------------------------------\n";
//     }
// }




// using namespace hls4csnn1d_cblk_sd_train;

// int main() {
//     //------------------------------------------------------------------
//     // 1.  Load ECG rows + labels from disk
//     //------------------------------------------------------------------
//     const std::string dataDir = "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_test/smallvhls";
//     FileReader reader;
//     reader.loadData(dataDir);

//     const int N = static_cast<int>(reader.X.size());
//     if (N == 0) {
//         std::cerr << "No rows loaded – aborting.\n";
//         return -1;
//     }
//     std::cout << "Loaded " << N << " rows from " << dataDir << "\n";

//     //------------------------------------------------------------------
//     // 2.  Convert vectors → Torch tensors  (float32, shape [N,1,180])
//     //------------------------------------------------------------------
//     torch::Tensor all_x = torch::empty({N, 1, 180}, torch::kFloat32);
//     torch::Tensor all_y = torch::from_blob(reader.y.data(), {N},
//                                            torch::kLong).clone();

//     /* --- collapse 4 labels → 2 classes ----------------------------- *
//     normal (0) stays 0 ;  all others (1,2,3) become 1                */
//     all_y = (all_y > 0).to(torch::kLong);   // 0 → 0,  {1,2,3} → 1

//     for (int n = 0; n < N; ++n) {
//         float *dst = all_x[n][0].data_ptr<float>();
//         for (int k = 0; k < 180; ++k)
//             dst[k] = static_cast<float>(reader.X[n][k]);
//     }

//     //------------------------------------------------------------------
//     // 3.  Mini-batch fetcher
//     //------------------------------------------------------------------
//     auto get_batch = [&](int B,
//                          torch::Tensor &bx,
//                          torch::Tensor &by) {
//         auto idx = torch::randperm(N).slice(0, 0, B);
//         bx = all_x.index_select(0, idx).clone();   // [B,1,180]
//         by = all_y.index_select(0, idx).clone();   // [B]
//     };

//     //------------------------------------------------------------------
//     // 4.  Build the network, optimiser, hyper-params
//     //------------------------------------------------------------------
//     Block1_SD_Train net;
//     net.train();

//     const int   batch  = 32;
//     const int   epochs = 10;
//     const float lr     = 5e-3f;

//     torch::optim::Adam opt(net.parameters(), lr);

//     // ------------------------------------------------------------------
//     // 5.  Training loop  (add metrics)
//     // ------------------------------------------------------------------
//     torch::Tensor bx, by;
//     for (int ep = 0; ep < epochs; ++ep) {

//         // reset epoch counters
//         int TP = 0, TN = 0, FP = 0, FN = 0;
//         float running_loss = 0.0f;
//         int   seen = 0;

//         // ------- single mini-batch version ----------------------------
//         get_batch(batch, bx, by);

//         torch::Tensor logits = net.forward(bx);          // [B,2]
//         torch::Tensor loss =
//             torch::nn::functional::cross_entropy(logits, by);

//         opt.zero_grad();
//         loss.backward();
//         torch::nn::utils::clip_grad_norm_(net.parameters(), 1.0);
//         opt.step();

//         running_loss += loss.item<float>();
//         seen         += batch;

//         // --- predictions: argmax over 2 logits ------------------------
//         torch::Tensor pred = logits.argmax(1);           // [B]

//         // --- confusion-matrix update ----------------------------------
//         for (int i = 0; i < batch; ++i) {
//             int p = pred[i].item<int>();
//             int a = by[i].item<int>();
//             if      (a==1 && p==1) ++TP;
//             else if (a==0 && p==0) ++TN;
//             else if (a==0 && p==1) ++FP;
//             else if (a==1 && p==0) ++FN;
//         }

//         // --- derive metrics -------------------------------------------
//         float acc  = static_cast<float>(TP + TN) / (TP + TN + FP + FN);
//         float prec = (TP + FP) ? static_cast<float>(TP) / (TP + FP) : 0.0f;
//         float rec  = (TP + FN) ? static_cast<float>(TP) / (TP + FN) : 0.0f;
//         float f1   = (prec + rec) ? 2 * prec * rec / (prec + rec) : 0.0f;

//         std::cout << "epoch " << std::setw(2) << ep
//                 << "  loss "      << std::fixed << std::setprecision(4)
//                 << running_loss / seen
//                 << "  acc "       << acc
//                 << "  prec "      << prec
//                 << "  rec "       << rec
//                 << "  f1 "        << f1
//                 << '\n';
//     }


   
//     //------------------------------------------------------------------
//     // 6.  Sanity: confirm every parameter got a gradient
//     //------------------------------------------------------------------
//     std::cout << "\nGrad check:\n";
//     for (const auto &kv : net.named_parameters()) {
//         std::cout << "  " << std::setw(18) << kv.key()
//                   << (kv.value().grad().defined() ? "  OK" : "  NONE") << '\n';
//     }
//     return 0;
// }


// // --------------------------------------------------------------
// // Helper: copy compile-time ap_fixed array → runtime torch tensor
// // --------------------------------------------------------------
// // torch::Tensor load_conv1_weights() {
// //     constexpr int OUT = qcsnet2_cblk1_qconv1d_OUT_CH;
// //     constexpr int IN  = qcsnet2_cblk1_qconv1d_IN_CH;
// //     constexpr int K   = qcsnet2_cblk1_qconv1d_K;

// //     torch::Tensor w = torch::empty({OUT, IN, K}, torch::kFloat32);

// //     auto acc = w.accessor<float, 3>();          // mutable view
// //     for (int o = 0; o < OUT; ++o)
// //         for (int i = 0; i < IN; ++i)
// //             for (int k = 0; k < K; ++k)
// //                 acc[o][i][k] = static_cast<float>(
// //                     qcsnet2_cblk1_qconv1d_weights[o][i][k]);  // cast ap_fixed→float

// //     return w;
// // }

// // int main() {

// //     // 1) Load ECG rows from folder
// //     std::string folderPath = "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_train/large";
// //     FileReader reader;
// //     reader.loadData(folderPath);


// //     // already have reader.X  (std::vector<array180_t>)  and reader.y (std::vector<int>)
// //     const int N = reader.X.size();
// //     torch::Tensor all_x = torch::empty({N, 1, 180}, torch::kFloat32);  // features
// //     torch::Tensor all_y = torch::from_blob(reader.y.data(), {N}, torch::kLong).clone();

// //     for (int n = 0; n < N; ++n) {
// //         // from_blob points to the fixed-point row, clone() makes its own copy
// //         torch::Tensor row = torch::from_blob(reader.X[n].data(), {180}, torch::kFloat32)
// //                                 .clone()
// //                                 .view({1, 180});              // [1,180]
// //         all_x[n] = row;
// //     }


// //      // ── 0.  make sure the final flatten length matches 1032 ─────────────
// //     // L_in = 178  →  Conv1(K=3) → 176  →  Pool/2 → 88
// //     //           →  Conv2(K=3) → 86   →  Pool/2 → 43
// //     // 43  ×  24  channels  ==  1032  (matches linear header)
// //     const int  seq_len   = 180;
// //     const int  batch     = 4;
// //     const int  epochs    = 20;
// //     const float lr       = 5e-3f;

// //     // ── 1.  build the network & set train mode ──────────────────────────
// //     Block1_SD_Train net;
// //     net.train();

// //     // ── 2.  simple Adam optimiser over *all* parameters ─────────────────
// //     torch::optim::Adam opt(net.parameters(), lr);

// //     // dummy target classes 0 / 1  (replace with real labels later)
// //     torch::Tensor target = torch::randint(0, 2, {batch}, torch::kLong);

// //     // ── 3.  mini training loop – just 3 steps to prove gradients flow ───
// //     for (int ep = 0; ep < epochs; ++ep) {
// //         torch::Tensor x = torch::randn({batch, 1, seq_len});

// //         torch::Tensor logits = net.forward(x);         // [B, 2]
// //         torch::Tensor loss =
// //             torch::nn::functional::cross_entropy(logits, target);

// //         opt.zero_grad();
// //         loss.backward();
// //         // /* after loss.backward();  (still inside the loop) */
// //         // torch::Tensor lin_w_grad =
// //         //         net.named_parameters()["linear1.weight"].grad();   // public lookup
// //         // float gmax = lin_w_grad.abs().max().item<float>();

// //         // std::cout << "logits[0]        " << logits[0] << '\n'
// //         //         << "linear grad max  " << gmax << '\n';

// //         // break;   // run only this first epoch for the probe
// //         torch::nn::utils::clip_grad_norm_(net.parameters(), 1.0); // keep stable
// //         opt.step();

        
       

// //         std::cout << "epoch " << ep
// //                   << "   loss " << loss.item<float>() << '\n';
// //     }

// //     // ── 4.  sanity: every layer got a gradient? ─────────────────────────
// //     std::cout << "\nGrad check:\n";
// //     for (const auto &kv : net.named_parameters()) {
// //         std::cout << "  " << kv.key()
// //                   << (kv.value().grad().defined() ? "  OK" : "  NONE")
// //                   << '\n';
// //     }
// //     return 0;
// //  std::cout << "✔️  LibTorch reachable\n"; 
// //  Conv1_SD_Module conv1;

// // torch::Tensor dummy = torch::randn({1, 1, 10});   // [N,C,L]
// // torch::Tensor out   = conv1.forward(dummy);

// // std::cout << "output shape: " << out.sizes() << '\n';   // expect [1,16,8]


// // flatten once
// // static const std::vector<float> w_blk1 = flatten_weights<16, 1, 3>(qcsnet2_cblk1_qconv1d_weights);

// // static const std::vector<float> w_blk2 = flatten_weights<24, 16, 3>(qcsnet2_cblk2_qconv1d_weights);

// // // two independent modules
// // Conv1D_Manual<1, 16, 3>   conv1(w_blk1.data());   // block-1
// // Conv1D_Manual<16, 24, 3>  conv2(w_blk2.data());   // block-2

// // // block-1 : 16 channels
// // BatchNorm1D_Manual<16> bn1(qcsnet2_cblk1_batch_norm_PARAMS);

// // // block-2 : 24 channels
// // BatchNorm1D_Manual<24> bn2(qcsnet2_cblk2_batch_norm_PARAMS);


// // Block1_SD_Train blk1;

// // torch::Tensor x = torch::randn({4, 1, 10});  // batch 4, len 10
// // torch::Tensor y = blk1.forward(x);

// // std::cout << "output shape: " << y.sizes() << '\n';   // expect [4,16,8]
       
// //     // Paths to input data and trained model weights/dimensions
// //     std::string folderPath = "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_test/smallvhls";
// //     // std::string jsonWeights4Path = "/home/velox-217533/Projects/fau_projects/research/trained_model/snn_models/extracted_weights_csnet4.json";
// //     // std::string jsonDims4Path = "/home/velox-217533/Projects/fau_projects/research/trained_model/snn_models/extracted_dimensions_csnet4.json";
    

// //     std::string jsonWeights2Path = "/home/velox-217533/Projects/fau_projects/research/trained_model/snn_quant_models/extracted_weights_model2.json";
// //     std::string jsonDims2Path = "/home/velox-217533/Projects/fau_projects/research/trained_model/snn_quant_models/extracted_dimensions_model2.json";

// //     // Instantiate FileReader to load input data
// //     FileReader reader;


    
// //     reader.loadData(folderPath);

// //     // Streams for data, labels, and predictions
// //     hls::stream<array180_t> dataStream;
// //     hls::stream<array240_t> outStream;
// //     hls::stream<ap_fixed_c> labelStream;

// //     // Stream the data into the input streams
// //     reader.streamData(dataStream);

    
// //     return 0;
// // }







