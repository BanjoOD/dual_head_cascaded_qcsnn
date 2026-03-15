// #ifndef BYTES_FOR_LAYER_H
// #define BYTES_FOR_LAYER_H

// #include <cstddef>
// #include "utils.h"             // hls_str_contains(...)
// #include "../../include/hls4csnn1d_sd_train/constants_sd.h"      // sizeof(ap_fixed_c)
// #include "../../../weights_header_bm/split_header_bm/qcsnet2_cblk1_batch_norm_p1_weights.h"
// #include "../../../weights_header_bm/split_header_bm/qcsnet2_cblk1_qconv1d_p1_weights.h"

// namespace hls4csnn1d_cblk_bm {

// /* --------------------------------------------------------------------------
//  * Return the number of bytes needed for the weight block of a given layer.
//  * --------------------------------------------------------------------------
//  *  ‑ `layerName` is the *same* string you already store in LayerInfo.names[i]
//  *  ‑ For each concrete layer we use the compile‑time macro constants that
//  *    the training script generated when it split the weights.
//  *  ‑ Size = (#elements) × sizeof(ap_fixed_c)   (int8 / int4 → change here)
//  * ------------------------------------------------------------------------ */
// inline std::size_t bytes_for_layer(const char* layerName)
// {
//     using namespace hls4csnn1d_sd_train;   // for hls_str_contains

//     /* ------------ Q‑conv1d ------------------------------------ */
//     if (hls_str_contains(layerName, "conv1d")) {

//         constexpr std::size_t num_elems =
//               qcsnet2_cblk1_qconv1d_p1_OUTPUT_CHANNELS *
//               qcsnet2_cblk1_qconv1d_p1_INPUT_CHANNELS  *
//               qcsnet2_cblk1_qconv1d_p1_KERNEL_SIZE;

//         return num_elems * sizeof(ap_fixed_c);
//     }

//     /* ------------ Batch‑norm ---------------------------------- */
//     if (hls_str_contains(layerName, "batch_norm")) {

//         constexpr std::size_t num_elems =
//               4 * qcsnet2_cblk1_batch_norm_p1_NUM_CHANNELS; // γ,β,μ,σ²

//         return num_elems * sizeof(ap_fixed_c);
//     }

//     /* ------------ Unsupported layer → 0 ----------------------- */
//     return 0;
// }

// } // namespace hls4csnn1d_cblk_bm
// #endif
