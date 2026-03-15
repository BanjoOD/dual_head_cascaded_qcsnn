// #include <hls_stream.h>
// #include <ap_fixed.h>
// #include <algorithm>
// #include <array>
// #include <string>

// #include <iostream>
// #include <chrono>
// #include <random>
// #include <cmath>
// #include <memory>

// #include "topclass_tb.h"
// #include "../include/hls4csnn1d_bm/constants.h"
// #include "../include/hls4csnn1d_bm/filereader.h"
// #include <nlohmann/json.hpp>


// // Include the header that defines topFunction, array180_t, ap_fixed_c, FIXED_LENGTH, and axi_fixed_t.

// // top_function.h should define:
// //   const int FIXED_LENGTH = 180;
// //   typedef ap_fixed<8, 4, AP_RND, AP_SAT> ap_fixed_c;
// //   typedef std::array<ap_fixed_c, FIXED_LENGTH> array180_t;  // same as input_data_type
// //   typedef ap_axiu<8, 0, 0, 0> axi_fixed_t;
// //   extern "C" void topFunction(hls::stream<axi_fixed_t>& dmaInStream, hls::stream<axi_fixed_t>& dmaOutStream);

// //----------------------------------------------------------------------
// // Conversion from internal (file-based) data stream to AXI input stream
// //----------------------------------------------------------------------
// // Converts each sample (row) in dataStreamInternal into FIXED_LENGTH AXI words,
// // setting TLAST on the final word.
// void input_data_to_axi(hls::stream<array180_t>& internalStream, hls::stream<axi_fixed_t>& axiStream) {
//     const int num_transfers = (FIXED_LENGTH + 7) / 8;  // For FIXED_LENGTH=180, equals 23 transfers
//     while (!internalStream.empty()) {
//         array180_t sample = internalStream.read();
//         // Pack the 180-byte sample into 23 64-bit words
//         for (int i = 0; i < num_transfers; i++) {
//             axi_fixed_t word;
//             // Pack 8 bytes into each 64-bit word
//             for (int j = 0; j < 8; j++) {
//                 int index = i * 8 + j;
//                 if (index < FIXED_LENGTH) {
//                     word.data.range(j * 8 + 7, j * 8) = sample[index].range(7, 0);
//                 } else {
//                     // Pad with zero if the sample doesn't fill the 64-bit word completely
//                     word.data.range(j * 8 + 7, j * 8) = 0;
//                 }
//             }
//             // Assert TLAST on the last word of the sample
//             word.last = (i == num_transfers - 1) ? 1 : 0;
//             axiStream.write(word);
//         }
//     }
// }


// //----------------------------------------------------------------------
// // Conversion from AXI output stream to internal label stream
// //----------------------------------------------------------------------
// // Each AXI word is converted back into an ap_fixed_c value.
// void axi_to_fixed(hls::stream<axi_fixed_t>& axiStream, hls::stream<ap_fixed_c>& fixedStream) {
//     while (!axiStream.empty()) {
//         axi_fixed_t word = axiStream.read();
//         // If TLAST is asserted, output only the remaining valid bytes.
//         if (word.last == 1) {
//             int valid_bytes = FIXED_LENGTH % 8;
//             // If FIXED_LENGTH is a multiple of 8, valid_bytes should be 8.
//             if (valid_bytes == 0) valid_bytes = 8;
//             for (int j = 0; j < valid_bytes; j++) {
//                 ap_fixed_c value = static_cast<ap_fixed_c>(word.data.range(j * 8 + 7, j * 8));
//                 fixedStream.write(value);
//             }
//         } else {
//             // For full words, output all 8 bytes.
//             for (int j = 0; j < 8; j++) {
//                 ap_fixed_c value = static_cast<ap_fixed_c>(word.data.range(j * 8 + 7, j * 8));
//                 fixedStream.write(value);
//             }
//         }
//     }
// }


// //----------------------------------------------------------------------
// // Main Testbench (for Vitis simulation)
// //----------------------------------------------------------------------
// int main() {
//     // Set up file paths.
//     std::string folderPath = "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_test/smallvhls";

//     std::string jsonWeights2Path = "/home/velox-217533/Projects/fau_projects/research/trained_model/snn_quant_models/extracted_weights_model2.json";
//     std::string jsonDims2Path = "/home/velox-217533/Projects/fau_projects/research/trained_model/snn_quant_models/extracted_dimensions_model2.json";

//     // std::string jsonWeights4Path = "/home/velox-217533/Projects/fau_projects/research/trained_model/snn_models/extracted_weights_csnet4.json";
//     // std::string jsonDims4Path = "/home/velox-217533/Projects/fau_projects/research/trained_model/snn_models/extracted_dimensions_csnet4.json";


//     // Create an instance of FileReader and load the data.
//     FileReader reader;
//     reader.loadData(folderPath);

//     // Internal streams from file-based data.
//     hls::stream<array180_t> dataStreamInternal;
//     hls::stream<ap_fixed_c> labelStreamInternal;
//     reader.streamData(dataStreamInternal, labelStreamInternal);
    
//     // Convert internal data stream to an AXI-compliant stream.
//     hls::stream<axi_fixed_t> dmaInStream;
//     input_data_to_axi(dataStreamInternal, dmaInStream);

//     // Create an AXI output stream for topFunction.
//     hls::stream<axi_fixed_t> dmaOutStream;

//     // Call the top-level function. This function will internally convert dmaInStream
//     // into the processing stream, run the neural network evaluation, and produce an AXI output.
//     topFunction(dmaInStream, dmaOutStream);

//     // Convert the AXI output stream back into an internal label stream.
//     hls::stream<ap_fixed_c> labelStreamFromAxi;
//     axi_to_fixed(dmaOutStream, labelStreamFromAxi);

//     // For verification, print the predicted labels.
//     //----------------------------------------------------------------------
// // Compare the AXI output labels with the original internal labels
// // Compute and print the classification accuracy
// //----------------------------------------------------------------------
// int correct_predictions = 0;
// int total_samples = 0;

// // Ensure both streams have the same number of samples
// while (!labelStreamFromAxi.empty() && !labelStreamInternal.empty()) {
//     ap_fixed_c predicted_label = labelStreamFromAxi.read();
//     ap_fixed_c actual_label = labelStreamInternal.read();

//     // Compare predictions
//     if (predicted_label == actual_label) {
//         correct_predictions++;
//     }
//     total_samples++;
// }

// // Compute and display accuracy
// if (total_samples > 0) {
//     float accuracy = (correct_predictions * 100.0) / total_samples;
//     std::cout << "Classification Accuracy: " << accuracy << "% (" 
//               << correct_predictions << "/" << total_samples << " correct)" 
//               << std::endl;
// } else {
//     std::cout << "No samples to evaluate accuracy." << std::endl;
// }

    
//     return 0;
// }
