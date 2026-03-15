#include <hls_stream.h>
#include <ap_fixed.h>
#include <algorithm>
#include <array>
#include <string>

#include <iostream>
#include <chrono>
#include <random>
#include <cmath>
#include <memory>

#include "topclass_cblk1_tb.h"
#include "../../include/hls4csnn1d_bm/constants.h"
#include "../../include/hls4csnn1d_bm/filereader.h"
#include <nlohmann/json.hpp>

// Conversion from internal (file-based) data stream to AXI input stream
void input_data_to_axi(hls::stream<array180_t>& internalStream, hls::stream<axi_fixed_t>& axiStream) {
    const int num_transfers = (FIXED_LENGTH1 + 7) / 8;  // For FIXED_LENGTH=180, equals 23 transfers
    
    while (!internalStream.empty()) {
        array180_t sample = internalStream.read();
        
        // Pack the 180-byte sample into 23 64-bit words
        for (int i = 0; i < num_transfers; i++) {
            axi_fixed_t word;
            
            // Pack 8 bytes into each 64-bit word
            for (int j = 0; j < 8; j++) {
                int index = i * 8 + j;
                if (index < FIXED_LENGTH1) {
                    // Ensure consistent bit representation with the other side
                    word.data.range(j * 8 + 7, j * 8) = sample[index].range(7, 0);
                } else {
                    // Pad with zero if the sample doesn't fill the 64-bit word completely
                    word.data.range(j * 8 + 7, j * 8) = 0;
                }
            }
            
            // Assert TLAST on the last word of the sample
            word.last = (i == num_transfers - 1) ? 1 : 0;
            
            // DO NOT assign values to keep, strb, or user - they're disabled in your configuration
            
            axiStream.write(word);
        }
    }
}

// Conversion from AXI output stream to internal label stream
void axi_to_fixed(hls::stream<axi_fixed_t>& axiStream, hls::stream<array240_t>& fixedStream) {
    const int words_per_sample = (FIXED_LENGTH3 + 7) / 8;  // For FIXED_LENGTH2=480, this is 60
    
    // Continue as long as there are words in the stream
    while (!axiStream.empty()) {
        array240_t sample;
        bool sample_complete = false;
        
        // Read enough words to form a complete sample
        for (int wordIdx = 0; wordIdx < words_per_sample && !sample_complete; wordIdx++) {
            if (axiStream.empty()) {
                std::cout << "Error: AXI stream unexpectedly empty at word " << wordIdx << std::endl;
                break;
            }
            
            axi_fixed_t word = axiStream.read();
            
            // Unpack the 64-bit word into 8 individual 8-bit elements
            for (int j = 0; j < 8; j++) {
                int index = wordIdx * 8 + j;
                if (index < FIXED_LENGTH3) {
                    sample[index] = static_cast<ap_fixed_c>(word.data.range(j*8+7, j*8));
                }
            }
            
            // Check if this is the last word of the sample
            if (word.last) {
                sample_complete = true;
                
                // If we received TLAST early, print a warning
                if (wordIdx != words_per_sample - 1) {
                    std::cout << "Warning: TLAST received early at word " << wordIdx 
                              << " (expected at " << (words_per_sample - 1) << ")" << std::endl;
                }
            }
        }
        
        // Only write a complete sample to the output stream
        if (sample_complete) {
            fixedStream.write(sample);
            // std::cout << "Wrote complete sample to fixedStream" << std::endl;
        }
    }
    
    std::cout << "Final fixedStream size: " << fixedStream.size() << std::endl;
}

// ------------------------------------------------------------
// Global weight memory for the first conv layer
// (make it large enough for *all* layers or compute sizes)
// ------------------------------------------------------------
static const int OC = 8;    // adjust to your actual first layer
static const int IC = 1;
static const int KS = 3;

static ap_fixed_c weight_mem[OC * IC * KS];

static void init_weights()
{
    for (int oc = 0; oc < OC; ++oc)
        for (int ic = 0; ic < IC; ++ic)
            for (int k  = 0; k  < KS; ++k) {
                const int idx = oc*IC*KS + ic*KS + k;
                weight_mem[idx] = ap_fixed_c((oc + ic + k) & 0x3); // dummy data
            }
}


// Main Testbench (for Vitis simulation)
int main() {
    init_weights();
    // Set up file paths.
    std::string folderPath = "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_test/smallvhls";

    // Create an instance of FileReader and load the data.
    FileReader reader;
    reader.loadData(folderPath);

    // Internal streams from file-based data.
    hls::stream<array180_t> dataStreamInternal;
    
    // Create a copy of the data stream for verification
    hls::stream<array180_t> dataStreamCopy;
    
    reader.streamData(dataStreamInternal);
    
    int dataSize = dataStreamInternal.size();
    std::cout << "Original dataStreamInternal size: " << dataSize << std::endl;
    
    // Make a copy of the data before conversion to AXI format
    while (!dataStreamInternal.empty()) {
        array180_t sample = dataStreamInternal.read();
        dataStreamCopy.write(sample);
    }
    
    // Now refill the original stream from our copy
    hls::stream<array180_t> processStream;
    int copySize = dataStreamCopy.size();
    
    while (!dataStreamCopy.empty()) {
        array180_t sample = dataStreamCopy.read();
        processStream.write(sample);
    }
    
    std::cout << "Process stream size: " << processStream.size() << std::endl;
    
    // Convert internal data stream to an AXI-compliant stream.
    hls::stream<axi_fixed_t> dmaInStream;
    input_data_to_axi(processStream, dmaInStream);
    std::cout << "dma instream size: " << dmaInStream.size() << std::endl;

    // Create an AXI output stream for topFunction.
    hls::stream<axi_fixed_t> dmaOutStream;

    // Call the top-level function
    topFunctionCblk1(dmaInStream, dmaOutStream, weight_mem);

    // Convert the AXI output stream back into an internal label stream.
    hls::stream<array240_t> labelStreamFromAxi;
    axi_to_fixed(dmaOutStream, labelStreamFromAxi);

    int labelSize = labelStreamFromAxi.size();
    std::cout << "Output label stream size: " << labelSize << std::endl;

    while (!labelStreamFromAxi.empty()) {
        labelStreamFromAxi.read();
    }

    if (dataSize == labelSize) {
        std::cout << "Test Passed!" << std::endl;
        return 0;
    } else {
        std::cout << "Test failed! Expected " << dataSize << " labels but got " << labelSize << std::endl;
        return -1;
    }
}



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

// #include "topclass_cblk1_tb.h"
// #include "../../include/hls4csnn1d_bm/constants.h"
// #include "../../include/hls4csnn1d_bm/filereader.h"
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
// void input_data_to_axi(hls::stream<array180_t>& internalStream, hls::stream<axi_fixed_t>& axiStream) {
//     const int num_transfers = (FIXED_LENGTH1 + 7) / 8;  // For FIXED_LENGTH=180, equals 23 transfers
    
//     while (!internalStream.empty()) {
//         array180_t sample = internalStream.read();
        
//         // Pack the 180-byte sample into 23 64-bit words
//         for (int i = 0; i < num_transfers; i++) {
//             axi_fixed_t word;
            
//             // Pack 8 bytes into each 64-bit word
//             for (int j = 0; j < 8; j++) {
//                 int index = i * 8 + j;
//                 if (index < FIXED_LENGTH1) {
//                     // Ensure consistent bit representation with the other side
//                     word.data.range(j * 8 + 7, j * 8) = sample[index].range(7, 0);
//                 } else {
//                     // Pad with zero if the sample doesn't fill the 64-bit word completely
//                     word.data.range(j * 8 + 7, j * 8) = 0;
//                 }
//             }
            
//             // Assert TLAST on the last word of the sample
//             word.last = (i == num_transfers - 1) ? 1 : 0;
//             word.keep = 0xFF;  // All bytes are valid
//             word.strb = 0xFF;  // All bytes are valid
//             word.user = 0;     // User field not used
            
//             axiStream.write(word);
//         }
//     }
// }
// // Converts each sample (row) in dataStreamInternal into FIXED_LENGTH AXI words,
// // setting TLAST on the final word.
// // void input_data_to_axi(hls::stream<array180_t>& internalStream, hls::stream<axi_fixed_t>& axiStream) {
// //     // Use a two-stage approach to break the FIFO dependency
// //         // First stage: Read all values from fixedStream into a buffer
// //         ap_fixed_c labelBuffer[FIXED_LENGTH1]; // Using FIXED_LENGTH as the buffer size

// //         int labelCount = 0;
// //         read_labels:
// //         while (!internalStream.empty()) {
// //             array180_t temp = internalStream.read();
// //             for (int i = 0; i < FIXED_LENGTH1; i++) {
// //                 labelBuffer[i] = temp[i];
// //             }
// //             labelCount++;
// //         }
// //         // Second stage: Process the buffer in 8-byte chunks
// //         int wordCount = 0;
// //         const int totalWords = (FIXED_LENGTH1 + 7) / 8;  // For FIXED_LENGTH=180, totalWords = 23
        
// //         process_labels:
// //         for (int wordIdx = 0; wordIdx < totalWords; wordIdx++) {
// //             axi_fixed_t word;
// //             // Pack 8 bytes per word (64-bit transfer)
// //             for (int i = 0; i < 8; i++) {
// //                 int labelIdx = wordIdx * 8 + i;
// //                 if (labelIdx < labelCount) {
// //                     word.data.range(i * 8 + 7, i * 8) = labelBuffer[labelIdx].range(7, 0);
// //                 }
// //             }
// //             // Set TLAST only on the last word of the record
// //             word.last = (wordCount == totalWords - 1);
// //             axiStream.write(word);
// //             wordCount++;
// //         }
//     // const int num_transfers = (FIXED_LENGTH1 + 7) / 8;  // For FIXED_LENGTH=180, equals 23 transfers
//     // while (!internalStream.empty()) {
//     //     array180_t sample = internalStream.read();
//     //     // Pack the 180-byte sample into 23 64-bit words
//     //     for (int i = 0; i < num_transfers; i++) {
//     //         axi_fixed_t word;
//     //         // Pack 8 bytes into each 64-bit word
//     //         for (int j = 0; j < 8; j++) {
//     //             int index = i * 8 + j;
//     //             if (index < FIXED_LENGTH1) {
//     //                 word.data.range(j * 8 + 7, j * 8) = sample[index].range(7, 0);
//     //             } else {
//     //                 // Pad with zero if the sample doesn't fill the 64-bit word completely
//     //                 word.data.range(j * 8 + 7, j * 8) = 0;
//     //             }
//     //         }
//     //         // Assert TLAST on the last word of the sample
//     //         word.last = (i == num_transfers - 1) ? 1 : 0;
//     //         axiStream.write(word);
//     //     }
//     // }
// // }


// //----------------------------------------------------------------------
// // Conversion from AXI output stream to internal label stream
// //----------------------------------------------------------------------
// // Each AXI word is converted back into an ap_fixed_c value.
// void axi_to_fixed(hls::stream<axi_fixed_t>& axiStream, hls::stream<array480_t>& fixedStream) {
//     const int num_transfers = (FIXED_LENGTH2 + 7) / 8;  // For FIXED_LENGTH=480, this equals 60.
        
//     while (!axiStream.empty()) {
//         array480_t sample;
//         int wordCount = 0;   // Use a word-level counter instead of a byte counter.
//         for (int i = 0; i < num_transfers; i++) {
//             axi_fixed_t word = axiStream.read();
//             // Unpack the 64-bit word into 8 individual 8-bit elements.
//             // This inner loop can be fully unrolled if the tool allows.
//             for (int j = 0; j < 8; j++) {
//                 int index = i * 8 + j;
//                 if (index < FIXED_LENGTH2) {  // Ensure we don’t go out of bounds.
//                     sample[index] = static_cast<ap_fixed_c>(word.data.range(j*8+7, j*8));
//                 }
//             }
//             // On the last word, assert that TLAST is set.
//             if (i == num_transfers - 1) {
//                 assert(word.last == 1);
//             }
//             wordCount++;  // Increase the word counter once per 64-bit word.
//         }
        
//         // Write the fully constructed sample (ECG record) into the output stream.
//         fixedStream.write(sample);
//     }
    
//     // while (!axiStream.empty()) {
//     //     axi_fixed_t word = axiStream.read();
//     //     // If TLAST is asserted, output only the remaining valid bytes.
//     //     if (word.last == 1) {
//     //         int valid_bytes = FIXED_LENGTH2 % 8;
//     //         // If FIXED_LENGTH is a multiple of 8, valid_bytes should be 8.
//     //         if (valid_bytes == 0) valid_bytes = 8;
//     //         for (int j = 0; j < valid_bytes; j++) {
//     //             ap_fixed_c value = static_cast<ap_fixed_c>(word.data.range(j * 8 + 7, j * 8));
//     //             fixedStream.write(value);
//     //         }
//     //     } else {
//     //         // For full words, output all 8 bytes.
//     //         for (int j = 0; j < 8; j++) {
//     //             ap_fixed_c value = static_cast<ap_fixed_c>(word.data.range(j * 8 + 7, j * 8));
//     //             fixedStream.write(value);
//     //         }
//     //     }
//     // }
// }


// //----------------------------------------------------------------------
// // Main Testbench (for Vitis simulation)
// //----------------------------------------------------------------------
// int main() {
//     // Set up file paths.
//     std::string folderPath = "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_test/smallvhls";

//     // Create an instance of FileReader and load the data.
//     FileReader reader;
//     reader.loadData(folderPath);

//     // Internal streams from file-based data.
//     hls::stream<array180_t> dataStreamInternal;
    
//     // Important: Create a copy of the data stream for verification
//     hls::stream<array180_t> dataStreamCopy;
    
//     reader.streamData(dataStreamInternal);
    
//     int dataSize = dataStreamInternal.size();
//     std::cout << "Original dataStreamInternal size: " << dataSize << std::endl;
    
//     // Make a copy of the data before conversion to AXI format
//     while (!dataStreamInternal.empty()) {
//         array180_t sample = dataStreamInternal.read();
//         dataStreamCopy.write(sample);
//     }
    
//     // Now refill the original stream from our copy
//     hls::stream<array180_t> processStream;
//     int copySize = dataStreamCopy.size();
    
//     while (!dataStreamCopy.empty()) {
//         array180_t sample = dataStreamCopy.read();
//         processStream.write(sample);
//     }
    
//     std::cout << "Process stream size: " << processStream.size() << std::endl;
    
//     // Convert internal data stream to an AXI-compliant stream.
//     hls::stream<axi_fixed_t> dmaInStream;
//     input_data_to_axi(processStream, dmaInStream);
//     std::cout << "dma instream size: " << dmaInStream.size() << std::endl;

//     // Create an AXI output stream for topFunction.
//     hls::stream<axi_fixed_t> dmaOutStream;

//     // Call the top-level function
//     topFunctionCblk1(dmaInStream, dmaOutStream);

//     // Convert the AXI output stream back into an internal label stream.
//     hls::stream<array480_t> labelStreamFromAxi;
//     axi_to_fixed(dmaOutStream, labelStreamFromAxi);

//     int labelSize = labelStreamFromAxi.size();
//     std::cout << "Output label stream size: " << labelSize << std::endl;

//     if (dataSize == labelSize) {
//         std::cout << "Test Passed!" << std::endl;
//         return 0;
//     } else {
//         std::cout << "Test failed! Expected " << dataSize << " labels but got " << labelSize << std::endl;
//         return -1;
//     }
// }
// // int main() {
// //     // Set up file paths.
// //     std::string folderPath = "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_test/smallvhls";

// //     // Create an instance of FileReader and load the data.
// //     FileReader reader;
// //     reader.loadData(folderPath);

// //     // Internal streams from file-based data.
// //     hls::stream<array180_t> dataStreamInternal;
// //     hls::stream<array480_t> labelStreamInternal;

// //     reader.streamData(dataStreamInternal);

// //     int dataSize = dataStreamInternal.size();
    
// //     // Convert internal data stream to an AXI-compliant stream.
// //     hls::stream<axi_fixed_t> dmaInStream;
// //     input_data_to_axi(dataStreamInternal, dmaInStream);
// //     std::cout <<"dma instream size: "<< dmaInStream.size() << std::endl;

// //     // Create an AXI output stream for topFunction.
// //     hls::stream<axi_fixed_t> dmaOutStream;

// //     // Call the top-level function. This function will internally convert dmaInStream
// //     // into the processing stream, run the neural network evaluation, and produce an AXI output.
// //     topFunctionCblk1(dmaInStream, dmaOutStream);

// //     // Convert the AXI output stream back into an internal label stream.
// //     hls::stream<array480_t> labelStreamFromAxi;
// //     axi_to_fixed(dmaOutStream, labelStreamFromAxi);

// //     int labelSize = labelStreamFromAxi.size();

// //     if (dataSize == labelSize) {
// //         std::cout << "Test Passed!" << std::endl;
// //         return 0;
// //     } else {
// //         std::cout << "Test failed!" << std::endl;
// //         return -1;
// //     }

//     // For verification, print the predicted labels.
//     //----------------------------------------------------------------------
//     // Compare the AXI output labels with the original internal labels
//     // Compute and print the classification accuracy
//     //----------------------------------------------------------------------
//     // int correct_predictions = 0;
//     // int total_samples = 0;

//     // // Ensure both streams have the same number of samples
//     // while (!labelStreamFromAxi.empty() && !labelStreamInternal.empty()) {
//     //     ap_fixed_c predicted_label = labelStreamFromAxi.read();
//     //     ap_fixed_c actual_label = labelStreamInternal.read();

//     //     // Compare predictions
//     //     if (predicted_label == actual_label) {
//     //         correct_predictions++;
//     //     }
//     //     total_samples++;
//     // }

//     // // Compute and display accuracy
//     // if (total_samples > 0) {
//     //     float accuracy = (correct_predictions * 100.0) / total_samples;
//     //     std::cout << "Classification Accuracy: " << accuracy << "% (" 
//     //             << correct_predictions << "/" << total_samples << " correct)" 
//     //             << std::endl;
//     // } else {
//     //     std::cout << "No samples to evaluate accuracy." << std::endl;
//     // }

// // }
