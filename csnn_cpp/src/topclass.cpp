#ifndef TOP_FUNCTION_H
#define TOP_FUNCTION_H

#include "../include/hls4csnn1d_bm/includeheaders.h"


using namespace hls4csnn1d_bm;

//---------------------------------------------------------------------
// TopClass: Instantiates, Initializes, and Evaluates the Network
//---------------------------------------------------------------------
class TopClass2 {
    public:
        void instantiate() {
            network2 = NeuralNetwork2_bm();
        }
    
        void initialize() {
            // Retrieve layer information and load corresponding weights.
            NeuralNetwork2_bm::LayerInfo layerInfo = network2.getLayerNames();
            for (int i = 0; i < layerInfo.count; ++i) {
                mapWeights(weights2, layerInfo.names[i]);
            }
            network2.setWeights(weights2);
        }
    
        void evaluate(hls::stream<input_data_type>& dataStream, hls::stream<ap_fixed_c>& labelStream) {
            evaluator2.evaluate(network2, dataStream, labelStream);
        }
    
    private:
        NeuralNetwork2_bm network2;      // The neural network instance.
        WeightsContainer weights2;      // Container for network weights.
        ModelEvaluation evaluator2;     // Object to evaluate the network.
    };

//---------------------------------------------------------------------
// TopClass: Instantiates, Initializes, and Evaluates the Network
//---------------------------------------------------------------------
// class TopClass4 {
//     public:
//         void instantiate() {
//             network = NeuralNetwork_bm();
//         }
    
//         void initialize() {
//             // Retrieve layer information and load corresponding weights.
//             NeuralNetwork_bm::LayerInfo layerInfo = network.getLayerNames();
//             for (int i = 0; i < layerInfo.count; ++i) {
//                 mapWeights(weights, layerInfo.names[i]);
//             }
//             network.setWeights(weights);
//         }
    
//         void evaluate(hls::stream<input_data_type>& dataStream, hls::stream<ap_fixed_c>& labelStream) {
//             evaluator.evaluate(network, dataStream, labelStream);
//         }
    
//     private:
//         NeuralNetwork_bm network;      // The neural network instance.
//         WeightsContainer weights;      // Container for network weights.
//         ModelEvaluation evaluator;     // Object to evaluate the network.
//     };
    
    //---------------------------------------------------------------------
    // Conversion Functions
    //---------------------------------------------------------------------
    
    // Convert the incoming AXI stream to an internal data stream of input_data_type.
    // Each sample is composed of FIXED_LENGTH words.
    void axi_to_input_data(hls::stream<axi_fixed_t>& axiStream, hls::stream<input_data_type>& dataStream) {
        // Calculate number of 64-bit transfers needed to pack FIXED_LENGTH bytes.
        const int num_transfers = (FIXED_LENGTH + 7) / 8;  // For FIXED_LENGTH=180, this equals 23.
        
        while (!axiStream.empty()) {
            input_data_type sample;
            // Partition the output array completely to allow parallel writes.
            #pragma HLS ARRAY_PARTITION variable=sample complete dim=1
            
            // Use a word-level counter instead of a byte counter.
            int wordCount = 0;
            
            // Read and unpack each 64-bit word
            for (int i = 0; i < num_transfers; i++) {
                // You can optionally pipeline the outer loop if desired:
                // #pragma HLS PIPELINE II=1
                axi_fixed_t word = axiStream.read();
                
                // Unpack the 64-bit word into 8 individual 8-bit elements.
                // This inner loop can be fully unrolled if the tool allows.
                for (int j = 0; j < 8; j++) {
                    int index = i * 8 + j;
                    if (index < FIXED_LENGTH) {  // Ensure we don’t go out of bounds.
                        sample[index] = static_cast<ap_fixed_c>(word.data.range(j*8+7, j*8));
                    }
                }
                
                // On the last word, assert that TLAST is set.
                if (i == num_transfers - 1) {
                    assert(word.last == 1);
                }
                wordCount++;  // Increase the word counter once per 64-bit word.
            }
            
            // Write the fully constructed sample (ECG record) into the output stream.
            dataStream.write(sample);
        }
    }


    
    
    // Convert the internal label stream (one ap_fixed_c per sample) to an AXI stream.
    // Here, each predicted label is sent as an 8-bit word with TLAST asserted.
    void fixed_to_axi(hls::stream<ap_fixed_c>& fixedStream, hls::stream<axi_fixed_t>& axiStream) {
        // Use a two-stage approach to break the FIFO dependency
        // First stage: Read all values from fixedStream into a buffer
        ap_fixed_c labelBuffer[FIXED_LENGTH]; // Using FIXED_LENGTH as the buffer size
        #pragma HLS ARRAY_PARTITION variable=labelBuffer cyclic factor=8
        
        int labelCount = 0;
        read_labels:
        while (!fixedStream.empty()) {
            #pragma HLS PIPELINE II=1
            labelBuffer[labelCount] = fixedStream.read();
            labelCount++;
        }
        
        // Second stage: Process the buffer in 8-byte chunks
        int wordCount = 0;
        const int totalWords = (FIXED_LENGTH + 7) / 8;  // For FIXED_LENGTH=180, totalWords = 23
        
        process_labels:
        for (int wordIdx = 0; wordIdx < totalWords; wordIdx++) {
            #pragma HLS PIPELINE II=1
            
            axi_fixed_t word;
            
            // Pack 8 bytes per word (64-bit transfer)
            for (int i = 0; i < 8; i++) {
                int labelIdx = wordIdx * 8 + i;
                if (labelIdx < labelCount) {
                    word.data.range(i * 8 + 7, i * 8) = labelBuffer[labelIdx].range(7, 0);
                }
            }
            
            // Set TLAST only on the last word of the record
            word.last = (wordCount == totalWords - 1);
            axiStream.write(word);
            wordCount++;
        }
    }

    // void fixed_to_axi(hls::stream<ap_fixed_c>& fixedStream, hls::stream<axi_fixed_t>& axiStream) {
    //     int wordCount = 0;
    //     const int totalWords = (FIXED_LENGTH + 7) / 8;  // For FIXED_LENGTH=180, totalWords = 23

    //     while (!fixedStream.empty()) {
    //         axi_fixed_t word;
    //         // Pack 8 bytes per word (64-bit transfer)
    //         for (int i = 0; i < 8; i++) {
    //             if (!fixedStream.empty()) {
    //                 ap_fixed_c label = fixedStream.read();
    //                 word.data.range(i * 8 + 7, i * 8) = label.range(7, 0);
    //             }
    //         }
    //         // Set TLAST only on the last word of the record
    //         word.last = (wordCount == totalWords - 1);
    //         axiStream.write(word);
    //         wordCount++;
    //     }
    // }
    
    //---------------------------------------------------------------------
    // Top-Level HLS Function
    //---------------------------------------------------------------------
    // This function is synthesized as the top-level module and interfaces with the DMA.
    extern "C" void topFunction(hls::stream<axi_fixed_t>& dmaInStream, hls::stream<axi_fixed_t>& dmaOutStream) {
        #pragma HLS INTERFACE axis port=dmaInStream
        #pragma HLS INTERFACE axis port=dmaOutStream
        #pragma HLS INTERFACE s_axilite port=return

        // #pragma HLS DATAFLOW
    
        // Internal streams for the network processing.
        hls::stream<input_data_type> dataStream;
        hls::stream<ap_fixed_c> labelStream;

        // Buffer to store the input data
        // input_data_type input_sample;
    
        // Convert the AXI input stream into the internal data stream.
        axi_to_input_data(dmaInStream, dataStream);
        // if (!dataStream.empty()) {
        //     input_sample = dataStream.read();
        // }

        // // Create a stream for the binary model
        // hls::stream<input_data_type> binaryInputStream;
        // binaryInputStream.write(input_sample);

        // hls::stream<input_data_type> multiInputStream;
        // multiInputStream.write(input_sample);
    
        // Instantiate, initialize, and evaluate the neural network.
        TopClass2 topClass2;
        topClass2.instantiate();
        topClass2.initialize();
        topClass2.evaluate(dataStream, labelStream);
    
        // Convert the computed label from the internal stream back to an AXI output stream.
        fixed_to_axi(labelStream, dmaOutStream);
    }
    
    #endif // TOP_FUNCTION_H

