#ifndef TOP_CLASS_CBLK1_H
#define TOP_CLASS_CBLK1_H

#include "../../include/hls4csnn1d_bm/cblk/debug_flag.h"
#include "../../include/hls4csnn1d_bm/cblk/includeheaders_cblk1_bm.h"

/* This creates the global object; only one definition allowed */
// const ap_fixed_c *g_weight_base = nullptr;


using namespace hls4csnn1d_cblk_bm;

//---------------------------------------------------------------------
// TopClass: Instantiates, Initializes, and Evaluates the Network
//---------------------------------------------------------------------
class TopClass2 {
    public:
        void instantiate() {
            network2_cblk1_p1 = NeuralNetwork2_Cblk1_P1_bm();
        }
    
        void initialize() {
            // Retrieve layer information and load corresponding weights.
            NeuralNetwork2_Cblk1_P1_bm::LayerInfo layerInfo = network2_cblk1_p1.getLayerNames();
            size_t offset_bytes    = 0;

            for (int i = 0; i < layerInfo.count; ++i) {
                const char* lname = layerInfo.names[i];
                // const ap_fixed_c* ddr_ptr = weight_ddr + offset_bytes / sizeof(ap_fixed_c);

                mapWeights(weights2, lname, offset_bytes);

                offset_bytes += bytes_for_layer(lname) / sizeof(ap_fixed_c);
            }

            network2_cblk1_p1.setWeights(weights2);
        }
    
        void evaluate(hls::stream<input180_data_type>& dataStream, 
                      hls::stream<input240_data_type>& labelStream,
                      const ap_fixed_c *weight_ddr) {
            evaluator2.evaluate(network2_cblk1_p1, dataStream, labelStream, weight_ddr);
        }
    
    private:
        NeuralNetwork2_Cblk1_P1_bm network2_cblk1_p1;      // The neural network instance.
        WeightsContainer_Cblk weights2;      // Container for network weights.
        ModelEvaluation evaluator2;     // Object to evaluate the network.
    };

//---------------------------------------------------------------------
// TopClass: Instantiates, Initializes, and Evaluates the Network
//---------------------------------------------------------------------
//---------------------------------------------------------------------
// AXI‑Stream → internal 180‑element sample(s)
//---------------------------------------------------------------------
void axi_to_input_data(hls::stream<axi_fixed_t>& axiStream, hls::stream<input180_data_type>& dataStream) {
    const int num_transfers = (FIXED_LENGTH1 + 7) / 8;   // 23 when 180 B

    #ifdef DEBUG
        unsigned sample_cnt = 0;
    #endif

    sample_loop:
    while (true) {

        /* ---------- check for end‑of‑stream BEFORE starting a sample ---- */
        if (axiStream.empty()) {
            #ifdef DEBUG
            #pragma HLS printf
                printf("[DBG] axi_to_input_data  samples_out = %u\n", sample_cnt);
                #ifndef __SYNTHESIS__  
                    fflush(stdout);
                #endif
            #endif
            return;  
        }                // no more words → assert ap_done

        input180_data_type sample;

        /* ---------- read exactly one sample (23 words) ------------------ */
        read_one:
        for (int w = 0; w < num_transfers; ++w) {
            #pragma HLS PIPELINE II=1

            axi_fixed_t word = axiStream.read();     // blocking read

            // unpack 8 bytes in this 64‑bit word
            
            for (int j = 0; j < 8; ++j) {
                #pragma HLS UNROLL
                int idx = w * 8 + j;
                if (idx < FIXED_LENGTH1)
                    sample[idx] =
                        ap_fixed_c(word.data.range(j * 8 + 7, j * 8));
            }
        }

        /* ---------- write the complete sample downstream ---------------- */
        dataStream.write(sample);
        #ifdef DEBUG
            ++sample_cnt;
        #endif
    }
   
}

// void axi_to_input_data(hls::stream<axi_fixed_t>& axiStream,
//                        hls::stream<input180_data_type>& dataStream) {
                           
//     // Continue processing as long as data arrives.
//     while (true) {
//         input180_data_type sample;
//         #pragma HLS ARRAY_PARTITION variable=sample complete dim=1

//         // Read words until TLAST is encountered,
//         // meaning the current sample is complete.
//         int word_index = 0;
//         bool sample_complete = false;
//         while (!sample_complete) {
//             axi_fixed_t word = axiStream.read();

//             // Unpack the word into the current sample.
//             for (int j = 0; j < 8; j++) {
//                 int index = word_index * 8 + j;
//                 if (index < FIXED_LENGTH1) {
//                     sample[index] = static_cast<ap_fixed_c>(word.data.range(j*8+7, j*8));
//                 }
//             }
//             word_index++;

//             // If TLAST is asserted, the sample is complete.
//             if (word.last == 1) {
//                 sample_complete = true;
//             }
//         }
//         // Write the complete sample into the data stream.
//         dataStream.write(sample);

//         // Optionally, decide when to break out of the loop.
//         // For example, if your testbench signals overall end-of-stream
//         // by not providing any further data for some timeout period,
//         // you might break here.
//         if (axiStream.empty()) {
//             // Depending on your system, you might implement a timeout here.
//             // For simulation, you might simply break if no new data is arriving.
//             break;
//         }
//     }
// }

// void axi_to_input_data(hls::stream<axi_fixed_t>& axiStream, hls::stream<input180_data_type>& dataStream) {
//     // Calculate number of 64-bit transfers needed to pack FIXED_LENGTH bytes.
//     const int num_transfers = (FIXED_LENGTH1 + 7) / 8;  // For FIXED_LENGTH=180, this equals 23.
    
//     while (!axiStream.empty()) {
//         input180_data_type sample;
//         // Partition the output array completely to allow parallel writes.
//         #pragma HLS ARRAY_PARTITION variable=sample complete dim=1
        
//         // Read and unpack each 64-bit word
//         for (int i = 0; i < num_transfers; i++) {
//             #pragma HLS PIPELINE II=1
//             axi_fixed_t word = axiStream.read();
            
//             // Unpack the 64-bit word into 8 individual 8-bit elements.
//             for (int j = 0; j < 8; j++) {
//                 int index = i * 8 + j;
//                 if (index < FIXED_LENGTH1) {  // Ensure we don't go out of bounds.
//                     sample[index] = static_cast<ap_fixed_c>(word.data.range(j*8+7, j*8));
//                 }
//             }
            
//             // Debug verification - enable temporarily if needed
//             // if (i == num_transfers - 1) {
//             //     if (word.last != 1) {
//             //         std::cout << "ERROR: Expected TLAST on final word!" << std::endl;
//             //     }
//             // }
//         }
//         // std::cout << "sample size before writing to datastream: " << sample.size() << std::endl;
//         // Write the fully constructed sample into the output stream.
//         dataStream.write(sample);
        
//         // Add debug print - optional
//         // std::cout << "Sample processed and added to dataStream" << std::endl;
//     }
// }

    //---------------------------------------------------------------------
    // Conversion Functions
    //---------------------------------------------------------------------
    
    // Convert the incoming AXI stream to an internal data stream of input_data_type.
    // Each sample is composed of FIXED_LENGTH words.
    // void axi_to_input_data(hls::stream<axi_fixed_t>& axiStream, hls::stream<input180_data_type>& dataStream) {
    //     // Calculate number of 64-bit transfers needed to pack FIXED_LENGTH bytes.
    //     const int num_transfers = (FIXED_LENGTH1 + 7) / 8;  // For FIXED_LENGTH=180, this equals 23.
        
    //     while (!axiStream.empty()) {
    //         input180_data_type sample;
    //         // Partition the output array completely to allow parallel writes.
    //         #pragma HLS ARRAY_PARTITION variable=sample complete dim=1
            
    //         // Use a word-level counter instead of a byte counter.
    //         int wordCount = 0;
            
    //         // Read and unpack each 64-bit word
    //         for (int i = 0; i < num_transfers; i++) {
    //             // You can optionally pipeline the outer loop if desired:
    //             // #pragma HLS PIPELINE II=1
    //             axi_fixed_t word = axiStream.read();
                
    //             // Unpack the 64-bit word into 8 individual 8-bit elements.
    //             // This inner loop can be fully unrolled if the tool allows.
    //             for (int j = 0; j < 8; j++) {
    //                 int index = i * 8 + j;
    //                 if (index < FIXED_LENGTH1) {  // Ensure we don’t go out of bounds.
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
    //         dataStream.write(sample);
    //     }
    // }

    
    // Convert the internal label stream (one ap_fixed_c per sample) to an AXI stream.
    // Here, each predicted label is sent as an 8-bit word with TLAST asserted.
    void fixed_to_axi(hls::stream<input240_data_type>& fixedStream, 
                      hls::stream<axi_fixed_t>& axiStream) {
        // Use a two-stage approach to break the FIFO dependency
        // First stage: Read all values from fixedStream into a buffer
        ap_fixed_c labelBuffer[FIXED_LENGTH3]; // Using FIXED_LENGTH as the buffer size
        #pragma HLS ARRAY_PARTITION variable=labelBuffer cyclic factor=8
        
        int labelCount = 0;
        #ifdef DEBUG
            unsigned axi_word_cnt = 0;
        #endif
        read_labels:
        while (!fixedStream.empty()) {
            input240_data_type temp = fixedStream.read();

            read_loop:
            for (int i = 0; i < FIXED_LENGTH3; i++) {
                #pragma HLS PIPELINE II=1
                labelBuffer[i] = temp[i];
            }

            // labelBuffer[labelCount] = fixedStream.read();
            labelCount++;
        
        
            // Second stage: Process the buffer in 8-byte chunks
            int wordCount = 0;
            const int totalWords = (FIXED_LENGTH3 + 7) / 8;  // For FIXED_LENGTH=180, totalWords = 23
            
            pack_loop:
            for (int wordIdx = 0; wordIdx < totalWords; wordIdx++) {
                #pragma HLS PIPELINE II=2
                
                axi_fixed_t word;
                
                // Pack 8 bytes per word (64-bit transfer)
                pack_values:
                for (int i = 0; i < 8; i++) {
                    #pragma HLS UNROLL
                    int labelIdx = wordIdx * 8 + i;
                    if (labelIdx < labelCount) {
                        word.data.range(i * 8 + 7, i * 8) = labelBuffer[labelIdx].range(7, 0);
                    }
                }
                
                // Set TLAST only on the last word of the record
                word.last = (wordCount == totalWords - 1);
                axiStream.write(word);
                #ifdef DEBUG
                    ++axi_word_cnt;
                #endif
                wordCount++;
            }
        }
        #ifdef DEBUG
        #pragma HLS printf
            printf("[DBG] fixed_to_axi  words_written = %u (expect 30)\n", axi_word_cnt);
            #ifndef __SYNTHESIS__  
                fflush(stdout);
            #endif
        #endif
    }

    
    
    //---------------------------------------------------------------------
    // Top-Level HLS Function
    //---------------------------------------------------------------------
    // This function is synthesized as the top-level module and interfaces with the DMA.
    extern "C" void topFunctionCblk1(hls::stream<axi_fixed_t>& dmaInStream, 
                                     hls::stream<axi_fixed_t>& dmaOutStream, 
                                     const ap_fixed_c *weight_ddr) {

        
        // g_weight_base = weight_ddr;

        #pragma HLS INTERFACE axis port=dmaInStream
        #pragma HLS INTERFACE axis port=dmaOutStream
        #pragma HLS INTERFACE s_axilite port=return

        #pragma HLS INTERFACE m_axi port=weight_ddr offset=slave bundle=gmem depth=24
        #pragma HLS INTERFACE s_axilite port=weight_ddr bundle=ctrl
        #pragma HLS READ_ONLY_PORT port=weight_ddr

        // Optional debug print - enable for debugging only
        // std::cout << "Starting topFunctionCblk1, dmaInStream size: " << dmaInStream.size() << std::endl;

        // Internal streams for the network processing.
        hls::stream<input180_data_type> dataStream;
        #pragma HLS STREAM variable=dataStream depth=64

        hls::stream<input240_data_type> labelStream;
        #pragma HLS STREAM variable=labelStream depth=64

       

        // Convert the AXI input stream into the internal data stream.
        axi_to_input_data(dmaInStream, dataStream);
        std::cout << "After axi_to_input_data, dataStream size: " << dataStream.size() << std::endl;

        // Instantiate, initialize, and evaluate the neural network.
        TopClass2 topClass2;
        topClass2.instantiate();
        topClass2.initialize();
        topClass2.evaluate(dataStream, labelStream, weight_ddr);
        
        std::cout << "After evaluation, labelStream size: " << labelStream.size() << std::endl;

        // Convert the computed label from the internal stream back to an AXI output stream.
        fixed_to_axi(labelStream, dmaOutStream);
        
        // std::cout << "After fixed_to_axi, dmaOutStream size: " << dmaOutStream.size() << std::endl;
        // #pragma HLS INTERFACE axis port=dmaInStream
        // #pragma HLS INTERFACE axis port=dmaOutStream
        // #pragma HLS INTERFACE s_axilite port=return

        // // #pragma HLS DATAFLOW
        // // std::cout <<"dma InStream size: "<< dmaInStream.size() << std::endl;
    
        // // Internal streams for the network processing.
        // hls::stream<input180_data_type> dataStream;
        // hls::stream<input480_data_type> labelStream;

        // // Buffer to store the input data
        // // input_data_type input_sample;
    
        // // Convert the AXI input stream into the internal data stream.
        // axi_to_input_data(dmaInStream, dataStream);
        // std::cout <<"data stream size: "<< dataStream.size() << std::endl;
    
        // // Instantiate, initialize, and evaluate the neural network.
        // TopClass2 topClass2;
        // topClass2.instantiate();
        // topClass2.initialize();
        // topClass2.evaluate(dataStream, labelStream);
    
        // // Convert the computed label from the internal stream back to an AXI output stream.
        // fixed_to_axi(labelStream, dmaOutStream);
    }
    
    #endif // TOP_FUNCTION_H

