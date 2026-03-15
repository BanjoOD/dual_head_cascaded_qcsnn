// #ifndef CONV1D_H
// #define CONV1D_H

// #include <iostream>
// #include <vector>
// #include <random>
// #include <algorithm>
// #include <string>
// #include <cmath>

// #include "layer.h"

// class Conv1D : public Layer {

//     private:
//         int input_channels;
//         int output_channels;
//         int kernel_size;
//         int stride;
//         std::vector<std::vector<float>> weights;


//     public:
//         Conv1D(int input_channels, int output_channels, int kernel_size, int stride)
//             : input_channels(input_channels), output_channels(output_channels), kernel_size(kernel_size), stride(stride) {
//             // Initialize weights (for simplicity, set to small random values)
//             weights.resize(output_channels);
//             for (auto& kernel : weights) {
//                 kernel.resize(input_channels * kernel_size);
//                 for (auto& weight : kernel) {
//                     weight = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);  // Random values between 0 and 1
//                 }
//             }
//         }

//         std::vector<float> forward(const std::vector<float>& input) override {
//             int input_length = input.size() / input_channels;
//             int output_length = (input_length - kernel_size) / stride + 1;
//             std::vector<float> output(output_channels * output_length, 0.0f);

//             for (int oc = 0; oc < output_channels; ++oc) {
//                 for (int ol = 0; ol < output_length; ++ol) {
//                     float sum = 0.0f;
//                     for (int ic = 0; ic < input_channels; ++ic) {
//                         for (int k = 0; k < kernel_size; ++k) {
//                             int input_index = ic * input_length + ol * stride + k;
//                             int weight_index = ic * kernel_size + k;
//                             sum += input[input_index] * weights[oc][weight_index];
//                         }
//                     }
//                     output[oc * output_length + ol] = sum;
//                 }
//             }

//             return output;
//         }


// };

// #endif