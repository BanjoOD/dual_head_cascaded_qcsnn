#ifndef MAXPOOL1D_H
#define MAXPOOL1D_H

#include "layer.h"
#include <vector>
#include <algorithm>
#include <string>

class MaxPool1D : public Layer {

    private:
        int kernel_size;
        int stride;

    public:
        MaxPool1D (int kernel_size, int stride)
            : kernel_size(kernel_size), stride(stride) {}

        std::vector<float> forward(const std::vector<float>& input) override {
            int input_length = input.size();
            int output_length = (input_length - kernel_size) / stride + 1;
            std::vector<float> output(output_length, 0.0f);

            for (int i = 0; i < output_length; ++i) {
                float max_val = -std::numeric_limits<float>::infinity();
                for (int j = 0; j < kernel_size; ++j) {
                    int input_index = i * stride + j;
                    if (input[input_index] > max_val) {
                        max_val = input[input_index];
                    }
                }
                output[i] = max_val;
            }

            return output;
        }

        std::string getType() const override {
            return "MaxPooling1d Layer";
        }

};

#endif // MAXPOOL1D_H
