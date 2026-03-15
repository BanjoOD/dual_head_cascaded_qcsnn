#ifndef LINEAR_H
#define LINEAR_H

#include "layer.h"
#include <vector>
#include <string>
#include <random>

class Linear : public Layer {

    private:
        int input_size;
        int output_size;
        bool use_bias;
        std::vector<float> weights;
        std::vector<float> biases;
        
    public:
        Linear(int input_size, int output_size, bool use_bias = true)
            : input_size(input_size), output_size(output_size), use_bias(use_bias) {
            // Initialize weights with small random values
            weights.resize(output_size * input_size);
            std::default_random_engine generator;
            std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
            for (auto& weight : weights) {
                weight = distribution(generator); // Random values between 0 and 1
            }

            // Initialize biases with small random values if use_bias is true
            if (use_bias) {
                biases.resize(output_size);
                for (auto& bias : biases) {
                    bias = distribution(generator); // Random values between 0 and 1
                }
            }
        }

        std::vector<float> forward(const std::vector<float>& input) override {
            std::vector<float> output(output_size, 0.0f);

            // Perform the linear transformation
            for (int i = 0; i < output_size; ++i) {
                for (int j = 0; j < input_size; ++j) {
                    output[i] += input[j] * weights[i * input_size + j];
                }
                if (use_bias) {
                    output[i] += biases[i];
                }
            }

            return output;
        }

        std::string getType() const override {
            return "Linear Layer";
        }


};

#endif // LINEAR_H
