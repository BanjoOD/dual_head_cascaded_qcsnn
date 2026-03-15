#ifndef BATCHNORM1D_H
#define BATCHNORM1D_H

#include "layer.h"
#include <vector>
#include <string>
#include <cmath>
#include <numeric>

class BatchNorm1D : public Layer {

    private:
        int num_features;
        float eps;
        float momentum;
        std::vector<float> gamma;
        std::vector<float> beta;
        std::vector<float> running_mean;
        std::vector<float> running_var;
        
    public:
        BatchNorm1D(int num_features, float eps = 1e-5, float momentum = 0.1)
            : num_features(num_features), eps(eps), momentum(momentum) {
            gamma.resize(num_features, 1.0f); // Scale parameter
            beta.resize(num_features, 0.0f);  // Shift parameter
            running_mean.resize(num_features, 0.0f);
            running_var.resize(num_features, 1.0f);
        }

        std::vector<float> forward(const std::vector<float>& input) override {
            int batch_size = input.size() / num_features;
            std::vector<float> output(input.size(), 0.0f);

            // Compute mean and variance for the batch
            std::vector<float> batch_mean(num_features, 0.0f);
            std::vector<float> batch_var(num_features, 0.0f);

            for (int i = 0; i < num_features; ++i) {
                for (int j = 0; j < batch_size; ++j) {
                    batch_mean[i] += input[j * num_features + i];
                }
                batch_mean[i] /= batch_size;

                for (int j = 0; j < batch_size; ++j) {
                    float diff = input[j * num_features + i] - batch_mean[i];
                    batch_var[i] += diff * diff;
                }
                batch_var[i] /= batch_size;
            }

            // Normalize
            for (int i = 0; i < num_features; ++i) {
                for (int j = 0; j < batch_size; ++j) {
                    int idx = j * num_features + i;
                    output[idx] = (input[idx] - batch_mean[i]) / std::sqrt(batch_var[i] + eps);
                    output[idx] = gamma[i] * output[idx] + beta[i];
                }
            }

            // Update running mean and variance
            for (int i = 0; i < num_features; ++i) {
                running_mean[i] = momentum * batch_mean[i] + (1 - momentum) * running_mean[i];
                running_var[i] = momentum * batch_var[i] + (1 - momentum) * running_var[i];
            }

            return output;
        }

        std::string getType() const override {
            return "BatchNorm1d Layer";
        }


};

#endif // BATCHNORM1D_H
