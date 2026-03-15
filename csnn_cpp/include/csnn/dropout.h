#ifndef DROPOUT_H
#define DROPOUT_H

#include "layer.h"
#include <vector>
#include <string>
#include <random>

class Dropout : public Layer {

    private:
        float dropout_prob;
        bool training;
        std::vector<float> mask;
        std::default_random_engine generator;
        std::uniform_real_distribution<float> distribution;


    public:
        Dropout(float dropout_prob): dropout_prob(dropout_prob), training(true) {
            // Seed the random number generator for reproducibility
            generator.seed(std::random_device{}());
            distribution = std::uniform_real_distribution<float>(0.0f, 1.0f);
        }

        void setTraining(bool is_training) {
            training = is_training;
        }

        std::vector<float> forward(const std::vector<float>& input) override {
            std::vector<float> output(input.size());
            
            if (training) {
                mask.clear();
                mask.resize(input.size());
                for (size_t i = 0; i < input.size(); ++i) {
                    mask[i] = (distribution(generator) >= dropout_prob) ? 1.0f : 0.0f;
                    output[i] = input[i] * mask[i];
                }
            } else {
                for (size_t i = 0; i < input.size(); ++i) {
                    output[i] = input[i] * (1.0f - dropout_prob);
                }
            }

            return output;
        }


        std::string getType() const override {
            return "Dropout Layer";
        }

    
};

#endif // DROPOUT_H
