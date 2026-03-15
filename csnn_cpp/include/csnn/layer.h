#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

class Layer {

    public:
        virtual ~Layer() {}

        // Pure virtual method for forward propagation
        virtual std::vector<float> forward(const std::vector<float>& input) = 0;

        // Pure virtual method to return the type of the layer
        virtual std::string getType() const = 0;
};

#endif