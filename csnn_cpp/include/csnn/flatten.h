#ifndef FLATTEN_H
#define FLATTEN_H

#include "layer.h"
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>

class Flatten : public Layer {
public:
    Flatten() {}

    std::vector<float> forward(const std::vector<float>& input) override {
        
        return input;
    }

    std::string getType() const override {
        return "Flatten Layer";
    }
};

#endif // FLATTEN_H
