#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"
#include <vector>
#include <memory>

class Network {

    private:
        std::vector<std::shared_ptr<Layer>> layers;

    public:
        void addLayer(std::shared_ptr<Layer> layer) {
            layers.push_back(layer);
        }

        std::vector<float> forward(const std::vector<float>& input) {
            std::vector<float> output = input;
            for (const auto& layer : layers) {
                output = layer->forward(output);
            }
            return output;
        }


};

#endif // NETWORK_H
