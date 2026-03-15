#ifndef DENSE_H
#define DENSE_H

#include <iostream>
#include <Eigen/Dense>
#include <cmath>

#include "layer.h"
#include "activation.h"

class Dense :  public Layer {
    private:
        string name;
        MatrixXd weights;
        VectorXd biases;
        VectorXd last_input;
        VectorXd last_output;
        VectorXi last_input_shape;

    public:
        Dense(const string& name, int nodes, int num_classes)
            : name(name), weights(MatrixXd::Random(nodes, num_classes) * 0.1), biases(VectorXd::Zero(num_classes)) {}

        VectorXd forward(const VectorXd& input) {
            last_input_shape = VectorXi(input.size(), 1);
            last_input = input;
            last_output = (weights.transpose() * last_input) + biases;
            return softmax(last_output);
        }

        VectorXd backward(const VectorXd& grad_output, double learning_rate = 0.005) {
            VectorXd softmax_output = last_output;
            VectorXd grad_softmax = softmax_derivative(softmax_output, grad_output);

            MatrixXd grad_weights = last_input * grad_softmax.transpose();
            VectorXd grad_biases = grad_softmax;

            VectorXd grad_input = weights * grad_softmax;

            weights -= learning_rate * grad_weights;
            biases -= learning_rate * grad_biases;

            return grad_input;
        }

        VectorXd get_weights() const {
            return Map<const VectorXd>(weights.data(), weights.size());
        }
};

#endif