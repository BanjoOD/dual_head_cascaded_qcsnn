#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

VectorXd leakyReLU(const VectorXd& x, double alpha = 0.01) {
    VectorXd result = x;
    for (int i = 0; i < x.size(); ++i) {
        result[i] = x[i] > 0 ? x[i] : alpha * x[i];
    }
    return result;
}

VectorXd leakyReLU_derivative(const VectorXd& x, double alpha = 0.01) {
    VectorXd result = x;
    for (int i = 0; i < x.size(); ++i) {
        result[i] = x[i] > 0 ? 1.0 : alpha;
    }
    return result;
}

VectorXd softmax(const VectorXd& x) {
    VectorXd e_x = (x.array() - x.maxCoeff()).exp();
    return e_x / e_x.sum();
}

VectorXd softmax_derivative(const VectorXd& output, const VectorXd& grad_output) {
    VectorXd s = output;
    MatrixXd jacobian = MatrixXd::Zero(s.size(), s.size());

    for (int i = 0; i < s.size(); ++i) {
        jacobian(i, i) = s[i] * (1 - s[i]);
    }

    for (int i = 0; i < s.size(); ++i) {
        for (int j = 0; j < s.size(); ++j) {
            if (i != j) {
                jacobian(i, j) = -s[i] * s[j];
            }
        }
    }

    return jacobian * grad_output;
};

#endif