#ifndef EVAL_UTILS_H
#define EVAL_UTILS_H

#include <ap_fixed.h>
#include "constants_sd.h"


// LUT-based approximation of exp(x) for small x (Taylor series-based)
// ap_fixed_c exp_fixed(ap_fixed_c x) {
//     #pragma HLS INLINE

//     float x_f = (float)x;  // Convert fixed to float
//     float result_f = 1.0f + x_f + (x_f * x_f) / 2.0f + (x_f * x_f * x_f) / 6.0f;

//     return ap_fixed_c(result_f);  // Convert back to fixed
// }

ap_fixed_c exp_fixed(ap_fixed_c x) {
    #pragma HLS INLINE
    // Approximate exp(x) using 3rd-order polynomial: exp(x) ≈ 1 + x + x²/2! + x³/6!
   return (ap_fixed_c(1.0) + x + (x * x) / ap_fixed_c(2) + (x * x * x) / ap_fixed_c(6));
}

// Softmax function optimized for HLS
template<int N>
void softmax(ap_fixed_c input[N], ap_fixed_c output[N]) {
    #pragma HLS INLINE

    // Step 1: Find max value for numerical stability
    ap_fixed_c max_val = input[0];
    for (int i = 1; i < N; i++) {
        #pragma HLS PIPELINE II=1
        if (input[i] > max_val) 
            max_val = input[i];
    }

    // Step 2: Compute exponentials and sum
    ap_fixed_c exp_vals[N];
    #pragma HLS ARRAY_PARTITION variable=exp_vals complete
    ap_fixed_c sum = 0;

    for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        exp_vals[i] = exp_fixed(input[i] - max_val);  // Use LUT-based exp function
        sum += exp_vals[i];
    }

    // Step 3: Compute reciprocal of sum (1/sum) using Newton-Raphson approximation
    ap_fixed_c inv_sum = 1 / sum; 

    // Step 4: Compute softmax probabilities
    for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        output[i] = exp_vals[i] * inv_sum;  // Avoid division, use multiplication
    }
}

#endif // EVAL_UTILS_H



// #ifndef EVAL_UTILS_H
// #define EVAL_UTILS_H

// #include "ap_fixed.h"
// #include <cmath>  // Using std::exp
// #include "constants.h"

// // Softmax function optimized for HLS
// template<int N>
// void softmax(ap_fixed_c input[N], ap_fixed_c output[N]) {
//     #pragma HLS INLINE

//     // Find max value for numerical stability
//     ap_fixed_c max_val = input[0];
//     for (int i = 1; i < N; i++) {
//         #pragma HLS PIPELINE II=1
//         if (input[i] > max_val) 
//             max_val = input[i];
//     }

//     // Compute exponentials and sum
//     ap_fixed_c exp_vals[N];
//     #pragma HLS ARRAY_PARTITION variable=exp_vals complete
//     ap_fixed_c sum = 0;

//     for (int i = 0; i < N; i++) {
//         #pragma HLS PIPELINE II=1
//         exp_vals[i] = std::exp(double(input[i] - max_val));  // Correctly using std::exp
//         sum += exp_vals[i];
//     }

//     // Compute softmax probabilities
//     for (int i = 0; i < N; i++) {
//         #pragma HLS PIPELINE II=1
//         output[i] = exp_vals[i] / sum;
//     }
// }

// #endif // EVAL_UTILS_H



// #ifndef EVAL_UTILS_H
// #define EVAL_UTILS_H

// #include "ap_fixed.h"
// #include <cmath>

// #include "constants.h"



// // Softmax function
// template<int N>
// void softmax(ap_fixed_c* input, float* output) {
//     #pragma HLS INLINE
    
//     float max_val = float(input[0]);
//     for (int i = 1; i < N; i++) {
//         #pragma HLS PIPELINE
//         float curr_val = float(input[i]);
//         if (curr_val > max_val) max_val = curr_val;
//     }
    
//     float sum = 0.0f;
//     float exp_vals[N];
//     for (int i = 0; i < N; i++) {
//         #pragma HLS PIPELINE
//         exp_vals[i] = std::exp(float(input[i]) - max_val);
//         sum += exp_vals[i];
//     }
    
//     for (int i = 0; i < N; i++) {
//         #pragma HLS PIPELINE
//         output[i] = exp_vals[i] / sum;
//     }
// }

// #endif // EVAL_UTILS_H
