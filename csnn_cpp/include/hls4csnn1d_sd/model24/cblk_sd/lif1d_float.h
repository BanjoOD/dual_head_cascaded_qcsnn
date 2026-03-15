#ifndef LIF1D_FLOAT_H
#define LIF1D_FLOAT_H

#include <hls_stream.h>
#include <ap_int.h>

namespace hls4csnn1d_cblk_sd {

enum ResetModeFloat { RESET_SUBTRACT_FLOAT = 0, RESET_ZERO_FLOAT = 1 };

template<
    int OUT_CH,
    int FEATURE_LEN,
    ResetModeFloat RESET_MODE = RESET_SUBTRACT_FLOAT
>
class LIF1D_Float {
public:
    LIF1D_Float() : bank(false) { reset(); }

    void reset() {
    #pragma HLS INLINE
        for (int c = 0; c < OUT_CH; ++c) {
            for (int t = 0; t < FEATURE_LEN; ++t) {
            #pragma HLS PIPELINE II=1
                V0[c][t] = 0.0f;
                V1[c][t] = 0.0f;
            }
        }
        bank = false;
    }

    // BN int8 → dequant (scale) → LIF float → spikes {0,1}
    void forward(hls::stream<ap_int<8>>& in,
                 hls::stream<ap_int<8>>& out_spk,
                 const float beta,
                 const float theta,
                 const float act_scale) {
    #pragma HLS INLINE off
        float b = beta;
        if (b < 0.0f) b = 0.0f;
        if (b > 1.0f) b = 1.0f;

        const bool rd = bank;
        const bool wr = !bank;

        for (int c = 0; c < OUT_CH; ++c) {
            for (int t = 0; t < FEATURE_LEN; ++t) {
            #pragma HLS PIPELINE II=1
                const ap_int<8> qi = in.read();
                const float x = (float)qi * act_scale; // dequant

                float v_prev = rd ? V1[c][t] : V0[c][t];

                // delayed reset from previous membrane
                const bool r_prev = (v_prev > theta);

                // base = b*v_prev + x
                float base = b * v_prev + x;

                float v_next;
                if (RESET_MODE == RESET_SUBTRACT_FLOAT) {
                    v_next = base;
                    if (r_prev) v_next -= theta;
                } else { // RESET_ZERO_FLOAT
                    if (r_prev) v_next = 0.0f;
                    else        v_next = base;
                }

                ap_int<8> spk = 0;
                if (v_next > theta) spk = 1;

                if (wr) V0[c][t] = v_next;
                else    V1[c][t] = v_next;

                out_spk.write(spk);
            }
        }

        bank = !bank;
    }

    // Optional: also stream membranes (for PyTorch output=True parity)
    void forward_with_mem(hls::stream<ap_int<8>>& in,
                          hls::stream<ap_int<8>>& out_spk,
                          hls::stream<float>&    out_mem,
                          const float beta,
                          const float theta,
                          const float act_scale) {
    #pragma HLS INLINE off
        float b = beta;
        if (b < 0.0f) b = 0.0f;
        if (b > 1.0f) b = 1.0f;

        const bool rd = bank;
        const bool wr = !bank;

        for (int c = 0; c < OUT_CH; ++c) {
            for (int t = 0; t < FEATURE_LEN; ++t) {
            #pragma HLS PIPELINE II=1
                const ap_int<8> qi = in.read();
                const float x = (float)qi * act_scale;

                float v_prev = rd ? V1[c][t] : V0[c][t];
                const bool r_prev = (v_prev > theta);

                float base = b * v_prev + x;

                float v_next;
                if (RESET_MODE == RESET_SUBTRACT_FLOAT) {
                    v_next = base;
                    if (r_prev) v_next -= theta;
                } else {
                    if (r_prev) v_next = 0.0f;
                    else        v_next = base;
                }

                ap_int<8> spk = 0;
                if (v_next > theta) spk = 1;

                if (wr) V0[c][t] = v_next;
                else    V1[c][t] = v_next;

                out_spk.write(spk);
                out_mem.write(v_next);
            }
        }

        bank = !bank;
    }

private:
    float V0[OUT_CH][FEATURE_LEN];
    float V1[OUT_CH][FEATURE_LEN];
    bool  bank;
};

} // namespace hls4csnn1d_cblk_sd
#endif // LIF1D_FLOAT_H





// #ifndef LIF1D_FLOAT_H
// #define LIF1D_FLOAT_H

// #include <hls_stream.h>
// #include <ap_int.h>

// namespace hls4csnn1d_cblk_sd {

// enum ResetModeFloat { RESET_SUBTRACT_FLOAT = 0, RESET_ZERO_FLOAT = 1 };

// template<int OUT_CH, int FEATURE_LEN, ResetModeFloat RESET_MODE = RESET_SUBTRACT_FLOAT>
// class LIF1D_Float {
// public:
//     LIF1D_Float() { reset(); }

//     // Call once per new sample (per batch element)
//     void reset() {
//     #pragma HLS INLINE
//         for (int c = 0; c < OUT_CH; ++c) {
//         #pragma HLS UNROLL
//             V[c] = 0.0f;  // membrane voltage (float)
//         }
//     }

//     // Float-based forward to match snnTorch exactly
//     // beta, theta, scale are all floats
//     void forward(hls::stream<ap_int<8>>& in,
//                  hls::stream<ap_int<8>>& out,
//                  const float beta,
//                  const float theta,
//                  const float scale) {
//     #pragma HLS INLINE off

//         // Clamp beta to [0, 1] as snnTorch does
//         float beta_clamped = beta;
//         if (beta_clamped < 0.0f) beta_clamped = 0.0f;
//         if (beta_clamped > 1.0f) beta_clamped = 1.0f;

//         for (int c = 0; c < OUT_CH; ++c) {
//             float v = V[c];  // Load membrane state
//             float r_delayed = 0.0f;  // Reset for first timestep is 0

//             for (int t = 0; t < FEATURE_LEN; ++t) {
//             #pragma HLS PIPELINE II=1
//                 const ap_int<8> x_i = in.read();  // int8 from BN
                
//                 // Convert int8 to float and scale (dequantize)
//                 float x_float = (float)x_i.to_int() * scale;

//                 // ---- Integrate with delayed reset ----
//                 // v_next = beta * v + x - r_delayed * theta
//                 float v_next = beta_clamped * v + x_float - r_delayed * theta;

//                 // ---- Generate spike (strict >) ----
//                 bool spike = (v_next > theta);

//                 // ---- Compute reset for NEXT timestep ----
//                 r_delayed = spike ? 1.0f : 0.0f;

//                 // Update membrane for next iteration
//                 v = v_next;

//                 // Emit binary spike {0, 1}
//                 out.write(spike ? ap_int<8>(1) : ap_int<8>(0));
//             }
//             V[c] = v;  // Store final membrane for this channel
//         }
//     }

// private:
//     float V[OUT_CH];  // Float membrane per channel
// };

// } // namespace hls4csnn1d_cblk_sd
// #endif // LIF1D_FLOAT_H