#ifndef LIF1D_SD_INTEGER_FINAL_H
#define LIF1D_SD_INTEGER_FINAL_H

#include <hls_stream.h>
#include <ap_int.h>
// #include "../constants_sd.h"  // if you centralize FRAC_BITS etc.

namespace hls4csnn1d_cblk_sd {

enum ResetModeInt { RESET_SUBTRACT_INT = 0, RESET_ZERO_INT = 1 };

template<
    int OUT_CH,
    int FEATURE_LEN,
    int FRAC_BITS = 12,
    ResetModeInt RESET_MODE = RESET_SUBTRACT_INT
>
class LIF1D_SD_Integer {
public:
    LIF1D_SD_Integer() : bank(false) { reset(); }

    // Call once per new sample/window
    void reset() {
    #pragma HLS INLINE
        for (int c = 0; c < OUT_CH; ++c) {
            for (int t = 0; t < FEATURE_LEN; ++t) {
            #pragma HLS PIPELINE II=1
                V0[c][t] = 0;
                V1[c][t] = 0;
            }
        }
        bank = false; // deterministic start: read V0, write V1
    }

    // Integer-only forward (delayed reset, strict '>')
    // Inputs: int8 from BN, scales/params in Q(FRAC_BITS)
    void forward(hls::stream<ap_int<8>>& in,
                 hls::stream<ap_int<8>>& out_spk,
                 const ap_int<16> beta_q,
                 const ap_int<16> theta_q,
                 const ap_int<16> act_scale_q) {
    #pragma HLS INLINE off
        const int SHIFT = FRAC_BITS;
        const ap_int<16> ONE_Q   = ap_int<16>(1) << SHIFT; // 1.0 in Q
        const ap_int<16> BETA_MAX= ONE_Q;
        const ap_int<40> ROUND   = ap_int<40>(1) << (SHIFT - 1);

        // clamp beta to [0, 1]
        ap_int<16> beta_c = beta_q;
        if (beta_c < 0)          beta_c = 0;
        else if (beta_c > BETA_MAX) beta_c = BETA_MAX;

        const bool rd = bank;     // read from this bank
        const bool wr = !bank;    // write to the other bank

        // Optional storage binding
        // #pragma HLS BIND_STORAGE variable=V0 type=ram_1p impl=bram
        // #pragma HLS BIND_STORAGE variable=V1 type=ram_1p impl=bram

        for (int c = 0; c < OUT_CH; ++c) {
            for (int t = 0; t < FEATURE_LEN; ++t) {
            #pragma HLS PIPELINE II=1
                // ---- Input: int8 → Q(FRAC_BITS) ----
                const ap_int<8>  qi   = in.read(); // [-128,127]
                const ap_int<24> x_q  = (ap_int<24>)qi * (ap_int<24>)act_scale_q; // QF

                // ---- Previous membrane at (c,t) from previous call ----
                ap_int<24> v_prev_q;
                if (rd) v_prev_q = V1[c][t];
                else     v_prev_q = V0[c][t];

                // ---- Delayed reset signal from previous membrane ----
                ap_int<16> r_prev_q = 0; // {0, ONE_Q}
                if (v_prev_q > theta_q) r_prev_q = ONE_Q;

                // ---- v_beta = round((beta * v_prev) >> F) ----
                ap_int<40> prod_b = (ap_int<40>)beta_c * (ap_int<40>)v_prev_q;
                if (prod_b >= 0) prod_b += ROUND; else prod_b -= ROUND;
                ap_int<24> v_beta_q = (ap_int<24>)(prod_b >> SHIFT); // QF

                // ---- base = v_beta + x ----
                ap_int<24> base_q = v_beta_q + x_q; // QF

                // ---- Apply reset mechanism (delayed) ----
                ap_int<24> v_next_q;
                if (RESET_MODE == RESET_SUBTRACT_INT) {
                    // subtract: base - round((r_prev * theta) >> F)
                    ap_int<40> prod_r = (ap_int<40>)r_prev_q * (ap_int<40>)theta_q;
                    if (prod_r >= 0) prod_r += ROUND; else prod_r -= ROUND;
                    ap_int<24> sub_reset_q = (ap_int<24>)(prod_r >> SHIFT); // 0 or theta_q
                    v_next_q = base_q - sub_reset_q;
                } else { // RESET_ZERO_INT
                    // zero: (1 - r_prev) * base   (since r_prev ∈ {0,1})
                    if (r_prev_q == ONE_Q) v_next_q = 0;
                    else                    v_next_q = base_q;
                }

                // ---- Spike from NEW membrane: strict '>' ----
                ap_int<8> spk = 0;
                if (v_next_q > theta_q) spk = 1;

                // ---- Write new membrane to the other bank ----
                if (wr) V0[c][t] = v_next_q;
                else    V1[c][t] = v_next_q;

                // ---- Emit spike {0,1} ----
                out_spk.write(spk);
            }
        }

        // advance "time" for next call
        bank = !bank;
    }

private:
    // Membrane per (channel, position), in Q(FRAC_BITS)
    ap_int<24> V0[OUT_CH][FEATURE_LEN];
    ap_int<24> V1[OUT_CH][FEATURE_LEN];
    bool bank;
};

} // namespace hls4csnn1d_cblk_sd
#endif // LIF1D_SD_INTEGER_FINAL_H
