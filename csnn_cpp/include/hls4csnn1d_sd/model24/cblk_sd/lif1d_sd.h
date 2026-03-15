#ifndef LIF1D_SD_INTEGER_H
#define LIF1D_SD_INTEGER_H

#include <hls_stream.h>
#include <ap_int.h>
#include "../constants24_sd.h"

namespace hls4csnn1d_cblk_sd {

// snnTorch defaults we mirror:
// - reset_delay = true  (delayed reset)
// - reset_mechanism = "subtract" (subtract θ from next step)
// - fire uses STRICT '>' comparison
// - beta is clamped to [0, 1]
// - supports graded_spikes_factor
enum ResetMode { RESET_SUBTRACT = 0, RESET_ZERO = 1 };

template<int OUT_CH, int FEATURE_LEN, int FRAC_BITS = 16, ResetMode RESET_MODE = RESET_SUBTRACT>
class LIF1D_SD {
public:
    LIF1D_SD() { reset(); }

    // Call once per new sample (per batch element)
    void reset() {
    #pragma HLS INLINE
        for (int c = 0; c < OUT_CH; ++c) {
        #pragma HLS UNROLL
            V_q[c] = 0;         // membrane in Q(FRAC_BITS)
        }
    }

    // Integer-only forward, matching snnTorch Leaky (reset_delay=True, subtract):
    // beta_q, theta_q, scale_q, graded_spikes_factor_q are all Q(FRAC_BITS).
    // Input stream order: channel-major over time: for c in C: for t in T: write x[c,t]
    void forward(hls::stream<ap_int<8>>& in,
                 hls::stream<ap_int<8>>& out,
                 const ap_int<16> beta_q,
                 const ap_int<16> theta_q,
                 const ap_int<16> scale_q,
                 const ap_int<16> graded_spikes_factor_q = ap_int<16>(1 << FRAC_BITS)) { // Default 1.0 in Q format
    #pragma HLS INLINE off

        const int SHIFT = FRAC_BITS;
        const ap_int<16> BETA_MAX = (1 << FRAC_BITS);  // 1.0 in Q format
        
        // Clamp beta to [0, 1] as snnTorch does
        ap_int<16> beta_clamped;
        if (beta_q < 0) {
            beta_clamped = 0;
        } else if (beta_q > BETA_MAX) {
            beta_clamped = BETA_MAX;
        } else {
            beta_clamped = beta_q;
        }

        for (int c = 0; c < OUT_CH; ++c) {
            // Load membrane state from previous sample
            ap_int<24> v = V_q[c];    // Q(FRAC_BITS) - membrane at end of previous sample
            
            // Initialize reset for first timestep (no spike before first timestep)
            ap_int<16> r_delayed = (ap_int<16>)0;  // Q(FRAC_BITS)

            for (int t = 0; t < FEATURE_LEN; ++t) {
            #pragma HLS PIPELINE II=1
                const ap_int<8> x_i = in.read();  // int8 from BN

                // ---- integrate using DELAYED reset from previous timestep ----
                // term A: beta * v  (Q*Q -> Q(2*FRAC) -> round >> FRAC -> Q)
                ap_int<40> prod = (ap_int<40>)beta_clamped * (ap_int<40>)v;
                const ap_int<40> ROUND = ((ap_int<40>)1) << (SHIFT - 1);
                const ap_int<40> prod_adj = (prod >= 0) ? (prod + ROUND) : (prod - ROUND);
                ap_int<24> v_beta = (ap_int<24>)(prod_adj >> SHIFT); // Q(FRAC_BITS)

                // term B: x * scale  (int8 * Q -> Q)
                ap_int<24> x_scaled = (ap_int<24>)x_i * (ap_int<24>)scale_q; // Q(FRAC_BITS)

                // term C: subtract delayed reset * theta  (Q * Q -> round >> SHIFT)
                // r_delayed was computed at END of previous timestep
                ap_int<40> reset_prod = (ap_int<40>)r_delayed * (ap_int<40>)theta_q;
                const ap_int<40> reset_adj = (reset_prod >= 0) ? (reset_prod + ROUND) : (reset_prod - ROUND);
                ap_int<24> sub_reset = (ap_int<24>)(reset_adj >> SHIFT);

                // u_t = beta*u_{t-1} + x_t - r_{t-1}*theta
                ap_int<24> v_next = v_beta + x_scaled - sub_reset; // Q(FRAC_BITS)

                // ---- Generate spike (matches fire() in snnTorch) ----
                // spike_raw = spike_grad(mem - threshold) in inference acts as Heaviside
                const bool spike_raw = (v_next > theta_q);  // STRICT >
                
                // ---- Compute reset signal for NEXT timestep (delayed reset) ----
                // This implements reset_delay=True: spike at t causes reset at t+1
                const ap_uint<1> mem_exceeds_threshold = (v_next > theta_q);
                r_delayed = mem_exceeds_threshold ? (ap_int<16>)(1 << SHIFT) : (ap_int<16>)0;
                
                // Update membrane for next iteration (NO immediate reset applied)
                v = v_next;
                // In Q(FRAC_BITS): spike * graded_spikes_factor_q >> SHIFT
                ap_int<8> spike_output;
                if (spike_raw) {
                    // For hardware efficiency, if graded_spikes_factor_q == (1 << SHIFT), output 1
                    // Otherwise scale and convert to 8-bit range
                    ap_int<16> one_q = (ap_int<16>)(1 << SHIFT);
                    if (graded_spikes_factor_q == one_q) {
                        spike_output = (ap_int<8>)1;  // Standard binary spike
                    } else {
                        // Scale spike: 1.0 * graded_spikes_factor in Q format
                        // Convert to int8 range: Q(FRAC_BITS) >> SHIFT
                        ap_int<16> scaled = graded_spikes_factor_q >> SHIFT;
                        ap_int<8> clamped = (scaled > 127) ? (ap_int<8>)127 : ((scaled < -128) ? (ap_int<8>)(-128) : (ap_int<8>)scaled);
                        spike_output = clamped;
                    }
                } else {
                    spike_output = (ap_int<8>)0;
                }

                // DO NOT apply immediate reset here (reset_delay=True in snnTorch)
                v = v_next;

                // emit spike
                out.write(spike_output);
            }
            V_q[c] = v; // store final membrane for channel c
        }
    }

private:
    // Q(FRAC_BITS) membrane per channel (24-bit gives headroom)
    ap_int<24> V_q[OUT_CH];
};

} // namespace hls4csnn1d_cblk_sd
#endif // LIF1D_SD_INTEGER_H




// #ifndef LIF1D_SD_INTEGER_H
// #define LIF1D_SD_INTEGER_H

// #include <hls_stream.h>
// #include <ap_int.h>
// #include "../constants_sd.h"

// namespace hls4csnn1d_cblk_sd {

// // snnTorch defaults we mirror:
// // - reset_delay = true  (delayed reset)
// // - reset_mechanism = "subtract" (subtract θ from next step)
// // - fire uses STRICT '>' comparison
// enum ResetMode { RESET_SUBTRACT = 0, RESET_ZERO = 1 };

// template<int OUT_CH, int FEATURE_LEN, int FRAC_BITS = 12, ResetMode RESET_MODE = RESET_SUBTRACT>
// class LIF1D_SD {
// public:
//     LIF1D_SD() { reset(); }

//     // Call once per new sample (per batch element)
//     void reset() {
//     #pragma HLS INLINE
//         for (int c = 0; c < OUT_CH; ++c) {
//         #pragma HLS UNROLL
//             V_q[c] = 0;         // membrane in Q(FRAC_BITS)
//         }
//     }

//     // Integer-only forward, matching snnTorch Leaky (reset_delay=True, subtract):
//     // beta_q, theta_q, scale_q are all Q(FRAC_BITS).
//     // Input stream order: channel-major over time: for c in C: for t in T: write x[c,t]
//     void forward(hls::stream<ap_int<8>>& in,
//                  hls::stream<ap_int<8>>& out,
//                  const ap_int<16> beta_q,
//                  const ap_int<16> theta_q,
//                  const ap_int<16> scale_q) {
//     #pragma HLS INLINE off

//         const int SHIFT = FRAC_BITS;

//         for (int c = 0; c < OUT_CH; ++c) {
//             // delayed reset uses previous membrane (u_{t-1})
//             ap_int<24> v = V_q[c];    // Q(FRAC_BITS)

//             for (int t = 0; t < FEATURE_LEN; ++t) {
//             #pragma HLS PIPELINE II=1
//                 const ap_int<8> x_i = in.read();  // int8 from BN

//                 // r_{t-1} = 1 if v > theta (STRICT '>')
//                 const ap_uint<1> r_prev = (v > theta_q);

//                 // ---- integrate in Q(FRAC_BITS) exactly ----
//                 // term A: beta * v  (Q*Q -> Q(2*FRAC) -> round >> FRAC -> Q)
//                 ap_int<40> prod = (ap_int<40>)beta_q * (ap_int<40>)v;
//                 const ap_int<40> ROUND = ((ap_int<40>)1) << (SHIFT - 1);
//                 const ap_int<40> prod_adj = (prod >= 0) ? (prod + ROUND) : (prod - ROUND);
//                 ap_int<24> v_beta = (ap_int<24>)(prod_adj >> SHIFT); // Q(FRAC_BITS)

//                 // term B: x * scale  (int8 * Q -> Q)
//                 ap_int<24> x_scaled = (ap_int<24>)x_i * (ap_int<24>)scale_q; // Q(FRAC_BITS)

//                 // term C: subtract delayed reset * theta  (Q)
//                 ap_int<24> sub_reset = r_prev ? (ap_int<24>)theta_q : (ap_int<24>)0;

//                 // u_t = beta*u_{t-1} + x_t - r_{t-1}*theta
//                 ap_int<24> v_next = v_beta + x_scaled - sub_reset; // Q(FRAC_BITS)

//                 // spike now: s_t = 1{u_t > theta} (STRICT)
//                 const bool spike = (v_next > theta_q);

//                 // DO NOT apply immediate reset here (reset_delay=True in snnTorch)
//                 v = v_next;

//                 // emit binary spike {0,1}
//                 out.write(spike ? ap_int<8>(1) : ap_int<8>(0));
//             }
//             V_q[c] = v; // store final membrane for channel c
//         }
//     }

// private:
//     // Q(FRAC_BITS) membrane per channel (24-bit gives headroom)
//     ap_int<24> V_q[OUT_CH];
// };

// } // namespace hls4csnn1d_cblk_sd
// #endif // LIF1D_SD_INTEGER_H




