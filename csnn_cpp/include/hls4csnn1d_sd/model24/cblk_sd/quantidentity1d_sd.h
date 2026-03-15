// quant_identity_int8.h
#ifndef QUANT_IDENTITY_INT8_H
#define QUANT_IDENTITY_INT8_H

#include <hls_stream.h>
#include <ap_int.h>

namespace hls4csnn1d_cblk_sd {

// -----------------------------------------------------------------------------
// Per-tensor QuantIdentity: spikes {0,1} -> int8 {0, q_one}, q_one = round(1/s)
// act_scale_next_q is s encoded in Q(FRAC_BITS) (per-tensor, zp = 0).
// -----------------------------------------------------------------------------
template<int OUT_CH, int FEATURE_LEN, int FRAC_BITS = 12>
class QuantIdentityPerTensor_Int8 {
public:
    QuantIdentityPerTensor_Int8() {}

    // Forward: in_spk -> out_q (II=1, channel-major C then L)
    void forward(hls::stream<ap_int<8>>& in_spk,     // {0,1}
                 hls::stream<ap_int<8>>& out_q,      // {0, q_one}
                 const ap_int<16> act_scale_next_q)  // Q(FRAC_BITS), s_next > 0
    {
    #pragma HLS INLINE off
        const ap_int<32> ONE_Q = ap_int<32>(1) << FRAC_BITS; // 1.0 in Q
        ap_int<16> q_one_tmp;

        // q_one = round((1<<F)/s) = (ONE_Q + s/2)/s for s>0
        if (act_scale_next_q > 0) {
            q_one_tmp = (ap_int<16>)((ONE_Q + (act_scale_next_q >> 1)) / act_scale_next_q);
        } else {
            q_one_tmp = 127;  // fallback (should not happen)
        }

        // Saturate to int8
        ap_int<8> q_one;
        if (q_one_tmp > 127)         q_one = 127;
        else if (q_one_tmp < -128)   q_one = -128;
        else                         q_one = (ap_int<8>)q_one_tmp;

        for (int c = 0; c < OUT_CH; ++c) {
            for (int t = 0; t < FEATURE_LEN; ++t) {
            #pragma HLS PIPELINE II=1
                ap_int<8> s = in_spk.read(); // 0 or 1
                ap_int<8> q = 0;
                if (s != 0) q = q_one;       // 1 -> q_one
                out_q.write(q);
            }
        }
    }
};

// -----------------------------------------------------------------------------
// Per-channel QuantIdentity: spikes {0,1} -> {0, q_one[c]}
// act_scale_next_q[c] is s_c in Q(FRAC_BITS), zp = 0.
// -----------------------------------------------------------------------------
// template<int OUT_CH, int FEATURE_LEN, int FRAC_BITS = 12>
// class QuantIdentityPerChannel_Int8 {
// public:
//     QuantIdentityPerChannel_Int8() {}

//     void forward(hls::stream<ap_int<8>>& in_spk,         // {0,1}
//                  hls::stream<ap_int<8>>& out_q,          // {0, q_one[c]}
//                  const ap_int<16> act_scale_next_q[OUT_CH]) // per-channel Q(FRAC_BITS)
//     {
//     #pragma HLS INLINE off
//         const ap_int<32> ONE_Q = ap_int<32>(1) << FRAC_BITS;

//         for (int c = 0; c < OUT_CH; ++c) {
//             // q_one[c] = round((1<<F) / s_c)
//             ap_int<16> q_one_tmp;
//             if (act_scale_next_q[c] > 0) {
//                 q_one_tmp = (ap_int<16>)((ONE_Q + (act_scale_next_q[c] >> 1)) / act_scale_next_q[c]);
//             } else {
//                 q_one_tmp = 127;
//             }

//             ap_int<8> q_one;
//             if (q_one_tmp > 127)         q_one = 127;
//             else if (q_one_tmp < -128)   q_one = -128;
//             else                         q_one = (ap_int<8>)q_one_tmp;

//             for (int t = 0; t < FEATURE_LEN; ++t) {
//             #pragma HLS PIPELINE II=1
//                 ap_int<8> s = in_spk.read();  // 0 or 1
//                 ap_int<8> q = 0;
//                 if (s != 0) q = q_one;
//                 out_q.write(q);
//             }
//         }
//     }
// };

} // namespace hls4csnn1d_cblk_sd

#endif // QUANT_IDENTITY_INT8_H
