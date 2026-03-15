#ifndef QCSNN24_SD_H
#define QCSNN24_SD_H

#include <hls_stream.h>
#include <ap_int.h>
#include "../constants24_sd.h"
#include "includeheaders24_sd.h"

#include "conv1d_sd.h"
#include "batchnorm1d_sd.h"
#include "lif1d_integer.h"
#include "maxpool1d_sd.h"
#include "linear1d_sd.h"
#include "quantidentity1d_sd.h"
#include "../utils_sd.h"

namespace hls4csnn1d_cblk_sd {

/* ----------------------------- helpers ----------------------------- */

static inline ap_int8_c gate_abnormal(ap_int<16> sum_norm, ap_int<16> sum_abn) {
#pragma HLS INLINE
    return (sum_abn > sum_norm) ? ap_int8_c(1) : ap_int8_c(0);
}

static inline ap_int8_c argmax_force_abnormal(ap_int<16> s1,
                                               ap_int<16> s2,
                                               ap_int<16> s3) {
#pragma HLS INLINE
    ap_int8_c best = 1;
    ap_int<16> bestv = s1;
    if (s2 > bestv) { bestv = s2; best = 2; }
    if (s3 > bestv) { bestv = s3; best = 3; }
    return best;
}

/* ===================================================================
 * TwoStageQCSNN24_OptionA_SD
 *   - shared trunk (net24) computed once per timestep
 *   - binary head (net2) always runs
 *   - multi head (net4) runs ONLY if abnormal (fabric gate)
 * =================================================================== */
template<int NUM_STEPS>
class QCSNN24_SD {
public:
    QCSNN24_SD()  = default;
    ~QCSNN24_SD() = default;

    void forward(hls::stream<ap_int8_c>& input_stream,
                 hls::stream<ap_int8_c>& pred2_stream,
                 hls::stream<ap_int8_c>& pred4_stream) {
#pragma HLS INTERFACE axis      port=input_stream
#pragma HLS INTERFACE axis      port=pred2_stream
#pragma HLS INTERFACE axis      port=pred4_stream
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

        /* ----------------------------------------------------------
         * 0) Buffer ONE record (180 samples) for replay
         * ---------------------------------------------------------- */
        ap_int8_c in_buf[FIXED_LENGTH1];
#pragma HLS ARRAY_PARTITION variable=in_buf cyclic factor=8 dim=1

        for (int k = 0; k < FIXED_LENGTH1; ++k) {
#pragma HLS PIPELINE II=1
            in_buf[k] = input_stream.read();
        }

        /* ----------------------------------------------------------
         * 1) Reset states ONCE per record (matches utils.reset)
         * ---------------------------------------------------------- */
        // trunk LIFs
        trunk_lif1.reset();
        trunk_lif2.reset();
        trunk_lif3.reset();

        // binary head LIF
        bin_lif.reset();

        /* ----------------------------------------------------------
         * 2) Cache trunk output (480) for each timestep t
         * ---------------------------------------------------------- */
        ap_int8_c body_cache[NUM_STEPS][FIXED_LENGTH7];   // 480
#pragma HLS ARRAY_PARTITION variable=body_cache cyclic factor=8 dim=2

        ap_int<16> sum_bin0 = 0;
        ap_int<16> sum_bin1 = 0;

        /* ==========================================================
         * Declare ALL streams ONCE (Option B)
         * ========================================================== */

        // replay input stream into trunk each timestep
        hls::stream<ap_int8_c> s_in("s_in");
#pragma HLS STREAM variable=s_in depth=FIXED_LENGTH1

        // trunk pipeline streams
        hls::stream<ap_int8_c> s0("s0");   // conv1 -> bn1
#pragma HLS STREAM variable=s0 depth=FIXED_LENGTH2
        hls::stream<ap_int8_c> s1("s1");   // bn1 -> lif1
#pragma HLS STREAM variable=s1 depth=FIXED_LENGTH2
        hls::stream<ap_int8_c> s2("s2");   // lif1 -> mp1
#pragma HLS STREAM variable=s2 depth=FIXED_LENGTH2
        hls::stream<ap_int8_c> s3("s3");   // mp1 -> qi2
#pragma HLS STREAM variable=s3 depth=FIXED_LENGTH3

        hls::stream<ap_int8_c> s4("s4");   // qi2 -> conv2
#pragma HLS STREAM variable=s4 depth=FIXED_LENGTH3
        hls::stream<ap_int8_c> s5("s5");   // conv2 -> bn2
#pragma HLS STREAM variable=s5 depth=FIXED_LENGTH4
        hls::stream<ap_int8_c> s6("s6");   // bn2 -> lif2
#pragma HLS STREAM variable=s6 depth=FIXED_LENGTH4
        hls::stream<ap_int8_c> s7("s7");   // lif2 -> mp2
#pragma HLS STREAM variable=s7 depth=FIXED_LENGTH4
        hls::stream<ap_int8_c> s8("s8");   // mp2 -> qi3
#pragma HLS STREAM variable=s8 depth=FIXED_LENGTH5

        hls::stream<ap_int8_c> s9("s9");   // qi3 -> conv3
#pragma HLS STREAM variable=s9 depth=FIXED_LENGTH5
        hls::stream<ap_int8_c> s10("s10"); // conv3 -> bn3
#pragma HLS STREAM variable=s10 depth=FIXED_LENGTH6
        hls::stream<ap_int8_c> s11("s11"); // bn3 -> lif3
#pragma HLS STREAM variable=s11 depth=FIXED_LENGTH6
        hls::stream<ap_int8_c> s12("s12"); // lif3 -> mp3
#pragma HLS STREAM variable=s12 depth=FIXED_LENGTH6
        hls::stream<ap_int8_c> s_body("s_body"); // mp3 output => 480
#pragma HLS STREAM variable=s_body depth=FIXED_LENGTH7

        // binary head streams (480 -> 2)
        hls::stream<ap_int8_c> s_bin_qi("s_bin_qi");
#pragma HLS STREAM variable=s_bin_qi depth=FIXED_LENGTH7
        hls::stream<ap_int8_c> s_bin_fc("s_bin_fc");
#pragma HLS STREAM variable=s_bin_fc depth=FIXED_LENGTH7
        hls::stream<ap_int8_c> s_bin_lif_in("s_bin_lif_in");
#pragma HLS STREAM variable=s_bin_lif_in depth=2
        hls::stream<ap_int8_c> s_bin_out("s_bin_out");
#pragma HLS STREAM variable=s_bin_out depth=2

        // multi head streams (480 -> 128 -> 4)
        hls::stream<ap_int8_c> s_m_qi1("s_m_qi1");
#pragma HLS STREAM variable=s_m_qi1 depth=FIXED_LENGTH7
        hls::stream<ap_int8_c> s_m_fc1("s_m_fc1");
#pragma HLS STREAM variable=s_m_fc1 depth=FIXED_LENGTH7
        hls::stream<ap_int8_c> s_m_lif1_out("s_m_lif1_out");
#pragma HLS STREAM variable=s_m_lif1_out depth=FIXED_LENGTH8

        hls::stream<ap_int8_c> s_m_qi2("s_m_qi2");
#pragma HLS STREAM variable=s_m_qi2 depth=FIXED_LENGTH8
        hls::stream<ap_int8_c> s_m_fc2("s_m_fc2");
#pragma HLS STREAM variable=s_m_fc2 depth=FIXED_LENGTH8
        hls::stream<ap_int8_c> s_m_lif2_in("s_m_lif2_in");
#pragma HLS STREAM variable=s_m_lif2_in depth=FIXED_LENGTH9
        hls::stream<ap_int8_c> s_m_out("s_m_out");
#pragma HLS STREAM variable=s_m_out depth=FIXED_LENGTH9

        /* ----------------------------------------------------------
         * 3) STAGE-1: trunk + binary head + cache body_t
         * ---------------------------------------------------------- */
    STAGE1_LOOP:
        for (int t = 0; t < NUM_STEPS; ++t) {

            // (a) refill s_in with same record
            for (int k = 0; k < FIXED_LENGTH1; ++k) {
#pragma HLS PIPELINE II=1
                s_in.write(in_buf[k]);
            }

            // (b) trunk (net24)
            trunk_conv1.forward(
                s_in, s0,
                qcsnet24_cblk1_qconv1d_weights,
                qcsnet24_cblk1_qconv1d_scale_multiplier,
                qcsnet24_cblk1_qconv1d_right_shift,
                qcsnet24_cblk1_qconv1d_bias,
                qcsnet24_cblk1_qconv1d_input_zero_point,
                qcsnet24_cblk1_qconv1d_weight_sum);

            trunk_bn1.forward(
                s0, s1,
                qcsnet24_cblk1_batch_norm_weight,
                qcsnet24_cblk1_batch_norm_bias,
                qcsnet24_cblk1_batch_norm_scale_multiplier,
                qcsnet24_cblk1_batch_norm_right_shift);

            trunk_lif1.forward(
                s1, s2,
                qcsnet24_cblk1_leaky_beta_int,
                qcsnet24_cblk1_leaky_theta_int,
                qcsnet24_cblk1_leaky_scale_int);

            trunk_mp1.forward(s2, s3);

            trunk_qi2.forward(s3, s4, qcsnet24_cblk2_input_act_scale_int);

            trunk_conv2.forward(
                s4, s5,
                qcsnet24_cblk2_qconv1d_weights,
                qcsnet24_cblk2_qconv1d_scale_multiplier,
                qcsnet24_cblk2_qconv1d_right_shift,
                qcsnet24_cblk2_qconv1d_bias,
                qcsnet24_cblk2_qconv1d_input_zero_point,
                qcsnet24_cblk2_qconv1d_weight_sum);

            trunk_bn2.forward(
                s5, s6,
                qcsnet24_cblk2_batch_norm_weight,
                qcsnet24_cblk2_batch_norm_bias,
                qcsnet24_cblk2_batch_norm_scale_multiplier,
                qcsnet24_cblk2_batch_norm_right_shift);

            trunk_lif2.forward(
                s6, s7,
                qcsnet24_cblk2_leaky_beta_int,
                qcsnet24_cblk2_leaky_theta_int,
                qcsnet24_cblk2_leaky_scale_int);

            trunk_mp2.forward(s7, s8);

            trunk_qi3.forward(s8, s9, qcsnet24_cblk3_input_act_scale_int);

            trunk_conv3.forward(
                s9, s10,
                qcsnet24_cblk3_qconv1d_weights,
                qcsnet24_cblk3_qconv1d_scale_multiplier,
                qcsnet24_cblk3_qconv1d_right_shift,
                qcsnet24_cblk3_qconv1d_bias,
                qcsnet24_cblk3_qconv1d_input_zero_point,
                qcsnet24_cblk3_qconv1d_weight_sum);

            trunk_bn3.forward(
                s10, s11,
                qcsnet24_cblk3_batch_norm_weight,
                qcsnet24_cblk3_batch_norm_bias,
                qcsnet24_cblk3_batch_norm_scale_multiplier,
                qcsnet24_cblk3_batch_norm_right_shift);

            trunk_lif3.forward(
                s11, s12,
                qcsnet24_cblk3_leaky_beta_int,
                qcsnet24_cblk3_leaky_theta_int,
                qcsnet24_cblk3_leaky_scale_int);

            trunk_mp3.forward(s12, s_body); // 480 output

            // (c) cache body + feed binary head
            for (int i = 0; i < FIXED_LENGTH7; ++i) {
#pragma HLS PIPELINE II=1
                ap_int8_c v = s_body.read();
                body_cache[t][i] = v;
                s_bin_qi.write(v);
            }

            // binary head (net2): qi -> linear -> lif
            bin_qi.forward(s_bin_qi, s_bin_fc, qcsnet2_lblk1_input_act_scale_int);

            bin_fc.forward(
                s_bin_fc, s_bin_lif_in,
                qcsnet2_lblk1_qlinear_weights,
                qcsnet2_lblk1_qlinear_scale_multiplier,
                qcsnet2_lblk1_qlinear_right_shift,
                qcsnet2_lblk1_qlinear_bias,
                qcsnet2_lblk1_qlinear_input_zero_point,
                qcsnet2_lblk1_qlinear_weight_sum);

            bin_lif.forward(
                s_bin_lif_in, s_bin_out,
                qcsnet2_lblk1_leaky_beta_int,
                qcsnet2_lblk1_leaky_theta_int,
                qcsnet2_lblk1_leaky_scale_int);

            ap_int8_c b0 = s_bin_out.read();
            ap_int8_c b1 = s_bin_out.read();

            sum_bin0 += (ap_int<16>)b0;
            sum_bin1 += (ap_int<16>)b1;
        }

        // (d) gate decision in hardware
        ap_int8_c pred2 = gate_abnormal(sum_bin0, sum_bin1);
        pred2_stream.write(pred2);

        if (pred2 == 0) {
            pred4_stream.write(ap_int8_c(0)); // Normal
            return;
        }

        /* ----------------------------------------------------------
         * 4) STAGE-2: multi head ONLY if abnormal
         * ---------------------------------------------------------- */
        multi_lif1.reset();
        multi_lif2.reset();

        ap_int<16> sum4_0 = 0, sum4_1 = 0, sum4_2 = 0, sum4_3 = 0;

    STAGE2_LOOP:
        for (int t = 0; t < NUM_STEPS; ++t) {

            // feed cached body_t (480) into multi head
            for (int i = 0; i < FIXED_LENGTH7; ++i) {
#pragma HLS PIPELINE II=1
                s_m_qi1.write(body_cache[t][i]);
            }

            // multi head: qi -> fc(480->128) -> lif -> qi -> fc(128->4) -> lif
            multi_qi1.forward(s_m_qi1, s_m_fc1, qcsnet4_lblk1_input_act_scale_int);

            multi_fc1.forward(
                s_m_fc1, s_m_lif1_out,
                qcsnet4_lblk1_qlinear_weights,
                qcsnet4_lblk1_qlinear_scale_multiplier,
                qcsnet4_lblk1_qlinear_right_shift,
                qcsnet4_lblk1_qlinear_bias,
                qcsnet4_lblk1_qlinear_input_zero_point,
                qcsnet4_lblk1_qlinear_weight_sum);

            multi_lif1.forward(
                s_m_lif1_out, s_m_qi2,
                qcsnet4_lblk1_leaky_beta_int,
                qcsnet4_lblk1_leaky_theta_int,
                qcsnet4_lblk1_leaky_scale_int);

            multi_qi2.forward(s_m_qi2, s_m_fc2, qcsnet4_lblk2_input_act_scale_int);

            multi_fc2.forward(
                s_m_fc2, s_m_lif2_in,
                qcsnet4_lblk2_qlinear_weights,
                qcsnet4_lblk2_qlinear_scale_multiplier,
                qcsnet4_lblk2_qlinear_right_shift,
                qcsnet4_lblk2_qlinear_bias,
                qcsnet4_lblk2_qlinear_input_zero_point,
                qcsnet4_lblk2_qlinear_weight_sum);

            multi_lif2.forward(
                s_m_lif2_in, s_m_out,
                qcsnet4_lblk2_leaky_beta_int,
                qcsnet4_lblk2_leaky_theta_int,
                qcsnet4_lblk2_leaky_scale_int);

            ap_int8_c y0 = s_m_out.read();
            ap_int8_c y1 = s_m_out.read();
            ap_int8_c y2 = s_m_out.read();
            ap_int8_c y3 = s_m_out.read();

            sum4_0 += (ap_int<16>)y0;
            sum4_1 += (ap_int<16>)y1;
            sum4_2 += (ap_int<16>)y2;
            sum4_3 += (ap_int<16>)y3;
        }

        // force abnormal => choose among {1,2,3}
        ap_int8_c pred4 = argmax_force_abnormal(sum4_1, sum4_2, sum4_3);
        pred4_stream.write(pred4);
    }

private:
    /* ====================== TRUNK (net24) ====================== */
    Conv1D_SD<1, 16, 3, 1, FIXED_LENGTH1>                trunk_conv1;
    BatchNorm1D_SD<16, 178>                              trunk_bn1;
    LIF1D_SD_Integer<16, 178>                            trunk_lif1;
    MaxPool1D_SD<2, 2, 16, 178>                          trunk_mp1;

    QuantIdentityPerTensor_Int8<16, 89>                  trunk_qi2;
    Conv1D_SD<16, 16, 3, 1, 89>                          trunk_conv2;
    BatchNorm1D_SD<16, 87>                               trunk_bn2;
    LIF1D_SD_Integer<16, 87>                             trunk_lif2;
    MaxPool1D_SD<2, 2, 16, 87>                           trunk_mp2;

    QuantIdentityPerTensor_Int8<16, 43>                  trunk_qi3;
    Conv1D_SD<16, 24, 3, 1, 43>                          trunk_conv3;
    BatchNorm1D_SD<24, 41>                               trunk_bn3;
    LIF1D_SD_Integer<24, 41>                             trunk_lif3;
    MaxPool1D_SD<2, 2, 24, 41>                           trunk_mp3; // output 24x20=480

    /* ==================== BINARY HEAD (net2) =================== */
    QuantIdentityPerTensor_Int8<24, 20>                  bin_qi;    // 24x20 flattened stream
    Linear1D_SD<FIXED_LENGTH7, 2>                        bin_fc;    // 480->2
    LIF1D_SD_Integer<2, 1>                               bin_lif;

    /* ===================== MULTI HEAD (net4) =================== */
    QuantIdentityPerTensor_Int8<24, 20>                  multi_qi1;  // 480
    Linear1D_SD<FIXED_LENGTH7, FIXED_LENGTH8>            multi_fc1;  // 480->128
    LIF1D_SD_Integer<FIXED_LENGTH8, 1>                   multi_lif1;

    QuantIdentityPerTensor_Int8<FIXED_LENGTH8, 1>        multi_qi2;
    Linear1D_SD<FIXED_LENGTH8, FIXED_LENGTH9>            multi_fc2;  // 128->4
    LIF1D_SD_Integer<FIXED_LENGTH9, 1>                   multi_lif2;
};

} // namespace hls4csnn1d_cblk_sd
#endif
