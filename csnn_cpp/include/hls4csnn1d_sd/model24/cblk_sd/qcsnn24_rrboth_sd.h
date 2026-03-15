#ifndef QCSNN24_RRBOTH_SD_H
#define QCSNN24_RRBOTH_SD_H

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

/* ----------------------------- constants ----------------------------- */
static const int RR_DIM = 4;
static const int SIGNAL_LEN = FIXED_LENGTH1;                    // 180
// ===== CHANGE 1: Input is now signal only =====
static const int INPUT_LEN = SIGNAL_LEN;                        // 180 (was 184)
static const int TRUNK_OUT = FIXED_LENGTH7;                     // 480
static const int STAGE1_IN = TRUNK_OUT + RR_DIM;                // 484
static const int STAGE2_IN = TRUNK_OUT + RR_DIM;                // 484

/* ----------------------------- helpers ----------------------------- */

static inline ap_int8_c gate_abnormal(ap_int<16> sum_norm, ap_int<16> sum_abn) {
#pragma HLS INLINE
    return (sum_abn > sum_norm) ? ap_int8_c(1) : ap_int8_c(0);
}

static inline ap_int8_c argmax4(ap_int<16> s0,
                                ap_int<16> s1,
                                ap_int<16> s2,
                                ap_int<16> s3) {
#pragma HLS INLINE
    ap_int8_c best = 0;
    ap_int<16> bestv = s0;
    if (s1 > bestv) { bestv = s1; best = 1; }
    if (s2 > bestv) { bestv = s2; best = 2; }
    if (s3 > bestv) { bestv = s3; best = 3; }
    return best;
}

/* ===================================================================
 * TwoStageQCSNN24_RRBoth_SD
 *   - shared trunk (net24) computed once per timestep
 *   - binary head (net2) gets trunk + RR features (484)
 *   - multi head (net4) gets trunk + RR features (484)
 *   
 *   ===== CHANGE 2: RR features now passed separately, bypass QuantIdentity =====
 * =================================================================== */
template<int NUM_STEPS>
class QCSNN24_RRBOTH_SD {
public:
    QCSNN24_RRBOTH_SD()  = default;
    ~QCSNN24_RRBOTH_SD() = default;

    // ===== CHANGE 3: New signature with separate RR inputs =====
    void forward(hls::stream<ap_int8_c>& input_stream,    // 180 signal samples only
                 const ap_int8_c rr_stage1[RR_DIM],       // RR quantized with Stage 1 scale
                 const ap_int8_c rr_stage2[RR_DIM],       // RR quantized with Stage 2 scale
                 hls::stream<ap_int8_c>& pred2_stream,
                 hls::stream<ap_int8_c>& pred4_stream) {
#pragma HLS INTERFACE axis      port=input_stream
#pragma HLS INTERFACE axis      port=pred2_stream
#pragma HLS INTERFACE axis      port=pred4_stream
#pragma HLS INTERFACE s_axilite port=rr_stage1 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=rr_stage2 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

        /* ----------------------------------------------------------
         * 0) Buffer signal samples (180) and copy RR to local arrays
         * ---------------------------------------------------------- */
        ap_int8_c sig_buf[SIGNAL_LEN];   // 180 signal samples
#pragma HLS ARRAY_PARTITION variable=sig_buf cyclic factor=8 dim=1

        ap_int8_c rr_s1[RR_DIM];         // RR for Stage 1 (pre-quantized)
#pragma HLS ARRAY_PARTITION variable=rr_s1 complete dim=1

        ap_int8_c rr_s2[RR_DIM];         // RR for Stage 2 (pre-quantized)
#pragma HLS ARRAY_PARTITION variable=rr_s2 complete dim=1

        // Read signal (180)
        for (int k = 0; k < SIGNAL_LEN; ++k) {
#pragma HLS PIPELINE II=1
            sig_buf[k] = input_stream.read();
        }

        // Copy RR features (already quantized with correct scales)
        for (int k = 0; k < RR_DIM; ++k) {
#pragma HLS UNROLL
            rr_s1[k] = rr_stage1[k];
            rr_s2[k] = rr_stage2[k];
        }

        /* ----------------------------------------------------------
         * 1) Reset states ONCE per record (matches utils.reset)
         * ---------------------------------------------------------- */
        trunk_lif1.reset();
        trunk_lif2.reset();
        trunk_lif3.reset();
        bin_lif.reset();

        /* ----------------------------------------------------------
         * 2) Cache trunk output (480) for each timestep t
         * ---------------------------------------------------------- */
        ap_int8_c body_cache[NUM_STEPS][TRUNK_OUT];   // 480
#pragma HLS ARRAY_PARTITION variable=body_cache cyclic factor=8 dim=2

        ap_int<16> sum_bin0 = 0;
        ap_int<16> sum_bin1 = 0;

        /* ==========================================================
         * Declare ALL streams ONCE
         * ========================================================== */

        hls::stream<ap_int8_c> s_in("s_in");
#pragma HLS STREAM variable=s_in depth=SIGNAL_LEN

        // trunk pipeline streams
        hls::stream<ap_int8_c> s0("s0");
#pragma HLS STREAM variable=s0 depth=FIXED_LENGTH2
        hls::stream<ap_int8_c> s1("s1");
#pragma HLS STREAM variable=s1 depth=FIXED_LENGTH2
        hls::stream<ap_int8_c> s2("s2");
#pragma HLS STREAM variable=s2 depth=FIXED_LENGTH2
        hls::stream<ap_int8_c> s3("s3");
#pragma HLS STREAM variable=s3 depth=FIXED_LENGTH3

        hls::stream<ap_int8_c> s4("s4");
#pragma HLS STREAM variable=s4 depth=FIXED_LENGTH3
        hls::stream<ap_int8_c> s5("s5");
#pragma HLS STREAM variable=s5 depth=FIXED_LENGTH4
        hls::stream<ap_int8_c> s6("s6");
#pragma HLS STREAM variable=s6 depth=FIXED_LENGTH4
        hls::stream<ap_int8_c> s7("s7");
#pragma HLS STREAM variable=s7 depth=FIXED_LENGTH4
        hls::stream<ap_int8_c> s8("s8");
#pragma HLS STREAM variable=s8 depth=FIXED_LENGTH5

        hls::stream<ap_int8_c> s9("s9");
#pragma HLS STREAM variable=s9 depth=FIXED_LENGTH5
        hls::stream<ap_int8_c> s10("s10");
#pragma HLS STREAM variable=s10 depth=FIXED_LENGTH6
        hls::stream<ap_int8_c> s11("s11");
#pragma HLS STREAM variable=s11 depth=FIXED_LENGTH6
        hls::stream<ap_int8_c> s12("s12");
#pragma HLS STREAM variable=s12 depth=FIXED_LENGTH6
        hls::stream<ap_int8_c> s_body("s_body");
#pragma HLS STREAM variable=s_body depth=TRUNK_OUT

        // ===== CHANGE 4: Binary head streams - QuantIdentity only processes trunk (480) =====
        hls::stream<ap_int8_c> s_bin_qi_in("s_bin_qi_in");   // Trunk spikes (480)
#pragma HLS STREAM variable=s_bin_qi_in depth=TRUNK_OUT
        hls::stream<ap_int8_c> s_bin_qi_out("s_bin_qi_out"); // Trunk remapped (480)
#pragma HLS STREAM variable=s_bin_qi_out depth=TRUNK_OUT
        hls::stream<ap_int8_c> s_bin_fc("s_bin_fc");         // Trunk + RR (484)
#pragma HLS STREAM variable=s_bin_fc depth=STAGE1_IN
        hls::stream<ap_int8_c> s_bin_lif_in("s_bin_lif_in");
#pragma HLS STREAM variable=s_bin_lif_in depth=2
        hls::stream<ap_int8_c> s_bin_out("s_bin_out");
#pragma HLS STREAM variable=s_bin_out depth=2

        // ===== CHANGE 5: Multi head streams - QuantIdentity only processes trunk (480) =====
        hls::stream<ap_int8_c> s_m_qi1_in("s_m_qi1_in");     // Trunk spikes (480)
#pragma HLS STREAM variable=s_m_qi1_in depth=TRUNK_OUT
        hls::stream<ap_int8_c> s_m_qi1_out("s_m_qi1_out");   // Trunk remapped (480)
#pragma HLS STREAM variable=s_m_qi1_out depth=TRUNK_OUT
        hls::stream<ap_int8_c> s_m_fc1("s_m_fc1");           // Trunk + RR (484)
#pragma HLS STREAM variable=s_m_fc1 depth=STAGE2_IN
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
         * 3) STAGE-1: trunk + binary head (with RR bypass)
         * ---------------------------------------------------------- */
    STAGE1_LOOP:
        for (int t = 0; t < NUM_STEPS; ++t) {

            // (a) refill s_in with signal only (180)
            for (int k = 0; k < SIGNAL_LEN; ++k) {
#pragma HLS PIPELINE II=1
                s_in.write(sig_buf[k]);
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

            trunk_mp3.forward(s12, s_body);

            // ===== CHANGE 6: Cache body + feed trunk to QuantIdentity (480 only) =====
            for (int i = 0; i < TRUNK_OUT; ++i) {
#pragma HLS PIPELINE II=1
                ap_int8_c v = s_body.read();
                body_cache[t][i] = v;
                s_bin_qi_in.write(v);   // Trunk spikes (480) → QuantIdentity
            }

            // ===== CHANGE 7: QuantIdentity processes trunk only (480) =====
            bin_qi.forward(s_bin_qi_in, s_bin_qi_out, qcsnet2_lblk1_input_act_scale_int);

            // ===== CHANGE 8: Concatenate remapped trunk + pre-quantized RR =====
            // First: trunk (480 remapped values)
            for (int i = 0; i < TRUNK_OUT; ++i) {
#pragma HLS PIPELINE II=1
                s_bin_fc.write(s_bin_qi_out.read());
            }
            // Then: RR (4 pre-quantized values, bypass QuantIdentity)
            for (int i = 0; i < RR_DIM; ++i) {
#pragma HLS PIPELINE II=1
                s_bin_fc.write(rr_s1[i]);  // Stage 1 RR
            }

            // binary head (net2): linear(484->2) -> lif
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

        // (d) gate decision
        ap_int8_c pred2 = gate_abnormal(sum_bin0, sum_bin1);
        pred2_stream.write(pred2);

        if (pred2 == 0) {
            pred4_stream.write(ap_int8_c(0));
            return;
        }

        /* ----------------------------------------------------------
         * 4) STAGE-2: multi head with RR bypass
         * ---------------------------------------------------------- */
        multi_lif1.reset();
        multi_lif2.reset();

        ap_int<16> sum4_0 = 0, sum4_1 = 0, sum4_2 = 0, sum4_3 = 0;

    STAGE2_LOOP:
        for (int t = 0; t < NUM_STEPS; ++t) {

            // ===== CHANGE 9: Feed cached trunk to QuantIdentity (480 only) =====
            for (int i = 0; i < TRUNK_OUT; ++i) {
#pragma HLS PIPELINE II=1
                s_m_qi1_in.write(body_cache[t][i]);
            }

            // ===== CHANGE 10: QuantIdentity processes trunk only (480) =====
            multi_qi1.forward(s_m_qi1_in, s_m_qi1_out, qcsnet4_lblk1_input_act_scale_int);

            // ===== CHANGE 11: Concatenate remapped trunk + pre-quantized RR =====
            // First: trunk (480 remapped values)
            for (int i = 0; i < TRUNK_OUT; ++i) {
#pragma HLS PIPELINE II=1
                s_m_fc1.write(s_m_qi1_out.read());
            }
            // Then: RR (4 pre-quantized values, bypass QuantIdentity)
            for (int i = 0; i < RR_DIM; ++i) {
#pragma HLS PIPELINE II=1
                s_m_fc1.write(rr_s2[i]);  // Stage 2 RR
            }

            // multi head: fc(484->128) -> lif -> qi -> fc(128->4) -> lif
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

        ap_int8_c pred4 = argmax4(sum4_0, sum4_1, sum4_2, sum4_3);
        pred4_stream.write(pred4);
    }

private:
    /* ====================== TRUNK (net24) ====================== */
    Conv1D_SD<1, 16, 3, 1, SIGNAL_LEN>                   trunk_conv1;
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
    MaxPool1D_SD<2, 2, 24, 41>                           trunk_mp3;

    /* ==================== BINARY HEAD (net2) =================== */
    // ===== CHANGE 12: QuantIdentity now processes trunk only (480) =====
    QuantIdentityPerTensor_Int8<TRUNK_OUT, 1>            bin_qi;       // 480 (was 484)
    Linear1D_SD<STAGE1_IN, 2>                            bin_fc;       // 484->2 (unchanged)
    LIF1D_SD_Integer<2, 1>                               bin_lif;

    /* ===================== MULTI HEAD (net4) =================== */
    // ===== CHANGE 13: QuantIdentity now processes trunk only (480) =====
    QuantIdentityPerTensor_Int8<TRUNK_OUT, 1>            multi_qi1;    // 480 (was 484)
    Linear1D_SD<STAGE2_IN, FIXED_LENGTH8>                multi_fc1;    // 484->128 (unchanged)
    LIF1D_SD_Integer<FIXED_LENGTH8, 1>                   multi_lif1;

    QuantIdentityPerTensor_Int8<FIXED_LENGTH8, 1>        multi_qi2;
    Linear1D_SD<FIXED_LENGTH8, FIXED_LENGTH9>            multi_fc2;
    LIF1D_SD_Integer<FIXED_LENGTH9, 1>                   multi_lif2;
};

} // namespace hls4csnn1d_cblk_sd
#endif
