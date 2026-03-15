#ifndef MODEL_EVAL24_REF_H
#define MODEL_EVAL24_REF_H

#include "../constants24_sd.h"
#include <hls_stream.h>
#include <ap_int.h>

#include "qcsnn24_rrboth_sd.h"
#include "../filereader24.h"

class ModelEvaluation {
public:
    // ===== CHANGE 1: Updated signature to take separate RR arrays =====
    void evaluate(
        hls4csnn1d_cblk_sd::QCSNN24_RRBOTH_SD<NUM_STEPS>& model24,
        hls::stream<ap_int8_c>& datastream,     // IN:  180 signal samples only
        const ap_int8_c rr_stage1[RR_FEATURE_LENGTH],  // IN: RR quantized with Stage 1 scale
        const ap_int8_c rr_stage2[RR_FEATURE_LENGTH],  // IN: RR quantized with Stage 2 scale
        hls::stream<ap_int8_c>& pred2stream,    // OUT: 1 scalar
        hls::stream<ap_int8_c>& pred4stream     // OUT: 1 scalar
    ) {
#pragma HLS INLINE

        // ----------------------------------------------------------
        // 1) Collect signal input (180 scalars) -> input_stream
        // ----------------------------------------------------------
        hls::stream<ap_int8_c> input_stream("input_stream");
#pragma HLS STREAM variable=input_stream depth=FIXED_LENGTH1  // 180

    in_write:
        for (int i = 0; i < FIXED_LENGTH1; ++i) {  // 180 only
#pragma HLS PIPELINE II=1
            ap_int8_c v = datastream.read();
            input_stream.write(v);
        }

        // ----------------------------------------------------------
        // 2) Model forward with separate RR arrays
        // ----------------------------------------------------------
        hls::stream<ap_int8_c> out2_stream("out2_stream");
#pragma HLS STREAM variable=out2_stream depth=2

        hls::stream<ap_int8_c> out4_stream("out4_stream");
#pragma HLS STREAM variable=out4_stream depth=2

        // ===== CHANGE 2: Pass RR arrays to forward() =====
        model24.forward(input_stream, rr_stage1, rr_stage2, out2_stream, out4_stream);

        // ----------------------------------------------------------
        // 3) Read ONE pred2 + ONE pred4 and push to output streams
        // ----------------------------------------------------------
        ap_int8_c pred2 = out2_stream.read();
        ap_int8_c pred4 = out4_stream.read();

        pred2stream.write(pred2);
        pred4stream.write(pred4);
    }
};

#endif // MODEL_EVAL24_REF_H