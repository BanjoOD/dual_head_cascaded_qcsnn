// topclass24_sd.cpp
#include <hls_stream.h>
#include "qcsnn24_rrboth_sd.h"
#include "modeleval24_sd.h"
#include "../constants24_sd.h"

using namespace hls4csnn1d_cblk_sd;

// ===== FIX: Use constants directly, no local redefinitions =====
static const int TOTAL_INPUT = FIXED_LENGTH1 + RR_FEATURE_LENGTH + RR_FEATURE_LENGTH;  // 188
static const int WORDS_PER_ROW = (TOTAL_INPUT + 7) / 8;                                // 24 AXI words

// ---------------------------------------------------------------------
// AXI -> signal stream + RR arrays
// ---------------------------------------------------------------------
static void axi_to_input_data(hls::stream<axi_fixed_t>& axiStream,
                              hls::stream<ap_int8_c>&   dataStream,
                              ap_int8_c                 rr_stage1[RR_FEATURE_LENGTH],
                              ap_int8_c                 rr_stage2[RR_FEATURE_LENGTH])
{
#pragma HLS INLINE

    ap_int8_c buf[TOTAL_INPUT];
#pragma HLS ARRAY_PARTITION variable=buf cyclic factor=8 dim=1

    for (int w = 0; w < WORDS_PER_ROW; ++w) {
#pragma HLS PIPELINE II=1
        axi_fixed_t word = axiStream.read();

        for (int j = 0; j < 8; ++j) {
#pragma HLS UNROLL
            int idx = w * 8 + j;
            if (idx < TOTAL_INPUT) {
                ap_int8_c temp;
                temp.range(7, 0) = word.data.range(j * 8 + 7, j * 8);
                buf[idx] = temp;
            }
        }

#ifndef __SYNTHESIS__
        if ((w == WORDS_PER_ROW - 1) && (word.last != 1)) {
            std::cerr << "[WARN] TLAST mismatch on input word " << w << '\n';
        }
#endif
    }

    // Signal (0-179) → dataStream
    for (int i = 0; i < FIXED_LENGTH1; ++i) {
#pragma HLS PIPELINE II=1
        dataStream.write(buf[i]);
    }

    // RR_s1 (180-183) → rr_stage1 array
    for (int i = 0; i < RR_FEATURE_LENGTH; ++i) {
#pragma HLS UNROLL
        rr_stage1[i] = buf[FIXED_LENGTH1 + i];
    }

    // RR_s2 (184-187) → rr_stage2 array
    for (int i = 0; i < RR_FEATURE_LENGTH; ++i) {
#pragma HLS UNROLL
        rr_stage2[i] = buf[FIXED_LENGTH1 + RR_FEATURE_LENGTH + i];
    }
}

// ---------------------------------------------------------------------
// ap_int8_c scalar prediction -> AXI word
// ---------------------------------------------------------------------
static void pred_to_axi(hls::stream<ap_int8_c>&   inPred,
                        hls::stream<axi_fixed_t>& outAxi)
{
#pragma HLS INLINE
    axi_fixed_t word;
    word.data = 0;
    word.keep = 0;
    word.strb = 0;
    word.last = 1;

    ap_int8_c p = inPred.read();
    word.data.range(7, 0) = p.range(7, 0);

    word.keep = 0x0F;
    word.strb = 0x0F;

    outAxi.write(word);
}


//---------------------------------------------------------------------
// TopClass24_SD — wraps model + evaluator
//---------------------------------------------------------------------
class TopClass24_SD {
public:
    void evaluate(hls::stream<ap_int8_c>& dataStream,
                  const ap_int8_c         rr_stage1[RR_FEATURE_LENGTH],
                  const ap_int8_c         rr_stage2[RR_FEATURE_LENGTH],
                  hls::stream<ap_int8_c>& out2Stream,
                  hls::stream<ap_int8_c>& out4Stream)
    {
#pragma HLS INLINE
        evaluator24.evaluate(qcsnn24, dataStream, rr_stage1, rr_stage2, out2Stream, out4Stream);
    }
private:
    QCSNN24_RRBOTH_SD<NUM_STEPS> qcsnn24;
    ModelEvaluation              evaluator24;
};

//---------------------------------------------------------------------
// Top-level HLS function exposed to Vitis / DMA wrapper
//---------------------------------------------------------------------
extern "C" void topFunction(hls::stream<axi_fixed_t>& dmaInStream,
                            hls::stream<axi_fixed_t>& dmaOut2Stream,
                            hls::stream<axi_fixed_t>& dmaOut4Stream)
{
#pragma HLS INTERFACE axis port=dmaInStream
#pragma HLS INTERFACE axis port=dmaOut2Stream
#pragma HLS INTERFACE axis port=dmaOut4Stream
#pragma HLS INTERFACE ap_ctrl_hs port=return
#pragma HLS DATAFLOW

    hls::stream<ap_int8_c> dataStream("dataStream");
#pragma HLS STREAM variable=dataStream depth=FIXED_LENGTH1

    ap_int8_c rr_stage1[RR_FEATURE_LENGTH];
#pragma HLS ARRAY_PARTITION variable=rr_stage1 complete dim=1

    ap_int8_c rr_stage2[RR_FEATURE_LENGTH];
#pragma HLS ARRAY_PARTITION variable=rr_stage2 complete dim=1

    hls::stream<ap_int8_c> pred2Stream("pred2Stream");
#pragma HLS STREAM variable=pred2Stream depth=2

    hls::stream<ap_int8_c> pred4Stream("pred4Stream");
#pragma HLS STREAM variable=pred4Stream depth=2

    // 1) Unpack AXI -> signal (180) + RR_s1 (4) + RR_s2 (4)
    axi_to_input_data(dmaInStream, dataStream, rr_stage1, rr_stage2);

    // 2) Evaluate network with separate RR arrays
    static TopClass24_SD topClass24;
    topClass24.evaluate(dataStream, rr_stage1, rr_stage2, pred2Stream, pred4Stream);

    // 3) Pack scalar preds back to AXI (1 word each)
    pred_to_axi(pred2Stream, dmaOut2Stream);
    pred_to_axi(pred4Stream, dmaOut4Stream);
}