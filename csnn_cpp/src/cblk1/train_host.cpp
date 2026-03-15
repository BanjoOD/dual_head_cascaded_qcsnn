/****************************************************************************************
 * train_host.cpp  –  host-side training stub with mutable weight copies
 * --------------------------------------------------------------------------------------
 *  • Keeps the immutable,  synth-time   `static const`  weights in nn2_cblk1_sd.h
 *  • Creates 1-for-1 mutable shadows    (w_…) for every parameter that must learn
 *  • Optimiser loops are redirected to those mutable buffers
 *  • After training, each buffer is dumped back to a `static const` header file
 *
 *  NOTE ▸ Only the sizes (C1, C2, …) are taken from the network header, so this file
 *         stays in sync if channel counts change.
 ***************************************************************************************/

#include <vector>
#include <cmath>
#include <memory>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>

#include "../../include/hls4csnn1d_sd/cblk_sd/nn2_cblk1_sd.h"
#include "../../include/hls4csnn1d_sd/filereader.h"

using namespace hls4csnn1d_cblk_sd;   // exposes ap_fixed_c and const weight arrays

/* ───────────────────────────────
   1.  Mutable shadow parameters
   ─────────────────────────────── */
static ap_fixed_c w_conv1[NeuralNetwork2_Cblk1_sd::C1][1][3];
static ap_fixed_c w_conv2[NeuralNetwork2_Cblk1_sd::C2]
                         [NeuralNetwork2_Cblk1_sd::C1][3];
static ap_fixed_c w_fc   [NeuralNetwork2_Cblk1_sd::FC_OUT]
                         [NeuralNetwork2_Cblk1_sd::FC_IN];

static ap_fixed_c gamma_bn1[NeuralNetwork2_Cblk1_sd::C1];
static ap_fixed_c  beta_bn1[NeuralNetwork2_Cblk1_sd::C1];
static ap_fixed_c gamma_bn2[NeuralNetwork2_Cblk1_sd::C2];
static ap_fixed_c  beta_bn2[NeuralNetwork2_Cblk1_sd::C2];

static ap_fixed_c lif1_beta, lif1_theta;
static ap_fixed_c lif2_beta, lif2_theta;
static ap_fixed_c head_beta, head_theta;

/* ------------  copy ROM → mutable  ----------- */
static void init_train_weights() {
    std::memcpy(w_conv1, qcsnet2_cblk1_qconv1d_weights, sizeof w_conv1);
    std::memcpy(w_conv2, qcsnet2_cblk2_qconv1d_weights, sizeof w_conv2);
    std::memcpy(w_fc,    qcsnet2_lblk1_qlinear_weights, sizeof w_fc);

    std::memcpy(gamma_bn1, qcsnet2_cblk1_batch_norm_PARAMS.gamma, sizeof gamma_bn1);
    std::memcpy(beta_bn1,  qcsnet2_cblk1_batch_norm_PARAMS.beta,  sizeof beta_bn1);
    std::memcpy(gamma_bn2, qcsnet2_cblk2_batch_norm_PARAMS.gamma, sizeof gamma_bn2);
    std::memcpy(beta_bn2,  qcsnet2_cblk2_batch_norm_PARAMS.beta,  sizeof beta_bn2);

    lif1_beta = qcsnet2_cblk1_leaky_beta;    lif1_theta = qcsnet2_cblk1_leaky_threshold;
    lif2_beta = qcsnet2_cblk2_leaky_beta;    lif2_theta = qcsnet2_cblk2_leaky_threshold;
    head_beta = qcsnet2_lblk1_leaky_beta;    head_theta = qcsnet2_lblk1_leaky_threshold;
}

/* ───────────────────────────────
   2.  Optimiser (Adam only for brevity)
   ─────────────────────────────── */
class OptimAdam {
public:
    OptimAdam(NeuralNetwork2_Cblk1_sd& net,
              float lr = 1e-3f, float b1 = 0.9f, float b2 = 0.999f,
              float eps = 1e-8f, float wd = 0.0f, float clip = 1.0f)
        : g_(net.grads()), lr_(lr), b1_(b1), b2_(b2),
          eps_(eps), wd_(wd), clip_(clip), t_(0)
    {
        alloc_moments();
    }

    void step() {
        ++t_;
        const float bc1 = 1.f - std::pow(b1_, t_);
        const float bc2 = 1.f - std::pow(b2_, t_);
        const float lr_hat = lr_ * std::sqrt(bc2) / bc1;

        /* -- Conv-1 ------------------------------------------------ */
        for (int oc = 0; oc < NeuralNetwork2_Cblk1_sd::C1; ++oc)
            for (int ic = 0; ic < 1; ++ic)
                for (int k = 0; k < 3; ++k)
                    adam_update(w_conv1[oc][ic][k],
                                g_.dW_conv1[oc][ic][k]);

        /* -- Conv-2 ------------------------------------------------ */
        for (int oc = 0; oc < NeuralNetwork2_Cblk1_sd::C2; ++oc)
            for (int ic = 0; ic < NeuralNetwork2_Cblk1_sd::C1; ++ic)
                for (int k = 0; k < 3; ++k)
                    adam_update(w_conv2[oc][ic][k],
                                g_.dW_conv2[oc][ic][k]);

        /* -- FC ---------------------------------------------------- */
        for (int o = 0; o < NeuralNetwork2_Cblk1_sd::FC_OUT; ++o)
            for (int i = 0; i < NeuralNetwork2_Cblk1_sd::FC_IN; ++i)
                adam_update(w_fc[o][i], g_.dW_fc[o][i]);

        /* -- BN-1 -------------------------------------------------- */
        for (int c = 0; c < NeuralNetwork2_Cblk1_sd::C1; ++c) {
            adam_update(gamma_bn1[c], g_.dγ_bn1[c]);
            adam_update(beta_bn1 [c], g_.dβ_bn1[c]);
        }

        /* -- BN-2 -------------------------------------------------- */
        for (int c = 0; c < NeuralNetwork2_Cblk1_sd::C2; ++c) {
            adam_update(gamma_bn2[c], g_.dγ_bn2[c]);
            adam_update(beta_bn2 [c], g_.dβ_bn2[c]);
        }

        /* -- LIF scalars ------------------------------------------ */
        adam_update(lif1_beta, g_.dβ_lif1);  adam_update(lif1_theta, g_.dθ_lif1);
        adam_update(lif2_beta, g_.dβ_lif2);  adam_update(lif2_theta, g_.dθ_lif2);
        adam_update(head_beta, g_.dβ_head);  adam_update(head_theta, g_.dθ_head);

        /* zero-grad ------------------------------------------------ */
        std::memset(&g_, 0, sizeof g_);
    }

    float lr()  const { return lr_; }
    int   step() const { return t_; }

private:
    /* ---- allocate moment buffers ----- */
    void alloc_moments() {
        auto zero3 = [](int a,int b,int c) {
            return std::vector<std::vector<std::vector<float>>>(a,
                   std::vector<std::vector<float>>(b,
                   std::vector<float>(c,0.f))); };

        m1_c1_ = zero3(NeuralNetwork2_Cblk1_sd::C1, 1, 3);
        m2_c1_ = zero3(NeuralNetwork2_Cblk1_sd::C1, 1, 3);
        m1_c2_ = zero3(NeuralNetwork2_Cblk1_sd::C2,
                       NeuralNetwork2_Cblk1_sd::C1, 3);
        m2_c2_ = zero3(NeuralNetwork2_Cblk1_sd::C2,
                       NeuralNetwork2_Cblk1_sd::C1, 3);

        m1_fc_.assign(NeuralNetwork2_Cblk1_sd::FC_OUT,
                      std::vector<float>(NeuralNetwork2_Cblk1_sd::FC_IN,0.f));
        m2_fc_ = m1_fc_;

        m1_bn1_.assign(NeuralNetwork2_Cblk1_sd::C1,0.f);
        m2_bn1_ = m1_bn1_;
        m1_bn2_.assign(NeuralNetwork2_Cblk1_sd::C2,0.f);
        m2_bn2_ = m1_bn2_;
    }

    /* ---- core update ----- */
    void adam_update(ap_fixed_c& w, const ap_fixed_c& g_fixed) {
        const int idx = idx_++;          // flat index across all params
        float& m1 = M1_[idx];
        float& m2 = M2_[idx];

        float g = clip(float(g_fixed) + wd_*float(w));
        m1 = b1_*m1 + (1.f-b1_)*g;
        m2 = b2_*m2 + (1.f-b2_)*g*g;
        w  -= ap_fixed_c(lr_*m1 / (std::sqrt(m2)+eps_));
    }

    inline float clip(float g) const {
        return std::max(std::min(g, clip_), -clip_);
    }

    /* ---- members ---- */
    NeuralNetwork2_Cblk1_sd::Grads& g_;
    float lr_, b1_, b2_, eps_, wd_, clip_;
    int   t_;

    /* moment buffers flattened for simplicity */
    std::vector<float> M1_, M2_;
    int idx_{0};

    /* layer-wise containers (not used directly after flatten) */
    std::vector<std::vector<std::vector<float>>> m1_c1_, m2_c1_, m1_c2_, m2_c2_;
    std::vector<std::vector<float>>              m1_fc_, m2_fc_;
    std::vector<float>                           m1_bn1_, m2_bn1_,
                                                m1_bn2_, m2_bn2_;
};

/* ───────────────────────────────
   3.  Dump trained values → header
   ─────────────────────────────── */
template<typename T, size_t N1, size_t N2, size_t N3>
static void dump3D(const char* name, const T (&arr)[N1][N2][N3], std::ofstream& ofs) {
    ofs << "static const ap_fixed_c " << name << "["<<N1<<"]["<<N2<<"]["<<N3<<"] = {\n";
    for (size_t i=0;i<N1;++i){ ofs<<" {"; for(size_t j=0;j<N2;++j){ ofs<<" {"; 
        for(size_t k=0;k<N3;++k){ ofs<<std::fixed<<std::setprecision(6)
            << float(arr[i][j][k]); if(k<N3-1) ofs<<", "; }
        ofs<<" }"; if(j<N2-1) ofs<<","; }
    ofs<<" }"; if(i<N1-1) ofs<<","; ofs<<"\n"; }
    ofs<<"};\n\n";
}

static void export_headers(const std::string& path) {
    std::ofstream h(path);
    h << "#pragma once\n\n";
    dump3D("qcsnet2_cblk1_qconv1d_weights", w_conv1, h);
    dump3D("qcsnet2_cblk2_qconv1d_weights", w_conv2, h);
    /*  …repeat for FC, BN, LIF (omitted for brevity) */
}

/* ───────────────────────────────
   4.  main()
   ─────────────────────────────── */
int main()
{
    /* 1.  Load everything once */

     // Set up file paths.
    std::string folderPath = "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_train/smaller";

    FileReader reader;
    reader.loadData(folderPath);
    std::cout << "#samples = " << reader.size() << '\n';

    /* 2.  Init network, weights, optimiser */
    NeuralNetwork2_Cblk1_sd net;
    init_train_weights();
    OptimAdam opt(net, 1e-3f);

    /* 3.  Streams used for every batch */
    hls::stream<array180_t> ecgStream("X");
    hls::stream<ap_fixed_c> labelStream("y");

    const int EPOCHS = 5;
    const int BATCH  = 32;
    const bool BIN   = false;          // set true for binary task

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        /* optional: reader.shuffleData(); */

        for (size_t idx = 0; idx < reader.size(); idx += BATCH) {

            /* (a) zero-accumulate loss / grads for this batch   */
            // (optional) your accum variables here

            for (size_t b = 0; b < BATCH && idx + b < reader.size(); ++b) {
                /* 1. Copy one sample from reader.X to std::vector */
                const array180_t& raw = reader.X[idx + b];
                std::vector<ap_fixed_c> x(raw.begin(), raw.end());

                /* 2. Forward */
                NeuralNetwork2_Cblk1_sd::Cache cache;
                std::vector<ap_fixed_c> y_hat;              // model output
                net.forward_train(x, cache, y_hat);

                /* 3. Compute loss gradient dY             */
                std::vector<ap_fixed_c> dY(y_hat.size());
                int lbl = reader.y[idx + b];
                /* ----- quick CrossEntropy gradient ----- */
                for (size_t k = 0; k < y_hat.size(); ++k) {
                    float target = (k == lbl) ? 1.f : 0.f;
                    ap_fixed_c tgt = ap_fixed_c(target);      // 0 or 1 in Q4.4
                    dY[k] = y_hat[k] - tgt;                   // ap_fixed_c – ap_fixed_c ✔

                    // dY[k] = ap_fixed_c(y_hat[k] - target);   // d L/dy = p - 1_hot
                }

                /* 4. Backward */
                std::vector<ap_fixed_c> dx;                 // not used further
                net.backward(cache, dY, dx);
            }

            /* (b) optimizer step once per batch */
            opt.step();
        }

        std::cout << "Epoch " << epoch << " done.\n";
    }

    export_headers("trained_weights.h");
    std::cout << "Training complete → weights dumped.\n";
}
