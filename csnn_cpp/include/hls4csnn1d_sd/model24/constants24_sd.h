#ifndef CONSTANT24_H
#define CONSTANT24_H

#include <hls_stream.h>
#include <ap_fixed.h>
#include <ap_axi_sdata.h>
#include <array>
#include <any>
#include <map>

#include <nlohmann/json.hpp>


typedef nlohmann::json json;
//typedef std::any any;
// In both FileReader.h and HydrateNetwork.h (at the top)
typedef std::map<std::string, json> JsonMap;
typedef ap_int<8> ap_int8_c; 
typedef ap_int<32> acc32_t;

// Constants
const int NUM_CLASSES_4 = 4;


/**
 * Maximum number of layers and max length of layer names.
 * Adjust these based on your design constraints.
 */
const int MAX_LAYERS = 10; // Maximum number of layers per type
static const int MAX_LAYER_NAME_LENGTH = 64;

// Upper‑bound for synthesis loop (change to suit your application)
static const int NUM_SAMPLES = 80; //4; //2489; //51002; //1385; //51013; 
static const int NUM_STEPS = 10;


const int FIXED_LENGTH1 = 180;
const int FIXED_LENGTH2 = 2848;
const int FIXED_LENGTH3 = 1424;

const int FIXED_LENGTH4 = 1392;
const int FIXED_LENGTH5 = 688;
const int FIXED_LENGTH6 = 984;
const int FIXED_LENGTH7 = 480;
const int FIXED_LENGTH8 = 128;
const int FIXED_LENGTH9 = 4;

const int RR_FEATURE_LENGTH = 4;
  


// ──────────── Dimensional bookkeeping ────────────
// Layer‑0 input row ...........................................
const int CONV_IN_LENGTH1   = 180;     // 1 × 180
const int STRIDE = 1;

// Block‑1 .....................................................
const int OUT_CH1           = 16;
const int FEATURE_LENGTH1   = (CONV_IN_LENGTH1 - 3) / STRIDE + 1;   // (180‑3)/1 + 1  = 178
const int POOL1_IN_LEN      = FEATURE_LENGTH1;                // 178
const int POOL1_OUT_LEN     = (POOL1_IN_LEN - 2) / 2 + 1;     // (178‑2)//2 + 1 = 89

// Block‑2 .....................................................
const int CONV_IN_LENGTH2   = POOL1_OUT_LEN;                  // 89
const int OUT_CH2           = 16;
const int FEATURE_LENGTH2   = (CONV_IN_LENGTH2 - 3) / STRIDE + 1;  // (89‑3)/1 + 1 = 87
const int POOL2_IN_LEN      = FEATURE_LENGTH2;                // 87
const int POOL2_OUT_LEN     = (POOL2_IN_LEN - 2) / 2 + 1;     // (87‑2)//2 + 1 = 43


// Block‑3 .....................................................
const int CONV_IN_LENGTH3   = POOL2_OUT_LEN;                  // 43
const int OUT_CH3           = 24;
const int FEATURE_LENGTH3   = (CONV_IN_LENGTH3 - 3) / STRIDE + 1;  // (43‑3)/1 + 1 = 41
const int POOL3_IN_LEN      = FEATURE_LENGTH3;                // 41
const int POOL3_OUT_LEN     = (POOL3_IN_LEN - 2) / 2 + 1;     // (41‑2)//2 + 1 = 20


// Dense head ..................................................
const int LINEAR_IN_SIZE1   = OUT_CH3 * POOL3_OUT_LEN;        // 24 × 20 = 480
const int LINEAR_OUT_SIZE1  = 128;


typedef std::array<ap_int8_c, FIXED_LENGTH1> array180_t;// Using std::array for compatibility
typedef std::array<ap_int8_c, FIXED_LENGTH1 + RR_FEATURE_LENGTH> array180rr_t;

typedef std::array<ap_int8_c, FIXED_LENGTH2> array2848_t;
typedef std::array<ap_int8_c, FIXED_LENGTH3> array1424_t;
typedef std::array<ap_int8_c, FIXED_LENGTH4> array1392_t;
typedef std::array<ap_int8_c, FIXED_LENGTH5> array688_t;
typedef std::array<ap_int8_c, FIXED_LENGTH6> array984_t;
typedef std::array<ap_int8_c, FIXED_LENGTH7> array480_t;
typedef std::array<ap_int8_c, FIXED_LENGTH8> array128_t;
typedef std::array<ap_int8_c, FIXED_LENGTH9> array4_t;

// Custom types
typedef std::array<ap_int8_c, FIXED_LENGTH1> input180_data_type;
typedef std::array<ap_int8_c, FIXED_LENGTH1 + RR_FEATURE_LENGTH> input180rr_data_type;

typedef std::array<ap_int8_c, FIXED_LENGTH2> input2848_data_type;
typedef std::array<ap_int8_c, FIXED_LENGTH3> input1424_data_type;
typedef std::array<ap_int8_c, FIXED_LENGTH4> input1392_data_type;
typedef std::array<ap_int8_c, FIXED_LENGTH5> input688_data_type;
typedef std::array<ap_int8_c, FIXED_LENGTH6> input984_data_type;
typedef std::array<ap_int8_c, FIXED_LENGTH7> input480_data_type;
typedef std::array<ap_int8_c, FIXED_LENGTH8> input128_data_type;
typedef std::array<ap_int8_c, FIXED_LENGTH9> input4_data_type;


// Define the AXI stream data type with an 8-bit wide data field.
typedef ap_axiu<64, 0, 0, 0> axi_fixed_t;

static inline ap_int8_c sat_clip_int8(ap_int<32> value) {
    if (value >  127) return  127;
    if (value < -128) return -128;
    return (ap_int8_c)value;
}

// y_q = clip((acc * scale_multiplier + 2^(right_shift-1)) >> right_shift)
static inline ap_int8_c requantize(acc32_t acc, ap_int<32> scale_multiplier, int right_shift) {
    ap_int<32> shifted_result;

    ap_int<64> scaled_product = (ap_int<64>)acc * (ap_int<64>)scale_multiplier;
    
    if (right_shift > 0) {
        ap_int<64> rounding_offset = (ap_int<64>)1 << (right_shift - 1);
        ap_int<64> rounded_value = scaled_product + rounding_offset;
        shifted_result = (ap_int<32>)(rounded_value >> right_shift);
    } else {
        shifted_result = (ap_int<32>)scaled_product;
    }
    
    return sat_clip_int8(shifted_result);
}

#endif