#ifndef UTILS_H
#define UTILS_H

#include "constants24_sd.h"
#include <vector>
#include <hls_stream.h>

namespace hls4csnn1d_cblk_sd {

// -----------------------------------------------------------------------------
// Helper: compile‑time‑bounded strlen (max MAX_LAYER_NAME_LENGTH)
// -----------------------------------------------------------------------------
inline int hls_strn_len(const char* s) {
    #pragma HLS INLINE
    if (!s) return 0;

    int len = 0;
LEN_LOOP:
    for (int i = 0; i < MAX_LAYER_NAME_LENGTH; ++i) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=0 max=MAX_LAYER_NAME_LENGTH
        if (s[i] == '\0') break;
        ++len;
    }
    return len;
}

// -----------------------------------------------------------------------------
// String contains  (fixed‑bound, synthesis‑friendly)
// -----------------------------------------------------------------------------
inline bool hls_str_contains(const char* str, const char* substr) {
    #pragma HLS INLINE
    if (!str || !substr) return false;

    const int sub_len = hls_strn_len(substr);

    // Empty substring always matches
    if (sub_len == 0) return true;

SEARCH_LOOP:
    for (int i = 0; i <= MAX_LAYER_NAME_LENGTH - sub_len; ++i) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_LAYER_NAME_LENGTH
        // Stop when end of main string is reached
        if (str[i] == '\0') return false;

        bool match = true;

COMP_LOOP:
        for (int j = 0; j < sub_len; ++j) {
            #pragma HLS UNROLL
            char s = str[i + j];
            char p = substr[j];
            if (s == '\0' || s != p) {
                match = false;
            }
        }

        if (match) return true;
    }

    return false;
}

// -----------------------------------------------------------------------------
// String equality  (fixed‑bound, synthesis‑friendly)
// -----------------------------------------------------------------------------
inline bool hls_str_equal(const char* str1, const char* str2) {
    #pragma HLS INLINE
    if (!str1 || !str2) return false;

EQUAL_LOOP:
    for (int i = 0; i < MAX_LAYER_NAME_LENGTH; ++i) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_LAYER_NAME_LENGTH
        char c1 = str1[i];
        char c2 = str2[i];

        if (c1 != c2) return false;
        if (c1 == '\0') return true; // both terminated
    }

    // Reached maximum length without mismatch or terminator
    return true;
}


template<int LEN>
inline void vec2stream(const std::vector<ap_int8_c>& v,
                       hls::stream<ap_int8_c>&      s) {
    for (int i = 0; i < LEN; ++i) s.write(v[i]);
}

template<int LEN>
inline void stream2vec(hls::stream<ap_int8_c>& s,
                       std::vector<ap_int8_c>& v) {
    v.resize(LEN);
    for (int i = 0; i < LEN; ++i) v[i] = s.read();
}

} // namespace hls4csnn1d_bm

#endif // UTILS_H



// #ifndef UTILS_H
// #define UTILS_H

// #include "constants.h"

// namespace hls4csnn1d_bm {

// // Redesigned string contains function with improved pipelining
// inline bool hls_str_contains(const char* str, const char* substr) {
//     #pragma HLS INLINE
    
//     // Handle null pointers
//     if (!str || !substr) return false;
    
//     // Check if substr is empty - early return
//     if (substr[0] == '\0') return true;
    
//     // Pre-fetch the first character of substring for faster comparison
//     const char first_char = substr[0];
//     bool found_match = false;
    
//     // Main outer loop with bounded length
//     OUTER_LOOP:
//     for (int i = 0; i < MAX_LAYER_NAME_LENGTH; i++) {
//         #pragma HLS PIPELINE II=6
        
//         // Load current character once
//         const char current = str[i];
        
//         // End of string check
//         if (current == '\0') break;
        
//         // Only proceed with inner loop if first character matches
//         if (current == first_char) {
//             // Check for full match at this position
//             bool potential_match = true;
            
//             // Bounded inner loop
//             for (int j = 1; j < MAX_LAYER_NAME_LENGTH && potential_match; j++) {
//                 // Load substring character once
//                 const char sub_char = substr[j];
                
//                 // End of substring means we found a match
//                 if (sub_char == '\0') {
//                     potential_match = true;
//                     break;
//                 }
                
//                 // Load corresponding string character
//                 const char str_char = str[i + j];
                
//                 // End of string or character mismatch
//                 if (str_char == '\0' || str_char != sub_char) {
//                     potential_match = false;
//                 }
//             }
            
//             // Update result
//             found_match = found_match || potential_match;
//         }
//     }
    
//     return found_match;
// }

// // Redesigned string equality function with improved pipelining
// inline bool hls_str_equal(const char* str1, const char* str2) {
//     #pragma HLS INLINE
    
//     // Handle null pointers
//     if (!str1 || !str2) return false;
    
//     // Use a flag to track equality rather than early returns
//     bool strings_equal = true;
//     bool end_reached = false;
    
//     // Compare characters with bounded loop
//     EQUAL_LOOP:
//     for (int i = 0; i < MAX_LAYER_NAME_LENGTH && !end_reached; i++) {
//         #pragma HLS PIPELINE II=6
        
//         // Load characters once
//         const char c1 = str1[i];
//         const char c2 = str2[i];
        
//         // Characters differ - strings not equal
//         if (c1 != c2) {
//             strings_equal = false;
//             end_reached = true;
//         }
        
//         // End of string - we're done
//         if (c1 == '\0') {
//             end_reached = true;
//         }
//     }
    
//     return strings_equal;
// }

// // Alternative implementation with flattened control flow
// // Use this if the above version still has issues
// // inline bool hls_str_equal_alt(const char* str1, const char* str2) {
// //     #pragma HLS INLINE
    
// //     // Handle null pointers
// //     if (!str1 || !str2) return false;
    
// //     // Array to store comparison results
// //     bool comparison[MAX_LAYER_NAME_LENGTH];
// //     #pragma HLS ARRAY_PARTITION variable=comparison complete
    
// //     // End flags
// //     bool end_flags[MAX_LAYER_NAME_LENGTH];
// //     #pragma HLS ARRAY_PARTITION variable=end_flags complete
    
// //     // Prefetch loop - load all comparison results
// //     PREFETCH_LOOP:
// //     for (int i = 0; i < MAX_LAYER_NAME_LENGTH; i++) {
// //         #pragma HLS PIPELINE II=6
        
// //         // Load characters
// //         const char c1 = str1[i];
// //         const char c2 = str2[i];
        
// //         // Compute comparison results
// //         comparison[i] = (c1 == c2);
// //         end_flags[i] = (c1 == '\0');
// //     }
    
// //     // Final result computation
// //     bool result = true;
// //     bool still_comparing = true;
    
// //     RESULT_LOOP:
// //     for (int i = 0; i < MAX_LAYER_NAME_LENGTH && still_comparing; i++) {
// //         #pragma HLS PIPELINE II=6
        
// //         // If not equal at this position, result is false
// //         if (!comparison[i]) {
// //             result = false;
// //             still_comparing = false;
// //         }
        
// //         // If we reached the end, stop comparing
// //         if (end_flags[i]) {
// //             still_comparing = false;
// //         }
// //     }
    
// //     return result;
// // }

// } // namespace hls4csnn1d_bm

// #endif // UTILS_H