#ifndef FILE_READER_H
#define FILE_READER_H

#include <algorithm>
// #include <ap_fixed.h>
#include <ap_int.h>
#include <any>
#include <array>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <hls_stream.h>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <string>

#include <sys/stat.h>   // for ::mkdir
#include <cerrno>
#include <sys/types.h>



#include <vector>

#include <dirent.h>
#include <sys/types.h>
#include <string.h>

#include "constants24_sd.h"
 // brings qcsnet4_input_scale, qcsnet4_input_zero_point
#include "./weights_sd/qparams_qcsnet24_cblk1_input.h"
#include "./weights_sd/rr_stats.h"  // ADD THIS LINE

// ===== CHANGE 1: Add includes for head QuantIdentity scales =====
#include "./weights_sd/qparams_qcsnet2_lblk1_input.h"  // Stage 1 head scale
#include "./weights_sd/qparams_qcsnet4_lblk1_input.h"  // Stage 2 head scale

using namespace hls4csnn1d_cblk_sd;

// ===== CHANGE 2: Add type alias for RR feature array =====
using array_rr_t = std::array<ap_int8_c, RR_FEATURE_LENGTH>;  // 4 elements


class FileReader {
    public:
    // const float      STEP_F    = 0.015625f;           // 2^-4
    // const ap_fixed_c STEP_Q    = ap_fixed_c(STEP_F);
    // const float      INV_STEP  = 1.0f / STEP_F;     // 16.0
    // toggle behavior for short rows
    static constexpr bool PAD_SHORT_ROWS = false;   // true = pad, false = skip

        std::vector<array180rr_t> X;  // To store all the data (180 signal + 4 RR placeholder)
        std::vector<int> y;         // To store labels

        // ===== CHANGE 3: Add storage for RR features quantized with head scales =====
        std::vector<array_rr_t> rr_stage1;  // RR quantized with Stage 1 head scale
        std::vector<array_rr_t> rr_stage2;  // RR quantized with Stage 2 head scale

        // Load data from multiple folders, concatenate, and assign labels
        void loadData(const std::string& basePath = "./mitbih_processed_test/smallvhls") {
            // Folders and label mapping
#ifndef __SYNTHESIS__
        // std::vector<std::string> folders = {"normalsmaller", "svebsmaller", "vebsmaller", "fsmaller"};
        // std::map<std::string, int> label_map = {{"normalsmaller", 0}, {"svebsmaller", 1}, {"vebsmaller", 2}, {"fsmaller", 3}};
        std::vector<std::string> folders = {"normal", "sveb", "veb", "f"};
        std::map<std::string, int> label_map = {{"normal", 0}, {"sveb", 1}, {"veb", 2}, {"f", 3}};
#else
        std::vector<std::string> folders = {"normalvhls", "svebvhls", "vebvhls", "fvhls"};
        std::map<std::string, int> label_map = {{"normalvhls", 0}, {"svebvhls", 1}, {"vebvhls", 2}, {"fvhls", 3}};

        // std::vector<std::string> folders = {"normal", "sveb", "veb", "f"};
        // std::map<std::string, int> label_map = {{"normal", 0}, {"sveb", 1}, {"veb", 2}, {"f", 3}};
        // std::vector<std::string> folders = {"normalsmaller"};
        // std::map<std::string, int> label_map = {{"normalsmaller", 0}};
            
#endif         
            for (const std::string& folder : folders) {
                std::string folderPath = basePath + "/" + folder;
                loadFolder(folderPath, label_map[folder]);
            }

            // Shuffle the data and labels together
            //shuffleData();
        }

        // Stream the loaded data into an hls_stream
        void streamData(hls::stream<array180rr_t>& outputStream) {
//             std::cout << "Data Size: " << X.size() << std::endl;
            for (size_t i = 0; i < X.size(); ++i) {
                outputStream.write(X[i]);  // Stream the row
            }
//             std::cout << "output stream Size: " << outputStream.size() << std::endl;
        }
        
        void streamLabel(hls::stream<ap_int8_c>& labelStream, bool binary) {
            if (binary) {
                for (size_t i = 0; i < y.size(); ++i) {
                    // Convert multiclass to binary: 0 stays 0, 1-3 become 1
                    int binary_label = (y[i] == 0) ? 0 : 1;
                    labelStream.write(ap_int8_c(binary_label));
                } 
            }else{ 
                for (size_t i = 0; i < y.size(); ++i) {
                    labelStream.write(y[i]);   // Stream the corresponding label
                } 
            }
        }

        // Function to print one row from the output stream
        void printOneRow(hls::stream<array180rr_t>& outputStream) {
            if (!outputStream.empty()) {
                // Read one row (array180_t) from the stream
                array180rr_t row = outputStream.read();

                // Print the values of the row
                std::cout << "Row: ";
                for (int i = 0; i < FIXED_LENGTH1+RR_FEATURE_LENGTH; ++i) {
                    std::cout << row[i] << " ";
                }
                std::cout << std::endl;
            } else {
                std::cout << "The stream is empty!" << std::endl;
            }
        }

        // Method to read weights from JSON file and return map<string, any>
        JsonMap readJsonWeightsOrDims(const std::string& filePath) {
            // Open the JSON file
            std::ifstream inputFile(filePath);
            if (!inputFile.is_open()) {
                std::cerr << "Error opening file: " << filePath << std::endl;
                return {};
            }

            // Parse the JSON content
            json jsonData;
            inputFile >> jsonData;

            // Check for parsing errors
            if (jsonData.is_discarded()) {
                std::cerr << "Parse error: Invalid JSON content." << std::endl;
                return {};
            }


            // Convert jsonData to map<string, any> with flattened structure
            return extractLayerWeightsOrDims(jsonData);
        }

        inline void checksum_row(const array180rr_t& row, double& sum, double& sumsq, double& maxabs) {
              sum = 0.0; sumsq = 0.0; maxabs = 0.0;
              for (int k = 0; k < FIXED_LENGTH1+RR_FEATURE_LENGTH; ++k) {
                  float v = (float)row[k];
                  sum   += v;
                  sumsq += v * v;
                  float a = std::fabs(v);
                  if (a > maxabs) maxabs = a;
              }
        }

        // Write an hls::stream to a text (CSV) file.
        // T is assumed to be a fixed-length container (e.g., std::array) with .size() and operator[].
        // Optionally, every 'rowsPerBatch' rows, an extra blank line is inserted.
        template <typename T>
        void write_hls_stream_to_text_file(hls::stream<T>& stream, const std::string& filename, int rowsPerBatch = 0) {
            std::ofstream ofs(filename);
            if (!ofs) {
                std::cerr << "Error opening file for writing: " << filename << "\n";
                return;
            }
            int rowCount = 0;
            while (!stream.empty()) {
                T data = stream.read();
                // Write each element of the container separated by commas.
                for (size_t i = 0; i < data.size(); i++) {
                    ofs << data[i];
                    if (i != data.size() - 1)
                        ofs << ",";
                }
                ofs << "\n";
                rowCount++;
                // Insert an extra blank line every rowsPerBatch rows, if requested.
                // if (rowsPerBatch > 0 && rowCount % rowsPerBatch == 0) {
                //     ofs << "\n";
                // }
            }
            ofs.close();
        }

        

        /* ---- number of samples loaded ---- */
		size_t size() const { return X.size(); }

		/* ---- push one mini-batch into a stream ---- */
		void streamBatch(hls::stream<array180rr_t>& out,
						size_t start, size_t batch)
		{
			for (size_t i = 0; i < batch && start + i < X.size(); ++i)
				out.write(X[start + i]);
		}

		/* ---- matching labels (binary or 4-class) ---- */
		void labelBatch(hls::stream<ap_int8_c>& out,
						size_t start, size_t batch, bool binary)
		{
			for (size_t i = 0; i < batch && start + i < y.size(); ++i) {
				int lbl = y[start + i];
				if (binary) lbl = (lbl == 0 ? 0 : 1);
				out.write(ap_int8_c(lbl));
			}
		}

        // ===== CHANGE 4: Add methods to access RR features =====
        const array_rr_t& getRRStage1(size_t idx) const { return rr_stage1[idx]; }
        const array_rr_t& getRRStage2(size_t idx) const { return rr_stage2[idx]; }


// Save one quantized row (1×180) to its own CSV file.
// Returns the full path written (or empty string on failure).

// create directory if missing (POSIX; safe to call repeatedly)
inline bool make_dir_if_needed(const std::string& dir) {
    struct stat st;
    if (stat(dir.c_str(), &st) == 0) return (st.st_mode & S_IFDIR) != 0;
    if (mkdir(dir.c_str(), 0755) == 0) return true;
    return (errno == EEXIST);
}

// get filename stem (basename without extension) from a path
inline std::string basename_stem(const std::string& path) {
    size_t slash = path.find_last_of("/\\");
    std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);
    size_t dot = name.find_last_of('.');
    if (dot != std::string::npos) name = name.substr(0, dot);
    return name;
}


// save one row (1x180) into its own CSV file
inline std::string saveRowToCSV(const array180rr_t& row,
                                const std::string& out_dir,
                                const std::string& prefix,
                                size_t idx)
{
    make_dir_if_needed(out_dir);  // ignore failure; we'll try open anyway

    std::ostringstream name;
    name << out_dir << '/' << prefix << '_'
         << std::setw(6) << std::setfill('0') << idx << ".csv";
    const std::string path = name.str();

    std::ofstream ofs(path.c_str());
    if (!ofs.is_open()) {
        std::cerr << "Failed to open for write: " << path << "\n";
        return std::string();
    }

    ofs << std::fixed << std::setprecision(6);
    for (int k = 0; k < FIXED_LENGTH1+RR_FEATURE_LENGTH; ++k) {
        if (k) ofs << ',';
        ofs << static_cast<float>(row[k]);   // 1×180 line
    }
    ofs << '\n';
    return path;
}


private:

// Helper function to check if string ends with .csv
bool endsWith(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() && 
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Helper function to load all CSV files from a folder and append data
void loadFolder(const std::string& folderPath, int label) {
    DIR* dir = opendir(folderPath.c_str());
    if (dir != nullptr) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string filename(entry->d_name);
            // Skip . and .. directory entries
            if (filename != "." && filename != "..") {
                // Check if file ends with .csv
                if (endsWith(filename, ".csv")) {
                    std::string fullPath = folderPath;
                    // Add trailing slash if needed
                    if (folderPath[folderPath.length()-1] != '/') {
                        fullPath += "/";
                    }
                    fullPath += filename;
                    readCSV(fullPath, label);
                }
            }
        }
        closedir(dir);
    }
}


// Quantize signal samples (0-179) using trunk input scale
static inline ap_int8_c quantize_input_sample(float x_real) {
    // x_q = round(x_real / scale) + zero_point, then clamp to int8
    const float inv_s = 1.0f / qcsnet24_cblk1_input_scale;
    float qf = std::nearbyintf(x_real * inv_s) + static_cast<float>(qcsnet24_cblk1_input_zero_point);
    int qi = static_cast<int>(qf);
    if (qi > 127)  qi = 127;
    if (qi < -128) qi = -128;
    return ap_int8_c(qi);
}

// ===== CHANGE 5: Add quantization function for RR using head scales =====
// Quantize RR features using head QuantIdentity scale (proper float->int8 quantization)
static inline ap_int8_c quantize_rr_sample(float x_real, ap_int<16> scale_int, int frac_bits) {
    // Convert scale_int back to float: scale = scale_int / 2^frac_bits
    float scale = static_cast<float>(scale_int) / static_cast<float>(1 << frac_bits);
    // x_q = round(x_real / scale), then clamp to int8
    float qf = std::nearbyintf(x_real / scale);
    int qi = static_cast<int>(qf);
    if (qi > 127)  qi = 127;
    if (qi < -128) qi = -128;
    return ap_int8_c(qi);
}


void readCSV(const std::string& filePath, int label){
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cout << "[readCSV] ERROR: Failed to open " << filePath << '\n';
        return;
    }

    constexpr int TOTAL_FEATURES = FIXED_LENGTH1 + RR_FEATURE_LENGTH;  // 184
    constexpr int FRAC_BITS = 12;  // ===== CHANGE 6: Define FRAC_BITS for scale conversion =====

    constexpr bool HAS_HEADER = false;
    constexpr bool HAS_INDEX_COLUMN = false;
    
    std::string line;
    if (HAS_HEADER) {
        std::getline(file, line);
        std::cout << "[readCSV] Skipped header line\n";
    }

    constexpr bool APPLY_MINMAX_NORM = false;

    int line_no = HAS_HEADER ? 1 : 0;

    while (std::getline(file, line)) {
        ++line_no;

        if (line.find_first_not_of(" \t\r\n,") == std::string::npos) {
            std::cout << "[readCSV] line " << line_no << " blank; skipping.\n";
            continue;
        }

        std::istringstream ls(line);

        float        buf[TOTAL_FEATURES] = {};
        array180rr_t row = {};                  // 184 elements (signal + placeholder for RR)
        array_rr_t   rr_s1 = {};                // ===== CHANGE 7: RR for Stage 1 =====
        array_rr_t   rr_s2 = {};                // ===== CHANGE 7: RR for Stage 2 =====
        int          col = 0;

        std::string val;
        while (std::getline(ls, val, ',')) {
            val = trim(val);
            if (val.empty()) continue;
            
            if (col == 0 && HAS_INDEX_COLUMN) {
                continue;
            }
            
            if (col < TOTAL_FEATURES) {
                buf[col++] = std::stof(val);
            }
        }

        if (col < TOTAL_FEATURES) {
            if (!PAD_SHORT_ROWS) {
                std::cout << "[readCSV] line " << line_no << ": only "
                          << col << " values (need " << TOTAL_FEATURES
                          << "); skipping row.\n";
                continue;
            } else {
                float fill = (col > 0) ? buf[col - 1] : 0.0f;
                for (int k = col; k < TOTAL_FEATURES; ++k) buf[k] = fill;
                std::cout << "[readCSV] line " << line_no << ": padded from "
                          << col << " to " << TOTAL_FEATURES << ".\n";
            }
        }

        float rmin = buf[0], rmax = buf[0];
        if (APPLY_MINMAX_NORM) {
            for (int k = 1; k < TOTAL_FEATURES; ++k) {
                if (buf[k] < rmin) rmin = buf[k];
                if (buf[k] > rmax) rmax = buf[k];
            }
        }

        // ===== CHANGE 8: Modified quantization loop =====
        for (int k = 0; k < TOTAL_FEATURES; ++k) {
            float x_pre;
            if (APPLY_MINMAX_NORM) {
                if (rmax != rmin)
                    x_pre = -2.0f + (buf[k] - rmin) * 4.0f / (rmax - rmin);
                else
                    x_pre = 0.0f;
            } else {
                x_pre = buf[k];
            }
            
            if (k < FIXED_LENGTH1) {
                // Signal samples (0-179): quantize with trunk input scale
                row[k] = quantize_input_sample(x_pre);
            } else {
                // RR features (180-183): normalize, then quantize with head scales
                int rr_idx = k - FIXED_LENGTH1;  // 0, 1, 2, 3
                float x_norm = (x_pre - RR_MEAN[rr_idx]) * RR_STD_INV[rr_idx];
                
                // Quantize with Stage 1 head scale
                rr_s1[rr_idx] = quantize_rr_sample(x_norm, qcsnet2_lblk1_input_act_scale_int, FRAC_BITS);
                
                // Quantize with Stage 2 head scale
                rr_s2[rr_idx] = quantize_rr_sample(x_norm, qcsnet4_lblk1_input_act_scale_int, FRAC_BITS);
                
                // Keep a copy in row for backward compatibility (using Stage 1 scale)
                row[k] = rr_s1[rr_idx];
            }
        }

        X.push_back(row);
        y.push_back(label);
        rr_stage1.push_back(rr_s1);  // ===== CHANGE 9: Store RR for Stage 1 =====
        rr_stage2.push_back(rr_s2);  // ===== CHANGE 9: Store RR for Stage 2 =====
    }

    file.close();
}


// Helper function to trim whitespace from both ends of a string
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(' ');
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, last - first + 1);
}

// Helper function to shuffle data and labels together
void shuffleData() {
    // Create a vector of indices and shuffle it
    std::vector<size_t> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // Apply the shuffle to both X and y
    std::vector<array180rr_t> X_shuffled;
    std::vector<int> y_shuffled;
    std::vector<array_rr_t> rr_s1_shuffled;  // ===== CHANGE 10: Shuffle RR too =====
    std::vector<array_rr_t> rr_s2_shuffled;
    for (size_t idx : indices) {
        X_shuffled.push_back(X[idx]);
        y_shuffled.push_back(y[idx]);
        rr_s1_shuffled.push_back(rr_stage1[idx]);
        rr_s2_shuffled.push_back(rr_stage2[idx]);
    }

    X = std::move(X_shuffled);
    y = std::move(y_shuffled);
    rr_stage1 = std::move(rr_s1_shuffled);
    rr_stage2 = std::move(rr_s2_shuffled);
}

// Function to extract the "weights" for each layer in the JSON
JsonMap extractLayerWeightsOrDims(const json& jsonData) {
    JsonMap resultMap;

    for (auto it = jsonData.begin(); it != jsonData.end(); ++it) {
        const std::string& layerName = it.key();
        const json& layerContent = it.value();

        // Check if the layer has a "weights" field and store it in the map
        if (layerContent.contains("weights")) {
            resultMap[layerName] = layerContent["weights"];
        } else if (layerContent.contains("dimensions")) {
            resultMap[layerName] = layerContent["dimensions"];
        } else{
            std::cerr << "Warning: Layer " << layerName << " does not contain 'weights'" << std::endl;
        }
    }

    return resultMap;
}


    };

        

#endif // FILE_READER_H