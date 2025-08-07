#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>  
#include <fstream>
#include <cmath>
#include <string>

#define BATCH_SIZE 16
#define CHANNELS 32
#define KERNEL_SIZE 3
#define STRIDE 2
#define PADDING 1
#define DILATION 3

__global__ void max_pool3d_kernel(const float* input, float* output,
    int in_d1, int in_d2, int in_d3,
    int out_d1, int out_d2, int out_d3) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BATCH_SIZE * CHANNELS * out_d1 * out_d2 * out_d3) return;

    // Calculate output indices
    int c = (idx / (out_d1 * out_d2 * out_d3)) % CHANNELS;
    int b = idx / (CHANNELS * out_d1 * out_d2 * out_d3);
    int od1 = (idx / (out_d2 * out_d3)) % out_d1;
    int od2 = (idx / out_d3) % out_d2;
    int od3 = idx % out_d3;

    // Calculate input window start coordinates
    int id1_start = od1 * STRIDE - PADDING;
    int id2_start = od2 * STRIDE - PADDING;
    int id3_start = od3 * STRIDE - PADDING;

    float max_val = -FLT_MAX;
    bool initialized = false;

    // Iterate through the pooling window
    for (int k1 = 0; k1 < KERNEL_SIZE; ++k1) {
        for (int k2 = 0; k2 < KERNEL_SIZE; ++k2) {
            for (int k3 = 0; k3 < KERNEL_SIZE; ++k3) {
                int id1 = id1_start + k1 * DILATION;
                int id2 = id2_start + k2 * DILATION;
                int id3 = id3_start + k3 * DILATION;

                // Check if the current position is within input bounds
                if (id1 >= 0 && id1 < in_d1 && 
                    id2 >= 0 && id2 < in_d2 && 
                    id3 >= 0 && id3 < in_d3) {
                    
                    int input_idx = b * (CHANNELS * in_d1 * in_d2 * in_d3) + 
                                   c * (in_d1 * in_d2 * in_d3) + 
                                   id1 * (in_d2 * in_d3) + 
                                   id2 * in_d3 + 
                                   id3;
                    
                    float val = input[input_idx];
                    if (!initialized || val > max_val) {
                        max_val = val;
                        initialized = true;
                    }
                }
            }
        }
    }

    output[idx] = initialized ? max_val : 0.0f;
}

void read_binary(const std::string& filename, float* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Can not open: " << filename << std::endl;
        exit(1);
    }
    in.read(reinterpret_cast<char*>(data), size * sizeof(float));
    in.close();
}

// test
bool compare_outputs(const float* output, const float* reference, size_t size, float tolerance = 1e-5f) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(output[i] - reference[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

int compute_out_dim(int dim) {
    return ((dim + 2 * PADDING - DILATION * (KERNEL_SIZE - 1) - 1) / STRIDE) + 1;
}

int main() {
    int dims[5] = {16, 24, 32, 40, 48};
    bool all_pass = true;

    for (int i = 0; i < 5; ++i) {
        int D = dims[i];
        int OD = compute_out_dim(D);

        size_t in_elems = BATCH_SIZE * CHANNELS * D * D * D;
        size_t out_elems = BATCH_SIZE * CHANNELS * OD * OD * OD;
        size_t in_bytes = in_elems * sizeof(float);
        size_t out_bytes = out_elems * sizeof(float);

        float* h_input = (float*)malloc(in_bytes);
        float* h_output = (float*)malloc(out_bytes);
        float* h_output_ref = (float*)malloc(out_bytes);

        // test
        std::string input_file = "./data/pool_input_" + std::to_string(i + 1) + ".bin";
        std::string ref_file = "./data/pool_output_ref_" + std::to_string(i + 1) + ".bin";

        read_binary(input_file, h_input, in_elems);
        read_binary(ref_file, h_output_ref, out_elems);

        float *d_input, *d_output;
        cudaMalloc(&d_input, in_bytes);
        cudaMalloc(&d_output, out_bytes);
        cudaMemcpy(d_input, h_input, in_bytes, cudaMemcpyHostToDevice);

        int total_outputs = out_elems;
        int threads = 256;
        int blocks = (total_outputs + threads - 1) / threads;

        max_pool3d_kernel<<<blocks, threads>>>(
            d_input, d_output,
            D, D, D,
            OD, OD, OD
        );

        cudaMemcpy(h_output, d_output, out_bytes, cudaMemcpyDeviceToHost);

        bool match = compare_outputs(h_output, h_output_ref, out_elems);
        if (!match) all_pass = false;

        cudaFree(d_input);
        cudaFree(d_output);
        free(h_input);
        free(h_output);
        free(h_output_ref);
    }

    std::cout << (all_pass ? "T" : "F") << std::endl;
    return 0;
}