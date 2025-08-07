#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <fstream>
#include <vector>

#define TILE_DIM 16
#define KERNEL_RADIUS 12  // For 24x24 kernel

__global__ void conv2d_kernel(const float* __restrict__ input, 
                             const float* __restrict__ kernel, 
                             float* __restrict__ output,
                             int in_rows, int in_cols, 
                             int k_rows, int k_cols) {
    // Output dimensions
    const int out_rows = in_rows - k_rows + 1;
    const int out_cols = in_cols - k_cols + 1;
    
    // Thread coordinates in output matrix
    const int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Only threads that produce valid output compute results
    if (out_row < out_rows && out_col < out_cols) {
        float sum = 0.0f;
        
        // Precompute input starting position
        const int input_start = (out_row * in_cols) + out_col;
        
        // Manual loop unrolling for 24x24 kernel
        #pragma unroll
        for (int m = 0; m < 24; m++) {
            const int input_row_offset = m * in_cols;
            #pragma unroll
            for (int n = 0; n < 24; n++) {
                sum += input[input_start + input_row_offset + n] * kernel[m * 24 + n];
            }
        }
        
        output[out_row * out_cols + out_col] = sum;
    }
}

void read_binary_float(const std::string& filename, float* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open: " << filename << std::endl;
        exit(1);
    }
    in.read(reinterpret_cast<char*>(data), size * sizeof(float));
    in.close();
}

// test
bool compare_array(const float* a, const float* b, size_t n, float tol = 1e-2f) {
    for (size_t i = 0; i < n; ++i) {
        if (fabs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

int main() {
    std::vector<int> Ns = {128, 256, 512, 1024, 2048};  
    const int k_rows = 24, k_cols = 24;
    bool all_match = true;

    for (int idx = 0; idx < Ns.size(); ++idx) {
        int in_rows = Ns[idx];
        int in_cols = Ns[idx];
        int out_rows = in_rows - k_rows + 1;
        int out_cols = in_cols - k_cols + 1;

        size_t in_size = in_rows * in_cols;
        size_t k_size = k_rows * k_cols;
        size_t out_size = out_rows * out_cols;

        // test
        std::string in_file = "data/conv_input_" + std::to_string(idx + 1) + ".bin";
        std::string k_file  = "data/conv_kernel_" + std::to_string(idx + 1) + ".bin";
        std::string ref_file = "data/conv_ref_" + std::to_string(idx + 1) + ".bin";

        float* h_input = (float*)malloc(in_size * sizeof(float));
        float* h_kernel = (float*)malloc(k_size * sizeof(float));
        float* h_output = (float*)malloc(out_size * sizeof(float));
        float* h_ref = (float*)malloc(out_size * sizeof(float));

        read_binary_float(in_file, h_input, in_size);
        read_binary_float(k_file, h_kernel, k_size);
        read_binary_float(ref_file, h_ref, out_size);

        float *d_input, *d_kernel, *d_output;
        cudaMalloc(&d_input, in_size * sizeof(float));
        cudaMalloc(&d_kernel, k_size * sizeof(float));
        cudaMalloc(&d_output, out_size * sizeof(float));

        cudaMemcpy(d_input, h_input, in_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernel, h_kernel, k_size * sizeof(float), cudaMemcpyHostToDevice);

        dim3 threads(TILE_DIM, TILE_DIM);
        dim3 blocks((out_cols + threads.x - 1) / threads.x,
                    (out_rows + threads.y - 1) / threads.y);

        conv2d_kernel<<<blocks, threads>>>(d_input, d_kernel, d_output, in_rows, in_cols, k_rows, k_cols);
        cudaMemcpy(h_output, d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

        if (!compare_array(h_output, h_ref, out_size)) {
            std::cout << "F" << std::endl;
            all_match = false;
            cudaFree(d_input); cudaFree(d_kernel); cudaFree(d_output);
            free(h_input); free(h_kernel); free(h_output); free(h_ref);
            break;
        }

        cudaFree(d_input); cudaFree(d_kernel); cudaFree(d_output);
        free(h_input); free(h_kernel); free(h_output); free(h_ref);
    }

    if (all_match) std::cout << "T" << std::endl;
    return 0;
}