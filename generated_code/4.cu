#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>

#define BATCH_SIZE 16
#define FEATURES 4
#define EPS 1e-5f
#define NUM_TESTS 5
#define WARP_SIZE 32
#define BLOCK_SIZE 256

const int DIMS[NUM_TESTS] = {64, 128, 256, 512, 1024};  // dim1 = dim2

__global__ void layer_norm_kernel(float* __restrict__ input, float* __restrict__ output, int dim1, int dim2) {
    // Warp-level reduction variables
    float sum = 0.0f;
    float var_sum = 0.0f;
    
    // Each thread handles multiple spatial positions for better utilization
    const int batch = blockIdx.x;
    const int spatial_idx = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    const int y = spatial_idx / dim2;
    const int z = spatial_idx % dim2;
    
    if (y >= dim1 || z >= dim2) return;
    
    // Base offset for this spatial position
    const int base_offset = batch * FEATURES * dim1 * dim2 + y * dim2 + z;
    
    // Load all features into registers
    float f0 = input[base_offset + 0 * dim1 * dim2];
    float f1 = input[base_offset + 1 * dim1 * dim2];
    float f2 = input[base_offset + 2 * dim1 * dim2];
    float f3 = input[base_offset + 3 * dim1 * dim2];
    
    // Compute mean
    sum = f0 + f1 + f2 + f3;
    float mean = sum / FEATURES;
    
    // Compute variance
    float diff0 = f0 - mean;
    float diff1 = f1 - mean;
    float diff2 = f2 - mean;
    float diff3 = f3 - mean;
    var_sum = diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3;
    float variance = var_sum / FEATURES;
    
    // Normalize
    float inv_std = rsqrtf(variance + EPS);
    output[base_offset + 0 * dim1 * dim2] = diff0 * inv_std;
    output[base_offset + 1 * dim1 * dim2] = diff1 * inv_std;
    output[base_offset + 2 * dim1 * dim2] = diff2 * inv_std;
    output[base_offset + 3 * dim1 * dim2] = diff3 * inv_std;
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

bool compare_outputs(const float* output, const float* reference, size_t size, float tol = 1e-3f) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(output[i] - reference[i]) > tol) {
            return false;
        }
    }
    return true;
}

int main() {
    bool all_passed = true;

    for (int test_id = 0; test_id < NUM_TESTS; ++test_id) {
        int dim = DIMS[test_id];
        int size = BATCH_SIZE * FEATURES * dim * dim;
        size_t bytes = size * sizeof(float);

        float* h_input = new float[size];
        float* h_output = new float[size];
        float* h_ref = new float[size];

        std::string input_file = "data/input_" + std::to_string(test_id + 1) + ".bin";
        std::string ref_file   = "data/reference_" + std::to_string(test_id + 1) + ".bin";

        read_binary(input_file, h_input, size);
        read_binary(ref_file, h_ref, size);

        float *d_input, *d_output;
        cudaMalloc(&d_input, bytes);
        cudaMalloc(&d_output, bytes);
        cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

        dim3 grid(BATCH_SIZE, (dim * dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);
        layer_norm_kernel<<<grid, block>>>(d_input, d_output, dim, dim);

        cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

        if (!compare_outputs(h_output, h_ref, size)) {
            std::cout << "F" << std::endl;
            all_passed = false;

            cudaFree(d_input);
            cudaFree(d_output);
            delete[] h_input;
            delete[] h_output;
            delete[] h_ref;
            break;
        }

        cudaFree(d_input);
        cudaFree(d_output);
        delete[] h_input;
        delete[] h_output;
        delete[] h_ref;
    }

    if (all_passed) std::cout << "T" << std::endl;
    return 0;
}