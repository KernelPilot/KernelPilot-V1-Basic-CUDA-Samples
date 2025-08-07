#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <cmath>
#include <vector>

__global__ void reverse_array_kernel(float* __restrict__ data, const int n) {
    // Using 4 elements per thread for better memory utilization
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx >= n/2) return;

    // Coalesced reads of 4 elements at a time using float4
    float4 val = reinterpret_cast<float4*>(data)[idx/4];
    float4 opposite_val = reinterpret_cast<float4*>(data)[(n - idx - 4)/4];

    // Reverse the elements within the float4
    float4 result;
    result.x = opposite_val.w;
    result.y = opposite_val.z;
    result.z = opposite_val.y;
    result.w = opposite_val.x;

    float4 opposite_result;
    opposite_result.x = val.w;
    opposite_result.y = val.z;
    opposite_result.z = val.y;
    opposite_result.w = val.x;

    // Coalesced writes
    reinterpret_cast<float4*>(data)[idx/4] = result;
    reinterpret_cast<float4*>(data)[(n - idx - 4)/4] = opposite_result;
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

bool compare_outputs(const float* output, const float* reference, size_t size, float tolerance = 1e-5f) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(output[i] - reference[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

int main() {
    std::vector<size_t> Ns = {1 << 20, 1 << 22, 1 << 24, 1 << 26, 1 << 28};
    bool all_passed = true;

    for (int t = 0; t < Ns.size(); ++t) {
        size_t N = Ns[t];
        size_t bytes = N * sizeof(float);

        std::string input_file = "data/reverse_input_" + std::to_string(t + 1) + ".bin";
        std::string ref_file = "data/reverse_ref_" + std::to_string(t + 1) + ".bin";

        float* h_input = (float*)malloc(bytes);
        float* h_output_ref = (float*)malloc(bytes);

        read_binary(input_file, h_input, N);
        read_binary(ref_file, h_output_ref, N);

        float* d_data;
        cudaMalloc(&d_data, bytes);
        cudaMemcpy(d_data, h_input, bytes, cudaMemcpyHostToDevice);

        // Using 256 threads per block, each handling 4 elements
        int blocks = (N/8 + 255) / 256;  // Each thread handles 4 elements (2 swaps)
        reverse_array_kernel<<<blocks, 256>>>(d_data, N);
        cudaMemcpy(h_input, d_data, bytes, cudaMemcpyDeviceToHost);

        if (!compare_outputs(h_input, h_output_ref, N)) {
            std::cout << "F" << std::endl;
            all_passed = false;

            cudaFree(d_data);
            free(h_input);
            free(h_output_ref);
            break;
        }

        cudaFree(d_data);
        free(h_input);
        free(h_output_ref);
    }

    if (all_passed) std::cout << "T" << std::endl;
    return 0;
}