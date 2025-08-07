#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <cmath>
#include <vector>

__global__ void relu_kernel(const float4* __restrict__ input, float4* __restrict__ output, int size) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 0x1F;
    const int warp_offset = (blockIdx.x * (blockDim.x >> 5)) + warp_id;
    const int elements_per_warp = 2;  // Process 2 float4 vectors per warp (8 floats)
    
    #pragma unroll
    for (int i = 0; i < elements_per_warp; i++) {
        const int idx = (warp_offset * elements_per_warp + i) * 32 + lane_id;
        if (idx < (size >> 2)) {
            float4 vec = __ldg(&input[idx]);
            vec.x = vec.x * (vec.x > 0.0f);
            vec.y = vec.y * (vec.y > 0.0f);
            vec.z = vec.z * (vec.z > 0.0f);
            vec.w = vec.w * (vec.w > 0.0f);
            output[idx] = vec;
        }
    }
}

void read_binary(const std::string& filename, float* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open file: " << filename << std::endl;
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
        size_t input_size = N * sizeof(float);

        std::string input_file = "data/relu_input_" + std::to_string(t + 1) + ".bin";
        std::string ref_file   = "data/relu_ref_" + std::to_string(t + 1) + ".bin";

        float* h_input      = (float*)malloc(input_size);
        float* h_output     = (float*)malloc(input_size);
        float* h_output_ref = (float*)malloc(input_size);

        read_binary(input_file, h_input, N);
        read_binary(ref_file, h_output_ref, N);

        float4 *d_input, *d_output;
        cudaMalloc(&d_input, input_size);
        cudaMalloc(&d_output, input_size);
        cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);

        const int threads = 256;  // 8 warps per block
        const int warps_per_block = threads >> 5;
        const int elements_per_warp = 2;
        const int elements_per_block = warps_per_block * elements_per_warp * 32;
        int blocks = ((N >> 2) + elements_per_block - 1) / elements_per_block;
        
        relu_kernel<<<blocks, threads>>>(d_input, d_output, N);
        cudaMemcpy(h_output, d_output, input_size, cudaMemcpyDeviceToHost);

        if (!compare_outputs(h_output, h_output_ref, N)) {
            std::cout << "F" << std::endl;
            all_passed = false;

            cudaFree(d_input); cudaFree(d_output);
            free(h_input); free(h_output); free(h_output_ref);
            break;
        }

        cudaFree(d_input); cudaFree(d_output);
        free(h_input); free(h_output); free(h_output_ref);
    }

    if (all_passed) std::cout << "T" << std::endl;
    return 0;
}