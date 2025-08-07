#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <cmath>
#include <vector>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

__global__ void topk_selection_kernel(const float* input, float* output, int k, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    __shared__ float shared_max[32];
    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid % 32;

    // Phase 1: Find global maximum
    float thread_max = -INFINITY;
    for (int i = tid; i < n; i += blockDim.x) {
        thread_max = fmaxf(thread_max, input[i]);
    }
    
    // Warp-level reduction
    float warp_max = cg::reduce(warp, thread_max, cg::greater<float>());
    
    if (lane == 0) {
        shared_max[wid] = warp_max;
    }
    block.sync();
    
    // Final reduction across warps
    if (wid == 0) {
        thread_max = (lane < blockDim.x / 32) ? shared_max[lane] : -INFINITY;
        warp_max = cg::reduce(warp, thread_max, cg::greater<float>());
        
        if (lane == 0) {
            output[0] = warp_max;
        }
    }
    block.sync();

    if (k == 1) return;

    // Phase 2: Find remaining top-k elements
    for (int current_k = 1; current_k < k; current_k++) {
        float next_max = -INFINITY;
        float threshold = output[current_k - 1];
        
        for (int i = tid; i < n; i += blockDim.x) {
            float val = input[i];
            if (val < threshold) {
                next_max = fmaxf(next_max, val);
            }
        }
        
        // Warp-level reduction
        warp_max = cg::reduce(warp, next_max, cg::greater<float>());
        
        if (lane == 0) {
            shared_max[wid] = warp_max;
        }
        block.sync();
        
        // Final reduction across warps
        if (wid == 0) {
            next_max = (lane < blockDim.x / 32) ? shared_max[lane] : -INFINITY;
            warp_max = cg::reduce(warp, next_max, cg::greater<float>());
            
            if (lane == 0) {
                output[current_k] = warp_max;
            }
        }
        block.sync();
    }
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

bool compare_outputs(const float* output, const float* reference, size_t size, float tol = 1e-5f) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(output[i] - reference[i]) > tol) return false;
    }
    return true;
}

int main() {
    std::vector<int> Ns = {1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18};
    std::vector<int> Ks = {32, 64, 128, 256, 512};

    bool all_passed = true;

    for (int t = 0; t < Ns.size(); ++t) {
        int N = Ns[t];
        int K = Ks[t];

        size_t input_bytes = N * sizeof(float);
        size_t output_bytes = K * sizeof(float);

        std::string input_file = "data/topk_input_" + std::to_string(t + 1) + ".bin";
        std::string ref_file   = "data/topk_ref_" + std::to_string(t + 1) + ".bin";

        float* h_input = (float*)malloc(input_bytes);
        float* h_output = (float*)malloc(output_bytes);
        float* h_ref = (float*)malloc(output_bytes);

        read_binary(input_file, h_input, N);
        read_binary(ref_file, h_ref, K);

        float* d_input;
        float* d_output;
        cudaMalloc(&d_input, input_bytes);
        cudaMalloc(&d_output, output_bytes);

        cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice);

        topk_selection_kernel<<<1, 256>>>(d_input, d_output, K, N);
        cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);

        if (!compare_outputs(h_output, h_ref, K)) {
            std::cout << "F" << std::endl;
            all_passed = false;
            cudaFree(d_input); cudaFree(d_output);
            free(h_input); free(h_output); free(h_ref);
            break;
        }

        cudaFree(d_input); cudaFree(d_output);
        free(h_input); free(h_output); free(h_ref);
    }

    if (all_passed) std::cout << "T" << std::endl;
    return 0;
}