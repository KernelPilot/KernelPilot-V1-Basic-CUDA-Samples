#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <cmath>
#include <vector>
#include <cooperative_groups.h>
#include <cuda_fp16.h>

namespace cg = cooperative_groups;

#define BLOCK_SIZE 256
#define UNROLL_FACTOR 8
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE/WARP_SIZE)

__device__ __forceinline__ float warpReduceSum(float val, cg::thread_block_tile<WARP_SIZE> tile) {
    #pragma unroll
    for (int offset = tile.size()/2; offset > 0; offset >>= 1) {
        val += tile.shfl_down(val, offset);
    }
    return val;
}

__global__ void dot_product_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ result,
                                  int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> tile = cg::tiled_partition<WARP_SIZE>(block);

    __shared__ float warp_sums[NUM_WARPS];
    
    float thread_sum = 0.0f;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x * UNROLL_FACTOR;

    // Main computation with aggressive unrolling and software prefetching
    int i = tid;
    for (; i + (UNROLL_FACTOR-1)*stride < n; i += stride) {
        #pragma unroll
        for (int j = 0; j < UNROLL_FACTOR; j++) {
            float a = A[i + j*stride];
            float b = B[i + j*stride];
            thread_sum += a * b;
        }
    }
    // Handle remaining elements
    for (; i < n; i += blockDim.x * gridDim.x) {
        thread_sum += A[i] * B[i];
    }

    // Warp-level reduction
    float warp_sum = warpReduceSum(thread_sum, tile);

    // Store warp sums to shared memory
    if (tile.thread_rank() == 0) {
        warp_sums[tile.meta_group_rank()] = warp_sum;
    }
    __syncthreads();

    // Final reduction by first warp
    if (tile.meta_group_rank() == 0) {
        float block_sum = tile.thread_rank() < NUM_WARPS ? warp_sums[tile.thread_rank()] : 0.0f;
        block_sum = warpReduceSum(block_sum, tile);
        
        if (tile.thread_rank() == 0) {
            atomicAdd(result, block_sum);
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

bool compare_scalar(float a, float b, float tol = 1e-1f) {
    return std::fabs(a - b) < tol;
}

int main() {
    std::vector<size_t> Ns = {1<<16, 1<<17, 1<<18, 1<<19, 1<<20};
    bool all_passed = true;

    for (int i = 0; i < Ns.size(); ++i) {
        size_t N = Ns[i];
        size_t bytes = N * sizeof(float);

        std::string prefix = "./data/dot_";
        std::string a_file = prefix + "input_a_" + std::to_string(i+1) + ".bin";
        std::string b_file = prefix + "input_b_" + std::to_string(i+1) + ".bin";
        std::string ref_file = prefix + "ref_" + std::to_string(i+1) + ".bin";

        float* h_A = (float*)malloc(bytes);
        float* h_B = (float*)malloc(bytes);
        float h_ref;

        read_binary(a_file, h_A, N);
        read_binary(b_file, h_B, N);
        read_binary(ref_file, &h_ref, 1);

        float *d_A, *d_B, *d_result;
        cudaMalloc(&d_A, bytes);
        cudaMalloc(&d_B, bytes);
        cudaMalloc(&d_result, sizeof(float));
        cudaMemset(d_result, 0, sizeof(float));

        cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

        int threads = BLOCK_SIZE;
        int blocks = (N + threads * UNROLL_FACTOR - 1) / (threads * UNROLL_FACTOR);
        blocks = max(blocks, 160);  // Ensure full GPU utilization
        
        // Configure kernel for optimal cache behavior
        cudaFuncSetCacheConfig(dot_product_kernel, cudaFuncCachePreferShared);
        cudaFuncSetAttribute(dot_product_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, BLOCK_SIZE*sizeof(float));
        
        dot_product_kernel<<<blocks, threads>>>(d_A, d_B, d_result, N);

        float h_result = 0.0f;
        cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

        if (!compare_scalar(h_result, h_ref)) {
            std::cout << "F" << std::endl;
            all_passed = false;
            cudaFree(d_A); cudaFree(d_B); cudaFree(d_result);
            free(h_A); free(h_B);
            break;
        }

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_result);
        free(h_A); free(h_B);
    }

    if (all_passed) std::cout << "T" << std::endl;
    return 0;
}