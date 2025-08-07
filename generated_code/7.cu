#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>

#define FULL_MASK 0xffffffff
#define UNROLL_FACTOR 8
#define BLOCK_SIZE 256  // Optimal for Ampere architecture

__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return val;
}

__global__ void mseKernelOpt(const float* __restrict__ predictions,
                            const float* __restrict__ targets,
                            size_t numElements,
                            float* __restrict__ result) {
    // Shared memory for partial sums (one per warp)
    __shared__ float warpSums[BLOCK_SIZE / 32];
    
    unsigned int tid = threadIdx.x;
    unsigned int warpId = tid / 32;
    unsigned int laneId = tid % 32;
    
    // Grid-stride loop with unrolling
    float threadSum = 0.0f;
    unsigned int idx = blockIdx.x * BLOCK_SIZE * UNROLL_FACTOR + tid;
    
    #pragma unroll
    for (int i = 0; i < UNROLL_FACTOR; i++) {
        unsigned int elementIdx = idx + i * BLOCK_SIZE;
        if (elementIdx < numElements) {
            float diff = predictions[elementIdx] - targets[elementIdx];
            threadSum += diff * diff;
        }
    }
    
    // Warp-level reduction
    float warpSum = warpReduceSum(threadSum);
    
    // First lane in each warp stores to shared memory
    if (laneId == 0) {
        warpSums[warpId] = warpSum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (warpId == 0) {
        float val = (laneId < BLOCK_SIZE / 32) ? warpSums[laneId] : 0.0f;
        float blockSum = warpReduceSum(val);
        
        // Single atomic add per block
        if (laneId == 0) {
            atomicAdd(result, blockSum);
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
    std::vector<size_t> sizes = {1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18};
    bool all_passed = true;

    for (int t = 0; t < sizes.size(); ++t) {
        size_t N = sizes[t];
        size_t input_size = N * sizeof(float);
        
        std::string pred_file = "data/mse_preds_" + std::to_string(t + 1) + ".bin";
        std::string target_file = "data/mse_targets_" + std::to_string(t + 1) + ".bin";
        std::string ref_file = "data/mse_ref_" + std::to_string(t + 1) + ".bin";

        float* h_preds = (float*)malloc(input_size);
        float* h_targets = (float*)malloc(input_size);
        float h_mse_ref;

        read_binary(pred_file, h_preds, N);
        read_binary(target_file, h_targets, N);
        read_binary(ref_file, &h_mse_ref, 1);

        float *d_preds, *d_targets, *d_sum;
        cudaMalloc(&d_preds, input_size);
        cudaMalloc(&d_targets, input_size);
        cudaMalloc(&d_sum, sizeof(float));
        cudaMemcpy(d_preds, h_preds, input_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_targets, h_targets, input_size, cudaMemcpyHostToDevice);
        cudaMemset(d_sum, 0, sizeof(float));

        // Calculate grid size with unrolling factor
        int gridSize = (N + BLOCK_SIZE * UNROLL_FACTOR - 1) / (BLOCK_SIZE * UNROLL_FACTOR);
        mseKernelOpt<<<gridSize, BLOCK_SIZE>>>(d_preds, d_targets, N, d_sum);

        float h_sum = 0.0f;
        cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
        float mse = h_sum / N;

        if (!compare_scalar(mse, h_mse_ref)) {
            std::cout << "F" << std::endl;
            all_passed = false;
            cudaFree(d_preds); cudaFree(d_targets); cudaFree(d_sum);
            free(h_preds); free(h_targets);
            break;
        }

        cudaFree(d_preds); cudaFree(d_targets); cudaFree(d_sum);
        free(h_preds); free(h_targets);
    }

    if (all_passed) std::cout << "T" << std::endl;
    return 0;
}