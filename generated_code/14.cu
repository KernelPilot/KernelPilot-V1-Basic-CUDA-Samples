#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <cmath>
#include <vector>

#define FULL_MASK 0xffffffff

template <int warpSize>
__inline__ __device__ float warpReduce(float val) {
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return val;
}

template <int blockSize, int elementsPerThread>
__global__ void reduction_kernel(const float* __restrict__ input, float* __restrict__ output, size_t n) {
    constexpr int warpsPerBlock = blockSize / 32;
    __shared__ float warpSums[warpsPerBlock];
    
    unsigned int tid = threadIdx.x;
    unsigned int warpId = tid / 32;
    unsigned int laneId = tid % 32;
    unsigned int i = blockIdx.x * (blockSize * elementsPerThread) + tid;
    unsigned int gridStride = blockSize * elementsPerThread * gridDim.x;
    
    float sum = 0.0f;
    
    // Process multiple elements per thread with unrolled loads
    #pragma unroll
    for (int j = 0; j < elementsPerThread; j++) {
        if (i + j * blockSize < n) {
            sum += input[i + j * blockSize];
        }
    }
    i += gridStride;
    
    // Grid-stride loop for remaining elements
    while (i < n) {
        #pragma unroll 4
        for (int j = 0; j < elementsPerThread && (i + j * blockSize) < n; j++) {
            sum += input[i + j * blockSize];
        }
        i += gridStride;
    }
    
    // Warp-level reduction
    sum = warpReduce<32>(sum);
    
    // Store warp sum to shared memory
    if (laneId == 0) {
        warpSums[warpId] = sum;
    }
    __syncthreads();
    
    // First warp reduces all warp sums
    if (warpId == 0) {
        sum = laneId < warpsPerBlock ? warpSums[laneId] : 0.0f;
        sum = warpReduce<warpsPerBlock>(sum);
        
        // Single atomic add per block
        if (laneId == 0) {
            atomicAdd(output, sum);
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

bool compare_scalar(float a, float b, float tol = 1e-2f) {
    return std::fabs(a - b) < tol;
}

int main() {
    std::vector<size_t> Ns = {1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18};
    bool all_passed = true;

    for (int t = 0; t < Ns.size(); ++t) {
        size_t N = Ns[t];
        size_t input_size = N * sizeof(float);

        std::string input_file = "data/reduce_input_" + std::to_string(t + 1) + ".bin";
        std::string ref_file   = "data/reduce_ref_" + std::to_string(t + 1) + ".bin";

        float* h_input = (float*)malloc(input_size);
        float h_ref_sum;
        read_binary(input_file, h_input, N);
        read_binary(ref_file, &h_ref_sum, 1);

        float *d_input, *d_sum;
        cudaMalloc(&d_input, input_size);
        cudaMalloc(&d_sum, sizeof(float));
        cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
        cudaMemset(d_sum, 0, sizeof(float));

        constexpr int blockSize = 256;
        constexpr int elementsPerThread = 8;
        int maxBlocks = 2048;
        int blocks = min(maxBlocks, (int)(N + blockSize * elementsPerThread - 1) / (blockSize * elementsPerThread));
        
        reduction_kernel<blockSize, elementsPerThread><<<blocks, blockSize>>>(d_input, d_sum, N);

        float h_sum = 0.0f;
        cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

        if (!compare_scalar(h_sum, h_ref_sum)) {
            std::cout << "F" << std::endl;
            all_passed = false;
            cudaFree(d_input); cudaFree(d_sum); free(h_input);
            break;
        }

        cudaFree(d_input);
        cudaFree(d_sum);
        free(h_input);
    }

    if (all_passed) std::cout << "T" << std::endl;
    return 0;
}