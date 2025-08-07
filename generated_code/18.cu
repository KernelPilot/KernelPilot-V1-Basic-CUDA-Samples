#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <fstream>
#include <vector>

#define THREADS_PER_BLOCK 256
#define ELEMENTS_PER_THREAD 16
#define UNROLL_FACTOR 4

__global__ void monte_carlo_kernel(const float* __restrict__ y_values, float* __restrict__ integral_sum, int N) {
    __shared__ float sdata[THREADS_PER_BLOCK];
    
    unsigned int tid = threadIdx.x;
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;
    unsigned int i = blockIdx.x * (blockDim.x * ELEMENTS_PER_THREAD) + threadIdx.x;
    
    float sum[UNROLL_FACTOR] = {0.0f};
    
    // Process multiple elements per thread with unrolled loops
    #pragma unroll
    for (int j = 0; j < ELEMENTS_PER_THREAD; j += UNROLL_FACTOR) {
        #pragma unroll
        for (int k = 0; k < UNROLL_FACTOR; k++) {
            unsigned int idx = i + (j + k) * blockDim.x;
            if (idx < N) {
                sum[k] += y_values[idx];
            }
        }
    }
    
    // Combine partial sums
    float thread_sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < UNROLL_FACTOR; k++) {
        thread_sum += sum[k];
    }
    
    // Warp-level reduction using shuffle instructions
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);
    }
    
    // First thread in each warp stores to shared memory
    if (lane_id == 0) {
        sdata[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (warp_id == 0) {
        float warp_sum = (lane_id < blockDim.x / 32) ? sdata[lane_id] : 0.0f;
        
        for (int offset = 8; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, offset);
        }
        
        if (lane_id == 0) {
            atomicAdd(integral_sum, warp_sum);
        }
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

bool compare_scalar(float a, float b, float tol = 1e-2f) {
    return fabs(a - b) < tol;
}

int main() {
    std::vector<size_t> Ns = {1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22};
    const float a = 0.0f, b = 1.0f;
    bool all_match = true;

    for (int idx = 0; idx < Ns.size(); ++idx) {
        size_t N = Ns[idx];
        size_t data_bytes = N * sizeof(float);

        std::string y_file   = "data/mc_y_"   + std::to_string(idx + 1) + ".bin";
        std::string ref_file = "data/mc_ref_" + std::to_string(idx + 1) + ".bin";

        float* h_y = (float*)malloc(data_bytes);
        float h_ref;

        read_binary_float(y_file, h_y, N);
        read_binary_float(ref_file, &h_ref, 1);

        float *d_y, *d_integral;
        cudaMalloc(&d_y, data_bytes);
        cudaMalloc(&d_integral, sizeof(float));
        cudaMemcpy(d_y, h_y, data_bytes, cudaMemcpyHostToDevice);
        cudaMemset(d_integral, 0, sizeof(float));

        int threads = THREADS_PER_BLOCK;
        int blocks = (N + threads * ELEMENTS_PER_THREAD - 1) / (threads * ELEMENTS_PER_THREAD);
        monte_carlo_kernel<<<blocks, threads>>>(d_y, d_integral, N);

        float h_result;
        cudaMemcpy(&h_result, d_integral, sizeof(float), cudaMemcpyDeviceToHost);
        h_result = h_result * (b - a) / N;

        if (!compare_scalar(h_result, h_ref)) {
            std::cout << "F" << std::endl;
            all_match = false;
            cudaFree(d_y); cudaFree(d_integral);
            free(h_y);
            break;
        }

        cudaFree(d_y); cudaFree(d_integral);
        free(h_y);
    }

    if (all_match) std::cout << "T" << std::endl;
    return 0;
}