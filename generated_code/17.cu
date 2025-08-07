#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <fstream>
#include <vector>

#define C 10
#define FULL_MASK 0xffffffff
#define BLOCK_SIZE 256  // Optimal for Ampere architecture

__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return val;
}

__global__ void cross_entropy_kernel(const float* __restrict__ logits,
                                   const int* __restrict__ labels,
                                   float* __restrict__ loss,
                                   int N) {
    // Block-level reduction in shared memory
    __shared__ float smem[BLOCK_SIZE / 32];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float thread_loss = 0.0f;

    if (idx < N) {
        // Coalesced memory access for logits and labels
        const float* sample_logits = logits + idx * C;
        int true_label = labels[idx];

        // Find max logit with vectorized comparison
        float max_logit = sample_logits[0];
        #pragma unroll
        for (int i = 1; i < C; ++i) {
            max_logit = fmaxf(max_logit, sample_logits[i]);
        }

        // Compute log(sum(exp(logits - max_logit))) + max_logit
        // Using fast math intrinsics where possible
        float log_sum_exp = 0.0f;
        #pragma unroll
        for (int i = 0; i < C; ++i) {
            log_sum_exp += __expf(sample_logits[i] - max_logit);
        }
        log_sum_exp = max_logit + __logf(log_sum_exp);

        // Compute loss for this sample
        thread_loss = log_sum_exp - sample_logits[true_label];
    }

    // Warp-level reduction
    float warp_sum = warp_reduce(thread_loss);

    // First thread in warp stores to shared memory
    if (tid % 32 == 0) {
        smem[tid / 32] = warp_sum;
    }
    __syncthreads();

    // Final reduction across warps in block
    if (tid < BLOCK_SIZE / 32) {
        float block_sum = smem[tid];
        block_sum = warp_reduce(block_sum);
        
        // Single atomic add per block
        if (tid == 0) {
            atomicAdd(loss, block_sum);
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

void read_binary_int(const std::string& filename, int* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open: " << filename << std::endl;
        exit(1);
    }
    in.read(reinterpret_cast<char*>(data), size * sizeof(int));
    in.close();
}

// test
bool compare_scalar(float a, float b, float tol = 1e-2f) {
    return fabs(a - b) < tol;
}

int main() {
    std::vector<size_t> Ns = {1<<14, 1<<16, 1<<18, 1<<20, 1<<22};
    bool all_match = true;

    for (int idx = 0; idx < Ns.size(); ++idx) {
        size_t N = Ns[idx];
        size_t logits_size = N * C;
        size_t logits_bytes = logits_size * sizeof(float);
        size_t labels_bytes = N * sizeof(int);

        // test
        std::string logits_file = "data/ce_logits_" + std::to_string(idx + 1) + ".bin";
        std::string labels_file = "data/ce_labels_" + std::to_string(idx + 1) + ".bin";
        std::string ref_file    = "data/ce_ref_"    + std::to_string(idx + 1) + ".bin";

        float* h_logits = (float*)malloc(logits_bytes);
        int* h_labels   = (int*)malloc(labels_bytes);
        float h_ref;

        read_binary_float(logits_file, h_logits, logits_size);
        read_binary_int(labels_file, h_labels, N);
        read_binary_float(ref_file, &h_ref, 1);

        float *d_logits, *d_loss;
        int* d_labels;
        cudaMalloc(&d_logits, logits_bytes);
        cudaMalloc(&d_labels, labels_bytes);
        cudaMalloc(&d_loss, sizeof(float));
        cudaMemcpy(d_logits, h_logits, logits_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_labels, h_labels, labels_bytes, cudaMemcpyHostToDevice);
        cudaMemset(d_loss, 0, sizeof(float));

        int threads = BLOCK_SIZE;
        int blocks = (N + threads - 1) / threads;
        cross_entropy_kernel<<<blocks, threads, (BLOCK_SIZE/32)*sizeof(float)>>>(d_logits, d_labels, d_loss, N);

        float h_loss;
        cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
        h_loss /= N;  // Compute average loss

        if (!compare_scalar(h_loss, h_ref)) {
            std::cout << "F" << std::endl;
            all_match = false;
            cudaFree(d_logits); cudaFree(d_labels); cudaFree(d_loss);
            free(h_logits); free(h_labels);
            break;
        }

        cudaFree(d_logits); cudaFree(d_labels); cudaFree(d_loss);
        free(h_logits); free(h_labels);
    }

    if (all_match) std::cout << "T" << std::endl;
    return 0;
}