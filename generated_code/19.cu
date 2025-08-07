#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <vector>

#define MAX_BINS 256
#define THREADS_PER_BLOCK 256
#define UNROLL_FACTOR 16
#define WARP_SIZE 32
#define MAX_WARPS_PER_BLOCK (THREADS_PER_BLOCK/WARP_SIZE)

__global__ void histogram_kernel(const int* __restrict__ input, int* __restrict__ histogram, int N, int num_bins) {
    // Warp-private histograms in shared memory (padded to avoid bank conflicts)
    __shared__ int warp_histograms[MAX_WARPS_PER_BLOCK][MAX_BINS + 32];
    
    // Initialize warp-private histograms
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    for (int bin = lane_id; bin < num_bins; bin += WARP_SIZE) {
        warp_histograms[warp_id][bin] = 0;
    }
    __syncthreads();
    
    // Process multiple elements per thread with aggressive unrolling
    const int tid = blockIdx.x * blockDim.x * UNROLL_FACTOR + threadIdx.x;
    
    #pragma unroll
    for (int i = 0; i < UNROLL_FACTOR; i++) {
        int idx = tid + i * blockDim.x;
        if (idx < N) {
            int val = input[idx];
            if (val >= 0 && val < num_bins) {
                atomicAdd(&warp_histograms[warp_id][val], 1);
            }
        }
    }
    __syncthreads();
    
    // Warp-level reduction to global memory (coalesced writes)
    for (int bin = lane_id; bin < num_bins; bin += WARP_SIZE) {
        int count = 0;
        
        // Reduce across warps in the block
        for (int w = 0; w < MAX_WARPS_PER_BLOCK; w++) {
            count += warp_histograms[w][bin];
        }
        
        // Single atomic per bin per block
        if (count > 0 && warp_id == 0) {
            atomicAdd(&histogram[bin], count);
        }
    }
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

bool compare_histogram(const int* a, const int* b, int num_bins) {
    for (int i = 0; i < num_bins; ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

int main() {
    std::vector<size_t> Ns = {1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24};
    int num_bins = 256;
    bool all_match = true;

    for (int idx = 0; idx < Ns.size(); ++idx) {
        size_t N = Ns[idx];
        size_t input_bytes = N * sizeof(int);
        size_t hist_bytes = num_bins * sizeof(int);

        std::string input_file = "data/hist_input_" + std::to_string(idx + 1) + ".bin";
        std::string ref_file   = "data/hist_ref_"   + std::to_string(idx + 1) + ".bin";

        int* h_input = (int*)malloc(input_bytes);
        int* h_ref   = (int*)malloc(hist_bytes);
        int* h_output = (int*)malloc(hist_bytes);
        memset(h_output, 0, hist_bytes);

        read_binary_int(input_file, h_input, N);
        read_binary_int(ref_file, h_ref, num_bins);

        int *d_input, *d_histogram;
        cudaMalloc(&d_input, input_bytes);
        cudaMalloc(&d_histogram, hist_bytes);
        cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice);
        cudaMemset(d_histogram, 0, hist_bytes);

        int threads = THREADS_PER_BLOCK;
        int blocks = (N + threads * UNROLL_FACTOR - 1) / (threads * UNROLL_FACTOR);
        histogram_kernel<<<blocks, threads>>>(d_input, d_histogram, N, num_bins);

        cudaMemcpy(h_output, d_histogram, hist_bytes, cudaMemcpyDeviceToHost);

        if (!compare_histogram(h_output, h_ref, num_bins)) {
            std::cout << "F" << std::endl;
            all_match = false;
            cudaFree(d_input); cudaFree(d_histogram);
            free(h_input); free(h_ref); free(h_output);
            break;
        }

        cudaFree(d_input); cudaFree(d_histogram);
        free(h_input); free(h_ref); free(h_output);
    }

    if (all_match) std::cout << "T" << std::endl;
    return 0;
}