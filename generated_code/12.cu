#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include <cmath>
#include <vector>

__global__ void sort_kernel(float* data, int n) {
    extern __shared__ float s_data[];
    
    // Each block handles a segment of the array
    int tid = threadIdx.x;
    int block_start = blockIdx.x * blockDim.x;
    int idx = block_start + tid;
    
    // Load data into shared memory
    if (idx < n) {
        s_data[tid] = data[idx];
    }
    __syncthreads();

    // Parallel bubble sort within the block
    for (int i = 0; i < blockDim.x; ++i) {
        int sort_pos = tid * 2 + (i % 2);
        if (sort_pos + 1 < blockDim.x && (block_start + sort_pos + 1) < n) {
            if (s_data[sort_pos] > s_data[sort_pos + 1]) {
                float temp = s_data[sort_pos];
                s_data[sort_pos] = s_data[sort_pos + 1];
                s_data[sort_pos + 1] = temp;
            }
        }
        __syncthreads();
    }

    // Store sorted block back to global memory
    if (idx < n) {
        data[idx] = s_data[tid];
    }

    // Global synchronization point
    __syncthreads();

    // Final merge step (single thread)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Simple bubble sort for final correctness
        for (int i = 0; i < n - 1; ++i) {
            for (int j = 0; j < n - i - 1; ++j) {
                if (data[j] > data[j + 1]) {
                    float temp = data[j];
                    data[j] = data[j + 1];
                    data[j + 1] = temp;
                }
            }
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

bool compare_outputs(const float* output, const float* reference, size_t size, float tol = 1e-5f) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(output[i] - reference[i]) > tol) {
            return false;
        }
    }
    return true;
}

int main() {
    std::vector<size_t> Ns = {1 << 9, 1 << 10, 1 << 11, 1 << 12, 1 << 13};
    bool all_passed = true;

    for (int t = 0; t < Ns.size(); ++t) {
        size_t N = Ns[t];
        size_t bytes = N * sizeof(float);

        std::string input_file = "data/sort_input_" + std::to_string(t + 1) + ".bin";
        std::string ref_file = "data/sort_ref_" + std::to_string(t + 1) + ".bin";

        float* h_input = (float*)malloc(bytes);
        float* h_output = (float*)malloc(bytes);
        float* h_ref = (float*)malloc(bytes);

        read_binary(input_file, h_input, N);
        read_binary(ref_file, h_ref, N);

        float* d_data;
        cudaMalloc(&d_data, bytes);
        cudaMemcpy(d_data, h_input, bytes, cudaMemcpyHostToDevice);

        int threads = 512;
        int blocks = (N + threads - 1) / threads;
        size_t shared_mem_size = threads * sizeof(float);
        sort_kernel<<<blocks, threads, shared_mem_size>>>(d_data, N);

        cudaMemcpy(h_output, d_data, bytes, cudaMemcpyDeviceToHost);

        if (!compare_outputs(h_output, h_ref, N)) {
            std::cout << "F" << std::endl;
            all_passed = false;
            cudaFree(d_data);
            free(h_input); free(h_output); free(h_ref);
            break;
        }

        cudaFree(d_data);
        free(h_input); free(h_output); free(h_ref);
    }

    if (all_passed) std::cout << "T" << std::endl;
    return 0;
}