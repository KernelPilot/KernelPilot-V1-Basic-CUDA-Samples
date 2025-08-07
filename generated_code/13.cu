#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <cmath>
#include <vector>

__global__ void matrix_copy_kernel(const float* __restrict__ A, float* __restrict__ B, int N) {
    // Reduced tile size to fit within shared memory limits (48KB)
    const int TILE_DIM = 32;
    const int BLOCK_ROWS = 8;
    const int PADDING = 1; // Avoid bank conflicts
    
    __shared__ float tile[TILE_DIM][TILE_DIM + PADDING];
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y_base = blockIdx.y * TILE_DIM;
    
    // Each thread copies multiple elements for better memory coalescing
    #pragma unroll
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        int y = y_base + threadIdx.y + i;
        if (x < N && y < N) {
            tile[threadIdx.y + i][threadIdx.x] = A[y * N + x];
        }
    }
    
    __syncthreads();
    
    // Store with same indexing for perfect coalescing
    #pragma unroll
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        int y = y_base + threadIdx.y + i;
        if (x < N && y < N) {
            B[y * N + x] = tile[threadIdx.y + i][threadIdx.x];
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
        if (std::fabs(output[i] - reference[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

int main() {
    std::vector<int> Ns = {1 << 10, 1 << 11, 1 << 12, 1 << 13, 1 << 14};
    bool all_passed = true;

    for (int t = 0; t < Ns.size(); ++t) {
        int N = Ns[t];
        size_t num_elements = static_cast<size_t>(N) * N;
        size_t size_bytes = num_elements * sizeof(float);

        std::string input_file = "data/matrix_input_" + std::to_string(t + 1) + ".bin";
        std::string ref_file   = "data/matrix_ref_" + std::to_string(t + 1) + ".bin";

        float* h_A = (float*)malloc(size_bytes);
        float* h_B = (float*)malloc(size_bytes);
        float* h_B_ref = (float*)malloc(size_bytes);

        read_binary(input_file, h_A, num_elements);
        read_binary(ref_file, h_B_ref, num_elements);

        float *d_A, *d_B;
        cudaMalloc(&d_A, size_bytes);
        cudaMalloc(&d_B, size_bytes);

        cudaMemcpy(d_A, h_A, size_bytes, cudaMemcpyHostToDevice);

        // 32x8 thread blocks (256 threads) - fits within shared memory limits
        dim3 threads(32, 8);
        dim3 blocks((N + 31) / 32, (N + 31) / 32);
        
        matrix_copy_kernel<<<blocks, threads>>>(d_A, d_B, N);
        
        cudaDeviceSynchronize();
        cudaMemcpy(h_B, d_B, size_bytes, cudaMemcpyDeviceToHost);

        if (!compare_outputs(h_B, h_B_ref, num_elements)) {
            std::cout << "F" << std::endl;
            all_passed = false;

            cudaFree(d_A);
            cudaFree(d_B);
            free(h_A); free(h_B); free(h_B_ref);
            break;
        }

        cudaFree(d_A);
        cudaFree(d_B);
        free(h_A); free(h_B); free(h_B_ref);
    }

    if (all_passed) std::cout << "T" << std::endl;
    return 0;
}