#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include <cmath>
#include <vector>

#define TILE_DIM 32
#define BLOCK_ROWS 8
#define UNROLL_FACTOR 4

__global__ void matrix_transpose_kernel(const float* __restrict__ A, float* __restrict__ B, int rows, int cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM+1]; // +1 padding to avoid bank conflicts
    
    int blockIdx_x = blockIdx.x;
    int blockIdx_y = blockIdx.y;
    
    // Coalesced read from global memory
    int x_in = blockIdx_x * TILE_DIM + threadIdx.x;
    int y_in = blockIdx_y * TILE_DIM + threadIdx.y;
    
    #pragma unroll
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (x_in < cols && (y_in + i) < rows) {
            tile[threadIdx.y + i][threadIdx.x] = A[(y_in + i) * cols + x_in];
        }
    }
    
    __syncthreads();
    
    // Transposed write with coalescing
    int x_out = blockIdx_y * TILE_DIM + threadIdx.x;
    int y_out = blockIdx_x * TILE_DIM + threadIdx.y;
    
    #pragma unroll
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (x_out < rows && (y_out + i) < cols) {
            B[(y_out + i) * rows + x_out] = tile[threadIdx.x][threadIdx.y + i];
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
    std::vector<std::pair<int, int>> sizes = {
        {1 << 10, 1 << 10},
        {1 << 11, 1 << 11},
        {1 << 12, 1 << 12},
        {1 << 13, 1 << 13},
        {1 << 14, 1 << 14}
    };

    bool all_passed = true;

    for (int t = 0; t < sizes.size(); ++t) {
        int ROWS = sizes[t].first;
        int COLS = sizes[t].second;

        size_t input_elems = ROWS * COLS;
        size_t output_elems = COLS * ROWS;
        size_t input_bytes = input_elems * sizeof(float);
        size_t output_bytes = output_elems * sizeof(float);
        
        std::string input_file = "data/matrix_input_" + std::to_string(t + 1) + ".bin";
        std::string ref_file   = "data/matrix_ref_" + std::to_string(t + 1) + ".bin";

        float* h_input = (float*)malloc(input_bytes);
        float* h_output = (float*)malloc(output_bytes);
        float* h_ref = (float*)malloc(output_bytes);

        read_binary(input_file, h_input, input_elems);
        read_binary(ref_file, h_ref, output_elems);

        float *d_input, *d_output;
        cudaMalloc(&d_input, input_bytes);
        cudaMalloc(&d_output, output_bytes);
        cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice);

        dim3 threads(TILE_DIM, BLOCK_ROWS);
        dim3 blocks((COLS + TILE_DIM - 1) / TILE_DIM, (ROWS + TILE_DIM - 1) / TILE_DIM);
        matrix_transpose_kernel<<<blocks, threads>>>(d_input, d_output, ROWS, COLS);
        cudaDeviceSynchronize();
        cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);

        if (!compare_outputs(h_output, h_ref, output_elems)) {
            std::cout << "F" << std::endl;
            all_passed = false;

            cudaFree(d_input);
            cudaFree(d_output);
            free(h_input);
            free(h_output);
            free(h_ref);
            break;
        }

        cudaFree(d_input);
        cudaFree(d_output);
        free(h_input);
        free(h_output);
        free(h_ref);
    }

    if (all_passed) std::cout << "T" << std::endl;

    return 0;
}