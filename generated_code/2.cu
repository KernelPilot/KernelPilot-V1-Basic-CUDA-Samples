#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>

#define K 4096
#define N 2048
#define TILE_WIDTH 32
#define NUM_TESTS 5
const int M_SIZES[NUM_TESTS] = {1024, 4096, 16384, 65536, 262144}; 

// Optimized matrix multiplication kernel with 4x4 tiling and prefetching
__global__ void Matrix_Multiplication_Kernel(const float* __restrict__ A, 
                                           const float* __restrict__ B, 
                                           float* __restrict__ C, 
                                           int M) {
    // 4x4 block tiling for better ILP and reduced shared memory bank conflicts
    const int TILE_SIZE = TILE_WIDTH / 4;
    __shared__ float As[TILE_WIDTH][TILE_WIDTH + 1];  // +1 for bank conflict avoidance
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH + 1];

    // Thread indices
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Each thread computes 4x4 elements of C
    int row = by * TILE_WIDTH + ty * 4;
    int col = bx * TILE_WIDTH + tx * 4;

    // Registers to hold the 4x4 block of C
    float c[4][4] = {{0}};

    // Loop over tiles
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        // Cooperative loading of 4x4 tiles into shared memory
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int load_row = ty * 4 + i;
            int load_col = tx * 4;
            int A_idx = row + i < M ? (row + i) * K + t * TILE_WIDTH + load_col : -1;
            int B_idx = col < N ? (t * TILE_WIDTH + load_row) * N + col : -1;

            if (A_idx >= 0 && load_col < TILE_WIDTH) {
                float4 A_val = reinterpret_cast<const float4*>(A)[A_idx / 4];
                As[load_row][load_col] = A_val.x;
                As[load_row][load_col + 1] = A_val.y;
                As[load_row][load_col + 2] = A_val.z;
                As[load_row][load_col + 3] = A_val.w;
            }

            if (B_idx >= 0 && load_row < TILE_WIDTH) {
                float4 B_val = reinterpret_cast<const float4*>(B)[B_idx / 4];
                Bs[load_row][load_col] = B_val.x;
                Bs[load_row][load_col + 1] = B_val.y;
                Bs[load_row][load_col + 2] = B_val.z;
                Bs[load_row][load_col + 3] = B_val.w;
            }
        }
        __syncthreads();

        // Compute 4x4 block matrix multiplication
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            float a[4], b[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = As[ty * 4 + i][k];
                b[i] = Bs[k][tx * 4 + i];
            }

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    c[i][j] += a[i] * b[j];
                }
            }
        }
        __syncthreads();
    }

    // Store the 4x4 block to global memory
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if ((row + i) < M && (col + j) < N) {
                C[(row + i) * N + (col + j)] = c[i][j];
            }
        }
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

bool compare_outputs(const float* output, const float* reference, size_t size, float tolerance = 1e-2f) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(output[i] - reference[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

int main() {
    bool all_passed = true;

    for (int test_id = 0; test_id < NUM_TESTS; ++test_id) {
        int M = M_SIZES[test_id];
        size_t size_A = M * K;
        size_t size_B = K * N;
        size_t size_C = M * N;

        float *h_A = new float[size_A];
        float *h_B = new float[size_B];
        float *h_C = new float[size_C];
        float *h_C_ref = new float[size_C];

        std::string A_file = "./data/matA_" + std::to_string(test_id + 1) + ".bin";
        std::string B_file = "./data/matB_" + std::to_string(test_id + 1) + ".bin";
        std::string C_file = "./data/matC_ref_" + std::to_string(test_id + 1) + ".bin";

        read_binary(A_file, h_A, size_A);
        read_binary(B_file, h_B, size_B);
        read_binary(C_file, h_C_ref, size_C);

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, size_A * sizeof(float));
        cudaMalloc(&d_B, size_B * sizeof(float));
        cudaMalloc(&d_C, size_C * sizeof(float));

        cudaMemcpy(d_A, h_A, size_A * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size_B * sizeof(float), cudaMemcpyHostToDevice);

        dim3 dimBlock(TILE_WIDTH / 4, TILE_WIDTH / 4);
        dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

        Matrix_Multiplication_Kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M);
        cudaMemcpy(h_C, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost);

        if (!compare_outputs(h_C, h_C_ref, size_C)) {
            std::cout << "F" << std::endl;
            all_passed = false;

            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            delete[] h_A;
            delete[] h_B;
            delete[] h_C;
            delete[] h_C_ref;
            break;
        }

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        delete[] h_C_ref;
    }

    if (all_passed) std::cout << "T" << std::endl;

    return 0;
}