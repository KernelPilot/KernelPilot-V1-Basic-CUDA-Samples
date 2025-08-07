#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <fstream>
#include <vector>

#define D 10
#define BLOCK_SIZE 256
#define WARPS_PER_BLOCK (BLOCK_SIZE / 32)
#define TILE_SIZE 4

__global__ void compute_XTX_XTy_kernel(const float* __restrict__ X, const float* __restrict__ y, 
                                      float* __restrict__ XTX, float* __restrict__ XTy, int N) {
    // Warp and lane indices
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    // Each block computes one element of XTX and contributes to XTy
    const int i = blockIdx.x / D;  // row index of XTX
    const int j = blockIdx.x % D;  // column index of XTX
    
    __shared__ float warp_sums_XTX[WARPS_PER_BLOCK];
    __shared__ float warp_sums_XTy[WARPS_PER_BLOCK];
    
    // Initialize accumulators
    float xtx_val[TILE_SIZE] = {0.0f};
    float xty_val = 0.0f;
    
    // Pre-compute the starting position for this thread
    const int start_idx = threadIdx.x;
    const int stride = BLOCK_SIZE * TILE_SIZE;
    
    // Process multiple elements per thread with tiling for better ILP
    for (int sample_base = start_idx; sample_base < N; sample_base += stride) {
        #pragma unroll
        for (int t = 0; t < TILE_SIZE; t++) {
            const int sample = sample_base + t * BLOCK_SIZE;
            if (sample < N) {
                // Pre-load the features we need
                const float x_i = __ldg(&X[sample * D + i]);
                const float x_j = __ldg(&X[sample * D + j]);
                
                // Accumulate XTX value
                xtx_val[t] = __fmaf_rn(x_i, x_j, xtx_val[t]);
                
                // Only compute XTy for diagonal elements (i == j)
                if (i == j) {
                    const float y_val = __ldg(&y[sample]);
                    xty_val = __fmaf_rn(x_i, y_val, xty_val);
                }
            }
        }
    }
    
    // Reduce the tile values using FMA
    float xtx_total = 0.0f;
    #pragma unroll
    for (int t = 0; t < TILE_SIZE; t++) {
        xtx_total += xtx_val[t];
    }
    
    // Warp-level reduction for XTX using shuffle instructions
    for (int offset = 16; offset > 0; offset >>= 1) {
        xtx_total += __shfl_down_sync(0xFFFFFFFF, xtx_total, offset);
    }
    
    // Warp-level reduction for XTy (only for diagonal elements)
    if (i == j) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            xty_val += __shfl_down_sync(0xFFFFFFFF, xty_val, offset);
        }
    }
    
    // First lane in each warp stores the partial sum
    if (lane_id == 0) {
        warp_sums_XTX[warp_id] = xtx_total;
        if (i == j) {
            warp_sums_XTy[warp_id] = xty_val;
        }
    }
    __syncthreads();
    
    // Final reduction across warps (single warp does this)
    if (warp_id == 0) {
        float xtx_final = (lane_id < WARPS_PER_BLOCK) ? warp_sums_XTX[lane_id] : 0.0f;
        float xty_final = (i == j && lane_id < WARPS_PER_BLOCK) ? warp_sums_XTy[lane_id] : 0.0f;
        
        // Warp-level reduction
        for (int offset = 16; offset > 0; offset >>= 1) {
            xtx_final += __shfl_down_sync(0xFFFFFFFF, xtx_final, offset);
            if (i == j) {
                xty_final += __shfl_down_sync(0xFFFFFFFF, xty_final, offset);
            }
        }
        
        // First thread stores the result
        if (lane_id == 0) {
            atomicAdd(&XTX[i * D + j], xtx_final);
            if (i == j) {
                atomicAdd(&XTy[j], xty_final);
            }
        }
    }
}

bool solve_linear_system_cpu(const float* A, const float* b, float* x, int n) {
    std::vector<std::vector<float>> mat(n, std::vector<float>(n + 1));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) mat[i][j] = A[i * n + j];
        mat[i][n] = b[i];
    }

    for (int i = 0; i < n; ++i) {
        if (fabs(mat[i][i]) < 1e-6) return false;
        for (int j = i + 1; j < n; ++j) {
            float f = mat[j][i] / mat[i][i];
            for (int k = i; k <= n; ++k) {
                mat[j][k] -= f * mat[i][k];
            }
        }
    }

    for (int i = n - 1; i >= 0; --i) {
        x[i] = mat[i][n];
        for (int j = i + 1; j < n; ++j) {
            x[i] -= mat[i][j] * x[j];
        }
        x[i] /= mat[i][i];
    }
    return true;
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

bool compare_vector(const float* a, const float* b, int len, float tol = 1e-2f) {
    for (int i = 0; i < len; ++i) {
        if (fabs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

int main() {
    std::vector<size_t> Ns = {1<<14, 1<<16, 1<<18, 1<<20, 1<<22};
    bool all_match = true;

    for (int idx = 0; idx < Ns.size(); ++idx) {
        size_t N = Ns[idx];
        size_t X_bytes = N * D * sizeof(float);
        size_t y_bytes = N * sizeof(float);
        size_t beta_bytes = D * sizeof(float);
        
        std::string X_file = "data/ols_X_" + std::to_string(idx + 1) + ".bin";
        std::string y_file = "data/ols_y_" + std::to_string(idx + 1) + ".bin";
        std::string ref_file = "data/ols_ref_" + std::to_string(idx + 1) + ".bin";

        float *h_X = (float*)malloc(X_bytes);
        float *h_y = (float*)malloc(y_bytes);
        float *h_beta = (float*)malloc(beta_bytes);
        float *h_ref = (float*)malloc(beta_bytes);

        read_binary_float(X_file, h_X, N * D);
        read_binary_float(y_file, h_y, N);
        read_binary_float(ref_file, h_ref, D);

        float *d_X, *d_y, *d_XTX, *d_XTy;
        cudaMalloc(&d_X, X_bytes);
        cudaMalloc(&d_y, y_bytes);
        cudaMalloc(&d_XTX, D * D * sizeof(float));
        cudaMalloc(&d_XTy, D * sizeof(float));
        cudaMemcpy(d_X, h_X, X_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, h_y, y_bytes, cudaMemcpyHostToDevice);
        cudaMemset(d_XTX, 0, D * D * sizeof(float));
        cudaMemset(d_XTy, 0, D * sizeof(float));

        // Launch D*D blocks, each with BLOCK_SIZE threads
        compute_XTX_XTy_kernel<<<D*D, BLOCK_SIZE>>>(d_X, d_y, d_XTX, d_XTy, N);

        float* h_XTX = (float*)malloc(D * D * sizeof(float));
        float* h_XTy = (float*)malloc(D * sizeof(float));
        cudaMemcpy(h_XTX, d_XTX, D * D * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_XTy, d_XTy, D * sizeof(float), cudaMemcpyDeviceToHost);

        if (!solve_linear_system_cpu(h_XTX, h_XTy, h_beta, D) ||
            !compare_vector(h_beta, h_ref, D)) {
            std::cout << "F" << std::endl;
            all_match = false;
            break;
        }

        cudaFree(d_X); cudaFree(d_y); cudaFree(d_XTX); cudaFree(d_XTy);
        free(h_X); free(h_y); free(h_beta); free(h_ref); free(h_XTX); free(h_XTy);
    }

    if (all_match) std::cout << "T" << std::endl;
    return 0;
}