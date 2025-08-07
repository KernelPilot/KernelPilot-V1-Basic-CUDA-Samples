#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <cooperative_groups.h>
#include <cuda_fp16.h>

struct Config {
    int N;
    int d_model;
    int h;
};

namespace cg = cooperative_groups;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

__device__ void softmax(float* input, int size, int tid, int block_size) {
    float max_val = -INFINITY;
    for (int i = tid; i < size; i += block_size) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    max_val = block_reduce_sum(max_val) / blockDim.x;

    float sum = 0.0f;
    for (int i = tid; i < size; i += block_size) {
        input[i] = expf(input[i] - max_val);
        sum += input[i];
    }
    sum = block_reduce_sum(sum) / blockDim.x;

    for (int i = tid; i < size; i += block_size) {
        input[i] /= sum;
    }
}

__global__ void multihead_attention_kernel(
    const float* __restrict__ Q, 
    const float* __restrict__ K, 
    const float* __restrict__ V,
    float* __restrict__ output, 
    int N, int d_model, int h)
{
    extern __shared__ float shared_mem[];
    float* attention_weights = shared_mem;
    float* head_results = shared_mem + N;

    const int d_k = d_model / h;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int head = tid / d_k;
    const int pos_in_head = tid % d_k;

    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(tb);

    // Initialize output
    if (tid < d_model) {
        output[row * d_model + tid] = 0.0f;
    }
    tb.sync();

    for (int hd = head; hd < h; hd += blockDim.x / d_k) {
        // Compute QK^T / sqrt(d_k) for this head
        for (int j = tid; j < N; j += blockDim.x) {
            float sum = 0.0f;
            for (int k = 0; k < d_k; ++k) {
                float q_val = Q[row * d_model + hd * d_k + k];
                float k_val = K[j * d_model + hd * d_k + k];
                sum += q_val * k_val;
            }
            attention_weights[j] = sum * __frsqrt_rn(static_cast<float>(d_k));
        }
        tb.sync();

        // Softmax
        softmax(attention_weights, N, tid, blockDim.x);
        tb.sync();

        // Compute attention * V
        if (tid % d_k == pos_in_head) {
            float result = 0.0f;
            for (int j = 0; j < N; ++j) {
                result += attention_weights[j] * V[j * d_model + hd * d_k + pos_in_head];
            }
            atomicAdd(&output[row * d_model + hd * d_k + pos_in_head], result);
        }
        tb.sync();
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

bool compare_outputs(const float* output, const float* reference, size_t size, float tol = 1e-2f) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(output[i] - reference[i]) > tol) return false;
    }
    return true;
}

int main() {
    std::vector<Config> configs = {
        {64,   32,  4},
        {128,  64,  8},
        {256, 128,  8},
        {384, 256, 16},
        {512, 512, 16},
    };

    bool all_passed = true;

    for (int t = 0; t < configs.size(); ++t) {
        int N = configs[t].N;
        int d_model = configs[t].d_model;
        int h = configs[t].h;
        size_t total = N * d_model;
        
        std::string qf = "data/Q_" + std::to_string(t + 1) + ".bin";
        std::string kf = "data/K_" + std::to_string(t + 1) + ".bin";
        std::string vf = "data/V_" + std::to_string(t + 1) + ".bin";
        std::string rf = "data/ref_out_" + std::to_string(t + 1) + ".bin";

        float *h_Q = new float[total];
        float *h_K = new float[total];
        float *h_V = new float[total];
        float *h_out = new float[total];
        float *h_ref = new float[total];

        read_binary(qf, h_Q, total);
        read_binary(kf, h_K, total);
        read_binary(vf, h_V, total);
        read_binary(rf, h_ref, total);

        float *d_Q, *d_K, *d_V, *d_out;
        cudaMalloc(&d_Q, total * sizeof(float));
        cudaMalloc(&d_K, total * sizeof(float));
        cudaMalloc(&d_V, total * sizeof(float));
        cudaMalloc(&d_out, total * sizeof(float));

        cudaMemcpy(d_Q, h_Q, total * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_K, h_K, total * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, h_V, total * sizeof(float), cudaMemcpyHostToDevice);

        dim3 grid(N);
        dim3 block(1024);  // Max threads per block for better occupancy
        size_t shared_mem_size = (N + d_model) * sizeof(float);
        multihead_attention_kernel<<<grid, block, shared_mem_size>>>(d_Q, d_K, d_V, d_out, N, d_model, h);
        cudaMemcpy(h_out, d_out, total * sizeof(float), cudaMemcpyDeviceToHost);

        if (!compare_outputs(h_out, h_ref, total)) {
            std::cout << "F" << std::endl;
            all_passed = false;

            cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_out);
            delete[] h_Q; delete[] h_K; delete[] h_V; delete[] h_out; delete[] h_ref;
            break;
        }

        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_out);
        delete[] h_Q; delete[] h_K; delete[] h_V; delete[] h_out; delete[] h_ref;
    }

    if (all_passed) std::cout << "T" << std::endl;
    return 0;
}