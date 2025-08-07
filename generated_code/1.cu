#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#define BATCH_SIZE 16
#define NUM_TESTS 5

const int DIMS[NUM_TESTS] = {1024, 4096, 16384, 65536, 262144};  
#define TOLERANCE 1e-5f

__global__ void sigmoid_kernel(const float* __restrict__ input, float* __restrict__ output, int total_size) {
    const int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    
    if (idx + 3 < total_size) {
        // Vectorized load of 4 contiguous floats
        float4 vec = reinterpret_cast<const float4*>(input)[idx/4];
        
        // Compute sigmoid using fast intrinsics
        float4 result;
        result.x = __fdividef(1.0f, 1.0f + __expf(-vec.x));
        result.y = __fdividef(1.0f, 1.0f + __expf(-vec.y));
        result.z = __fdividef(1.0f, 1.0f + __expf(-vec.z));
        result.w = __fdividef(1.0f, 1.0f + __expf(-vec.w));
        
        // Vectorized store
        reinterpret_cast<float4*>(output)[idx/4] = result;
    }
    else {
        // Handle remaining elements (less than 4)
        for (int i = idx; i < min(idx + 4, total_size); i++) {
            output[i] = __fdividef(1.0f, 1.0f + __expf(-input[i]));
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

// test
bool compare_outputs(const float* output, const float* reference, size_t size, float tolerance = TOLERANCE) {
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
        int dim = DIMS[test_id];
        int total_size = BATCH_SIZE * dim;

        float *h_input = new float[total_size];
        float *h_output = new float[total_size];
        float *h_output_ref = new float[total_size];

        // test
        std::string input_file = "./data/input_" + std::to_string(test_id + 1) + ".bin";
        std::string ref_file = "./data/reference_" + std::to_string(test_id + 1) + ".bin";

        read_binary(input_file, h_input, total_size);
        read_binary(ref_file, h_output_ref, total_size);

        float *d_input, *d_output;
        cudaMalloc(&d_input, total_size * sizeof(float));
        cudaMalloc(&d_output, total_size * sizeof(float));
        cudaMemcpy(d_input, h_input, total_size * sizeof(float), cudaMemcpyHostToDevice);

        // Optimized launch configuration for RTX 3090 Ti
        const int threads_per_block = 256;  // Optimal for Ampere
        const int blocks = (total_size + 4 * threads_per_block - 1) / (4 * threads_per_block);

        sigmoid_kernel<<<blocks, threads_per_block>>>(d_input, d_output, total_size);
        cudaMemcpy(h_output, d_output, total_size * sizeof(float), cudaMemcpyDeviceToHost);

        if (!compare_outputs(h_output, h_output_ref, total_size)) {
            std::cout << "F" << std::endl;
            all_passed = false;

            cudaFree(d_input);
            cudaFree(d_output);
            delete[] h_input;
            delete[] h_output;
            delete[] h_output_ref;
            break;
        }

        cudaFree(d_input);
        cudaFree(d_output);
        delete[] h_input;
        delete[] h_output;
        delete[] h_output_ref;
    }

    if (all_passed) std::cout << "T" << std::endl;

    return 0;
}