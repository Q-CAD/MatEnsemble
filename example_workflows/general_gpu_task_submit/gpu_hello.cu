// file: gpu_hello_auto.cu
#include <iostream>
#include <cuda_runtime.h>

__global__ void hello_from_all(int total_threads) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id < total_threads) {
        printf("Hello from GPU | Block %d | Thread %d | Global ID %d\n",
               blockIdx.x, threadIdx.x, global_id);
    }
}

int main() {
    int device_id = 0;
    cudaSetDevice(device_id);

    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    std::cout << "Using GPU: " << prop.name << " (device " << device_id << ")\n";

    // Set a reasonable total number of threads based on GPU
    int total_threads = prop.multiProcessorCount * 32 * 4;  // 4 warps per SM
    int max_threads_per_block = prop.maxThreadsPerBlock;

    // Choose block size and number of blocks
    int threads_per_block = (max_threads_per_block > 256) ? 256 : max_threads_per_block;
    int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    std::cout << "Launching " << num_blocks << " blocks of "
              << threads_per_block << " threads (total: "
              << (num_blocks * threads_per_block) << " threads)\n";

    // Launch kernel
    hello_from_all<<<num_blocks, threads_per_block>>>(total_threads);
    cudaDeviceSynchronize();

    return 0;
}

