#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "timer.h"
#include "vector.h"

__global__ void reduce_1D_global_memory_kernel(int *d_out, int *d_in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            d_in[idx] += d_in[idx + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = d_in[idx];
    }
}

__global__ void reduce_1D_shared_memory_kernel(int *d_out, int *d_in) {
    extern __shared__ int sharedMemory[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sharedMemory[threadIdx.x] = d_in[idx];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sharedMemory[threadIdx.x] += sharedMemory[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = sharedMemory[0];
    }
}

int reduce_1D(int *h_in, int n, int sharedMemory) {
    assert((n & (n - 1)) == 0);
    const int BLOCK_THREADS = 16;
    const int BLOCKS = n / BLOCK_THREADS;
    int h_out;

    int *d_in, *d_temp, *d_out;
    checkCudaErrors(cudaMalloc((void **) &d_in, n * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &d_temp, BLOCKS * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &d_out, sizeof(int)));

    checkCudaErrors(cudaMemcpy(d_in, h_in, n * sizeof(int), cudaMemcpyHostToDevice));

    struct GpuTimer *timer = NewGpuTimer();
    if (sharedMemory) {
        StartTimer(timer);
        reduce_1D_shared_memory_kernel<<<BLOCKS, BLOCK_THREADS, BLOCK_THREADS * sizeof(int)>>>(d_temp, d_in);
        reduce_1D_shared_memory_kernel<<<1, BLOCKS, BLOCKS * sizeof(int)>>>(d_out, d_temp);
        StopTimer(timer);
    } else {
        StartTimer(timer);
        reduce_1D_global_memory_kernel<<<BLOCKS, BLOCK_THREADS>>>(d_temp, d_in);
        reduce_1D_global_memory_kernel<<<1, BLOCKS>>>(d_out, d_temp);
        StopTimer(timer);
    }
    printf("Shared memory? (%s) - Execution time: %f msecs.\n", sharedMemory ? "T" : "F", Elapsed(timer));
    DestroyTimer(timer);

    checkCudaErrors(cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_temp));
    checkCudaErrors(cudaFree(d_out));

    return h_out;
}

int main(int argc, char **argv) {
    int n = 1024, low = 0, high = 10;
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) low = atoi(argv[2]);
    if (argc > 3) high = atoi(argv[3]);
    assert(high >= low);

    int *h_in = create_random_vector(n, low, high);

    assert(vector_sum(h_in, n) == reduce_1D(h_in, n, 0));
    assert(vector_sum(h_in, n) == reduce_1D(h_in, n, 1));

    free(h_in);
    return 0;
}
