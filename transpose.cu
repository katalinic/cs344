#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "math_utils.h"
#include "timer.h"
#include "vector.h"

__global__ void scan_1D_inclusive_add_kernel(int *d_out, const int *d_in, int distance, int n) {
    int tIdx = blockIdx.x * blockDim.x + threadIdx.x;

    int neighbour = tIdx - distance;
    if (neighbour >= 0 && tIdx < n) {
        d_out[tIdx] += d_in[neighbour];
    }
}

void scan_1D_inclusive_add(int *h_out, const int *h_in, const int n) {
    const int BLOCK_THREADS = 1024;
    const int BLOCKS = min_div(n, BLOCK_THREADS);

    int *d_in, *d_out;
    checkCudaErrors(cudaMalloc((void **) &d_in, n * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &d_out, n * sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_out, h_in, n * sizeof(int), cudaMemcpyHostToDevice));

    struct GpuTimer *timer = NewGpuTimer();
    StartTimer(timer);
    for (unsigned int s = 1; s < n; s <<= 1) {
      checkCudaErrors(cudaMemcpy(d_in, d_out, n * sizeof(int), cudaMemcpyDeviceToDevice));
      scan_1D_inclusive_add_kernel<<<BLOCKS, BLOCK_THREADS>>>(d_out, d_in, s, n);
    }
    StopTimer(timer);
    printf("Execution time: %f msecs.\n", Elapsed(timer));
    DestroyTimer(timer);

    checkCudaErrors(cudaMemcpy(h_out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_in);
    cudaFree(d_out);
}

int validate_scan_1D_inclusive_add(const int *answer, const int *input, int n) {
    int running_total = 0;
    for (int i = 0; i < n; i++) {
        running_total += input[i];
        if (answer[i] != running_total) {
            printf("Mismatch encountered at index %d: expected actual %d %d.\n", i, running_total, answer[i]);
            return 1;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    int n = 1024, low = 0, high = 10;
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) low = atoi(argv[2]);
    if (argc > 3) high = atoi(argv[3]);
    assert(high >= low);

    int *h_in = create_random_vector(n, low, high);
    int *h_out = (int *)calloc(n, sizeof(int));

    scan_1D_inclusive_add(h_out, h_in, n);
    assert(!validate_scan_1D_inclusive_add(h_out, h_in, n));

    free(h_in);
    free(h_out);
    return 0;
}
