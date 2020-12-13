#ifndef GPU_TIMER_H__
#define GPU_TIMER_H__

#include <cuda_runtime.h>

struct GpuTimer {
    cudaEvent_t start;
    cudaEvent_t stop;
};

struct GpuTimer *NewGpuTimer() {
    struct GpuTimer *timer = (GpuTimer*)malloc(sizeof(*timer));
    if (!timer) {
        perror("malloc failed");
        exit(1);
    }
    cudaEventCreate(&timer->start);
    cudaEventCreate(&timer->stop);
    return timer;
}

void DestroyTimer(struct GpuTimer *timer) {
    if (!timer) return;
    cudaEventDestroy(timer->start);
    cudaEventDestroy(timer->stop);
    free(timer);
}

void StartTimer(struct GpuTimer *timer) {
    cudaEventRecord(timer->start, 0);
}

void StopTimer(struct GpuTimer *timer) {
    cudaEventRecord(timer->stop, 0);
}

float Elapsed(struct GpuTimer *timer) {
    float elapsed;
    cudaEventSynchronize(timer->stop);
    cudaEventElapsedTime(&elapsed, timer->start, timer->stop);
    return elapsed;
}

#endif  /* GPU_TIMER_H__ */
