// vector_add.cuh - the "hello world" of CUDA: one thread per element.
#pragma once
#include "common.cuh"

__global__ void vector_add(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // global index of this thread
    if (i < n) c[i] = a[i] + b[i];                     // the last block may be partly outside the array
}
