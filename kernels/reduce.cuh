// reduce.cuh - sum of an array, three versions that show the classic optimisations.
// Each kernel writes one partial sum per block; reduce_sum() runs a second pass.
#pragma once
#include "common.cuh"

// v1: shared-memory tree with INTERLEAVED addressing. Correct but slow:
// `tid % (2*stride)` keeps only scattered threads busy, so most of every warp idles (divergence).
template <int BLOCK>
__global__ void reduce_interleaved(const float* __restrict__ x, float* __restrict__ out, int n) {
    __shared__ float s[BLOCK];
    const unsigned tid = threadIdx.x;
    const unsigned i = blockIdx.x * BLOCK + tid;
    s[tid] = i < (unsigned)n ? x[i] : 0.0f;
    __syncthreads();
    for (unsigned stride = 1; stride < BLOCK; stride *= 2) {
        if (tid % (2 * stride) == 0) s[tid] += s[tid + stride];
        __syncthreads();                       // every thread reaches every barrier
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

// v2: SEQUENTIAL addressing. Active threads are contiguous (whole warps retire together)
// and neighbouring threads read neighbouring words (no shared-memory bank conflicts).
template <int BLOCK>
__global__ void reduce_sequential(const float* __restrict__ x, float* __restrict__ out, int n) {
    __shared__ float s[BLOCK];
    const unsigned tid = threadIdx.x;
    const unsigned i = blockIdx.x * BLOCK + tid;
    s[tid] = i < (unsigned)n ? x[i] : 0.0f;
    __syncthreads();
    for (unsigned stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

// v3: GRID-STRIDE loop + WARP SHUFFLES. Each thread first sums many elements in a register
// (few blocks, lots of work per thread), then warps reduce with shuffles (no shared memory,
// no __syncthreads), and only one value per warp goes through shared memory.
template <int BLOCK>
__global__ void reduce_warp(const float* __restrict__ x, float* __restrict__ out, int n) {
    static_assert(BLOCK % 32 == 0 && BLOCK <= 1024, "BLOCK must be a multiple of 32");
    __shared__ float warp_sums[BLOCK / 32];
    const unsigned tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    float v = 0.0f;
    for (unsigned i = blockIdx.x * BLOCK + tid; i < (unsigned)n; i += gridDim.x * BLOCK) v += x[i];
    v = warp_sum_down(v);
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    if (warp == 0) {                            // the whole first warp takes part in the shuffles
        v = lane < BLOCK / 32 ? warp_sums[lane] : 0.0f;
        v = warp_sum_down(v);
        if (lane == 0) out[blockIdx.x] = v;
    }
}

enum class ReduceKind { Interleaved, Sequential, Warp };

// Host helper: device array -> sum (two passes; deterministic, no atomics).
template <int BLOCK = 256>
float reduce_sum(const float* d_x, int n, ReduceKind kind) {
    unsigned blocks = kind == ReduceKind::Warp ? std::min(ceil_div(n, BLOCK), 1024u) : ceil_div(n, BLOCK);
    if (blocks == 0) blocks = 1;
    float* d_partial = nullptr;
    CUDA_CHECK(cudaMalloc(&d_partial, blocks * sizeof(float)));
    switch (kind) {
        case ReduceKind::Interleaved: LAUNCH(reduce_interleaved<BLOCK>, dim3(blocks), dim3(BLOCK), d_x, d_partial, n); break;
        case ReduceKind::Sequential:  LAUNCH(reduce_sequential<BLOCK>, dim3(blocks), dim3(BLOCK), d_x, d_partial, n); break;
        case ReduceKind::Warp:        LAUNCH(reduce_warp<BLOCK>, dim3(blocks), dim3(BLOCK), d_x, d_partial, n); break;
    }
    CUDA_CHECK(cudaGetLastError());
    float* d_total = nullptr;
    CUDA_CHECK(cudaMalloc(&d_total, sizeof(float)));
    LAUNCH(reduce_warp<BLOCK>, dim3(1), dim3(BLOCK), (const float*)d_partial, d_total, (int)blocks);
    CUDA_CHECK(cudaGetLastError());
    float total = 0.0f;
    CUDA_CHECK(cudaMemcpy(&total, d_total, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFree(d_total));
    return total;
}
