// softmax.cuh - row-wise softmax of a [rows x cols] row-major matrix, three versions.
#pragma once
#include "common.cuh"

// v1: one THREAD per row, three passes over the row (max, sum, write).
// Simple, but adjacent threads read addresses `cols` apart: uncoalesced, and few threads for few rows.
__global__ void softmax_naive(const float* __restrict__ x, float* __restrict__ y, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;                    // safe: this kernel has no __syncthreads
    const float* xr = x + (size_t)r * cols;
    float* yr = y + (size_t)r * cols;
    float m = -INFINITY;
    for (int c = 0; c < cols; ++c) m = fmaxf(m, xr[c]);
    float sum = 0.0f;
    for (int c = 0; c < cols; ++c) sum += expf(xr[c] - m);
    for (int c = 0; c < cols; ++c) yr[c] = expf(xr[c] - m) / sum;
}

// Block-wide reductions built from warp reductions + one shared slot per warp.
template <int BLOCK>
__device__ float block_max(float v) {
    __shared__ float tmp[BLOCK / 32];
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    v = warp_max_all(v);
    if (lane == 0) tmp[warp] = v;
    __syncthreads();
    v = lane < BLOCK / 32 ? tmp[lane] : -INFINITY;
    v = warp_max_all(v);                      // every warp computes the same final value
    __syncthreads();                          // tmp may be reused right after
    return v;
}
template <int BLOCK>
__device__ float block_sum(float v) {
    __shared__ float tmp[BLOCK / 32];
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    v = warp_sum_all(v);
    if (lane == 0) tmp[warp] = v;
    __syncthreads();
    v = lane < BLOCK / 32 ? tmp[lane] : 0.0f;
    v = warp_sum_all(v);
    __syncthreads();
    return v;
}

// v2: one BLOCK per row. Threads stride over the columns, so consecutive threads read
// consecutive addresses (coalesced). Still three passes over the row in global memory.
template <int BLOCK>
__global__ void softmax_block(const float* __restrict__ x, float* __restrict__ y, int rows, int cols) {
    const float* xr = x + (size_t)blockIdx.x * cols;
    float* yr = y + (size_t)blockIdx.x * cols;
    float m = -INFINITY;
    for (int c = threadIdx.x; c < cols; c += BLOCK) m = fmaxf(m, xr[c]);
    m = block_max<BLOCK>(m);
    float s = 0.0f;
    for (int c = threadIdx.x; c < cols; c += BLOCK) s += expf(xr[c] - m);
    s = block_sum<BLOCK>(s);
    for (int c = threadIdx.x; c < cols; c += BLOCK) yr[c] = expf(xr[c] - m) / s;
}

// Online softmax state: running max m and running sum d of exp(x - m).
struct MD { float m, d; };
__device__ __forceinline__ MD md_merge(MD a, MD b) {
    float m = fmaxf(a.m, b.m);
    if (m == -INFINITY) return {m, 0.0f};     // both empty: avoid exp(-inf - -inf) = NaN
    return {m, a.d * expf(a.m - m) + b.d * expf(b.m - m)};
}

// v3: ONLINE softmax (Milakov and Gimelshein, 2018). Max and sum are computed together in
// ONE pass, by rescaling the running sum whenever the max grows. Two passes instead of three:
// the same trick FlashAttention uses (Chapter 2.3).
template <int BLOCK>
__global__ void softmax_online(const float* __restrict__ x, float* __restrict__ y, int rows, int cols) {
    __shared__ float tm[BLOCK / 32], td[BLOCK / 32];
    const float* xr = x + (size_t)blockIdx.x * cols;
    float* yr = y + (size_t)blockIdx.x * cols;
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    MD st{-INFINITY, 0.0f};
    for (int c = threadIdx.x; c < cols; c += BLOCK) st = md_merge(st, MD{xr[c], 1.0f});
    for (int mask = 16; mask > 0; mask >>= 1)
        st = md_merge(st, MD{__shfl_xor_sync(FULL_MASK, st.m, mask), __shfl_xor_sync(FULL_MASK, st.d, mask)});
    if (lane == 0) { tm[warp] = st.m; td[warp] = st.d; }
    __syncthreads();
    st = lane < BLOCK / 32 ? MD{tm[lane], td[lane]} : MD{-INFINITY, 0.0f};
    for (int mask = 16; mask > 0; mask >>= 1)
        st = md_merge(st, MD{__shfl_xor_sync(FULL_MASK, st.m, mask), __shfl_xor_sync(FULL_MASK, st.d, mask)});
    const float inv = 1.0f / st.d;
    for (int c = threadIdx.x; c < cols; c += BLOCK) yr[c] = expf(xr[c] - st.m) * inv;
}
