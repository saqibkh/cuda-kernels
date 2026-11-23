// common.cuh - warp-level building blocks shared by the kernels.
#pragma once
#include <algorithm>
#include "launch.h"

constexpr unsigned FULL_MASK = 0xffffffffu;

// Sum of `v` over the 32 lanes of a warp; lane 0 receives the result.
__device__ __forceinline__ float warp_sum_down(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(FULL_MASK, v, offset);
    return v;
}

// Butterfly reductions: EVERY lane receives the result.
__device__ __forceinline__ float warp_sum_all(float v) {
    for (int mask = 16; mask > 0; mask >>= 1) v += __shfl_xor_sync(FULL_MASK, v, mask);
    return v;
}
__device__ __forceinline__ float warp_max_all(float v) {
    for (int mask = 16; mask > 0; mask >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, mask));
    return v;
}
