// launch.h - one launch syntax for both builds.
//   nvcc:  LAUNCH(k, grid, block, args...)  ->  k<<<grid, block>>>(args...)
//   g++:   LAUNCH(k, grid, block, args...)  ->  run k on CPU threads (cudemu.h)
#pragma once
#include <cstdio>
#include <cstdlib>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define LAUNCH(kernel, grid, block, ...) kernel<<<(grid), (block)>>>(__VA_ARGS__)
#else
#include "cudemu.h"
#define LAUNCH(kernel, grid, block, ...) cudemu::launch((grid), (block), kernel, __VA_ARGS__)
#endif

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            std::fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(err_),  \
                         __FILE__, __LINE__);                                             \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

inline unsigned ceil_div(unsigned a, unsigned b) { return (a + b - 1) / b; }
