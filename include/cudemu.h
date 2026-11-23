// cudemu.h - run CUDA kernels on the CPU to check that they are CORRECT (not fast).
//
// Each thread of a block is a std::thread; blocks run one after another, so a
// function-local `static` array behaves exactly like __shared__ memory.
// __syncthreads() is a barrier across the block; warp shuffles use one barrier per warp.
// Used only when the code is compiled by a normal C++ compiler (no __CUDACC__).
#pragma once
#include <algorithm>
#include <barrier>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <math.h>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

struct dim3 {
    unsigned x, y, z;
    constexpr dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};

inline thread_local dim3 threadIdx, blockIdx;
inline dim3 blockDim, gridDim;

#define __global__
#define __host__
#define __device__ inline
#define __forceinline__
#define __shared__ static            // blocks run sequentially, so static == per-block shared memory
#define __expf expf

namespace cudemu {
inline std::barrier<>* block_barrier = nullptr;
inline std::vector<std::unique_ptr<std::barrier<>>>* warp_barriers = nullptr;
inline std::vector<double>* exchange = nullptr;   // one slot per thread for shuffles
inline unsigned nthreads = 0;
inline thread_local unsigned tid_linear = 0;
inline std::mutex atomic_mu;

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wdangling-pointer"
#endif
template <class Kernel, class... Args>
void launch(dim3 grid, dim3 block, Kernel kernel, Args... args) {
    const unsigned nt = block.x * block.y * block.z;
    if (nt == 0 || nt > 1024) {
        std::fprintf(stderr, "cudemu: invalid block size %u (must be 1..1024)\n", nt);
        std::abort();
    }
    blockDim = block;
    gridDim = grid;
    std::barrier<> bar(nt);
    std::vector<std::unique_ptr<std::barrier<>>> warps;
    for (unsigned w = 0; w * 32 < nt; ++w) warps.push_back(std::make_unique<std::barrier<>>(std::min(32u, nt - w * 32)));
    std::vector<double> xchg(nt);
    block_barrier = &bar;
    warp_barriers = &warps;
    exchange = &xchg;
    nthreads = nt;
    std::vector<std::thread> threads;
    threads.reserve(nt);
    for (unsigned t = 0; t < nt; ++t) {
        threads.emplace_back([=, &bar] {
            threadIdx = dim3(t % block.x, (t / block.x) % block.y, t / (block.x * block.y));
            tid_linear = t;
            for (unsigned bz = 0; bz < grid.z; ++bz)
                for (unsigned by = 0; by < grid.y; ++by)
                    for (unsigned bx = 0; bx < grid.x; ++bx) {
                        blockIdx = dim3(bx, by, bz);
                        kernel(args...);
                        bar.arrive_and_wait();   // the whole block finishes before the next one starts
                    }
        });
    }
    for (auto& th : threads) th.join();
    block_barrier = nullptr;
    warp_barriers = nullptr;
    exchange = nullptr;
}

template <class T>
T shuffle(T v, unsigned src_lane_of(unsigned lane, unsigned arg), unsigned arg, int width) {
    const unsigned t = tid_linear, warp = t / 32, lane = t % 32;
    const unsigned warp_size = std::min(32u, nthreads - warp * 32);
    (*exchange)[t] = static_cast<double>(v);
    (*warp_barriers)[warp]->arrive_and_wait();
    const unsigned seg = lane / width * width;               // sub-warp segment for width < 32
    const unsigned src = src_lane_of(lane - seg, arg);
    T r = v;
    if (src < static_cast<unsigned>(width) && seg + src < warp_size) r = static_cast<T>((*exchange)[warp * 32 + seg + src]);
    (*warp_barriers)[warp]->arrive_and_wait();
    return r;
}
}  // namespace cudemu

inline void __syncthreads() { cudemu::block_barrier->arrive_and_wait(); }

template <class T>
T __shfl_down_sync(unsigned, T v, unsigned delta, int width = 32) {
    return cudemu::shuffle(v, +[](unsigned l, unsigned d) { return l + d; }, delta, width);
}
template <class T>
T __shfl_xor_sync(unsigned, T v, unsigned mask, int width = 32) {
    return cudemu::shuffle(v, +[](unsigned l, unsigned m) { return l ^ m; }, mask, width);
}

template <class T>
T atomicAdd(T* addr, T v) {
    std::lock_guard<std::mutex> g(cudemu::atomic_mu);
    T old = *addr;
    *addr = old + v;
    return old;
}

// ---- the small part of the CUDA runtime API the tests use ----
typedef int cudaError_t;
enum { cudaSuccess = 0, cudaErrorMemoryAllocation = 2 };
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice };
template <class T>
inline cudaError_t cudaMalloc(T** p, size_t bytes) {
    *p = static_cast<T*>(std::malloc(bytes ? bytes : 1));
    return *p ? cudaSuccess : cudaErrorMemoryAllocation;
}
inline cudaError_t cudaFree(void* p) { std::free(p); return cudaSuccess; }
inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) { std::memcpy(d, s, n); return cudaSuccess; }
inline cudaError_t cudaMemset(void* d, int v, size_t n) { std::memset(d, v, n); return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t e) { return e ? "cudemu error" : "no error"; }
