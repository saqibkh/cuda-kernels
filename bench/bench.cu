// bench.cu - GPU timing for every kernel (needs an NVIDIA GPU; build with `make bench`).
// Reports GFLOP/s for matmul (vs cuBLAS) and effective GB/s for memory-bound kernels.
#include <cublas_v2.h>
#include <cstdio>
#include <vector>

#include "../kernels/matmul.cuh"
#include "../kernels/reduce.cuh"
#include "../kernels/softmax.cuh"
#include "../kernels/vector_add.cuh"

#define CUBLAS_CHECK(call)                                                       \
    do {                                                                         \
        cublasStatus_t st_ = (call);                                             \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                      \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st_, __FILE__, __LINE__); \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

// Median-free but robust enough: 3 warm-up runs, then the mean of `iters` timed runs.
template <class F>
float time_ms(F fn, int iters = 20) {
    for (int i = 0; i < 3; ++i) fn();
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) fn();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iters;
}

static float* device_random(size_t n) {
    std::vector<float> h(n);
    unsigned s = 12345;
    for (auto& x : h) { s = s * 1664525u + 1013904223u; x = (s >> 8) / float(1u << 24) - 0.5f; }
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

int main() {
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    std::printf("device,%s,SMs,%d,memory_GB,%.1f,cc,%d.%d\n", p.name, p.multiProcessorCount,
                p.totalGlobalMem / 1e9, p.major, p.minor);
    std::printf("kernel,size,ms,GFLOPs_or_GBs\n");

    // ---- memory-bound kernels: effective bandwidth = bytes that must move / time ----
    const int n = 1 << 26;                                   // 64 Mi floats = 256 MiB
    float *a = device_random(n), *b = device_random(n), *c = nullptr;
    CUDA_CHECK(cudaMalloc(&c, (size_t)n * sizeof(float)));
    float ms = time_ms([&] { LAUNCH(vector_add, dim3(ceil_div(n, 256)), dim3(256), (const float*)a, (const float*)b, c, n); });
    std::printf("vector_add,%d,%.3f,%.1f\n", n, ms, 3.0 * n * 4 / ms / 1e6);
    const char* rnames[] = {"reduce_interleaved", "reduce_sequential", "reduce_warp"};
    for (int k = 0; k < 3; ++k) {
        ms = time_ms([&] { reduce_sum<256>(a, n, static_cast<ReduceKind>(k)); });
        std::printf("%s,%d,%.3f,%.1f\n", rnames[k], n, ms, 1.0 * n * 4 / ms / 1e6);
    }
    const int rows = 8192, cols = 8192;                      // reuse a and c as [8192 x 8192]
    const char* snames[] = {"softmax_naive", "softmax_block", "softmax_online"};
    for (int k = 0; k < 3; ++k) {
        ms = time_ms([&] {
            if (k == 0) LAUNCH(softmax_naive, dim3(ceil_div(rows, 128)), dim3(128), (const float*)a, c, rows, cols);
            if (k == 1) LAUNCH(softmax_block<256>, dim3(rows), dim3(256), (const float*)a, c, rows, cols);
            if (k == 2) LAUNCH(softmax_online<256>, dim3(rows), dim3(256), (const float*)a, c, rows, cols);
        });
        std::printf("%s,%dx%d,%.3f,%.1f\n", snames[k], rows, cols, ms, 2.0 * rows * cols * 4 / ms / 1e6);  // ideal: read once, write once
    }

    // ---- matmul: GFLOP/s = 2*M*N*K / time ----
    cublasHandle_t h;
    CUBLAS_CHECK(cublasCreate(&h));
    const char* mnames[] = {"matmul_naive", "matmul_tiled", "matmul_regtile"};
    for (int sz : {1024, 2048, 4096}) {
        const double flops = 2.0 * sz * sz * sz;
        for (int k = 0; k < 3; ++k) {
            ms = time_ms([&] { matmul(a, b, c, sz, sz, sz, static_cast<MatmulKind>(k)); }, 5);
            std::printf("%s,%d,%.3f,%.1f\n", mnames[k], sz, ms, flops / ms / 1e6);
        }
        const float alpha = 1.0f, beta = 0.0f;
        // cuBLAS is column-major: row-major C = A*B is column-major C^T = B^T * A^T.
        ms = time_ms([&] {
            CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, sz, sz, sz, &alpha, b, sz, a, sz, &beta, c, sz));
        }, 5);
        std::printf("cublas_sgemm,%d,%.3f,%.1f\n", sz, ms, flops / ms / 1e6);
    }
    CUBLAS_CHECK(cublasDestroy(h));
    cudaFree(a); cudaFree(b); cudaFree(c);
    return 0;
}
