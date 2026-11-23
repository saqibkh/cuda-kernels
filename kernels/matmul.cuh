// matmul.cuh - C[M x N] = A[M x K] * B[K x N], row-major fp32, three versions.
#pragma once
#include "common.cuh"

// v1: NAIVE. One thread per output; every thread reads a full row of A and column of B
// from global memory: 2K loads per 2K flops, so arithmetic intensity is ~0.25 flop/byte.
__global__ void matmul_naive(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // x = column: neighbouring threads read neighbouring B and C
    if (row >= M || col >= N) return;                  // no __syncthreads here, so an early return is safe
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += A[(size_t)row * K + k] * B[(size_t)k * N + col];
    C[(size_t)row * N + col] = acc;
}

// v2: SHARED-MEMORY TILING. A TILE x TILE block loads one tile of A and one of B into
// shared memory, then every thread reuses them TILE times: global traffic falls by ~TILE x.
template <int TILE>
__global__ void matmul_tiled(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                             int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        // Out-of-range threads still load (zeros) and still reach both barriers.
        As[ty][tx] = (row < M && k0 + tx < K) ? A[(size_t)row * K + k0 + tx] : 0.0f;
        Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(size_t)(k0 + ty) * N + col] : 0.0f;
        __syncthreads();                     // tile fully loaded before anyone reads it
        for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();                     // everyone done before the tile is overwritten
    }
    if (row < M && col < N) C[(size_t)row * N + col] = acc;
}

// v3: 2D REGISTER TILING. A 256-thread block computes a 64 x 64 tile of C; each thread
// computes a 4 x 4 patch held in registers. Per k step a thread loads 4 + 4 values from
// shared memory and does 16 multiply-adds, so shared-memory traffic per flop falls 4x again.
constexpr int RT_BM = 64, RT_BN = 64, RT_BK = 8, RT_TM = 4, RT_TN = 4;
constexpr int RT_THREADS = (RT_BM / RT_TM) * (RT_BN / RT_TN);   // 256

__global__ void matmul_regtile(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                               int M, int N, int K) {
    __shared__ float As[RT_BM][RT_BK];
    __shared__ float Bs[RT_BK][RT_BN];
    const int tid = threadIdx.x;                     // 1-D block of 256 threads
    const int tr = tid / (RT_BN / RT_TN), tc = tid % (RT_BN / RT_TN);
    const int row0 = blockIdx.y * RT_BM, col0 = blockIdx.x * RT_BN;
    float acc[RT_TM][RT_TN] = {};
    for (int k0 = 0; k0 < K; k0 += RT_BK) {
        for (int e = tid; e < RT_BM * RT_BK; e += RT_THREADS) {          // 512 values of A, 2 per thread
            int r = e / RT_BK, k = e % RT_BK;
            As[r][k] = (row0 + r < M && k0 + k < K) ? A[(size_t)(row0 + r) * K + k0 + k] : 0.0f;
        }
        for (int e = tid; e < RT_BK * RT_BN; e += RT_THREADS) {          // 512 values of B, coalesced rows
            int k = e / RT_BN, c = e % RT_BN;
            Bs[k][c] = (k0 + k < K && col0 + c < N) ? B[(size_t)(k0 + k) * N + col0 + c] : 0.0f;
        }
        __syncthreads();
        for (int k = 0; k < RT_BK; ++k) {
            float a[RT_TM], b[RT_TN];
            for (int i = 0; i < RT_TM; ++i) a[i] = As[tr * RT_TM + i][k];
            for (int j = 0; j < RT_TN; ++j) b[j] = Bs[k][tc * RT_TN + j];
            for (int i = 0; i < RT_TM; ++i)
                for (int j = 0; j < RT_TN; ++j) acc[i][j] += a[i] * b[j];   // outer product in registers
        }
        __syncthreads();
    }
    for (int i = 0; i < RT_TM; ++i)
        for (int j = 0; j < RT_TN; ++j) {
            int r = row0 + tr * RT_TM + i, c = col0 + tc * RT_TN + j;
            if (r < M && c < N) C[(size_t)r * N + c] = acc[i][j];
        }
}

enum class MatmulKind { Naive, Tiled, RegTile };

inline void matmul(const float* dA, const float* dB, float* dC, int M, int N, int K, MatmulKind kind) {
    switch (kind) {
        case MatmulKind::Naive:
            LAUNCH(matmul_naive, dim3(ceil_div(N, 16), ceil_div(M, 16)), dim3(16, 16), dA, dB, dC, M, N, K); break;
        case MatmulKind::Tiled:
            LAUNCH(matmul_tiled<16>, dim3(ceil_div(N, 16), ceil_div(M, 16)), dim3(16, 16), dA, dB, dC, M, N, K); break;
        case MatmulKind::RegTile:
            LAUNCH(matmul_regtile, dim3(ceil_div(N, RT_BN), ceil_div(M, RT_BM)), dim3(RT_THREADS), dA, dB, dC, M, N, K); break;
    }
    CUDA_CHECK(cudaGetLastError());
}
