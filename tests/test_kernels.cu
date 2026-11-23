// test_kernels.cu - every kernel against a simple CPU reference, on awkward sizes.
// Builds two ways:  nvcc (runs on a GPU)  or  g++ -x c++ (runs on the CPU emulator, cudemu.h).
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "../kernels/matmul.cuh"
#include "../kernels/reduce.cuh"
#include "../kernels/softmax.cuh"
#include "../kernels/vector_add.cuh"

static int failures = 0, checks = 0;
#define EXPECT(cond, ...)                                  \
    do {                                                   \
        ++checks;                                          \
        if (!(cond)) {                                     \
            ++failures;                                    \
            std::printf("  FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                      \
            std::printf("\n");                             \
        }                                                  \
    } while (0)

static std::vector<float> random_vec(size_t n, unsigned seed, float lo = -1.0f, float hi = 1.0f) {
    std::vector<float> v(n);
    unsigned s = seed * 2654435761u + 1u;
    for (auto& x : v) {
        s = s * 1664525u + 1013904223u;                  // fixed LCG: same data on every machine
        x = lo + (hi - lo) * (s >> 8) / float(1u << 24);
    }
    return v;
}

template <class T>
static T* to_device(const std::vector<T>& h) {
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(T) + 1));
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <class T>
static std::vector<T> to_host(const T* d, size_t n) {
    std::vector<T> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

static void test_vector_add() {
    for (int n : {1, 255, 256, 257, 10001}) {
        auto a = random_vec(n, 1), b = random_vec(n, 2);
        float *da = to_device(a), *db = to_device(b), *dc = nullptr;
        CUDA_CHECK(cudaMalloc(&dc, n * sizeof(float)));
        LAUNCH(vector_add, dim3(ceil_div(n, 256)), dim3(256), (const float*)da, (const float*)db, dc, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto c = to_host(dc, n);
        int bad = 0;
        for (int i = 0; i < n; ++i) bad += c[i] != a[i] + b[i];
        EXPECT(bad == 0, "vector_add n=%d: %d wrong", n, bad);
        cudaFree(da); cudaFree(db); cudaFree(dc);
    }
}

static void test_reduce() {
    const char* names[] = {"interleaved", "sequential", "warp"};
    for (int n : {1, 31, 256, 1000, 100003}) {
        auto x = random_vec(n, 3, 0.0f, 1.0f);
        double ref = 0;
        for (float v : x) ref += v;
        float* dx = to_device(x);
        for (int k = 0; k < 3; ++k) {
            float got = reduce_sum<256>(dx, n, static_cast<ReduceKind>(k));
            EXPECT(std::fabs(got - ref) <= 1e-4 * std::fabs(ref) + 1e-5, "reduce %s n=%d: got %.6f want %.6f", names[k], n, got, ref);
        }
        cudaFree(dx);
    }
}

static std::vector<float> softmax_ref(const std::vector<float>& x, int rows, int cols) {
    std::vector<float> y(x.size());
    for (int r = 0; r < rows; ++r) {
        double m = -INFINITY, s = 0;
        for (int c = 0; c < cols; ++c) m = std::max<double>(m, x[r * cols + c]);
        for (int c = 0; c < cols; ++c) s += std::exp(x[r * cols + c] - m);
        for (int c = 0; c < cols; ++c) y[r * cols + c] = float(std::exp(x[r * cols + c] - m) / s);
    }
    return y;
}

static void test_softmax() {
    const char* names[] = {"naive", "block", "online"};
    for (int cols : {1, 31, 257, 1000}) {
        const int rows = 7;
      for (int masked = 0; masked < 2; ++masked) {
        auto x = random_vec(rows * cols, 4, -30.0f, 30.0f);   // large range: tests max subtraction
        if (masked)                                            // causal-style mask: -inf after a cut-off
            for (int r = 0; r < rows; ++r)
                for (int c = 1 + r * cols / rows; c < cols; ++c) x[r * cols + c] = -INFINITY;
        auto ref = softmax_ref(x, rows, cols);
        float* dx = to_device(x);
        float* dy = nullptr;
        CUDA_CHECK(cudaMalloc(&dy, x.size() * sizeof(float)));
        for (int k = 0; k < 3; ++k) {
            if (k == 0) LAUNCH(softmax_naive, dim3(ceil_div(rows, 64)), dim3(64), (const float*)dx, dy, rows, cols);
            if (k == 1) LAUNCH(softmax_block<128>, dim3(rows), dim3(128), (const float*)dx, dy, rows, cols);
            if (k == 2) LAUNCH(softmax_online<128>, dim3(rows), dim3(128), (const float*)dx, dy, rows, cols);
            CUDA_CHECK(cudaDeviceSynchronize());
            auto y = to_host(dy, x.size());
            double err = 0, rowsum_err = 0;
            for (size_t i = 0; i < y.size(); ++i) err = std::max(err, std::isnan(y[i]) ? 1e9 : (double)std::fabs(y[i] - ref[i]));
            for (int r = 0; r < rows; ++r) {
                double s = 0;
                for (int c = 0; c < cols; ++c) s += y[r * cols + c];
                rowsum_err = std::max(rowsum_err, std::fabs(s - 1.0));
            }
            EXPECT(err < 1e-6 && rowsum_err < 1e-5, "softmax %s cols=%d masked=%d: max err %.2e, row-sum err %.2e",
                   names[k], cols, masked, err, rowsum_err);
        }
        cudaFree(dx); cudaFree(dy);
      }
    }
}

static void test_matmul() {
    const char* names[] = {"naive", "tiled", "regtile"};
    struct Shape { int M, N, K; };
    for (Shape s : {Shape{1, 1, 1}, Shape{17, 33, 9}, Shape{64, 64, 64}, Shape{100, 70, 130}, Shape{65, 129, 8}}) {
        auto A = random_vec((size_t)s.M * s.K, 5), B = random_vec((size_t)s.K * s.N, 6);
        std::vector<float> ref((size_t)s.M * s.N);
        for (int i = 0; i < s.M; ++i)
            for (int j = 0; j < s.N; ++j) {
                double acc = 0;
                for (int k = 0; k < s.K; ++k) acc += (double)A[(size_t)i * s.K + k] * B[(size_t)k * s.N + j];
                ref[(size_t)i * s.N + j] = float(acc);
            }
        float *dA = to_device(A), *dB = to_device(B), *dC = nullptr;
        CUDA_CHECK(cudaMalloc(&dC, ref.size() * sizeof(float)));
        for (int k = 0; k < 3; ++k) {
            CUDA_CHECK(cudaMemset(dC, 0xff, ref.size() * sizeof(float)));    // NaN everywhere: unwritten outputs fail
            matmul(dA, dB, dC, s.M, s.N, s.K, static_cast<MatmulKind>(k));
            CUDA_CHECK(cudaDeviceSynchronize());
            auto C = to_host(dC, ref.size());
            double err = 0;
            for (size_t i = 0; i < C.size(); ++i) err = std::max(err, std::isnan(C[i]) ? 1e9 : std::fabs(C[i] - ref[i]));
            EXPECT(err < 1e-4, "matmul %s %dx%dx%d: max err %.2e", names[k], s.M, s.N, s.K, err);
        }
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
}

int main() {
    struct T { const char* name; std::function<void()> fn; };
    for (auto& t : {T{"vector_add", test_vector_add}, T{"reduce", test_reduce}, T{"softmax", test_softmax}, T{"matmul", test_matmul}}) {
        int before = failures;
        t.fn();
        std::printf("%-12s %s\n", t.name, failures == before ? "ok" : "FAILED");
    }
    std::printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
