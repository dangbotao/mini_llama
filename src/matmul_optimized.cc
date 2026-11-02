#include "mini_llama/matmul_dispatch.h"
#include "mini_llama/thread_pool.h"

#include <stdexcept>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define MINI_LLAMA_USE_AVX2 1
#endif

namespace mini_llama
{

static Tensor MatmulNaive(const Tensor& a, const Tensor& b) {
    int row = a.shape[0];
    int col = b.shape[1];
    Tensor c = MakeTensor2D(row, col, 0.0f);
    int mid_dim = a.shape[1];

    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            for (int k = 0; k < mid_dim; ++k) {
                c[i * col + j] += a[i * mid_dim + k] * b[k * col + j];
            }
        }
    }

    return c;
}

// linear
// x: [1, in_features] or [in_features]
// weight: [out_features, in_features]
// result: [1, out_features] or [out_features]
static Tensor LinearNaive(const Tensor& x, const Tensor& weight) {
    int in_features = weight.shape[1];
    int out_features = weight.shape[0];
    bool is_1d = (x.num_dims() == 1);

    Tensor result(is_1d ? std::vector<int>{out_features}
                        : std::vector<int>{1, out_features},
                    0.0f); // keep input x dim

    for (int i = 0; i < out_features; ++i) {
        for (int j = 0; j < in_features; ++j) {
            result[i] += x[j] * weight[i * in_features + j];
        }
    }

    return result;
}

static Tensor MatmulThreaded(const Tensor& a, const Tensor& b) {
    int row = a.shape[0];
    int col = b.shape[1];
    // Tensor c =MakeTensor2D(row, col, 0.0f);
    Tensor c({row, col}, 0.0f);
    int mid_dim = a.shape[1];

    ParallelFor(row, [&](int begin, int end) {
        for (int i = begin; i < end; ++i) {
            for (int j = 0; j < col; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < mid_dim; ++k) {
                    sum += a[i * mid_dim + k] * b[k * col + j];
                }
                c[i * col + j] = sum;
            }
        }
    });
    return c;
}

static Tensor LinearThreaded(const Tensor& x, const Tensor& weight) {
    int in_features = weight.shape[1];
    int out_features = weight.shape[0];
    bool is_1d = (x.num_dims() == 1);

    Tensor result(is_1d ? std::vector<int>{out_features}
                        : std::vector<int>{1, out_features},
                    0.0f); // keep input x dim

    ParallelFor(out_features, [&](int begin, int end) {
        for (int i = begin; i < end; ++i) { // weight矩阵按行拆分到子线程
            for (int j = 0; j < in_features; ++j) {
                result[i] += x[j] * weight[i * in_features + j];
            }
        }
    });

    return result;
}

// 两个长度为size的数组的点积
#ifdef MINI_LLAMA_USE_AVX2
static float DotSimdAvx2(const float* x, const float* weight, int size) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();

    int k = 0;

    // 两个累加器降低 FMA 依赖链带来的延迟。
    for (; k + 16 <= size; k += 16) { // 处理16元素块
        const __m256 x0 = _mm256_loadu_ps(x + k);
        const __m256 w0 = _mm256_loadu_ps(weight + k);
        const __m256 x1 = _mm256_loadu_ps(x + k + 8);
        const __m256 w1 = _mm256_loadu_ps(weight + k + 8);

        sum0 = _mm256_fmadd_ps(x0, w0, sum0);
        sum1 = _mm256_fmadd_ps(x1, w1, sum1);
    }

    sum0 = _mm256_add_ps(sum0, sum1); // 8个通道逐一相加

    for (; k + 8 <= size; k += 8) { // 处理8元素快
        const __m256 xv = _mm256_loadu_ps(x + k);
        const __m256 wv = _mm256_loadu_ps(weight + k);
        sum0 = _mm256_fmadd_ps(xv, wv, sum0);
    }

    // 横向求和
    __m128 sum = _mm_add_ps( // 取出高低四位逐一求和
        _mm256_castps256_ps128(sum0),
        _mm256_extractf128_ps(sum0, 1));

    sum = _mm_hadd_ps(sum, sum); // 相邻元素求和
    sum = _mm_hadd_ps(sum, sum);
    float result = _mm_cvtss_f32(sum); // 从寄存器最低位取最终和

    // 处理不能被 8 整除的尾部。
    for (; k < size; ++k) {
        result += x[k] * weight[k];
    }

    return result;
}
#endif

static float DotSimd(const float* x, const float* weight, int size) {
    #ifdef MINI_LLAMA_USE_AVX2
        return DotSimdAvx2(x, weight, size);
    #else
        float sum = 0.0f;
        for (int i = 0; i < size; ++i) {
            sum += x[i] * weight[i];
        }
        return sum;
    #endif
}

static Tensor LinearSimd(const Tensor& x, const Tensor& weight) {
    int in_features = weight.shape[1];
    int out_features = weight.shape[0];
    bool is_1d = (x.num_dims() == 1);

    Tensor result(is_1d ? std::vector<int>{out_features}
                        : std::vector<int>{1, out_features},
                    0.0f); // keep input x dim

    for (int i = 0; i < out_features; ++i) {
        // for (int j = 0; j < in_features; ++j) {
        //     result[i] += x[j] * weight[i * in_features + j];
        // }
        result[i] = DotSimd(x.data.data(),
                    weight.data.data() + i * in_features, in_features);
    }
    return result;
}

static Tensor LinearThreadedSimd(const Tensor& x, const Tensor& weight) {
    int in_features = weight.shape[1];
    int out_features = weight.shape[0];
    bool is_1d = (x.num_dims() == 1);

    Tensor result(is_1d ? std::vector<int>{out_features}
                        : std::vector<int>{1, out_features},
                    0.0f); // keep input x dim

    ParallelFor(out_features, [&](int begin, int end) {
        for (int i = begin; i < end; ++i) { // weight矩阵按行拆分到子线程
            // for (int j = 0; j < in_features; ++j) {
            //     result[i] += x[j] * weight[i * in_features + j];
            // }
            result[i] = DotSimd(x.data.data(),
                    weight.data.data() + i * in_features, in_features);
        }
    });

    return result;
}

MatmulMode DefaultMatmulMode() { return MatmulMode::kThreadedSimd; }

Tensor MatmulDispatch(const Tensor& a, const Tensor& b, MatmulMode mode) {
    if (a.num_dims() != 2 || b.num_dims() != 2) {
        throw std::runtime_error("matrix must be 2D");
    }

    if (a.shape[1] != b.shape[0]) {
        throw std::runtime_error("two matrix shape dismatch");
    }

    switch (mode) {
    case MatmulMode::kNaive:
        return MatmulNaive(a, b);
    case MatmulMode::kThreaded:
        return MatmulThreaded(a, b);
    case MatmulMode::kSimd:
    case MatmulMode::kThreadedSimd:
        return MatmulThreaded(a, b);
    }

    return MatmulThreaded(a, b);
}

Tensor LinearDispatch(const Tensor& x, const Tensor& weight, MatmulMode mode) {
    if (weight.num_dims() != 2) {
        throw std::runtime_error(
            "LinearDispatch: expected weight shape [out_features, in_features]");
    }
    int in_features;
    if (x.num_dims() == 1) {
        in_features = x.shape[0];
    } else if (x.num_dims() == 2 && x.shape[0] == 1) {
        in_features = x.shape[1];
    } else {
        throw std::runtime_error(
            "LinearDispatch: expected x shape [in_features] or [1, in_features]");
    }
    if (weight.shape[1] != in_features) {
        throw std::runtime_error("LinearDispatch: dimension mismatch");
    }

    switch (mode) {
    case MatmulMode::kNaive:
        return LinearNaive(x, weight);
    case MatmulMode::kThreaded:
        return LinearThreaded(x, weight);
    case MatmulMode::kSimd:
        return LinearSimd(x, weight);
    case MatmulMode::kThreadedSimd:
        return LinearThreadedSimd(x, weight);
    }

    return LinearThreadedSimd(x, weight);
}

} // namespace mini-llama
