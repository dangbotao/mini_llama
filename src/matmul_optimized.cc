#include "mini_llama/matmul_dispatch.h"

#include <stdexcept>

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

MatmulMode DefaultMatmulMode() { return MatmulMode::kNaive; }

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
    }

    return MatmulNaive(a, b);
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
    }

    return LinearNaive(x, weight);
}

} // namespace mini_llama
