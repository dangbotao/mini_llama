#pragma once

#include "mini_llama/tensor.h"
#include "mini_llama/matmul_dispatch.h"
#include "mini_llama/model.h"
#include "mini_llama/quantized_tensor.h"

namespace mini_llama {

// Matrix multiplication: c = a * b
// a: [m, k], b: [k, n], c: [m, n]
Tensor Matmul(const Tensor& a, const Tensor& b);

// Linear projection: y = x * weight^T
// x: [in_features] or [1, in_features]
// weight: [out_features, in_features]
// result keeps x rank: [out_features] or [1, out_features]
Tensor Linear(const Tensor& x, const Tensor& weight);

// Linear projection with quantized weight.
// Dispatches to the F32 or Q8_0 path based on weight.type.
Tensor Linear(const Tensor& x, const QuantizedTensor& weight);

// Softmax over a 1D tensor.
// Formula: y_i = exp(x_i) / sum_j(exp(x_j)).
// For numerical stability, the implementation uses: 防止溢出
// y_i = exp(x_i - max(x)) / sum_j(exp(x_j - max(x))).
// x: [n], result: [n], and sum_i(y_i) = 1.
Tensor SoftMax(const Tensor& x);

// RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
// x: [dim], weight: [dim], result: [dim]
Tensor RmsNorm(const Tensor& x, const Tensor& weight, float eps);

// SwiGLU: SwiGlu(gate, up) = Silu(gate) * up
Tensor SwiGLU(const Tensor& gate, const Tensor& up);

// SiLU activation: Silu(x) = x * sigmoid(x)
Tensor Silu(const Tensor& x);

void Rope(Tensor& q, Tensor& k, int pos, float theta, RopeType type);

// Argmax: return FlatIndex of maximum value
int ArgMax(const Tensor& x);
}  // namespace mini_llama
