#pragma once

#include "mini_llama/tensor.h"
#include "mini_llama/matmul_dispatch.h"
#include "mini_llama/model.h"

namespace mini_llama {

// Matrix multiplication: c = a * b
// a: [m, k], b: [k, n], c: [m, n]
Tensor Matmul(const Tensor& a, const Tensor& b);

// Linear projection: y = x * weight^T
// x: [in_features] or [1, in_features]
// weight: [out_features, in_features]
// result keeps x rank: [out_features] or [1, out_features]
Tensor Linear(const Tensor& x, const Tensor& weight);

// Softmax over a 1D tensor
// x: [n], result: [n]
Tensor SoftMax(const Tensor& x);

// RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
// x: [dim], weight: [dim], result: [dim]
Tensor RmsNorm(const Tensor& x, const Tensor& weight, float eps);

// SwiGLU: SwiGlu(gate, up) = Silu(gate) * up
Tensor SwiGLU(const Tensor& gate, const Tensor& up);

// SiLU activation: Silu(x) = x * sigmoid(x)
Tensor Silu(const Tensor& x);

void Rope(Tensor& q, Tensor& k, int pos, float theta, RopeType type);
}  // namespace mini_llama
