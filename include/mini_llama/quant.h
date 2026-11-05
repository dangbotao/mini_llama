#pragma once

#include <vector>

#include "mini_llama/quantized_tensor.h"

namespace mini_llama {

// Expand GGML-compatible Q4_0 blocks into an F32 Tensor.
Tensor DequantizeFromQ40(const std::vector<BlockQ40>& blocks,
                         const std::vector<int>& shape);

// Compute y = x * weight^T directly from Q4_0 blocks.
Tensor LinearQ40(const Tensor& x, const std::vector<BlockQ40>& weight,
                 const std::vector<int>& weight_shape);

// Expand GGML-compatible Q6_K blocks into an F32 Tensor.
Tensor DequantizeFromQ6K(const std::vector<BlockQ6K>& blocks,
                         const std::vector<int>& shape);

// Compute y = x * weight^T directly from Q6_K blocks.
Tensor LinearQ6K(const Tensor& x, const std::vector<BlockQ6K>& weight,
                 const std::vector<int>& weight_shape);

// Expand GGML-compatible Q8_0 blocks into an F32 Tensor.
Tensor DequantizeFromQ80(const std::vector<BlockQ80>& blocks,
                         const std::vector<int>& shape);

// Compute y = x * weight^T without expanding the complete Q8_0 weight.
// x: [in_features] or [batch, in_features]
// weight_shape: [out_features, in_features]
Tensor LinearQ80(const Tensor& x, const std::vector<BlockQ80>& weight,
                 const std::vector<int>& weight_shape);

}  // namespace mini_llama
