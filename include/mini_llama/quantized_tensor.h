#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "mini_llama/tensor.h"

namespace mini_llama {

// Quantization formats supported by the inference kernels.
enum class QuantType {
  kF32,
  kQ40,
  kQ6K,
  kQ80,
};

// GGML Q4_0 stores 32 signed 4-bit values and one FP16 scale per block.
// The first 16 values use the low nibbles; the remaining 16 use the high
// nibbles. Stored nibbles are biased by 8.
constexpr int kQ40BlockSize = 32;

struct BlockQ40 {
  uint16_t d;
  uint8_t qs[kQ40BlockSize / 2];
};

static_assert(sizeof(BlockQ40) == 18, "invalid GGML Q4_0 block layout");

// GGML Q6_K stores 256 values in sixteen 16-value groups. Each value has
// four low bits and two high bits, while each group has a signed scale.
constexpr int kQ6KBlockSize = 256;

struct BlockQ6K {
  uint8_t ql[kQ6KBlockSize / 2];
  uint8_t qh[kQ6KBlockSize / 4];
  int8_t scales[kQ6KBlockSize / 16];
  uint16_t d;
};

static_assert(sizeof(BlockQ6K) == 210, "invalid GGML Q6_K block layout");

// GGML Q8_0 stores 32 signed 8-bit values and one FP16 scale per block.
constexpr int kQ80BlockSize = 32;

struct BlockQ80 {
  uint16_t d;
  int8_t qs[kQ80BlockSize];
};

static_assert(sizeof(BlockQ80) == 34, "invalid GGML Q8_0 block layout");

// Only the vector matching `type` contains weight data.
struct QuantizedTensor {
  QuantType type = QuantType::kF32;
  std::vector<int> shape;
  std::vector<float> f32_data;
  std::vector<BlockQ40> q4_0_data;
  std::vector<BlockQ6K> q6_k_data;
  std::vector<BlockQ80> q8_0_data;

  std::size_t num_elements() const;
  int num_dims() const { return static_cast<int>(shape.size()); }
  std::string ShapeString() const;
  void AssertShape(const std::vector<int>& expected,
                   const char* caller) const;
};

// These conversions do not quantize or dequantize. They only move F32 data
// between the two container types.
Tensor ToTensor(const QuantizedTensor& tensor);
QuantizedTensor ToQuantizedTensor(const Tensor& tensor);

}  // namespace mini_llama
