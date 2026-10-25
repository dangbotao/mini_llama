#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "mini_llama/tensor.h"

namespace mini_llama {

// Minimal set needed by an F32/Q8_0 GGUF model.
enum class QuantType {
  kF32,
  kQ80,
};

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
