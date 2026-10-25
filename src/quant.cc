#include "mini_llama/quant.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace mini_llama {
namespace {

float Fp16ToFloat(uint16_t value) {
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
  uint32_t exponent = (value >> 10) & 0x1fu;
  uint32_t mantissa = value & 0x03ffu;
  uint32_t bits = 0;

  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      int shift = 0;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        ++shift;
      }
      mantissa &= 0x03ffu;
      const uint32_t exponent32 =
          static_cast<uint32_t>(127 - 15 + 1 - shift);
      bits = sign | (exponent32 << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    exponent += 127 - 15;
    bits = sign | (exponent << 23) | (mantissa << 13);
  }

  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

std::size_t CheckedNumElements(const std::vector<int>& shape,
                               const char* caller) {
  if (shape.empty()) {
    throw std::runtime_error(std::string(caller) + ": shape is empty");
  }

  std::size_t total = 1;
  for (int dimension : shape) {
    if (dimension <= 0) {
      throw std::runtime_error(std::string(caller) +
                               ": dimensions must be positive");
    }

    const std::size_t size = static_cast<std::size_t>(dimension);
    if (total > std::numeric_limits<std::size_t>::max() / size) {
      throw std::runtime_error(std::string(caller) +
                               ": element count overflow");
    }
    total *= size;
  }
  return total;
}

}  // namespace

// 将 GGML Q8_0 量化块反量化为普通 F32 Tensor
Tensor DequantizeFromQ80(const std::vector<BlockQ80>& blocks,
                         const std::vector<int>& shape) {
  const std::size_t total = CheckedNumElements(shape, "DequantizeFromQ80"); // 元素数，不是block数
  if (total % kQ80BlockSize != 0) { // 可被32整除
    throw std::runtime_error(
        "DequantizeFromQ80: element count must be divisible by 32");
  }

  // 验证通过元素数计算得到的期望块数是否与存储的块数一致
  const std::size_t expected_blocks = total / kQ80BlockSize;
  if (blocks.size() != expected_blocks) {
    throw std::runtime_error(
        "DequantizeFromQ80: block count mismatch: expected " +
        std::to_string(expected_blocks) + ", got " +
        std::to_string(blocks.size()));
  }

  // 逐个处理 Q8_0 block
  Tensor result(shape, 0.0f);
  for (std::size_t block_index = 0; block_index < blocks.size();
       ++block_index) {
    const BlockQ80& block = blocks[block_index];

    // block.d 保存 FP16 的二进制位，需要先转换成 F32 缩放系数（不可直接强转）
    const float scale = Fp16ToFloat(block.d);
    // 当前 block 在最终 F32 数组中的起始位置。
    const std::size_t offset = block_index * kQ80BlockSize;

    // 使用 value = scale × quantized_value 还原 32 个元素，放入result
    for (int i = 0; i < kQ80BlockSize; ++i) {
      result.data[offset + static_cast<std::size_t>(i)] =
          scale * static_cast<float>(block.qs[i]);
    }
  }
  return result;
}

//
Tensor LinearQ80(const Tensor& x, const std::vector<BlockQ80>& weight,
                 const std::vector<int>& weight_shape) {
  if (weight_shape.size() != 2 || weight_shape[0] <= 0 ||
      weight_shape[1] <= 0) {
    throw std::runtime_error(
        "LinearQ80: weight shape must be [out_features, in_features]");
  }

  int batch = 1;
  int in_features = 0;
  const bool input_is_1d = x.num_dims() == 1;
  if (input_is_1d) {
    in_features = x.shape[0];
  } else if (x.num_dims() == 2) {
    batch = x.shape[0];
    in_features = x.shape[1];
  } else {
    throw std::runtime_error(
        "LinearQ80: input must be [in_features] or [batch, in_features]");
  }

  const int out_features = weight_shape[0];
  if (weight_shape[1] != in_features) {
    throw std::runtime_error("LinearQ80: input and weight dimensions mismatch");
  }
  if (in_features % kQ80BlockSize != 0) {
    throw std::runtime_error(
        "LinearQ80: in_features must be divisible by 32");
  }

  const int blocks_per_row = in_features / kQ80BlockSize;
  const std::size_t expected_blocks =
      static_cast<std::size_t>(out_features) * blocks_per_row;
  if (weight.size() != expected_blocks) {
    throw std::runtime_error(
        "LinearQ80: block count mismatch: expected " +
        std::to_string(expected_blocks) + ", got " +
        std::to_string(weight.size()));
  }

  Tensor result(input_is_1d ? std::vector<int>{out_features}
                            : std::vector<int>{batch, out_features},
                0.0f);

  for (int row = 0; row < batch; ++row) {
    const float* input =
        x.data.data() + static_cast<std::size_t>(row) * in_features;
    for (int output = 0; output < out_features; ++output) {
      float sum = 0.0f;
      const std::size_t block_base =
          static_cast<std::size_t>(output) * blocks_per_row;

      for (int block_index = 0; block_index < blocks_per_row;
           ++block_index) {
        const BlockQ80& block = weight[block_base + block_index];
        const float scale = Fp16ToFloat(block.d);
        const int input_offset = block_index * kQ80BlockSize;
        for (int i = 0; i < kQ80BlockSize; ++i) {
          sum += input[input_offset + i] * scale *
                 static_cast<float>(block.qs[i]);
        }
      }

      result.data[static_cast<std::size_t>(row) * out_features + output] =
          sum;
    }
  }

  return result;
}

}  // namespace mini_llama
