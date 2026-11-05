#include "mini_llama/quant.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define MINI_LLAMA_QUANT_USE_AVX2 1
#endif

#include "mini_llama/thread_pool.h"

namespace mini_llama {
namespace {

// Convert the FP16 scale stored in GGML quantization blocks to FP32.
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

float DotQ40(const float* input, const BlockQ40* blocks, int block_count) {
#ifdef MINI_LLAMA_QUANT_USE_AVX2
  __m256 sum0 = _mm256_setzero_ps();
  __m256 sum1 = _mm256_setzero_ps();
  const __m128i nibble_mask = _mm_set1_epi8(0x0f);
  const __m128i zero_point = _mm_set1_epi8(8);

  for (int block_index = 0; block_index < block_count; ++block_index) {
    const BlockQ40& block = blocks[block_index];
    const __m128i packed =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(block.qs));
    const __m128i low =
        _mm_sub_epi8(_mm_and_si128(packed, nibble_mask), zero_point);
    const __m128i high = _mm_sub_epi8(
        _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask), zero_point);

    const __m256 scale = _mm256_set1_ps(Fp16ToFloat(block.d));
    const __m256 q0 = _mm256_mul_ps(
        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(low)), scale);
    const __m256 q1 = _mm256_mul_ps(
        _mm256_cvtepi32_ps(
            _mm256_cvtepi8_epi32(_mm_srli_si128(low, 8))),
        scale);
    const __m256 q2 = _mm256_mul_ps(
        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(high)), scale);
    const __m256 q3 = _mm256_mul_ps(
        _mm256_cvtepi32_ps(
            _mm256_cvtepi8_epi32(_mm_srli_si128(high, 8))),
        scale);

    const float* x = input + block_index * kQ40BlockSize;
    sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(x), q0, sum0);
    sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + 8), q1, sum1);
    sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + 16), q2, sum0);
    sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + 24), q3, sum1);
  }

  const __m256 total = _mm256_add_ps(sum0, sum1);
  __m128 sum = _mm_add_ps(_mm256_castps256_ps128(total),
                          _mm256_extractf128_ps(total, 1));
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  return _mm_cvtss_f32(sum);
#else
  float sum = 0.0f;
  for (int block_index = 0; block_index < block_count; ++block_index) {
    const BlockQ40& block = blocks[block_index];
    const float scale = Fp16ToFloat(block.d);
    const float* x = input + block_index * kQ40BlockSize;
    for (int i = 0; i < kQ40BlockSize / 2; ++i) {
      const uint8_t packed = block.qs[i];
      sum += x[i] * scale * static_cast<float>((packed & 0x0f) - 8);
      sum += x[i + kQ40BlockSize / 2] * scale *
             static_cast<float>((packed >> 4) - 8);
    }
  }
  return sum;
#endif
}

float DotQ6K(const float* input, const BlockQ6K* blocks, int block_count) {
  float sum = 0.0f;
  for (int block_index = 0; block_index < block_count; ++block_index) {
    const BlockQ6K& block = blocks[block_index];
    const float super_scale = Fp16ToFloat(block.d);
    const float* x = input + block_index * kQ6KBlockSize;
    const uint8_t* ql = block.ql;
    const uint8_t* qh = block.qh;
    const int8_t* scales = block.scales;

    for (int group = 0; group < kQ6KBlockSize; group += 128) {
      for (int lane = 0; lane < 32; ++lane) {
        const int scale_index = lane / 16;
        const int q1 =
            ((ql[lane] & 0x0f) | (((qh[lane] >> 0) & 3) << 4)) - 32;
        const int q2 =
            ((ql[lane + 32] & 0x0f) |
             (((qh[lane] >> 2) & 3) << 4)) -
            32;
        const int q3 =
            ((ql[lane] >> 4) | (((qh[lane] >> 4) & 3) << 4)) - 32;
        const int q4 =
            ((ql[lane + 32] >> 4) |
             (((qh[lane] >> 6) & 3) << 4)) -
            32;

        sum += x[group + lane] * super_scale * scales[scale_index] * q1;
        sum += x[group + lane + 32] * super_scale *
               scales[scale_index + 2] * q2;
        sum += x[group + lane + 64] * super_scale *
               scales[scale_index + 4] * q3;
        sum += x[group + lane + 96] * super_scale *
               scales[scale_index + 6] * q4;
      }
      ql += 64;
      qh += 32;
      scales += 8;
    }
  }
  return sum;
}

}  // namespace

Tensor DequantizeFromQ40(const std::vector<BlockQ40>& blocks,
                         const std::vector<int>& shape) {
  const std::size_t total = CheckedNumElements(shape, "DequantizeFromQ40");
  if (total % kQ40BlockSize != 0) {
    throw std::runtime_error(
        "DequantizeFromQ40: element count must be divisible by 32");
  }
  const std::size_t expected_blocks = total / kQ40BlockSize;
  if (blocks.size() != expected_blocks) {
    throw std::runtime_error(
        "DequantizeFromQ40: block count mismatch: expected " +
        std::to_string(expected_blocks) + ", got " +
        std::to_string(blocks.size()));
  }

  Tensor result(shape, 0.0f);
  for (std::size_t block_index = 0; block_index < blocks.size();
       ++block_index) {
    const BlockQ40& block = blocks[block_index];
    const float scale = Fp16ToFloat(block.d);
    const std::size_t offset = block_index * kQ40BlockSize;
    for (int i = 0; i < kQ40BlockSize / 2; ++i) {
      const uint8_t packed = block.qs[i];
      result.data[offset + i] =
          scale * static_cast<float>((packed & 0x0f) - 8);
      result.data[offset + i + kQ40BlockSize / 2] =
          scale * static_cast<float>((packed >> 4) - 8);
    }
  }
  return result;
}

Tensor LinearQ40(const Tensor& x, const std::vector<BlockQ40>& weight,
                 const std::vector<int>& weight_shape) {
  if (weight_shape.size() != 2 || weight_shape[0] <= 0 ||
      weight_shape[1] <= 0) {
    throw std::runtime_error(
        "LinearQ40: weight shape must be [out_features, in_features]");
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
        "LinearQ40: input must be [in_features] or [batch, in_features]");
  }

  const int out_features = weight_shape[0];
  if (weight_shape[1] != in_features) {
    throw std::runtime_error("LinearQ40: input and weight dimensions mismatch");
  }
  if (in_features % kQ40BlockSize != 0) {
    throw std::runtime_error(
        "LinearQ40: in_features must be divisible by 32");
  }

  const int blocks_per_row = in_features / kQ40BlockSize;
  const std::size_t expected_blocks =
      static_cast<std::size_t>(out_features) * blocks_per_row;
  if (weight.size() != expected_blocks) {
    throw std::runtime_error(
        "LinearQ40: block count mismatch: expected " +
        std::to_string(expected_blocks) + ", got " +
        std::to_string(weight.size()));
  }

  Tensor result(input_is_1d ? std::vector<int>{out_features}
                            : std::vector<int>{batch, out_features},
                0.0f);
  ParallelFor(batch * out_features, [&](int begin, int end) {
    for (int task = begin; task < end; ++task) {
      const int row = task / out_features;
      const int output = task % out_features;
      const float* input =
          x.data.data() + static_cast<std::size_t>(row) * in_features;
      const BlockQ40* blocks =
          weight.data() + static_cast<std::size_t>(output) * blocks_per_row;
      result.data[static_cast<std::size_t>(row) * out_features + output] =
          DotQ40(input, blocks, blocks_per_row);
    }
  });
  return result;
}

Tensor DequantizeFromQ6K(const std::vector<BlockQ6K>& blocks,
                         const std::vector<int>& shape) {
  const std::size_t total = CheckedNumElements(shape, "DequantizeFromQ6K");
  if (total % kQ6KBlockSize != 0) {
    throw std::runtime_error(
        "DequantizeFromQ6K: element count must be divisible by 256");
  }
  const std::size_t expected_blocks = total / kQ6KBlockSize;
  if (blocks.size() != expected_blocks) {
    throw std::runtime_error(
        "DequantizeFromQ6K: block count mismatch: expected " +
        std::to_string(expected_blocks) + ", got " +
        std::to_string(blocks.size()));
  }

  Tensor result(shape, 0.0f);
  for (std::size_t block_index = 0; block_index < blocks.size();
       ++block_index) {
    const BlockQ6K& block = blocks[block_index];
    const float super_scale = Fp16ToFloat(block.d);
    const std::size_t block_offset = block_index * kQ6KBlockSize;
    const uint8_t* ql = block.ql;
    const uint8_t* qh = block.qh;
    const int8_t* scales = block.scales;

    for (int group = 0; group < kQ6KBlockSize; group += 128) {
      for (int lane = 0; lane < 32; ++lane) {
        const int scale_index = lane / 16;
        const int q1 =
            ((ql[lane] & 0x0f) | (((qh[lane] >> 0) & 3) << 4)) - 32;
        const int q2 =
            ((ql[lane + 32] & 0x0f) |
             (((qh[lane] >> 2) & 3) << 4)) -
            32;
        const int q3 =
            ((ql[lane] >> 4) | (((qh[lane] >> 4) & 3) << 4)) - 32;
        const int q4 =
            ((ql[lane + 32] >> 4) |
             (((qh[lane] >> 6) & 3) << 4)) -
            32;

        result.data[block_offset + group + lane] =
            super_scale * scales[scale_index] * q1;
        result.data[block_offset + group + lane + 32] =
            super_scale * scales[scale_index + 2] * q2;
        result.data[block_offset + group + lane + 64] =
            super_scale * scales[scale_index + 4] * q3;
        result.data[block_offset + group + lane + 96] =
            super_scale * scales[scale_index + 6] * q4;
      }
      ql += 64;
      qh += 32;
      scales += 8;
    }
  }
  return result;
}

Tensor LinearQ6K(const Tensor& x, const std::vector<BlockQ6K>& weight,
                 const std::vector<int>& weight_shape) {
  if (weight_shape.size() != 2 || weight_shape[0] <= 0 ||
      weight_shape[1] <= 0) {
    throw std::runtime_error(
        "LinearQ6K: weight shape must be [out_features, in_features]");
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
        "LinearQ6K: input must be [in_features] or [batch, in_features]");
  }

  const int out_features = weight_shape[0];
  if (weight_shape[1] != in_features) {
    throw std::runtime_error("LinearQ6K: input and weight dimensions mismatch");
  }
  if (in_features % kQ6KBlockSize != 0) {
    throw std::runtime_error(
        "LinearQ6K: in_features must be divisible by 256");
  }

  const int blocks_per_row = in_features / kQ6KBlockSize;
  const std::size_t expected_blocks =
      static_cast<std::size_t>(out_features) * blocks_per_row;
  if (weight.size() != expected_blocks) {
    throw std::runtime_error(
        "LinearQ6K: block count mismatch: expected " +
        std::to_string(expected_blocks) + ", got " +
        std::to_string(weight.size()));
  }

  Tensor result(input_is_1d ? std::vector<int>{out_features}
                            : std::vector<int>{batch, out_features},
                0.0f);
  ParallelFor(batch * out_features, [&](int begin, int end) {
    for (int task = begin; task < end; ++task) {
      const int row = task / out_features;
      const int output = task % out_features;
      const float* input =
          x.data.data() + static_cast<std::size_t>(row) * in_features;
      const BlockQ6K* blocks =
          weight.data() + static_cast<std::size_t>(output) * blocks_per_row;
      result.data[static_cast<std::size_t>(row) * out_features + output] =
          DotQ6K(input, blocks, blocks_per_row);
    }
  });
  return result;
}

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
