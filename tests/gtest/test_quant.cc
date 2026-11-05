#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "mini_llama/quant.h"
#include "mini_llama/quantized_tensor.h"

namespace mini_llama {
namespace {

BlockQ80 MakeUnitScaleBlock(int8_t value) {
  BlockQ80 block{};
  block.d = 0x3c00;  // FP16 representation of 1.0.
  for (int i = 0; i < kQ80BlockSize; ++i) {
    block.qs[i] = value;
  }
  return block;
}

BlockQ40 MakeUnitScaleQ40Block(int value) {
  BlockQ40 block{};
  block.d = 0x3c00;  // FP16 representation of 1.0.
  const uint8_t nibble = static_cast<uint8_t>(value + 8);
  const uint8_t packed = static_cast<uint8_t>(nibble | (nibble << 4));
  for (int i = 0; i < kQ40BlockSize / 2; ++i) {
    block.qs[i] = packed;
  }
  return block;
}

BlockQ6K MakeUnitScaleQ6KBlock(int value) {
  BlockQ6K block{};
  block.d = 0x3c00;  // FP16 representation of 1.0.
  const uint8_t encoded = static_cast<uint8_t>(value + 32);
  const uint8_t low = encoded & 0x0f;
  const uint8_t high = (encoded >> 4) & 0x03;
  const uint8_t packed_low = static_cast<uint8_t>(low | (low << 4));
  const uint8_t packed_high = static_cast<uint8_t>(
      high | (high << 2) | (high << 4) | (high << 6));
  for (uint8_t& byte : block.ql) {
    byte = packed_low;
  }
  for (uint8_t& byte : block.qh) {
    byte = packed_high;
  }
  for (int8_t& scale : block.scales) {
    scale = 1;
  }
  return block;
}

TEST(QuantizedTensorTest, ConvertsF32ContainerToAndFromTensor) {
  Tensor source({2, 2}, 0.0f);
  source.data = {1.0f, 2.0f, 3.0f, 4.0f};

  QuantizedTensor quantized = ToQuantizedTensor(source);
  EXPECT_EQ(quantized.type, QuantType::kF32);
  EXPECT_EQ(quantized.shape, source.shape);
  EXPECT_EQ(quantized.f32_data, source.data);

  Tensor restored = ToTensor(quantized);
  EXPECT_EQ(restored.shape, source.shape);
  EXPECT_EQ(restored.data, source.data);
}

TEST(QuantTest, DequantizesQ80Block) {
  BlockQ80 block{};
  block.d = 0x3800;  // FP16 representation of 0.5.
  for (int i = 0; i < kQ80BlockSize; ++i) {
    block.qs[i] = static_cast<int8_t>(i - 16);
  }

  Tensor result = DequantizeFromQ80({block}, {kQ80BlockSize});
  ASSERT_EQ(result.size(), static_cast<std::size_t>(kQ80BlockSize));
  EXPECT_FLOAT_EQ(result.data[0], -8.0f);
  EXPECT_FLOAT_EQ(result.data[16], 0.0f);
  EXPECT_FLOAT_EQ(result.data[31], 7.5f);
}

TEST(QuantTest, DequantizesQ40BlockInGgmlNibbleOrder) {
  BlockQ40 block{};
  block.d = 0x3800;  // FP16 representation of 0.5.
  for (int i = 0; i < kQ40BlockSize / 2; ++i) {
    const uint8_t low = static_cast<uint8_t>(i);
    const uint8_t high = static_cast<uint8_t>(15 - i);
    block.qs[i] = static_cast<uint8_t>(low | (high << 4));
  }

  Tensor result = DequantizeFromQ40({block}, {kQ40BlockSize});
  ASSERT_EQ(result.size(), static_cast<std::size_t>(kQ40BlockSize));
  EXPECT_FLOAT_EQ(result.data[0], -4.0f);
  EXPECT_FLOAT_EQ(result.data[15], 3.5f);
  EXPECT_FLOAT_EQ(result.data[16], 3.5f);
  EXPECT_FLOAT_EQ(result.data[31], -4.0f);
}

TEST(QuantTest, ComputesQ40LinearForVectorAndBatch) {
  std::vector<BlockQ40> weight{
      MakeUnitScaleQ40Block(1), MakeUnitScaleQ40Block(-1)};

  Tensor vector({kQ40BlockSize}, 1.0f);
  Tensor vector_result =
      LinearQ40(vector, weight, {2, kQ40BlockSize});
  EXPECT_EQ(vector_result.shape, std::vector<int>({2}));
  EXPECT_FLOAT_EQ(vector_result.data[0], 32.0f);
  EXPECT_FLOAT_EQ(vector_result.data[1], -32.0f);

  Tensor batch({2, kQ40BlockSize}, 1.0f);
  Tensor batch_result = LinearQ40(batch, weight, {2, kQ40BlockSize});
  EXPECT_EQ(batch_result.shape, std::vector<int>({2, 2}));
  EXPECT_EQ(batch_result.data,
            (std::vector<float>{32.0f, -32.0f, 32.0f, -32.0f}));
}

TEST(QuantTest, DequantizesAndComputesQ6KLinear) {
  const BlockQ6K positive = MakeUnitScaleQ6KBlock(1);
  const BlockQ6K negative = MakeUnitScaleQ6KBlock(-1);

  Tensor restored = DequantizeFromQ6K({positive}, {kQ6KBlockSize});
  ASSERT_EQ(restored.size(), static_cast<std::size_t>(kQ6KBlockSize));
  for (float value : restored.data) {
    EXPECT_FLOAT_EQ(value, 1.0f);
  }

  Tensor input({kQ6KBlockSize}, 1.0f);
  Tensor result = LinearQ6K(input, {positive, negative},
                            {2, kQ6KBlockSize});
  EXPECT_EQ(result.shape, std::vector<int>({2}));
  EXPECT_FLOAT_EQ(result.data[0], 256.0f);
  EXPECT_FLOAT_EQ(result.data[1], -256.0f);
}

TEST(QuantTest, ComputesQ80LinearForVectorAndBatch) {
  std::vector<BlockQ80> weight{
      MakeUnitScaleBlock(1), MakeUnitScaleBlock(-1)};

  Tensor vector({kQ80BlockSize}, 1.0f);
  Tensor vector_result =
      LinearQ80(vector, weight, {2, kQ80BlockSize});
  EXPECT_EQ(vector_result.shape, std::vector<int>({2}));
  EXPECT_FLOAT_EQ(vector_result.data[0], 32.0f);
  EXPECT_FLOAT_EQ(vector_result.data[1], -32.0f);

  Tensor batch({2, kQ80BlockSize}, 1.0f);
  Tensor batch_result = LinearQ80(batch, weight, {2, kQ80BlockSize});
  EXPECT_EQ(batch_result.shape, std::vector<int>({2, 2}));
  EXPECT_EQ(batch_result.data,
            (std::vector<float>{32.0f, -32.0f, 32.0f, -32.0f}));
}

TEST(QuantTest, RejectsInvalidQ80BlockCount) {
  EXPECT_THROW(DequantizeFromQ80({}, {kQ80BlockSize}),
               std::runtime_error);
}

}  // namespace
}  // namespace mini_llama
