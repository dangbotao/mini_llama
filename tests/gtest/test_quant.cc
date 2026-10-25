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
