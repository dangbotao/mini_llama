#include "mini_llama/quantized_tensor.h"

#include <limits>
#include <stdexcept>

namespace mini_llama {

std::size_t QuantizedTensor::num_elements() const {
  if (shape.empty()) {
    return 0;
  }

  std::size_t total = 1;
  for (int dimension : shape) {
    if (dimension <= 0) {
      throw std::runtime_error(
          "QuantizedTensor: dimensions must be positive");
    }

    const std::size_t size = static_cast<std::size_t>(dimension);
    if (total > std::numeric_limits<std::size_t>::max() / size) {
      throw std::runtime_error("QuantizedTensor: element count overflow");
    }
    total *= size;
  }
  return total;
}

std::string QuantizedTensor::ShapeString() const {
  std::string result = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      result += ", ";
    }
    result += std::to_string(shape[i]);
  }
  result += "]";
  return result;
}

void QuantizedTensor::AssertShape(const std::vector<int>& expected,
                                  const char* caller) const {
  if (shape == expected) {
    return;
  }

  QuantizedTensor expected_tensor;
  expected_tensor.shape = expected;
  throw std::runtime_error(std::string(caller) +
                           ": expected shape " +
                           expected_tensor.ShapeString() + ", got " +
                           ShapeString());
}

// F32 QuantizedTensor 转换为普通 Tensor
Tensor ToTensor(const QuantizedTensor& tensor) {
  if (tensor.type != QuantType::kF32) {
    throw std::runtime_error(
        "ToTensor: cannot convert Q8_0 without dequantization");
  }

  Tensor result(tensor.shape, 0.0f);
  if (tensor.f32_data.size() != result.data.size()) {
    throw std::runtime_error("ToTensor: F32 data size does not match its shape");
  }
  result.data = tensor.f32_data;
  return result;
}

// 普通 Tensor 转换为 F32 QuantizedTensor
QuantizedTensor ToQuantizedTensor(const Tensor& tensor) {
  QuantizedTensor result;
  result.type = QuantType::kF32;
  result.shape = tensor.shape;
  result.f32_data = tensor.data;
  return result;
}

}  // namespace mini_llama
