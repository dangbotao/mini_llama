#include "mini_llama/tensor.h"

#include <limits>
#include <stdexcept>

namespace mini_llama {


// Validates a tensor shape and returns the number of elements it describes.
size_t CheckedElementNum(const std::vector<int>& input_shape) {
    if (input_shape.empty()) {
        throw std::runtime_error("tensor shape must not be empty");
    }

    std::size_t total = 1;
    for (const int dimension : input_shape) {
        if (dimension <= 0) {
            throw std::runtime_error(
                "every tensor dimension must be greater than zero");
        }

        const auto dimension_size = static_cast<std::size_t>(dimension);
        if (total > std::numeric_limits<std::size_t>::max() / dimension_size) {
            throw std::runtime_error("tensor element count overflows size_t");
        }
        total *= dimension_size;
    }

    return total;
}

void CheckAxisIndex(const Tensor& t, size_t axis, int indice) {
    const int dim = t.shape[axis];
    if (indice < 0 || indice >= dim) {
        throw std::out_of_range("indices out of range");
    }
}

void CheckRank(const Tensor& t, size_t expected_size) {
    if (t.num_dims() != expected_size) {
        throw std::runtime_error("rank does not match");
    }
}

Tensor::Tensor(const std::vector<int> input_shape, float fill)
    : shape(input_shape) {
    const std::size_t total = CheckedElementNum(input_shape);
    data.resize(total, fill);
}

size_t Tensor::FlatIndex(const std::vector<int>& indices) const {
    if (indices.size() != shape.size()) {
        throw std::runtime_error(" Indices number does not match the dimensions of the Tensor. ");
    }
    // CheckRank(*this, indices.size());

    for (size_t axis = 0; axis < indices.size(); ++axis) {
        CheckAxisIndex(*this, axis, indices[axis]);
    }

    size_t flat_index = 0;
    size_t stride = 1;
    for (int axis = shape.size() - 1; axis >=0; axis--) {
        flat_index += indices[axis] * stride;
        stride *= shape[axis];
    }

    return flat_index;
}

float& Tensor::At(const std::vector<int>& indices) {
    return data[FlatIndex(indices)];
}

float& Tensor::At1(int i) {
    CheckRank(*this, 1);
    CheckAxisIndex(*this, 0, i);
    return data[i];
}
float& Tensor::At2(int i, int j) {
    CheckRank(*this, 2);
    CheckAxisIndex(*this, 0, i);
    CheckAxisIndex(*this, 1, j);
    return data[i * shape[1] + j];
}
float& Tensor::At3(int i, int j, int k) {
    CheckRank(*this, 3);
    CheckAxisIndex(*this, 0, i);
    CheckAxisIndex(*this, 1, j);
    CheckAxisIndex(*this, 2, k);
    return data[i * shape[1] * shape[2] + j * shape[2] + k];
}
float& Tensor::At4(int i, int j, int k, int l) {
    CheckRank(*this, 4);
    CheckAxisIndex(*this, 0, i);
    CheckAxisIndex(*this, 1, j);
    CheckAxisIndex(*this, 2, k);
    CheckAxisIndex(*this, 3, l);
    return data[i * shape[1] * shape[2] * shape[3] +
    j * shape[2] * shape[3] + k * shape[3] + l];
}

float* Tensor::RowPtr(int row) {
    CheckRank(*this, 2);
    CheckAxisIndex(*this, 0, row);
    return data.data() + row * shape[1];
}

const float* Tensor::RowPtr(int row) const {
    CheckRank(*this, 2);
    CheckAxisIndex(*this, 0, row);
    return data.data() + row * shape[1];
}

void Tensor::AssertShape(const std::vector<int>& expected_shape) {
    if (shape != expected_shape) {
        throw std::runtime_error("shape does not match !");
    }
}

Tensor Tensor::ReshapeChecked(const std::vector<int>& new_shape) {
    const size_t new_total = CheckedElementNum(new_shape);
    if (new_total != num_elements()) {
        throw std::runtime_error("data num does not match !");
    }

    Tensor new_tensor = *this;
    new_tensor.shape = new_shape;
    return new_tensor;
}

Tensor MakeTensor1D(int size, float fill) {
    return Tensor({size}, fill);
}

Tensor MakeTensor2D(int rows, int cols, float fill) {
    return Tensor({rows, cols}, fill);
}

Tensor MakeTensor3D(
    int dim0, int dim1, int dim2, float fill) {
    return Tensor({dim0, dim1, dim2}, fill);
}

Tensor MakeTensor4D(
    int dim0, int dim1, int dim2, int dim3, float fill) {
    return Tensor({dim0, dim1, dim2, dim3}, fill);
}

}  // namespace mini_llama
