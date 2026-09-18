#pragma once

#include <cstddef>
#include <vector>

namespace mini_llama {

struct Tensor {
    std::vector<float> data;
    std::vector<int> shape;

    Tensor() = default;
    Tensor(const std::vector<int> input_shape, float fill = 0.0f);

    size_t size() const { return data.size(); }
    size_t num_elements() const { return data.size(); }
    size_t num_dims() const { return shape.size(); };

    // Convert the multi-dimensional indices into the one-dimensional data index
    size_t FlatIndex(const std::vector<int>& indices) const;

    float& At(const std::vector<int>& indices);

    float& At1(int i);
    float& At2(int i, int j);
    float& At3(int i, int j, int k);
    float& At4(int i, int j, int k, int l);

    float& operator[](int i) { return data[i]; };
    const float& operator[](int i) const { return data[i]; };// read only

    float* RowPtr(int row);
    const float* RowPtr(int row) const;

    void AssertShape(const std::vector<int>& expected_shape);
    // Reshape to new shape after verifying element count stays unchanged
    Tensor ReshapeChecked(const std::vector<int>& new_shape);
};

Tensor MakeTensor1D(int size, float fill = 0.0f);
Tensor MakeTensor2D(int rows, int cols, float fill = 0.0f);
Tensor MakeTensor3D(
    int dim0, int dim1, int dim2, float fill = 0.0f);
Tensor MakeTensor4D(
    int dim0, int dim1, int dim2, int dim3, float fill = 0.0f);

}
