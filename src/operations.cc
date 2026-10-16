#include "mini_llama/operations.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace mini_llama {

Tensor Matmul(const Tensor& a, const Tensor& b) {
    return MatmulDispatch(a, b, DefaultMatmulMode());
}

Tensor Linear(const Tensor& x, const Tensor& weight) {
    return LinearDispatch(x, weight, DefaultMatmulMode());
}

Tensor SoftMax(const Tensor& x) {
    // check
    if (x.num_dims() != 1) {
        throw std::runtime_error("SoftMax: x must be 1D!");
    }
    if (x.size() == 0) {
        throw std::runtime_error("SoftMax: x can not be zero!");
    }

    // max_val
    int n = x.size();
    float max_val = x[0];
    for (int i = 0; i < n; ++i) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    // Tensor res = MakeTensor1D(n);
    Tensor res({n});
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        res[i] = std::exp(x[i] - max_val);
        sum += res[i];
    }

    for (int i = 0; i < n; ++i) {
        res[i] /= sum;
    }

    return res;
}

// RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
Tensor RmsNorm(const Tensor& x, const Tensor& weight, float eps) {
    if (x.num_dims() != 1 || weight.num_dims() != 1) {
        throw std::runtime_error("RmsNorm: input dim must be 1D!");
    }
    if (x.size() != weight.size()) {
        throw std::runtime_error("RmsNorm: input dim dismatch!");
    }

    int n = x.size();
    float ss = 0.0f;

    // 1 / sqrt(mean(x^2) + eps)
    for (int i = 0; i < n; ++i) {
        ss += x[i] * x[i];
    }
    float rms = std::sqrt(ss / n + eps);
    float scale = 1 / rms;

    // x / sqrt(mean(x^2) + eps) * weight
    Tensor res({n}, 0.0f);
    for (int i = 0; i <n; ++i) {
        res[i] = x[i] * scale * weight[i];
    }
    return res;
}

Tensor SwiGLU(const Tensor& gate, const Tensor& up) {
    if (gate.shape != up.shape) {
        throw std::runtime_error("SwiGLU: gate and up shape dismatch!");
    }

    Tensor res(gate.shape, 0.0f);
    int dim = gate.size();
    for (int i = 0; i < dim; ++i) {
        // res_i = gate_i * sigmoid(gate_i) * up_i
        float g = gate[i];
        float sigmoid = 1.0f / (1.0f + std::exp(-g));
        res[i] = g * sigmoid * up[i];
    }
    return res;
}

Tensor Silu(const Tensor& x) {
  Tensor y(x.shape, 0.0f);
  for (size_t i = 0; i < x.data.size(); ++i) {
    float val = x.data[i];
    y.data[i] = val / (1.0f + std::exp(-val));
  }
  return y;
}

static void ApplyRopeNormalToHeads(Tensor& x, int pos, float theta) {
    int heads = x.shape[0];
    int hidden_dim = x.shape[1];

    for (int i = 0; i < heads; ++i) {
        for (int j = 0; j < hidden_dim; j += 2) { // 相邻部分匹配
            float freq = 1.0f / std::pow(theta, static_cast<float>(j) /
                static_cast<float>(hidden_dim));

            float cos_val = std::cos(pos * freq);
            float sin_val = std::sin(pos * freq);
            float x0 = x[i * hidden_dim + j];
            float x1 = x[i * hidden_dim + j + 1];

            x[i * hidden_dim + j] = x0 * cos_val - x1 * sin_val;
            x[i * hidden_dim + j + 1] = x0 * sin_val + x1 * cos_val;
        }
    }
}

static void ApplyRopeNeoXToHeads(Tensor& x, int pos, float theta) {
    int heads = x.shape[0];
    int hidden_dim = x.shape[1];

    for (int i = 0; i < heads; ++i) {
        for (int j = 0; j < hidden_dim / 2; j += 1) { // 前半部分和后半部分匹配
            // freq = theta ^ (- 2i / d)
            float freq = 1.0f / std::pow(theta, static_cast<float>(2 * j) /
                static_cast<float>(hidden_dim));

            // x' = R(pos * freq) * x
            float cos_val = std::cos(pos * freq);
            float sin_val = std::sin(pos * freq);
            float x0 = x[i * hidden_dim + j];
            float x1 = x[i * hidden_dim + j + hidden_dim / 2];

            x[i * hidden_dim + j] = x0 * cos_val - x1 * sin_val;
            x[i * hidden_dim + j + hidden_dim / 2] = x0 * sin_val + x1 * cos_val;
        }
    }
}

void Rope(Tensor& q, Tensor& k, int pos, float theta, RopeType type) {
    if (q.num_dims() != 2 || k.num_dims() != 2) {
        throw std::runtime_error("Rope: expected 2D tensors");
    }
    if (pos < 0) {
        throw std::out_of_range("Rope: position must be non-negative");
    }
    if (!std::isfinite(theta) || theta <= 0.0f) {
    throw std::runtime_error("Rope: theta must be finite and positive");
    }

    if (type == RopeType::kNormal) {
        ApplyRopeNormalToHeads(q, pos, theta);
        ApplyRopeNormalToHeads(k, pos, theta);
    } else if (type == RopeType::kNeox) {
        ApplyRopeNeoXToHeads(q, pos, theta);
        ApplyRopeNeoXToHeads(k, pos, theta);
    }
}

int ArgMax(const Tensor& x) {
  if (x.size() == 0) {
    throw std::runtime_error("ArgMax: cannot compute ArgMax of empty tensor");
  }
  int best = 0;
  float best_val = x.data[0];
  for (size_t i = 1; i < x.data.size(); ++i) {
    if (x.data[i] > best_val) {
      best_val = x.data[i];
      best = static_cast<int>(i);
    }
  }
  return best;
}
}
