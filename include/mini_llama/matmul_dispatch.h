#pragma once

#include "tensor.h"

namespace mini_llama {

enum class MatmulMode {
    kNaive,
    kThreaded,
    kSimd,
    kThreadedSimd,
};

// Default mode used by the main inference path.
MatmulMode DefaultMatmulMode();

Tensor MatmulDispatch(const Tensor& a, const Tensor& b, MatmulMode mode);

Tensor LinearDispatch(const Tensor& x, const Tensor& weight, MatmulMode mode);

} // namespace mini_llama
