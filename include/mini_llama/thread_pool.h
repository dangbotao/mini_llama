#pragma once

#include "mini_llama/tensor.h"

#include <functional>

namespace mini_llama {
    int GetThreadCount();
    void SetThreadCount(int n);
    // Todo:为什么function接受lambda能接受它引用捕获的参数
    void ParallelFor(int task_nums, const std::function<void(int, int)>& fn);
} // namespace mini_llama