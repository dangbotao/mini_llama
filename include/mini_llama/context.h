#pragma once

#include "mini_llama/tensor.h"
#include "mini_llama/model.h"
#include "kv_cache.h"

namespace mini_llama {

struct MiniLlamaContext {
  const MiniLlamaModel* model = nullptr;
  KvCache kv_cache;
  int pos = 0;

  // All tokens that have been fed through this context (including prefill).
  std::vector<int> token_history;

  MiniLlamaContext() = default;
  explicit MiniLlamaContext(const MiniLlamaModel* model);

  MiniLlamaContext(const MiniLlamaContext&) = delete;
  MiniLlamaContext& operator=(const MiniLlamaContext&) = delete;

  MiniLlamaContext(MiniLlamaContext&&) noexcept = default;
  MiniLlamaContext& operator=(MiniLlamaContext&&) noexcept = default;
};

}  // namespace mini_llama
