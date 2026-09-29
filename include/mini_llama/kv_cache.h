#pragma once

#include "mini_llama/tensor.h"

namespace mini_llama {

// KV cache stores keys and values for all layers and positions
struct KvCache {
  // keys:   [n_layers, max_seq_len, n_kv_heads, head_dim]
  // values: [n_layers, max_seq_len, n_kv_heads, head_dim]
  Tensor keys;
  Tensor values;

  KvCache() = default;
  KvCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim);

  // Write k and v for a specific layer and position
  // k: [n_kv_heads, head_dim]
  // v: [n_kv_heads, head_dim]
  void Write(int layer, int pos, const Tensor& k, const Tensor& v);

  // Return a pointer to the key/value vector for one layer,
  // one token position, and one KV head.
  // The returned vector has shape [head_dim].
  // return: &keys[layer][pos][kv_head][0]  &values[layer][pos][kv_head][0]
  const float* KeyPtr(int layer, int pos, int kv_head) const;
  const float* ValuePtr(int layer, int pos, int kv_head) const;

};

}  // namespace mini_llama