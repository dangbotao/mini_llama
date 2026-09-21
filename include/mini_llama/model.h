#pragma once

#include "mini_llama/tensor.h"

namespace mini_llama {

enum class RopeType {
    kNormal,
    kNeox
};

struct ModelConfig {
  int vocab_size = 128;
  int dim = 32; // token_id -> embedding -> dim
  int hidden_dim = 86; // dim -> FFN -> hidden dim
  int n_layers = 2;
  int n_heads = 4; // q heads count
  int n_kv_heads = 4;
  int head_dim = 8;
  int max_seq_len = 128;

  float rope_theta = 10000.0f;
  float rms_norm_eps = 1e-5f;
  RopeType rope_type = RopeType::kNormal;
};

// a transformer layer weight
struct LayerWeights {
  // weight of rmsnorm at attention beginning
  Tensor attention_norm;   // [dim] F32

  Tensor wq;      // [n_heads * head_dim, dim]  F32/Q8_0/Q4_0
  Tensor wk;      // [n_kv_heads * head_dim, dim]
  Tensor wv;      // [n_kv_heads * head_dim, dim]
  Tensor wo;      // [dim, n_heads * head_dim]

  Tensor ffn_norm;         // [dim] F32

  Tensor w_gate;  // [hidden_dim, dim]
  Tensor w_up;    // [hidden_dim, dim]
  Tensor w_down;  // [dim, hidden_dim]
};

struct MiniLlamaModel {
    ModelConfig config;

    Tensor token_embedding;            // [vocab_size, dim]   token_id -> dim
    std::vector<LayerWeights> layers;
    Tensor final_norm;                 // [dim]
    Tensor lm_head;                    // [vocab_size, dim]
};

}  // namespace mini_llama
