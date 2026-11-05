# pragma once

#include <string>

#include "mini_llama/tensor.h"
#include "mini_llama/quantized_tensor.h"

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

  // Quantized matrix weights supported by the GGUF loader.
  QuantizedTensor wq;      // [n_heads * head_dim, dim] F32/Q4_0/Q6_K/Q8_0
  QuantizedTensor wk;      // [n_kv_heads * head_dim, dim]
  QuantizedTensor wv;      // [n_kv_heads * head_dim, dim]

  Tensor bq;
  Tensor bk;
  Tensor bv;

  QuantizedTensor wo;      // [dim, n_heads * head_dim]

  Tensor ffn_norm;         // [dim] F32

  QuantizedTensor w_gate;  // [hidden_dim, dim]
  QuantizedTensor w_up;    // [hidden_dim, dim]
  QuantizedTensor w_down;  // [dim, hidden_dim]
};

struct MiniLlamaModel {
    ModelConfig config;
    bool loaded = false;
    std::string load_error = "";

    // weights
    Tensor token_embedding;            // [vocab_size, dim]   token_id -> dim
    std::vector<LayerWeights> layers;
    Tensor final_norm;                 // [dim]
    QuantizedTensor lm_head;                    // [vocab_size, dim]
};

}  // namespace mini_llama
