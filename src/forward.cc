#include "mini_llama/forward.h"
#include "mini_llama/context.h"
#include "mini_llama/operations.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace mini_llama {

static void RequireShape(const Tensor& tensor,
                         const std::vector<int>& expected,
                         const std::string& name) {
  size_t expected_size = 1;
  for (int dim : expected) {
    expected_size *= static_cast<size_t>(dim);
  }
  if (tensor.shape != expected || tensor.size() != expected_size) {
    throw std::runtime_error("ForwardToken: invalid shape for " + name);
  }
}

// 检查：
// - 配置参数必须为正数
// - n_heads 能被 n_kv_heads 整除
// - head_dim 必须为偶数
// - RoPE、RMSNorm 参数合法
// - token 和 position 范围
// - Context 是否属于当前模型
// - 模型层数
// - Embedding、Norm、LM Head 形状
// - KV Cache 形状
// - 每层 Q/K/V、Attention、FFN 权重形状
static void ValidateForwardInputs(const MiniLlamaContext& ctx,
                                  const MiniLlamaModel& model,
                                  int token_id) {
  const ModelConfig& cfg = model.config;
  if (cfg.vocab_size <= 0 || cfg.dim <= 0 || cfg.hidden_dim <= 0 ||
      cfg.n_layers <= 0 || cfg.n_heads <= 0 || cfg.n_kv_heads <= 0 ||
      cfg.head_dim <= 0 || cfg.max_seq_len <= 0) {
    throw std::runtime_error("ForwardToken: model config must be positive");
  }
  if (cfg.n_heads % cfg.n_kv_heads != 0) {
    throw std::runtime_error(
        "ForwardToken: n_heads must be divisible by n_kv_heads");
  }
  if (cfg.head_dim % 2 != 0) {
    throw std::runtime_error("ForwardToken: head_dim must be even for RoPE");
  }
  if (!std::isfinite(cfg.rope_theta) || cfg.rope_theta <= 0.0f ||
      !std::isfinite(cfg.rms_norm_eps) || cfg.rms_norm_eps <= 0.0f) {
    throw std::runtime_error("ForwardToken: invalid RoPE or RMSNorm config");
  }
  if (token_id < 0 || token_id >= cfg.vocab_size) {
    throw std::out_of_range("ForwardToken: token id out of range");
  }
  if (ctx.pos < 0 || ctx.pos >= cfg.max_seq_len) {
    throw std::out_of_range("ForwardToken: position out of range");
  }
  if (ctx.model != &model) {
    throw std::runtime_error("ForwardToken: context belongs to another model");
  }
  if (model.layers.size() != static_cast<size_t>(cfg.n_layers)) {
    throw std::runtime_error("ForwardToken: layer count mismatch");
  }

  RequireShape(model.token_embedding, {cfg.vocab_size, cfg.dim},
               "token_embedding");
  RequireShape(model.final_norm, {cfg.dim}, "final_norm");
  RequireShape(model.lm_head, {cfg.vocab_size, cfg.dim}, "lm_head");
  RequireShape(ctx.kv_cache.keys,
               {cfg.n_layers, cfg.max_seq_len, cfg.n_kv_heads, cfg.head_dim},
               "kv_cache.keys");
  RequireShape(ctx.kv_cache.values,
               {cfg.n_layers, cfg.max_seq_len, cfg.n_kv_heads, cfg.head_dim},
               "kv_cache.values");

  for (int layer = 0; layer < cfg.n_layers; ++layer) {
    const LayerWeights& weights = model.layers[layer];
    const std::string prefix = "layers." + std::to_string(layer) + ".";
    RequireShape(weights.attention_norm, {cfg.dim},
                 prefix + "attention_norm");
    RequireShape(weights.wq, {cfg.n_heads * cfg.head_dim, cfg.dim},
                 prefix + "wq");
    RequireShape(weights.wk, {cfg.n_kv_heads * cfg.head_dim, cfg.dim},
                 prefix + "wk");
    RequireShape(weights.wv, {cfg.n_kv_heads * cfg.head_dim, cfg.dim},
                 prefix + "wv");
    RequireShape(weights.wo, {cfg.dim, cfg.n_heads * cfg.head_dim},
                 prefix + "wo");
    RequireShape(weights.ffn_norm, {cfg.dim}, prefix + "ffn_norm");
    RequireShape(weights.w_gate, {cfg.hidden_dim, cfg.dim},
                 prefix + "w_gate");
    RequireShape(weights.w_up, {cfg.hidden_dim, cfg.dim}, prefix + "w_up");
    RequireShape(weights.w_down, {cfg.dim, cfg.hidden_dim},
                 prefix + "w_down");
  }
}

// Elementwise residual addition: [dim] + [dim] -> [dim].
static Tensor Add(const Tensor& a, const Tensor& b) {
  if (a.shape != b.shape) {
    throw std::runtime_error("Add: shape mismatch");
  }

  Tensor out(a.shape, 0.0f);
  for (size_t i = 0; i < a.size(); ++i) {
    out.data[i] = a.data[i] + b.data[i];
  }
  return out;
}

// function: token_id -> [dim]
// input: token_embedding [vocab_size, dim], token_id [0, vocab_size - 1]
// output: [dim]
Tensor EmbedToken(const MiniLlamaModel& model, int token_id) {
    if (token_id >= model.config.vocab_size) {
        throw std::runtime_error("EmbedToken: token_id exceed vocab_size!");
    }
    if (token_id < 0) {
        throw std::runtime_error("EmbedToken: token_id can not be negative!");
    }
    int dim = model.config.dim;
    Tensor x({dim}, 0.0f);
    for (int d = 0; d < dim; ++d) {
        x[d] = model.token_embedding[token_id * dim + d];
    }
    return x;
}


// Single layer forward
//   x: input hidden state [dim]
//   cur_layer: layer FlatIndex
//   ctx: inference context (for pos and kv_cache)
// Returns updated hidden state [dim]
static Tensor ForwardLayer(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                const Tensor& x, int cur_layer) {
    const LayerWeights& lw = model.layers[cur_layer];
    const ModelConfig& cfg = model.config;
    const int cur_pos = ctx.pos;
    const int n_heads = cfg.n_heads;
    const int n_kv_heads = cfg.n_kv_heads;
    const int head_dim = cfg.head_dim;

    // ------------------------------------------------------------------
    // 1. Attention pre-norm and Q/K/V projections.
    // ------------------------------------------------------------------

    // rmsnorm
    Tensor normed = RmsNorm(x, lw.attention_norm, cfg.rms_norm_eps);

    // q\k\v projection:
    Tensor q_flat, k_flat, v_flat; // q*w_q, k*w_k, v*w_v
    q_flat = Linear(normed, lw.wq); // x [in_dim] * wq [out_dim, in_dim] -> q_flat [out_dim]
    k_flat = Linear(normed, lw.wk);
    v_flat = Linear(normed, lw.wv);

    // reshape to multiheads  [head_num, head_dim]
    Tensor q = q_flat.ReshapeChecked({n_heads, head_dim}); // normally n_heads > n_kv_heads
    Tensor k = k_flat.ReshapeChecked({n_kv_heads, head_dim});
    Tensor v = v_flat.ReshapeChecked({n_kv_heads, head_dim});

    // Apply RoPE to q and k
    Rope(q, k, cur_pos, cfg.rope_theta, cfg.rope_type);

    // write current k, v to kv_cache
    ctx.kv_cache.Write(cur_layer, cur_pos, k, v);

    // Attention
    // softmax(Q_head * K_head^T / sqrt(d_k)) * V   d_k: dim of k
    Tensor attention({n_heads, head_dim}, 0.0f);
    const float scale = 1.0f / sqrt(head_dim);

    for (int h = 0; h < n_heads; ++h) {
        const int kv_head = h / (n_heads / n_kv_heads);
        Tensor scores({cur_pos + 1}, 0.0f);

        for (int p = 0; p <= cur_pos; ++p) {
            const float* cached_k = ctx.kv_cache.KeyPtr(cur_layer, p, kv_head);
            float dot = 0.0f;

            for (int d = 0; d < head_dim; ++d) {
                dot += q[h * head_dim + d] * cached_k[d];
            }
            scores[p] = dot * scale;
        }

        Tensor probs = SoftMax(scores);
        for (int d = 0; d < head_dim; ++d) {
            float value = 0.0f;

            for (int p = 0; p <= cur_pos; ++p) {
                const float* cached_v = ctx.kv_cache.ValuePtr(cur_layer, p, kv_head);
                value += probs[p] * cached_v[d];
            }
            attention[h * head_dim + d] = value;
        }
    }

    // Projection
    Tensor attention_flat = attention.ReshapeChecked({n_heads * head_dim});
    Tensor attention_proj = Linear(attention_flat, lw.wo); // output [dim]

    // Residual Add
    Tensor attention_added = Add(x, attention_proj);

    // ------- FFN ------
    Tensor ffn = RmsNorm(attention_added, lw.ffn_norm, cfg.rms_norm_eps);
    Tensor gate = Linear(ffn, lw.w_gate); // [hidden_dim]
    Tensor up = Linear(ffn, lw.w_up); // [hidden_dim]
    Tensor activated = SwiGLU(gate, up);
    Tensor ffn_output = Linear(activated, lw.w_down); // [dim]

    return Add(attention_added, ffn_output);
}

// ForwardToken: input one token (token_id)
//            -> embedding [dim]
//            -> multi transformer_layer
//            -> logits, output a tensor [vocab_size]

// Embedding
// Forward Layer
//    → RMSNorm
//    → Q/K/V 投影
//    → RoPE
//    → KV Cache
//    → Attention
//    → 残差连接
//    → FFN/SwiGLU
//    → 残差连接
// Logits
//    → Final RMSNorm
//    → LM Head
//    → logits
Tensor ForwardToken(MiniLlamaContext& ctx, const MiniLlamaModel& model, int token_id) {
    ValidateForwardInputs(ctx, model, token_id);

    const ModelConfig& cfg = model.config;
    int n_layers = cfg.n_layers;

    // embedding
    Tensor x = EmbedToken(model, token_id);

    // forward_layer
    for (int n = 0; n < n_layers; ++n) {
        x = ForwardLayer(ctx, model, x, n);
    }

    // logits
        // input: x model [dim]
        // output: logits [vocab_size]
    Tensor normed = RmsNorm(x, model.final_norm, cfg.rms_norm_eps);
    Tensor logits = Linear(normed, model.lm_head);

    return logits.ReshapeChecked({cfg.vocab_size});
}

}
