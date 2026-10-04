#include "mini_llama/forward.h"
#include "mini_llama/context.h"
#include "mini_llama/operations.h"

#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace mini_llama {

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