#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "mini_llama/forward.h"

namespace mini_llama {

namespace {

Tensor MakeIdentity(int size) {
  Tensor matrix({size, size}, 0.0f);
  for (int i = 0; i < size; ++i) {
    matrix.At2(i, i) = 1.0f;
  }
  return matrix;
}

// make test model

// - 词表大小为 3。
// - Embedding 维度为 4。
// - 只有 1 个 Transformer 层。
// - 只有 1 个 Attention Head。
// - FFN 隐藏维度也是 4。
// - Norm 权重全部设为 1。
// - Q/K/V、Attention 输出和 FFN 矩阵都使用单位矩阵。
// - lm_head 使用固定数据，将 4 维隐藏状态转换为 3 个 logits。
// 使用单位矩阵的好处是数据经过各层后不会轻易变成全零，方便确认完整链路和 KV Cache 确实执行

MiniLlamaModel MakeTestModel() {
  MiniLlamaModel model;
  model.config.vocab_size = 3;
  model.config.dim = 4;
  model.config.hidden_dim = 4;
  model.config.n_layers = 1;
  model.config.n_heads = 1;
  model.config.n_kv_heads = 1;
  model.config.head_dim = 4;
  model.config.max_seq_len = 4;
  model.token_embedding = Tensor({model.config.vocab_size, model.config.dim});

  const std::vector<float> embeddings = {
      0.1f, 0.2f, 0.3f, 0.4f,
      1.1f, 1.2f, 1.3f, 1.4f,
      2.1f, 2.2f, 2.3f, 2.4f,
  };
  model.token_embedding.data = embeddings;

  LayerWeights layer;
  layer.attention_norm = Tensor({model.config.dim}, 1.0f);
  layer.wq = MakeIdentity(model.config.dim);
  layer.wk = MakeIdentity(model.config.dim);
  layer.wv = MakeIdentity(model.config.dim);
  layer.wo = MakeIdentity(model.config.dim);
  layer.ffn_norm = Tensor({model.config.dim}, 1.0f);
  layer.w_gate = MakeIdentity(model.config.hidden_dim);
  layer.w_up = MakeIdentity(model.config.hidden_dim);
  layer.w_down = MakeIdentity(model.config.dim);
  model.layers.push_back(layer);

  model.final_norm = Tensor({model.config.dim}, 1.0f);
  model.lm_head = Tensor({model.config.vocab_size, model.config.dim});
  model.lm_head.data = embeddings;
  return model;
}

}  // namespace

TEST(ForwardTest, EmbedTokenReturnsMatchingEmbeddingRow) {
  const MiniLlamaModel model = MakeTestModel();

  const Tensor embedding = EmbedToken(model, 1);

  EXPECT_EQ(embedding.shape, std::vector<int>({model.config.dim}));
  ASSERT_EQ(embedding.size(), 4);
  EXPECT_FLOAT_EQ(embedding[0], 1.1f);
  EXPECT_FLOAT_EQ(embedding[1], 1.2f);
  EXPECT_FLOAT_EQ(embedding[2], 1.3f);
  EXPECT_FLOAT_EQ(embedding[3], 1.4f);

  EXPECT_THROW(EmbedToken(model, model.config.vocab_size), std::runtime_error);
}

TEST(ForwardTest, ForwardTokenRunsCompletePath) {
  const MiniLlamaModel model = MakeTestModel();
  MiniLlamaContext ctx(&model);

  const Tensor logits = ForwardToken(ctx, model, 1);

  EXPECT_EQ(logits.shape, std::vector<int>({model.config.vocab_size}));
  const std::vector<float> expected_logits = {
      1.0451331f, 5.0181026f, 8.9910717f};
  ASSERT_EQ(logits.size(), expected_logits.size());
  for (size_t i = 0; i < logits.size(); ++i) {
    EXPECT_TRUE(std::isfinite(logits[i]));
    EXPECT_NEAR(logits[i], expected_logits[i], 1e-5f);
  }

  bool key_written = false;
  bool value_written = false;
  const float* key = ctx.kv_cache.KeyPtr(0, 0, 0);
  const float* value = ctx.kv_cache.ValuePtr(0, 0, 0);
  for (int d = 0; d < model.config.head_dim; ++d) {
    key_written |= std::abs(key[d]) > 0.0f;
    value_written |= std::abs(value[d]) > 0.0f;
  }
  EXPECT_TRUE(key_written);
  EXPECT_TRUE(value_written);
}

TEST(ForwardTest, TwoTokensPreserveAndExtendKvCache) {
  const MiniLlamaModel model = MakeTestModel();
  MiniLlamaContext ctx(&model);

  ctx.pos = 0;
  ForwardToken(ctx, model, 1);
  const float* key0 = ctx.kv_cache.KeyPtr(0, 0, 0);
  const float* value0 = ctx.kv_cache.ValuePtr(0, 0, 0);
  const std::vector<float> saved_key0(key0, key0 + model.config.head_dim);
  const std::vector<float> saved_value0(value0,
                                        value0 + model.config.head_dim);

  ctx.pos = 1;
  ForwardToken(ctx, model, 2);

  key0 = ctx.kv_cache.KeyPtr(0, 0, 0);
  value0 = ctx.kv_cache.ValuePtr(0, 0, 0);
  const float* key1 = ctx.kv_cache.KeyPtr(0, 1, 0);
  const float* value1 = ctx.kv_cache.ValuePtr(0, 1, 0);
  bool key1_written = false;
  bool value1_written = false;
  for (int d = 0; d < model.config.head_dim; ++d) {
    EXPECT_FLOAT_EQ(key0[d], saved_key0[d]);
    EXPECT_FLOAT_EQ(value0[d], saved_value0[d]);
    key1_written |= std::abs(key1[d]) > 0.0f;
    value1_written |= std::abs(value1[d]) > 0.0f;
  }
  EXPECT_TRUE(key1_written);
  EXPECT_TRUE(value1_written);
}

TEST(ForwardTest, RejectsInvalidTokenAndPosition) {
  const MiniLlamaModel model = MakeTestModel();
  MiniLlamaContext ctx(&model);

  EXPECT_THROW(ForwardToken(ctx, model, -1), std::out_of_range);
  EXPECT_THROW(ForwardToken(ctx, model, model.config.vocab_size),
               std::out_of_range);

  ctx.pos = -1;
  EXPECT_THROW(ForwardToken(ctx, model, 1), std::out_of_range);
  ctx.pos = model.config.max_seq_len;
  EXPECT_THROW(ForwardToken(ctx, model, 1), std::out_of_range);
}

TEST(ForwardTest, RejectsInvalidModelBeforeWritingKvCache) {
  MiniLlamaModel model = MakeTestModel();
  MiniLlamaContext ctx(&model);
  model.layers[0].wq = Tensor({3, model.config.dim}, 1.0f);

  EXPECT_THROW(ForwardToken(ctx, model, 1), std::runtime_error);
  for (float value : ctx.kv_cache.keys.data) {
    EXPECT_FLOAT_EQ(value, 0.0f);
  }
  for (float value : ctx.kv_cache.values.data) {
    EXPECT_FLOAT_EQ(value, 0.0f);
  }
}

}  // namespace mini_llama
