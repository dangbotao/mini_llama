#include <gtest/gtest.h>

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

TEST(BatchTest, BuildHelpersCreateExpectedPositions) {
  const MiniBatch single = MiniBatch::BuildSingleTokenBatch(2, 3);
  EXPECT_EQ(single.tokens, std::vector<int>({2}));
  EXPECT_EQ(single.positions, std::vector<int>({3}));

  const MiniBatch multiple =
      MiniBatch::BuildMultiTokensBatch({0, 1, 2}, 1);
  EXPECT_EQ(multiple.tokens, std::vector<int>({0, 1, 2}));
  EXPECT_EQ(multiple.positions, std::vector<int>({1, 2, 3}));
}

TEST(BatchTest, ForwardBatchMatchesSequentialForward) {
  const MiniLlamaModel model = MakeTestModel();
  const MiniBatch batch = MiniBatch::BuildMultiTokensBatch({1, 2});

  MiniLlamaContext batch_ctx(&model);
  const Tensor actual = ForwardBatch(batch_ctx, model, batch);

  MiniLlamaContext sequential_ctx(&model);
  sequential_ctx.pos = 0;
  ForwardToken(sequential_ctx, model, 1);
  sequential_ctx.pos = 1;
  const Tensor expected = ForwardToken(sequential_ctx, model, 2);

  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], 1e-5f);
  }
  EXPECT_EQ(batch_ctx.pos, 1);
  EXPECT_EQ(batch_ctx.token_history, std::vector<int>({1, 2}));
}

TEST(BatchTest, RejectsEmptyAndMismatchedBatch) {
  const MiniLlamaModel model = MakeTestModel();
  MiniLlamaContext ctx(&model);

  EXPECT_THROW(ForwardBatch(ctx, model, MiniBatch{}), std::runtime_error);

  MiniBatch mismatched;
  mismatched.tokens = {1, 2};
  mismatched.positions = {0};
  EXPECT_THROW(ForwardBatch(ctx, model, mismatched), std::runtime_error);
  EXPECT_TRUE(ctx.token_history.empty());
}

}  // namespace mini_llama
