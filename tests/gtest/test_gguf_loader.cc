#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "mini_llama/gguf_loader.h"

namespace mini_llama {
namespace {

TEST(GgufModelLoaderTest, LoadsTinyF32Model) {
  GgufReader reader;
  ASSERT_TRUE(reader.Load("models/tiny/model.gguf")) << reader.load_error;

  GgufModelLoader loader(reader);
  const MiniLlamaModel model = loader.Load();

  ASSERT_TRUE(model.loaded) << model.load_error;
  EXPECT_TRUE(model.load_error.empty());
  EXPECT_EQ(model.config.vocab_size, 128);
  EXPECT_EQ(model.config.dim, 32);
  EXPECT_EQ(model.config.hidden_dim, 86);
  EXPECT_EQ(model.config.n_layers, 2);
  ASSERT_EQ(model.layers.size(), 2U);

  EXPECT_EQ(model.token_embedding.shape, (std::vector<int>{128, 32}));
  EXPECT_EQ(model.layers[0].wq.type, QuantType::kF32);
  EXPECT_EQ(model.layers[0].wq.shape, (std::vector<int>{32, 32}));
  EXPECT_TRUE(model.layers[0].bq.data.empty());
  EXPECT_EQ(model.lm_head.shape, (std::vector<int>{128, 32}));
}

TEST(GgufModelLoaderTest, ReportsMissingFile) {
  GgufReader reader;
  ASSERT_FALSE(reader.Load("models/tiny/does-not-exist.gguf"));

  GgufModelLoader loader(reader);
  const MiniLlamaModel model = loader.Load();

  EXPECT_FALSE(model.loaded);
  EXPECT_NE(model.load_error.find("failed to open"), std::string::npos);
}

}  // namespace
}  // namespace mini_llama
