#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <vector>

#include "mini_llama/gguf.h"
#include "mini_llama/gguf_tokenizer.h"

namespace mini_llama {
namespace {

TEST(GgufTokenizerTest, LoadsFromExistingReaderWhenModelIsAvailable) {
  const std::filesystem::path path =
      "models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf";
  if (!std::filesystem::exists(path) ||
      std::filesystem::file_size(path) < 1024 * 1024) {
    GTEST_SKIP() << "bundled Qwen2 model is not available";
  }

  GgufReader reader;
  ASSERT_TRUE(reader.Load(path.string())) << reader.load_error;

  std::unique_ptr<GgufTokenizer> tokenizer = CreateGgufTokenizer(reader);
  ASSERT_NE(tokenizer, nullptr);
  EXPECT_GT(tokenizer->vocab_size(), 0);
  EXPECT_GE(tokenizer->eos_id(), 0);

  std::unique_ptr<BpeTokenizer> file_tokenizer = CreateBpeTokenizer(
      "models/chat/vocab.json", "models/chat/merges.txt",
      "models/chat/special_tokens.json");
  ASSERT_NE(file_tokenizer, nullptr);

  const std::string text = "<|im_start|>user\nHello, 世界<|im_end|>";
  const std::vector<int> tokens = tokenizer->Encode(text);
  ASSERT_FALSE(tokens.empty());
  EXPECT_EQ(tokens, file_tokenizer->Encode(text));
  EXPECT_EQ(tokenizer->Decode(tokens), text);
}

}  // namespace
}  // namespace mini_llama
