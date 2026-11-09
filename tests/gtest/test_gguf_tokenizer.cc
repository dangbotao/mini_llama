#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <vector>

#include "mini_llama/gguf.h"
#include "mini_llama/gguf_tokenizer.h"

namespace mini_llama {
namespace {

TEST(GgufTokenizerTest, LoadsFromExistingReaderWhenModelIsAvailable) {
  const std::vector<std::filesystem::path> candidates = {
      "models/chat/Qwen2.5-3B-Instruct-Q8_0.gguf",
      "models/chat/Qwen2.5-3B-Instruct-Q4_0.gguf",
  };

  std::filesystem::path path;
  for (const auto& candidate : candidates) {
    if (std::filesystem::exists(candidate) &&
        std::filesystem::file_size(candidate) >= 1024 * 1024) {
      path = candidate;
      break;
    }
  }
  if (path.empty()) {
    GTEST_SKIP() << "a local Qwen2.5 GGUF model is not available";
  }

  GgufReader reader;
  ASSERT_TRUE(reader.Load(path.string())) << reader.load_error;

  std::unique_ptr<GgufTokenizer> tokenizer = CreateGgufTokenizer(reader);
  ASSERT_NE(tokenizer, nullptr);
  EXPECT_GT(tokenizer->vocab_size(), 0);
  EXPECT_GE(tokenizer->eos_id(), 0);

  const std::string text = "<|im_start|>user\nHello, 世界<|im_end|>";
  const std::vector<int> tokens = tokenizer->Encode(text);
  ASSERT_FALSE(tokens.empty());
  EXPECT_EQ(tokenizer->Decode(tokens), text);
}

}  // namespace
}  // namespace mini_llama
