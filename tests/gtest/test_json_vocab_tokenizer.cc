#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "mini_llama/tokenizer.h"

namespace mini_llama {

TEST(JsonVocabTokenizerTest, EncodesAndDecodesTinyVocabulary) {
  std::unique_ptr<ITokenizer> tokenizer =
      CreateTokenizer("models/tiny/vocab.json");
  ASSERT_NE(tokenizer, nullptr);

  EXPECT_EQ(tokenizer->bos_id(), 1);
  EXPECT_EQ(tokenizer->eos_id(), 2);
  EXPECT_EQ(tokenizer->unk_id(), 0);

  const std::vector<int> tokens = tokenizer->Encode("hi!");
  EXPECT_EQ(tokens, (std::vector<int>{1, 104, 105, 33}));
  EXPECT_EQ(tokenizer->Decode(tokens), "<bos>hi!");
}

TEST(JsonVocabTokenizerTest, UsesUnknownTokenForMissingContent) {
  JsonVocabTokenizer tokenizer("models/tiny/vocab.json");
  EXPECT_EQ(tokenizer.Encode("?"), (std::vector<int>{1, 0}));
  EXPECT_EQ(tokenizer.DecodeToken(0), "<unk>");
}

}  // namespace mini_llama
