#include <gtest/gtest.h>

#include <string>

#include "mini_llama/chat.h"
#include "mini_llama/prompt_builder.h"

namespace mini_llama {

TEST(PromptBuilderTest, BuildsPlainMultiTurnPrompt) {
  ChatSession session;
  session.AddMessage("user", "hello");
  session.AddMessage("assistant", "hi");
  session.AddMessage("user", "hello again");

  PromptBuilder builder;
  EXPECT_EQ(builder.Build(session.GetMessages()),
            "User: hello\n"
            "Assistant: hi\n"
            "User: hello again\n"
            "Assistant:");
}

TEST(PromptBuilderTest, BuildsQwen2GenerationPrompt) {
  ChatSession session;
  session.AddMessage("user", "hello");

  PromptBuilder builder;
  builder.SetChatTemplate("qwen2");
  EXPECT_EQ(builder.Build(session.GetMessages()),
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "hello<|im_end|>\n"
            "<|im_start|>assistant\n");
}

}  // namespace mini_llama
