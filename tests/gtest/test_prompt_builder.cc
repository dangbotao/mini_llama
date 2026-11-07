#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <stdexcept>
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

TEST(PromptBuilderTest, ExecutesBundledQwen2JinjaTemplate) {
  std::ifstream input("models/chat/chat_template.txt");
  ASSERT_TRUE(input.is_open());
  std::stringstream buffer;
  buffer << input.rdbuf();

  ChatSession session;
  session.AddMessage("user", "hello");

  PromptBuilder builder;
  builder.SetChatTemplate(buffer.str());
  EXPECT_EQ(builder.Build(session.GetMessages()),
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "hello<|im_end|>\n"
            "<|im_start|>assistant\n");
}

TEST(PromptBuilderTest, ExecutesQwen25DotSyntaxJinjaTemplate) {
  const std::string qwen25_template =
      R"({%- for message in messages %}{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}You are Qwen, created by Alibaba Cloud. You are a helpful assistant.)";

  PromptBuilder builder;
  builder.SetChatTemplate(qwen25_template);

  EXPECT_EQ(builder.Build({{"user", "hello"}}),
            "<|im_start|>system\n"
            "You are Qwen, created by Alibaba Cloud. You are a helpful "
            "assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "hello<|im_end|>\n"
            "<|im_start|>assistant\n");
}

TEST(PromptBuilderTest, RejectsUnsupportedJinjaTemplate) {
  PromptBuilder builder;
  builder.SetChatTemplate("{{ unsupported_expression }}");
  EXPECT_THROW(builder.Build({{"user", "hello"}}), std::runtime_error);
}

}  // namespace mini_llama
