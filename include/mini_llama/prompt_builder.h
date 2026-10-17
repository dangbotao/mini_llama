#pragma once

#include <string>
#include <vector>

#include "mini_llama/chat.h"

namespace mini_llama {

// Builds a text prompt from a chat message history.
// Supports two modes:
//   1. Plain text (default): System:/User:/Assistant: format
//   2. Chat template: applies a model-specific template (currently Qwen2)
//
// The trailing assistant prefix tells the model to start generating.
class PromptBuilder {
 public:
  PromptBuilder() = default;

  // Set a chat template. If empty, uses plain text mode.
  void SetChatTemplate(const std::string& template_str);

  // Build prompt from messages.
  std::string Build(const std::vector<ChatMessage>& messages) const;

 private:
  std::string chat_template_;

  std::string BuildPlain(const std::vector<ChatMessage>& messages) const;
  std::string BuildQwen2(const std::vector<ChatMessage>& messages) const;
};

}  // namespace mini_llama
