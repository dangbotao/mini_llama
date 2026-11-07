#include "mini_llama/prompt_builder.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace mini_llama {
namespace {

enum class TemplateKind {
  kQwen2,
  kQwen25,
};

struct TemplateProgram {
  TemplateKind kind;
};

std::string RenderQwen2(const std::vector<ChatMessage>& messages,
                        bool add_generation_prompt,
                        const std::string& default_system_message) {
  std::string prompt;
  const bool has_system =
      !messages.empty() && messages.front().role == "system";

  if (!has_system) {
    prompt += "<|im_start|>system\n" + default_system_message +
              "<|im_end|>\n";
  }

  for (const ChatMessage& message : messages) {
    if (message.role != "system" && message.role != "user" &&
        message.role != "assistant") {
      continue;
    }
    prompt += "<|im_start|>" + message.role + "\n" + message.content +
              "<|im_end|>\n";
  }

  if (add_generation_prompt) {
    prompt += "<|im_start|>assistant\n";
  }
  return prompt;
}

// Compile the Jinja subset used by the bundled Qwen2 GGUF template. A full
// general-purpose Jinja interpreter is intentionally out of scope; rejecting
// unknown templates is safer than silently constructing the wrong prompt.
TemplateProgram CompileTemplate(const std::string& template_text) {
  const bool iterates_messages =
      template_text.find("for message in messages") != std::string::npos;
  const bool reads_role =
      template_text.find("message['role']") != std::string::npos ||
      template_text.find("message[\"role\"]") != std::string::npos ||
      template_text.find("message.role") != std::string::npos;
  const bool reads_content =
      template_text.find("message['content']") != std::string::npos ||
      template_text.find("message[\"content\"]") != std::string::npos ||
      template_text.find("message.content") != std::string::npos;
  const bool has_chatml_tokens =
      template_text.find("<|im_start|>") != std::string::npos &&
      template_text.find("<|im_end|>") != std::string::npos;
  const bool supports_generation_prompt =
      template_text.find("add_generation_prompt") != std::string::npos;

  if (iterates_messages && reads_role && reads_content && has_chatml_tokens &&
      supports_generation_prompt) {
    if (template_text.find("You are Qwen, created by Alibaba Cloud.") !=
        std::string::npos) {
      return {TemplateKind::kQwen25};
    }
    return {TemplateKind::kQwen2};
  }

  throw std::runtime_error(
      "PromptBuilder: unsupported GGUF Jinja chat template");
}

std::string ExecuteTemplate(const TemplateProgram& program,
                            const std::vector<ChatMessage>& messages,
                            bool add_generation_prompt) {
  switch (program.kind) {
    case TemplateKind::kQwen2:
      return RenderQwen2(messages, add_generation_prompt,
                         "You are a helpful assistant.");
    case TemplateKind::kQwen25:
      return RenderQwen2(
          messages, add_generation_prompt,
          "You are Qwen, created by Alibaba Cloud. You are a helpful "
          "assistant.");
  }
  throw std::runtime_error("PromptBuilder: unknown compiled template");
}

}  // namespace

void PromptBuilder::SetChatTemplate(const std::string& template_str) {
  chat_template_ = template_str;
}

std::string PromptBuilder::Build(
    const std::vector<ChatMessage>& messages) const {
  if (chat_template_.empty()) {
    return BuildPlain(messages);
  }

  // GGUF stores full Jinja templates. Compile the supported subset and
  // execute it with add_generation_prompt=true for chat generation.
  if (chat_template_.size() >= 2 && chat_template_[0] == '{' &&
      (chat_template_[1] == '%' || chat_template_[1] == '{')) {
    const TemplateProgram program = CompileTemplate(chat_template_);
    return ExecuteTemplate(program, messages, true);
  }

  if (chat_template_ == "qwen2") {
    return BuildQwen2(messages);
  }

  return BuildPlain(messages);
}

std::string PromptBuilder::BuildPlain(
    const std::vector<ChatMessage>& messages) const {
  std::string prompt;
  for (const ChatMessage& message : messages) {
    if (message.role == "system") {
      prompt += "System: " + message.content + "\n";
    } else if (message.role == "user") {
      prompt += "User: " + message.content + "\n";
    } else if (message.role == "assistant") {
      prompt += "Assistant: " + message.content + "\n";
    }
  }
  prompt += "Assistant:";
  return prompt;
}

std::string PromptBuilder::BuildQwen2(
    const std::vector<ChatMessage>& messages) const {
  return RenderQwen2(messages, true, "You are a helpful assistant.");
}

}  // namespace mini_llama
