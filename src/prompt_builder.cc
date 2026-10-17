  #include "mini_llama/prompt_builder.h"


  namespace mini_llama {

  void PromptBuilder::SetChatTemplate(const std::string& template_str) {
    chat_template_ = template_str;
  }

  // Build prompt from messages.
  std::string PromptBuilder::Build(const std::vector<ChatMessage>& messages) const {
    // chat_template_ 为空, tiny模型走这个分支
    if (chat_template_.empty()) {
        return BuildPlain(messages);
    }

    // Jinja 模板解析- chat_template_ 以 {% 或 {{ 开头
    // 当前 Qwen2 GGUF 自带完整 Jinja 模板，因此实际走这里
    // If the template string starts with {%, it's a full Jinja2 template.
    // Otherwise it's a shorthand like "qwen2" for built-in templates.
    // if (chat_template_.size() >= 2 && chat_template_[0] == '{' &&
    //     (chat_template_[1] == '%' || chat_template_[1] == '{')) {
    //     auto instrs = CompileTemplate(chat_template_);
    //     return ExecuteTemplate(instrs, messages, true);
    // }

    // "qwen2"
    if (chat_template_ == "qwen2") {
        return BuildQwen2(messages);
    }

    return BuildPlain(messages);
}

// for tiny model test
std::string PromptBuilder::BuildPlain(const std::vector<ChatMessage>& messages) const {
    std::string prompt;
    for (const auto& msg : messages) {
        if (msg.role == "system") {
        prompt += "System: " + msg.content + "\n";
        } else if (msg.role == "user") {
        prompt += "User: " + msg.content + "\n";
        } else if (msg.role == "assistant") {
        prompt += "Assistant: " + msg.content + "\n";
        }
    }
    prompt += "Assistant:";
    return prompt;
}

// Qwen2 chat template (built-in fallback):
//   If no system message, inject default system message.
//   Each message: <|im_start|>{role}\n{content}<|im_end|>\n
//   Final assistant prompt: <|im_start|>assistant\n
std::string PromptBuilder::BuildQwen2(const std::vector<ChatMessage>& messages) const {
    std::string prompt;

    // 检测头部是否有system，若无则补一个默认的
    bool has_system =
    !messages.empty() && messages.front().role == "system";

    if (!has_system) {
        prompt += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
    }

    for (const auto& msg : messages) {
        if (msg.role != "system" && msg.role != "user" && msg.role != "assistant") {
            continue;
        }
        prompt += "<|im_start|>" + msg.role + "\n" + msg.content + "<|im_end|>\n";
    }
    prompt += "<|im_start|>assistant\n";
    return prompt;
}

}