#pragma once

#include <string>
#include <vector>

#include "mini_llama/sampler.h"

namespace mini_llama {

struct ChatMessage {
  std::string role;  // "system", "user", "assistant"
  std::string content;
};

struct ChatSession {
public:
    // Add a message to the history.
    void AddMessage(const std::string& role, const std::string& content);

    // Clear all messages in the current chat session.
    void Clear();

    const std::vector<ChatMessage>& GetMessages() const {
        return messages_;
    }
private:
    std::vector<ChatMessage> messages_;
};
}  // namespace mini_llama
