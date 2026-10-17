#include "mini_llama/chat.h"

#include <string>
#include <vector>

namespace mini_llama {

void ChatSession::AddMessage(const std::string& role,
                             const std::string& content) {
  messages_.push_back({role, content});
}

void ChatSession::Clear() { messages_.clear(); }


}  // namespace mini_llama
