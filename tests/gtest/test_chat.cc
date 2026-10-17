#include <gtest/gtest.h>

#include <string>

#include "mini_llama/chat.h"

namespace mini_llama {

TEST(ChatSessionTest, AddsAndClearsMessages) {
  ChatSession session;
  session.AddMessage("user", "hello");
  session.AddMessage("assistant", "hi");

  ASSERT_EQ(session.GetMessages().size(), 2U);
  EXPECT_EQ(session.GetMessages()[0].role, "user");
  EXPECT_EQ(session.GetMessages()[0].content, "hello");

  session.Clear();
  EXPECT_TRUE(session.GetMessages().empty());
}

}  // namespace mini_llama
