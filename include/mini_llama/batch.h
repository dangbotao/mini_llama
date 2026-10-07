#pragma once

#include <vector>

namespace mini_llama {

// MiniBatch mirrors llama_batch: a collection of tokens with positions.
// For this educational implementation, each token maps to one position.
struct MiniBatch {
  std::vector<int> tokens;     // token ids
  std::vector<int> positions;  // position for each token

  int num_tokens() const { return static_cast<int>(tokens.size()); }

  static MiniBatch BuildSingleTokenBatch(int token, int pos);
  static MiniBatch BuildMultiTokensBatch(std::vector<int> tokens, int start_pos = 0);
};

}  // namespace mini_llama
