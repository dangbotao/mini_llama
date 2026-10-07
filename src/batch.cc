#include "mini_llama/batch.h"

#include <utility>

namespace mini_llama {
MiniBatch MiniBatch::BuildSingleTokenBatch(int token, int pos) {
  MiniBatch batch;
  batch.tokens.push_back(token);
  batch.positions.push_back(pos);
  return batch;
}

MiniBatch MiniBatch::BuildMultiTokensBatch(std::vector<int> tokens, int start_pos) {
  MiniBatch batch;
  batch.tokens = std::move(tokens); // avoid copy
  batch.positions.reserve(batch.tokens.size());
  for (std::size_t i = 0; i < batch.tokens.size(); ++i) {
    batch.positions.push_back(start_pos + static_cast<int>(i));
  }
  return batch;
}
}  // namespace mini_llama
