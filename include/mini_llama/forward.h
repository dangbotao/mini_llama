#pragma once

#include "mini_llama/tensor.h"
#include "mini_llama/model.h"
#include "mini_llama/context.h"
#include "mini_llama/batch.h"

namespace mini_llama {

// token_id -> embedding -> dim
// embedding: [vocab_size, dim]
Tensor EmbedToken(const MiniLlamaModel& model, int token_id);

Tensor ForwardToken(MiniLlamaContext& ctx, const MiniLlamaModel& model, int token_id);

Tensor ForwardBatch(MiniLlamaContext& ctx, const MiniLlamaModel& model, const MiniBatch& batch);
}