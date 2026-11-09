// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/gguf_tokenizer.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/gguf.h"

namespace mini_llama {

// ---------------------------------------------------------------------------
// GgufTokenizer
// ---------------------------------------------------------------------------

void GgufTokenizer::BuildByteMappings() {
  b2u_ = BuildByteToUtf8Table();
  u2b_.clear();
  for (int b = 0; b < 256; ++b) {
    u2b_[b2u_[b]] = static_cast<uint8_t>(b);
  }
}

bool GgufTokenizer::Load(const GgufReader& reader) {
  vocab_.clear();
  id_to_token_.clear();
  merge_ranks_.clear();
  special_tokens_.clear();
  bos_id_ = -1;
  eos_id_ = -1;
  unk_id_ = -1;

  BuildByteMappings();

  std::string tokenizer_model;
  if (reader.GetMetadataString("tokenizer.ggml.model", tokenizer_model) &&
      tokenizer_model != "gpt2") {
    std::cerr << "Unsupported GGUF tokenizer model: " << tokenizer_model
              << std::endl;
    return false;
  }

  // Read vocab (token pieces)
  std::vector<std::string> tokens;
  if (!reader.GetMetadataStringArray("tokenizer.ggml.tokens", tokens)) {
    std::cerr << "GGUF missing tokenizer.ggml.tokens" << std::endl;
    return false;
  }

  // Read token types (to identify special/control tokens)
  std::vector<int32_t> token_types;
  bool has_token_types =
      reader.GetMetadataIntArray("tokenizer.ggml.token_type", token_types);
  if (has_token_types && token_types.size() != tokens.size()) {
    std::cerr << "GGUF tokenizer token/type count mismatch" << std::endl;
    return false;
  }

  // Build vocab map and id-to-token table
  id_to_token_.resize(tokens.size());
  for (size_t i = 0; i < tokens.size(); ++i) {
    vocab_[tokens[i]] = static_cast<int>(i);
    id_to_token_[i] = tokens[i];
  }

  // Identify special tokens from token_type (3 = control/special)
  if (has_token_types && token_types.size() == tokens.size()) {
    for (size_t i = 0; i < tokens.size(); ++i) {
      if (token_types[i] == 3) {
        special_tokens_.push_back({tokens[i], static_cast<int>(i)});
      }
    }
  }
  std::sort(special_tokens_.begin(), special_tokens_.end(),
            [](const auto& a, const auto& b) {
              return a.first.size() > b.first.size();
            });

  // Read merges
  std::vector<std::string> merges;
  if (!reader.GetMetadataStringArray("tokenizer.ggml.merges", merges)) {
    std::cerr << "GGUF missing tokenizer.ggml.merges" << std::endl;
    return false;
  }
  for (int rank = 0; rank < static_cast<int>(merges.size()); ++rank) {
    const std::string& merge = merges[rank];
    size_t space_pos = merge.find(' ');
    if (space_pos == std::string::npos) {
      continue;
    }
    std::string left = merge.substr(0, space_pos);
    std::string right = merge.substr(space_pos + 1);
    merge_ranks_[{left, right}] = rank;
  }

  // Read BOS / EOS ids
  int64_t bos = -1, eos = -1;
  if (reader.GetMetadataInt("tokenizer.ggml.bos_token_id", bos)) {
    bos_id_ = static_cast<int>(bos);
  }
  if (reader.GetMetadataInt("tokenizer.ggml.eos_token_id", eos)) {
    eos_id_ = static_cast<int>(eos);
  }

  // unk_id is optional in some GGUF tokenizers.
  int64_t unk = -1;
  if (reader.GetMetadataInt("tokenizer.ggml.unknown_token_id", unk)) {
    unk_id_ = static_cast<int>(unk);
  }

  std::cout << "GGUF tokenizer loaded: vocab=" << vocab_.size()
            << " merges=" << merge_ranks_.size()
            << " special=" << special_tokens_.size() << " bos=" << bos_id_
            << " eos=" << eos_id_ << " unk=" << unk_id_ << std::endl;
  return true;
}

// input(byte): E4 | B8 | AD
// b2u_: ä(C3 A4) | ¸(C2 B8) | Ń（C5 83）
// merge(token): ä¸Ń
// token_to_id
// output() : id
std::vector<int> GgufTokenizer::Encode(const std::string& text) const {
  if (text.empty()) {
    return {};
  }

  std::vector<int> result;
  size_t pos = 0;

  while (pos < text.size()) {
    // Match special tokens before normal text, longest first.
    bool is_special = false;
    for (const auto& st : special_tokens_) {
      size_t token_size = st.first.size();
      if (pos + token_size <= text.size() &&
          text.compare(pos, token_size, st.first) == 0) {
        result.push_back(st.second);
        pos += token_size;
        is_special = true;
        break;
      }
    }
    if (is_special) {
      continue;
    }

    // Encode normal text up to the next special token.
    size_t end = text.size();
    for (const auto& st : special_tokens_) {
      size_t found = text.find(st.first, pos);
      if (found != std::string::npos && found < end) {
        end = found;
      }
    }

    std::string segment = text.substr(pos, end - pos);
    EncodeNormalSegment(segment, result);
    pos = end;
  }

  return result;
}

void GgufTokenizer::EncodeNormalSegment(
    const std::string& segment, std::vector<int>& result) const {
  std::vector<std::string> word;
  word.reserve(segment.size());
  for (unsigned char byte : segment) {
    word.push_back(b2u_[byte]);
  }

  while (word.size() > 1) {
    int best_rank = std::numeric_limits<int>::max();
    size_t best_idx = word.size();

    for (size_t i = 0; i + 1 < word.size(); ++i) {
      auto it = merge_ranks_.find({word[i], word[i + 1]});
      if (it != merge_ranks_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_idx = i;
      }
    }

    if (best_idx == word.size()) {
      break;
    }

    word[best_idx] += word[best_idx + 1];
    word.erase(word.begin() + static_cast<std::ptrdiff_t>(best_idx) + 1);
  }

  for (const std::string& piece : word) {
    auto it = vocab_.find(piece);
    if (it != vocab_.end()) {
      result.push_back(it->second);
    } else {
      result.push_back(unk_id_);
    }
  }
}

std::string GgufTokenizer::DecodeToken(int token) const {
  return Decode({token});
}

std::string GgufTokenizer::Decode(const std::vector<int>& tokens) const {
  std::string text;
  for (int id : tokens) {
    if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) {
      throw std::out_of_range("invalid token id");
    }
    text += id_to_token_[id];
  }

  std::string result;
  size_t pos = 0;
  while (pos < text.size()) {
    unsigned char first_byte = text[pos];
    size_t len = 1;
    if ((first_byte & 0x80) == 0) {
      len = 1;
    } else if ((first_byte & 0xE0) == 0xC0) {
      len = 2;
    } else if ((first_byte & 0xF0) == 0xE0) {
      len = 3;
    } else if ((first_byte & 0xF8) == 0xF0) {
      len = 4;
    } else {
      ++pos;
      continue;
    }

    if (pos + len > text.size()) {
      break;
    }

    std::string unicode = text.substr(pos, len);
    pos += len;
    auto it = u2b_.find(unicode);
    if (it != u2b_.end()) {
      result += static_cast<char>(it->second);
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<GgufTokenizer> CreateGgufTokenizer(
    const GgufReader& reader) {
  auto tok = std::make_unique<GgufTokenizer>();
  if (tok->Load(reader)) {
    return tok;
  }
  return nullptr;
}

}  // namespace mini_llama
