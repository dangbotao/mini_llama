#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <map>

namespace mini_llama {

// ---------------------------------------------------------------------------
// Tokenizer interface
// ---------------------------------------------------------------------------
class ITokenizer {
 public:
  virtual ~ITokenizer() = default;

  // Encode text to token ids. BOS placement is tokenizer-specific.
  virtual std::vector<int> Encode(const std::string& text) const = 0;

  // Decode a single token id to string.
  virtual std::string DecodeToken(int token) const = 0;

  // Decode a sequence of tokens.
  virtual std::string Decode(const std::vector<int>& tokens) const = 0;

  virtual int vocab_size() const = 0;
  virtual int bos_id() const = 0;
  virtual int eos_id() const = 0;
  virtual int unk_id() const = 0;
};

std::string UnicodePointToUtf8(char32_t cp);
std::vector<std::string> BuildByteToUtf8Table();

// ---------------------------------------------------------------------------
// JSON vocab tokenizer: loads vocabulary from a JSON file.
// Format: [{"id": 0, "content": "<unk>", "special": true}, ...]
// Encode uses greedy character-level fallback (exact match then UNK).
// ---------------------------------------------------------------------------
class JsonVocabTokenizer : public ITokenizer {
 public:
  explicit JsonVocabTokenizer(const std::string& vocab_path);

  std::vector<int> Encode(const std::string& text) const override;
  std::string DecodeToken(int token) const override;
  std::string Decode(const std::vector<int>& tokens) const override;

  int vocab_size() const override { return vocab_size_; }
  int bos_id() const override { return bos_id_; }
  int eos_id() const override { return eos_id_; }
  int unk_id() const override { return unk_id_; }

 private:
  struct VocabEntry {
    int id = 0;
    std::string content;
    bool special = false;
  };

  int vocab_size_ = 0;
  int bos_id_ = 1;
  int eos_id_ = 2;
  int unk_id_ = 0;

  // id -> entry
  std::vector<VocabEntry> id_to_entry_;

  // content -> id (for Encode)
  // Use longest match: store lengths and try longest first.
  std::vector<std::pair<std::string, int>> content_to_id_;
};

// ---------------------------------------------------------------------------
// BPE tokenizer (GPT-2 style).
// Loads vocab.json + merges.txt + special_tokens.json.
// ---------------------------------------------------------------------------
class BpeTokenizer : public ITokenizer {
private:
    std::vector<std::string> b2u_; // byte -> utf8
    std::unordered_map<std::string, uint8_t> u2b_;

    std::unordered_map<std::string, int> vocab_; // token_to_id_
    std::vector<std::string> id_to_token_;
    std::vector<std::pair<std::string, int>> special_tokens_; // 由长到短，遍历时优先匹配最长的

    std::map<std::pair<std::string, std::string>, int> merge_ranks_; // 这里为什么用map

    int bos_id_ = -1;
    int eos_id_ = -1;
    int unk_id_ = -1;

    void BuildByteMappings();
    void EncodeNormalSegment(const std::string& segment,
                             std::vector<int>& result) const;
public:
    int bos_id() const override { return bos_id_; }
    int eos_id() const override { return eos_id_; }
    int unk_id() const override { return unk_id_; }

    bool Load(const std::string& vocab_path,
                        const std::string& merges_path,
                        const std::string& special_path);

    // std::unordered_map<std::string, int> GetVocab_() { return vocab_; }
    // std::vector<std::pair<std::string, int>> GetSpecialTokens()
    //             { return special_tokens_; }

    int vocab_size() const override {
        return static_cast<int>(id_to_token_.size());}

    // text -> token_id
    std::vector<int> Encode(const std::string& text) const override;
    std::string DecodeToken(int token) const override;
    // token_id -> text
    std::string Decode(const std::vector<int>& token_ids) const override;

};

std::unique_ptr<BpeTokenizer> CreateBpeTokenizer(
    const std::string& vocab_path, const std::string& merges_path,
    const std::string& special_path);

std::unique_ptr<ITokenizer> CreateTokenizer(const std::string& vocab_path);

}  // namespace mini_llama
