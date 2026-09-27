#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <map>

namespace mini_llama {

std::string UnicodePointToUtf8(char32_t cp);
std::vector<std::string> BuildByteToUtf8Table();

class BpeTokenizer {
private:
    std::vector<std::string> b2u_; // byte -> utf8
    std::unordered_map<std::string, uint8_t> u2b_; //

    std::unordered_map<std::string, int> vocab_; // token_to_id_
    std::vector<std::string> id_to_token_;
    std::vector<std::pair<std::string, int>> special_tokens_; // 由长到短，遍历时优先匹配最长的

    std::map<std::pair<std::string, std::string>, int> merge_ranks_; // 这里为什么用map

    int bos_id_ = -1;
    int eos_id_ = -1;
    int unk_id_ = -1;

    void BuildByteMappings();
    void EncodeNormalSegment(const std::string& segment, std::vector<int>& result);
public:
    int bos_id() { return bos_id_; }
    int eos_id() { return eos_id_; }
    int unk_id() { return unk_id_; }

    bool Load(const std::string& vocab_path,
                        const std::string& merges_path,
                        const std::string& special_path);

    // std::unordered_map<std::string, int> GetVocab_() { return vocab_; }
    // std::vector<std::pair<std::string, int>> GetSpecialTokens()
    //             { return special_tokens_; }

    int vocab_size() const {
        return static_cast<int>(id_to_token_.size());}

    // text -> token_id
    std::vector<int> Encode(const std::string& text);
    // token_id -> text
    std::string Decode(const std::vector<int>& token_ids);

};

std::unique_ptr<BpeTokenizer> CreateBpeTokenizer(
    const std::string& vocab_path, const std::string& merges_path,
    const std::string& special_path);
}  // namespace mini_llama
