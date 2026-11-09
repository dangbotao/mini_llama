#include "mini_llama/tokenizer.h"

#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cerrno>
#include <sstream>
#include <fstream>
#include <memory>
#include <cstring>

namespace mini_llama {

// ---------------------------------------------------------------------------
// GPT-2 bytes_to_unicode mapping
// ---------------------------------------------------------------------------

// 将数值形式的unicode码点，转换成实际的utf-8字节串
std::string UnicodePointToUtf8(char32_t cp) {
// static std::string UnicodePointToUtf8(char32_t cp) {
    // 1byte 0xxxxxxx 0~0x7F 7个有效位
    // 2byte 110xxxxx 10xxxxxx 0x80~0x7FF 11个有效位
    // 3byte 1110xxxx 10xxxxxx 10xxxxxx 0x800~0xFFFF
    // 4byte 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx 0x10000~0x10FFFF(10FFFF，unicode最大值)
    if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
        throw std::runtime_error("UnicodePointToUft8: input value exceed");
    }

    std::string result;
    if (cp <= 0x7F) {
        result += static_cast<char>(cp);
    } else if (cp <= 0x7FF) {
        result += static_cast<char>(0xC0 | (cp >> 6));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp <= 0xFFFF) {
        result += 0xE0 | (cp >> 12);
        result += 0x80 | (cp >> 6 & 0x3F);
        result += 0x80 | (cp & 0x3F);
    } else {
        result += 0xF0 | (cp >> 18);
        result += 0x80 | (cp >> 12 & 0x3F);
        result += 0x80 | (cp >> 6 & 0x3F);
        result += 0x80 | (cp & 0x3F);
    }
    return result;
}

std::vector<std::string> BuildByteToUtf8Table() {
    /*将每个 byte（0～255）映射为一个可打印、可逆的 Unicode 字符，
      并以 UTF-8 字符串保存。
    其中：
    - 0x21～0x7E、0xA1～0xAC、0xAE～0xFF 直接映射到同值 Unicode 码点。
    - 其余 68 个不便直接表示的 byte 依次映射到 U+0100～U+0143。
    - 最后把 Unicode 码点编码成 UTF-8字符（string)。*/

    // '!' -> '~' 0x21-0x7E
    // 0xA1-0xAC
    // 0xAE-0xFF
    std::vector<int> m;
    for (int i = '!'; i <= '~'; ++i) {
        m.push_back(i);
    }
    for (int i = 0xA1; i <= 0xAC; ++i) {
        m.push_back(i);
    }
    for (int i = 0xAE; i <= 0xFF; ++i) {
        m.push_back(i);
    }

    std::vector<int> n = m;
    int k = 0;
    for (int i = 0; i < 256; ++i) {
        if (std::find(m.begin(), m.end(), i) == m.end()) {
            m.push_back(i);
            n.push_back(256 + k);
            ++k;
        }
    }

    std::vector<std::string> map(256);
    for (int i = 0; i < 256; ++i) {
        map[m[i]] = UnicodePointToUtf8(static_cast<char32_t>(n[i]));
    }

    return map;
}

// ---------------------------------------------------------------------------
// Simple JSON number parser helper
// ---------------------------------------------------------------------------

// 解析整数
static bool ParseJsonInt(const std::string& s, size_t& pos, int& out) {
  // s：完整 JSON 文本。
  // pos：当前解析位置，会被函数向后移动。
  // out：保存解析出的整数。
  // 返回值：成功返回 true，失败返回 false。

  // 跳过空白
  while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) {
    ++pos;
  }
  if (pos >= s.size()) {
    return false;
  }

  size_t start = pos; // 记录数字起点
  if (s[pos] == '-') {
    ++pos;
  }
  // 让pos指向数字末尾
  while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) {
    ++pos;
  }

  // 数据长度为0则解析失败
  if (start == pos) {
    return false;
  }

  // 截取数字并转为整数
  try {
    out = std::stoi(s.substr(start, pos - start));
    return true;
  } catch (...) {
    return false;
  }
}

// 解析字符串
static bool ParseJsonString(const std::string& s, size_t& pos,
                            std::string& out) {
  // 跳过空白并检查双引号
  while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) {
    ++pos;
  }
  if (pos >= s.size() || s[pos] != '"') {
    return false;
  }

  ++pos;  // skip opening quote
  out.clear();

  // 循环读取字符
  while (pos < s.size() && s[pos] != '"') {
    // 遇到反斜杠，处理转义字符
    if (s[pos] == '\\' && pos + 1 < s.size()) {
      char esc = s[pos + 1];
      switch (esc) {
        case '"':
          out += '"';
          break;
        case '\\':
          out += '\\';
          break;
        case '/':
          out += '/';
          break;
        case 'b':
          out += '\b';
          break;
        case 'f':
          out += '\f';
          break;
        case 'n':
          out += '\n';
          break;
        case 'r':
          out += '\r';
          break;
        case 't':
          out += '\t';
          break;
        case 'u': { // 16进制，对应输入 "\u00E4\u00B8\u0143" ，解析为 C3 A4 C2 B8 C5 83
          if (pos + 5 < s.size()) { // 处理4位十六进制，例如"\u4E2D"
            std::string hex = s.substr(pos + 2, 4); // 取出子串
            try {
              // 转换为Unicode码点，例如U+4E2D
              char32_t cp = static_cast<char32_t>(std::stoul(hex, nullptr, 16));
              // 编码为UTF-8
              out += UnicodePointToUtf8(cp); // 例如：最终out中保存"中" 的 UTF-8 字节
            } catch (...) {
              out += '?';
            }
            pos += 4;
          }
          break;
        }
        default:
          out += esc; // 对应输入 "ä¸Ń"，直接逐字节读取 C3 A4 C2 B8 C5 83
          break;
      }
      pos += 2;
    } else { // 将普通字符直接添加到 out
      out += s[pos];
      ++pos;
    }
  }
  if (pos < s.size() && s[pos] == '"') {
    ++pos;
  }
  return true;
}

// ---------------------------------------------------------------------------
// BpeTokenizer
// ---------------------------------------------------------------------------

void BpeTokenizer::BuildByteMappings() {
    b2u_ = BuildByteToUtf8Table();
    for (int i = 0; i < 256; ++i) {
        u2b_[b2u_[i]] = static_cast<uint8_t>(i);
    }
}

bool BpeTokenizer::Load(const std::string& vocab_path,
                        const std::string& merges_path,
                        const std::string& special_path) {
  BuildByteMappings();

  // Load vocab.json: { "token": id, ... } -> vocab_、id_to_token_、special_tokens_
  {
    // 打开文件，把文件内容读取到content中
    std::ifstream f(vocab_path);
    if (!f.is_open()) {
      std::cerr << "Failed to open vocab: " << vocab_path << std::endl;
      return false;
    }
    std::stringstream buf;
    buf << f.rdbuf();
    std::string content = buf.str();

    // -------------- 构建vocab_(token_to_id_) --------------

    // 跳过开头空白（空格、换行、Tab、回车）和‘{’
    size_t pos = 0;
    while (pos < content.size() &&
           std::isspace(static_cast<unsigned char>(content[pos]))) {
      ++pos;
    }
    if (pos < content.size() && content[pos] == '{') {
      ++pos;
    }

    // 循环解析词表项 "token": id
    while (pos < content.size()) {
      // 跳过词条前空白
      while (pos < content.size() &&
             std::isspace(static_cast<unsigned char>(content[pos]))) {
        ++pos;
      }

      // 遇到'}'结束
      if (pos < content.size() && content[pos] == '}') {
        break;
      }

      // 解析双引号中的字符串
      std::string key;
      if (!ParseJsonString(content, pos, key)) {
        break;
      }

      // 跳过token后的空白
      while (pos < content.size() &&
             std::isspace(static_cast<unsigned char>(content[pos]))) {
        ++pos;
      }

      // 检查':'（合法词表项必须是"hello": 7，如果token后不是冒号，停止解析）
      if (pos >= content.size() || content[pos] != ':') {
        break;
      }
      ++pos;

      // 解析token ID
      int val = 0;
      if (!ParseJsonInt(content, pos, val)) {
        break;
      }

      // 存入词表（token字符串 → token ID）
      vocab_[key] = val;

      // 跳过空白和逗号，随后进入下一轮
      while (pos < content.size() &&
             std::isspace(static_cast<unsigned char>(content[pos]))) {
        ++pos;
      }
      if (pos < content.size() && content[pos] == ',') {
        ++pos;
      }
    }

    if (vocab_.empty()) {
      std::cerr << "Vocab is empty" << std::endl;
      return false;
    }

    // -------------- 构造id_to_token_ ---------------
    // 找到最大token ID
    int max_id = 0;
    for (const auto& kv : vocab_) {
      if (kv.second > max_id) {
        max_id = kv.second;
      }
    }

    id_to_token_.resize(max_id + 1);
    for (const auto& kv : vocab_) {
      id_to_token_[kv.second] = kv.first;
    }

    // -------------- 提取special_token ---------------
    // Extract special tokens (e.g. <|im_start|>) from vocab for longest-match
    // Encode.
    for (const auto& kv : vocab_) {
      if (kv.first.size() >= 4 && kv.first.front() == '<' &&
          kv.first[1] == '|' && kv.first[kv.first.size() - 2] == '|' &&
          kv.first.back() == '>') { // 特殊token的形式：<|...|>
        special_tokens_.push_back({kv.first, kv.second});
      }
    }
    // 降序排列，实现优先最长匹配
    std::sort(special_tokens_.begin(), special_tokens_.end(),
              [](const auto& a, const auto& b) {
                return a.first.size() > b.first.size();
              });
  }

    // Load merges.txt: "token1 token2" per line -> merge_ranks_
  {
    std::ifstream f(merges_path);
    if (!f.is_open()) {
      std::cerr << "Failed to open merges: " << merges_path << std::endl;
      return false;
    }
    std::string line;
    int rank = 0;
    while (std::getline(f, line)) {
      std::stringstream ss(line);
      std::string a, b;
      if (ss >> a >> b) {
        merge_ranks_[{a, b}] = rank++;
      }
    }
  }

  // Load special_tokens.json -> bos_id_、eos_id_、unk_id_
  {
    std::ifstream f(special_path);
    if (f.is_open()) {
      std::stringstream buf;
      buf << f.rdbuf();
      std::string content = buf.str();
      auto find_int = [&](const std::string& key) -> int {
        size_t p = content.find('"' + key + '"');
        if (p == std::string::npos) {
          return -1;
        }
        p = content.find(':', p);
        if (p == std::string::npos) {
          return -1;
        }
        size_t start = p + 1;
        while (start < content.size() &&
               std::isspace(static_cast<unsigned char>(content[start]))) {
          ++start;
        }
        size_t end = start;
        while (end < content.size() &&
               (std::isdigit(static_cast<unsigned char>(content[end])) ||
                content[end] == '-')) {
          ++end;
        }
        if (start == end) {
          return -1;
        }
        try {
          return std::stoi(content.substr(start, end - start));
        } catch (...) {
          return -1;
        }
      };
      bos_id_ = find_int("bos_id");
      eos_id_ = find_int("eos_id");
      unk_id_ = find_int("unk_id");
    }
}

  return true;
}

// input(byte): E4 | B8 | AD
// b2u_: ä(C3 A4) | ¸(C2 B8) | Ń（C5 83）
// merge(token): ä¸Ń
// token_to_id
// output() : id

// 这段代码的作用是把输入文本分成“特殊 token”和“普通文本片段”两类，再分别编码：
// 1. 如果输入为空，直接返回空的 token ID 数组。
// 2. 创建 result 保存编码结果，pos 表示当前处理到的字节位置。
// 3. 在循环中，先遍历 special_tokens_，检查当前位置是否以某个特殊 token 开头。
// 4. 如果匹配，例如 <|im_start|>，就直接将其 token ID 加入 result，并让 pos 跳过整个特殊 token；它不会参与普通 BPE。
// 5. 如果当前位置不是特殊 token，就查找当前位置之后最早出现的特殊 token，将它的位置记为 end。
// 6. 截取 [pos, end) 之间的普通文本，并交给 EncodeNormalSegment() 完成 byte 映射、BPE 合并和词表查询。
// 7. 将 pos 移动到 end，继续处理后面的特殊 token 或普通文本。
// 8. 整个文本处理完后，返回 token ID 序列。
std::vector<int> BpeTokenizer::Encode(const std::string& text) const {
  if (text.empty()) {
    return {};
  }

  std::vector<int> result;
  int pos = 0;

  while (pos < text.size()) {
    // ------------ process special token -------------
    // find first special token
    bool is_special = false;
    for (const auto& st : special_tokens_) {
      // if (pos + st.first.size() <= text.size() &&
      //     std::memcmp(st.first.data(), text.data() + pos, st.first.size()) == 0) {
      size_t token_size = st.first.size();
      if (pos + token_size <= text.size() &&
          text.compare(pos, token_size, st.first, 0 , token_size) == 0) {
        is_special = true;
        pos += token_size;
        result.push_back(st.second);
        break;
      }
    }

    // check whether next pos is special
    if (is_special) {
      continue;
    }

    // current pos is normal token, find end of normal token
    std::size_t end = text.size();
    for (const auto& st : special_tokens_) {
      std::size_t found = text.find(st.first, pos);
      if (found != std::string::npos && found < end) {
        end = found;
      }
    }

    // normal segment bpe [pos, end)
    std::string segment = text.substr(pos, end - pos);
    EncodeNormalSegment(segment, result);

    pos = end;
  }
  return result;
}

void BpeTokenizer::EncodeNormalSegment(
    const std::string& segment, std::vector<int>& result) const {
  // --------------- process normal token --------------
  // b2u_: text[i] (byte:0-255) -> utf8
  std::vector<std::string> word;
  for (unsigned char c : segment) { // char：-128 - 127 会越界
      word.push_back(b2u_[c]);
  }

  // merge
  while (word.size() > 1) {
      int best_rank = INT32_MAX;
      size_t best_idx = word.size();

      for (int i = 0; i + 1 < word.size(); ++i) {
          auto it = merge_ranks_.find({word[i], word[i+1]});
          if (it != merge_ranks_.end() && it->second < best_rank) {
              best_rank = it->second;
              best_idx = i;
          }
      }

      if (best_idx == word.size()) {
          break;
      }

      word[best_idx] += word[best_idx + 1];
      word.erase(word.begin() + best_idx + 1);
  }

  // token -> token id
  for (const auto& c : word) {
      auto it = vocab_.find(c);
      if (it != vocab_.end()) {
        result.push_back(it->second);
      } else {
        result.push_back(unk_id_);
      }
  }
}

std::string BpeTokenizer::DecodeToken(int token) const {
    return Decode({token});
}

std::string BpeTokenizer::Decode(const std::vector<int>& token_ids) const {
    // {7, 20, 100} → "hello" + "Ġ" + "ä¸Ń" → "helloĠä¸Ń"
    // "helloĠä¸Ń" divide to "h" "e" "l" "l" "o" "Ġ" "ä" "¸" "Ń"
    // u2b_
        // hello → 68 65 6C 6C 6F
        // Ġ     → 20
        // ä(C3 A4)     → E4
        // ¸     → B8
        // Ń     → AD
        // 68 65 6C 6C 6F 20 E4 B8 AD -> 显示为： hello 中

    // 1. id -> token string
    std::string text;
    for (const auto& id : token_ids) {
        if (id < 0 ||
            static_cast<std::size_t>(id) >= id_to_token_.size()) {
            throw std::out_of_range("invalid token id");
        }
        text += id_to_token_[id];
    }

    // 2. traverse string， unicode -> byte
    int pos = 0;
    std::string result;
    while (pos < text.size()) {
        // extract next utf8 codepoint as a string
        unsigned char first_byte = text[pos];
        int len = 1;
        if ((first_byte & 0x80) == 0) { // 0xxx-xxxx
            len = 1;
        } else if ((first_byte & 0xE0) == 0xC0) { // 110x-xxxx
            len = 2;
        } else if ((first_byte & 0xF0) == 0xE0) { // 1110-xxxx
            len = 3;
        } else if ((first_byte & 0xF8) == 0xF0) {// 1111-0xxx
            len = 4;
        } else {
            ++pos; // skip
            continue;;
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

std::unique_ptr<BpeTokenizer> CreateBpeTokenizer(
    const std::string& vocab_path, const std::string& merges_path,
    const std::string& special_path) {
  auto tok = std::make_unique<BpeTokenizer>();
  if (tok->Load(vocab_path, merges_path, special_path)) {
    return tok;
  }
  return nullptr;
}

}
