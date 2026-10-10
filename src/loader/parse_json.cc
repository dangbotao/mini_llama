#include "mini_llama/loader/parse_json.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mini_llama {

// 将 shape 格式化为错误信息使用的 "[d0, d1, ...]" 字符串。
static std::string ShapeToString(const std::vector<int>& shape) {
  std::string result = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      result += ", ";
    }
    result += std::to_string(shape[i]);
  }
  result += "]";
  return result;
}

// 定位必填键对应值的第一个非空白字符；键缺失或结构不完整时抛异常。
static size_t FindConfigKey(const std::string& content,
                            const std::string& key) {
  const std::string marker = "\"" + key + "\"";
  size_t key_pos = content.find(marker);
  if (key_pos == std::string::npos) {
    throw std::runtime_error("missing config key: " + key);
  }
  size_t colon_pos = content.find(":", key_pos + marker.size());
  if (colon_pos == std::string::npos) {
    throw std::runtime_error("missing ':' after config key: " + key);
  }
  size_t value_pos = colon_pos + 1;
  while (value_pos < content.size() &&
         std::isspace(static_cast<unsigned char>(content[value_pos]))) {
    ++value_pos;
  }
  if (value_pos >= content.size()) {
    throw std::runtime_error("missing value for config key: " + key);
  }
  return value_pos;
}

// 安全计算 F32 张量应占用的字节数，同时检查非法维度和乘法溢出。
static size_t CheckedTensorByteSize(const TensorInfo& info) {
  size_t num_elements = 1;
  for (int dim : info.shape) {
    if (dim <= 0) {
      throw std::runtime_error("invalid tensor shape for " + info.name);
    }
    const size_t d = static_cast<size_t>(dim);
    if (num_elements > std::numeric_limits<size_t>::max() / d) {
      throw std::runtime_error("tensor size overflow for " + info.name);
    }
    num_elements *= d;
  }
  if (num_elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
    throw std::runtime_error("tensor byte size overflow for " + info.name);
  }
  return num_elements * sizeof(float);
}

// 数字后只允许出现空白、逗号或容器结束符。
static bool IsJsonValueTerminator(char c) {
  return std::isspace(static_cast<unsigned char>(c)) || c == ',' || c == '}' ||
         c == ']';
}

// 检查数值转换实际消费的字符边界，防止把 "4.5" 当成整数 4 接受。
static void ValidateNumberTerminator(const std::string& content,
                                     size_t value_pos, size_t parsed,
                                     const std::string& key) {
  size_t pos = value_pos + parsed;
  while (pos < content.size() &&
         std::isspace(static_cast<unsigned char>(content[pos]))) {
    ++pos;
  }
  if (pos < content.size() && !IsJsonValueTerminator(content[pos])) {
    throw std::runtime_error("invalid numeric value for config key: " + key);
  }
}

// 读取必填的 int 字段，并要求整个 JSON 数值 token 都是合法整数。
static int ParseRequiredInt(const std::string& content,
                            const std::string& key) {
  size_t value_pos = FindConfigKey(content, key);
  size_t parsed = 0;
  int value = 0;
  try {
    value = std::stoi(content.substr(value_pos), &parsed);
  } catch (const std::exception&) {
    throw std::runtime_error("invalid integer for config key: " + key);
  }
  if (parsed == 0) {
    throw std::runtime_error("invalid integer for config key: " + key);
  }
  ValidateNumberTerminator(content, value_pos, parsed, key);
  return value;
}

static float ParseRequiredFloat(const std::string& content,
                                const std::string& key) {
  size_t value_pos = FindConfigKey(content, key);
  size_t parsed = 0;
  float value = 0.0f;
  try {
    value = std::stof(content.substr(value_pos), &parsed);
  } catch (const std::exception&) {
    throw std::runtime_error("invalid float for config key: " + key);
  }
  if (parsed == 0) {
    throw std::runtime_error("invalid float for config key: " + key);
  }
  ValidateNumberTerminator(content, value_pos, parsed, key);
  return value;
}

// 校验会影响张量布局和前向计算的模型结构约束。
static void ValidateConfig(const ModelConfig& config) {
  if (config.vocab_size <= 0 || config.dim <= 0 || config.hidden_dim <= 0 ||
      config.n_layers <= 0 || config.n_heads <= 0 || config.n_kv_heads <= 0 ||
      config.max_seq_len <= 0) {
    throw std::runtime_error("all model dimensions must be positive");
  }
  if (config.dim % config.n_heads != 0) {
    throw std::runtime_error("dim must be divisible by n_heads");
  }
  if (config.n_heads % config.n_kv_heads != 0) {
    throw std::runtime_error("n_heads must be divisible by n_kv_heads");
  }
  // RoPE 按相邻的两个通道组成旋转对，因此每个 head 的维度必须为偶数。
  if (config.head_dim <= 0 || config.head_dim % 2 != 0) {
    throw std::runtime_error("head_dim must be a positive even number");
  }
  if (!std::isfinite(config.rope_theta) || config.rope_theta <= 0.0f) {
    throw std::runtime_error("rope_theta must be positive");
  }
  if (!std::isfinite(config.rms_norm_eps) || config.rms_norm_eps <= 0.0f) {
    throw std::runtime_error("rms_norm_eps must be positive");
  }
}

// 从 model.json 文本提取模型结构，并推导 JSON 中未单独保存的 head_dim。
static ModelConfig ParseConfigJson(const std::string& content) {
  ModelConfig config;
  config.vocab_size = ParseRequiredInt(content, "vocab_size");
  config.dim = ParseRequiredInt(content, "dim");
  config.hidden_dim = ParseRequiredInt(content, "hidden_dim");
  config.n_layers = ParseRequiredInt(content, "n_layers");
  config.n_heads = ParseRequiredInt(content, "n_heads");
  config.n_kv_heads = ParseRequiredInt(content, "n_kv_heads");
  config.max_seq_len = ParseRequiredInt(content, "max_seq_len");
  config.rope_theta = ParseRequiredFloat(content, "rope_theta");
  config.rms_norm_eps = ParseRequiredFloat(content, "rms_norm_eps");
  // head_dim 不在 model.json 中单独保存，这里由模型维度和头数推导。
  if (config.n_heads <= 0) {
    throw std::runtime_error("n_heads must be positive");
  }
  config.head_dim = config.dim / config.n_heads;

  ValidateConfig(config);
  return config;
}

// ---------------------------------------------------------------------------
// 解析 JSON 中的 tensors 数组
// ---------------------------------------------------------------------------
// 将每个固定结构的张量对象转换为 TensorInfo；这里只解析元数据，
// dtype、shape、文件区间和模型必需权重由 ValidateManifest 统一校验。
static std::vector<TensorInfo> ParseTensorMetadata(const std::string& content) {
  std::vector<TensorInfo> result;

  const std::string tensors_marker = "\"tensors\"";
  size_t tensors_pos = content.find(tensors_marker);
  if (tensors_pos == std::string::npos) {
    throw std::runtime_error("missing 'tensors' field in model.json");
  }

  size_t bracket_open = content.find("[", tensors_pos);
  if (bracket_open == std::string::npos) {
    throw std::runtime_error("missing '[' after 'tensors'");
  }

  size_t i = bracket_open + 1;
  int depth = 1;
  while (i < content.size() && depth > 0) {
    // 在 tensors 数组中寻找下一个张量对象，同时识别数组结束位置。
    while (i < content.size() && content[i] != '{') {
      if (content[i] == '[') {
        ++depth;
      } else if (content[i] == ']') {
        --depth;
      }
      ++i;
    }
    if (depth <= 0 || i >= content.size()) {
      break;
    }

    // 当前格式的张量条目不包含嵌套对象，因此第一个 '}' 即为条目结尾。
    TensorInfo info;
    size_t obj_start = i;
    size_t obj_end = content.find("}", obj_start);
    if (obj_end == std::string::npos) {
      throw std::runtime_error("unclosed tensor object");
    }
    std::string obj = content.substr(obj_start, obj_end - obj_start + 1);

    // 单独记录字段是否出现，因为 offset 和 byte_size 的合法值可以为 0。
    bool has_name = false;
    bool has_dtype = false;
    bool has_shape = false;
    bool has_offset = false;
    bool has_byte_size = false;

    // 解析内部张量名，例如 "layers.0.wq"。
    size_t name_pos = obj.find("\"name\"");
    if (name_pos != std::string::npos) {
      size_t name_colon = obj.find(":", name_pos);
      size_t name_quote = obj.find("\"", name_colon + 1);
      size_t name_end = obj.find("\"", name_quote + 1);
      if (name_quote != std::string::npos && name_end != std::string::npos) {
        info.name = obj.substr(name_quote + 1, name_end - name_quote - 1);
        has_name = true;
      }
    }

    // 解析磁盘数据类型；当前格式后续只接受 float32。
    size_t dtype_pos = obj.find("\"dtype\"");
    if (dtype_pos != std::string::npos) {
      size_t dtype_colon = obj.find(":", dtype_pos);
      size_t dtype_quote = obj.find("\"", dtype_colon + 1);
      size_t dtype_end = obj.find("\"", dtype_quote + 1);
      if (dtype_quote != std::string::npos && dtype_end != std::string::npos) {
        info.dtype = obj.substr(dtype_quote + 1, dtype_end - dtype_quote - 1);
        has_dtype = true;
      }
    }

    // 解析按 row-major 约定记录的 shape 数组。
    size_t shape_pos = obj.find("\"shape\"");
    if (shape_pos != std::string::npos) {
      size_t shape_bracket = obj.find("[", shape_pos);
      size_t shape_end = obj.find("]", shape_bracket);
      if (shape_bracket != std::string::npos &&
          shape_end != std::string::npos) {
        std::string shape_string_short =
            obj.substr(shape_bracket + 1, shape_end - shape_bracket - 1);
        std::stringstream ss(shape_string_short);
        std::string token;
        while (std::getline(ss, token, ',')) {
          // 去除逗号分隔的维度文本两端空白。
          size_t start = token.find_first_not_of(" \t\n\r");
          size_t end = token.find_last_not_of(" \t\n\r");
          if (start != std::string::npos) {
            int dim = std::stoi(token.substr(start, end - start + 1));
            info.shape.push_back(dim);
          }
        }
        has_shape = true;
      }
    }

    // offset 是相对 model.bin 文件开头的绝对字节偏移。
    size_t offset_pos = obj.find("\"offset\"");
    if (offset_pos != std::string::npos) {
      size_t offset_colon = obj.find(":", offset_pos);
      if (offset_colon == std::string::npos) {
        throw std::runtime_error("tensor offset field is malformed");
      }
      info.offset =
          static_cast<size_t>(std::stoull(obj.substr(offset_colon + 1)));
      has_offset = true;
    }

    // byte_size 是该张量在 model.bin 中占用的原始字节数。
    size_t bs_pos = obj.find("\"byte_size\"");
    if (bs_pos != std::string::npos) {
      size_t bs_colon = obj.find(":", bs_pos);
      if (bs_colon == std::string::npos) {
        throw std::runtime_error("tensor byte_size field is malformed");
      }
      info.byte_size =
          static_cast<size_t>(std::stoull(obj.substr(bs_colon + 1)));
      has_byte_size = true;
    }

    // 五个字段共同构成从逻辑张量到二进制区间的完整索引。
    if (!has_name || info.name.empty()) {
      throw std::runtime_error("tensor missing name field");
    }
    if (!has_dtype) {
      throw std::runtime_error("tensor missing dtype field: " + info.name);
    }
    if (!has_shape) {
      throw std::runtime_error("tensor missing shape field: " + info.name);
    }
    if (!has_offset) {
      throw std::runtime_error("tensor missing offset field: " + info.name);
    }
    if (!has_byte_size) {
      throw std::runtime_error("tensor missing byte_size field: " + info.name);
    }

    result.push_back(info);
    i = obj_end + 1;
  }

  return result;
}

/**
 * @brief 根据模型配置生成运行时必需张量的名称和期望形状。
 *
 * 结果包含 token_embedding、final_norm、lm_head，以及每个
 * Transformer 层的 9 个必需张量。二维 Linear 权重统一使用
 * [out_features, in_features] 布局。
 *
 * @param config 用于推导张量名称和形状的模型配置。
 * @return 按模型装配顺序排列的 (tensor_name, expected_shape) 列表。
 * @pre config 中相关整数参数为正数，且 head 数与 head_dim 的乘积
 *      可以由 int 表示。
 *
 * @note 返回值只描述运行时必需的张量，manifest 可以包含额外张量。
 */
static std::vector<std::pair<std::string, std::vector<int>>>
ExpectedTensorShapes(const ModelConfig& config) {
  // 二维 Linear 权重统一使用 [out_features, in_features] 布局。
  std::vector<std::pair<std::string, std::vector<int>>> expected = {
      {"token_embedding", {config.vocab_size, config.dim}},
      {"final_norm", {config.dim}},
      {"lm_head", {config.vocab_size, config.dim}},
  };

  for (int layer = 0; layer < config.n_layers; ++layer) {
    const std::string prefix = "layers." + std::to_string(layer) + ".";
    expected.push_back({prefix + "attention_norm", {config.dim}});
    expected.push_back(
        {prefix + "wq", {config.n_heads * config.head_dim, config.dim}});
    expected.push_back(
        {prefix + "wk", {config.n_kv_heads * config.head_dim, config.dim}});
    expected.push_back(
        {prefix + "wv", {config.n_kv_heads * config.head_dim, config.dim}});
    expected.push_back(
        {prefix + "wo", {config.dim, config.n_heads * config.head_dim}});
    expected.push_back({prefix + "ffn_norm", {config.dim}});
    expected.push_back({prefix + "w_gate", {config.hidden_dim, config.dim}});
    expected.push_back({prefix + "w_up", {config.hidden_dim, config.dim}});
    expected.push_back({prefix + "w_down", {config.dim, config.hidden_dim}});
  }

  return expected;
}

// 对模型清单ModelManifest做不依赖 model.bin 内容的完整一致性校验。
// 实际上对比的是json中config推导的值与tensors提供的元数据是否一致
void ValidateManifest(const ModelManifest& manifest) {
  ValidateConfig(manifest.config);
  if (manifest.config.head_dim !=
      manifest.config.dim / manifest.config.n_heads) {
    throw std::runtime_error("n_heads * head_dim must equal dim");
  }

  if (manifest.tensors.empty()) {
    throw std::runtime_error("model.json contains no tensors");
  }

  std::map<std::string, TensorInfo> tensor_map;
  std::vector<TensorInfo> by_offset;
  by_offset.reserve(manifest.tensors.size());

  // 1. 逐个tensor校验 dtype、shape、字节数、区间溢出和名称唯一性。
  for (const auto& info : manifest.tensors) {
    if (info.dtype != "float32") {
      throw std::runtime_error("unsupported dtype '" + info.dtype +
                               "' for tensor: " + info.name);
    }
    if (info.shape.empty()) {
      throw std::runtime_error("tensor shape must not be empty: " + info.name);
    }
    size_t expected_bytes = CheckedTensorByteSize(info);
    if (expected_bytes != info.byte_size) {
      throw std::runtime_error("tensor byte_size mismatch for " + info.name +
                               ": expected " + std::to_string(expected_bytes) +
                               ", manifest says " +
                               std::to_string(info.byte_size));
    }
    if (info.offset > std::numeric_limits<size_t>::max() - info.byte_size) {
      throw std::runtime_error("tensor offset overflow for " + info.name);
    }
    auto inserted = tensor_map.emplace(info.name, info);
    if (!inserted.second) {
      throw std::runtime_error("duplicate tensor name: " + info.name);
    }
    by_offset.push_back(info);
  }

  // 2. 检查张量位置是否存在重叠
  // 张量之间允许存在未使用的间隙，不允许重叠，重叠则报错
  std::sort(by_offset.begin(), by_offset.end(),
            [](const TensorInfo& a, const TensorInfo& b) {
              return a.offset < b.offset;
            });
  for (size_t i = 1; i < by_offset.size(); ++i) {
    size_t previous_end = by_offset[i - 1].offset + by_offset[i - 1].byte_size;
    if (by_offset[i].offset < previous_end) {
      throw std::runtime_error(
          "tensor ranges overlap: " + by_offset[i - 1].name + " and " +
          by_offset[i].name);
    }
  }

  // 3.确认模型运行必需的张量全部存在，且 shape 与配置严格一致。
  for (const auto& expected : ExpectedTensorShapes(manifest.config)) {
    auto it = tensor_map.find(expected.first);
    if (it == tensor_map.end()) {
      throw std::runtime_error("missing required tensor: " + expected.first);
    }
    if (it->second.shape != expected.second) {
      throw std::runtime_error("tensor shape mismatch for " + expected.first +
                               ": expected " + ShapeToString(expected.second) +
                               ", manifest says " +
                               ShapeToString(it->second.shape));
    }
  }
}

// ---------------------------------------------------------------------------
// ModelManifest
// ---------------------------------------------------------------------------
// 读取并解析 model.json，返回前完成配置、tokenizer 和张量布局校验。
// 此函数只处理清单，不会打开或验证实际的 model.bin。
ModelManifest ParseManifest(const std::string& config_path) {
  ModelManifest manifest;

  std::ifstream cf(config_path);
  if (!cf.is_open()) {
    throw std::runtime_error("failed to open config: " + config_path);
  }
  std::stringstream buffer;
  buffer << cf.rdbuf();
  if (cf.bad()) {
    throw std::runtime_error("failed to read config: " + config_path);
  }

  std::string content = buffer.str();
  // 三部分分别解析，最后进行配置与张量元数据之间的交叉校验。
  manifest.config = ParseConfigJson(content);
  // manifest.tokenizer = ParseTokenizerJson(content);
  manifest.tensors = ParseTensorMetadata(content);
  ValidateManifest(manifest);

  return manifest;
}

}
