
#include "mini_llama/gguf_loader.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "mini_llama/gguf.h"
#include "mini_llama/model.h"
#include "mini_llama/quant.h"

namespace mini_llama {

// ---------------------------------------------------------------------------
// Tensor name mapping: GGUF (llama.cpp style) -> this project naming
// ---------------------------------------------------------------------------

static std::string MapTensorName(const std::string& gguf_name) {
  // Embedding
  if (gguf_name == "token_embd.weight") {
    return "token_embedding";
  }
  // Output norm
  if (gguf_name == "output_norm.weight") {
    return "final_norm";
  }
  // Output projection (lm_head) — may be tied with token_embd
  if (gguf_name == "output.weight") {
    return "lm_head";
  }

  // Layer tensors: blk.{L}.{name}.weight
  // Parse "blk.n.xxx.weight"
  if (gguf_name.rfind("blk.", 0) == 0) {
    size_t dot1 = gguf_name.find('.', 4);
    if (dot1 == std::string::npos) {
      return "";
    }
    std::string layer_str = gguf_name.substr(4, dot1 - 4);
    size_t dot2 = gguf_name.find('.', dot1 + 1);
    if (dot2 == std::string::npos) {
      return "";
    }
    std::string op = gguf_name.substr(dot1 + 1, dot2 - dot1 - 1);
    std::string suffix = gguf_name.substr(dot2 + 1);
    std::string mapped_op;
    if (suffix == "bias") {
      if (op == "attn_q") {
        mapped_op = "bq";
      } else if (op == "attn_k") {
        mapped_op = "bk";
      } else if (op == "attn_v") {
        mapped_op = "bv";
      } else {
        return "";
      }
    } else if (suffix != "weight") {
      return "";
    } else if (op == "attn_norm") {
      mapped_op = "attention_norm";
    } else if (op == "attn_q") {
      mapped_op = "wq";
    } else if (op == "attn_k") {
      mapped_op = "wk";
    } else if (op == "attn_v") {
      mapped_op = "wv";
    } else if (op == "attn_output") {
      mapped_op = "wo";
    } else if (op == "ffn_norm") {
      mapped_op = "ffn_norm";
    } else if (op == "ffn_gate") {
      mapped_op = "w_gate";
    } else if (op == "ffn_up") {
      mapped_op = "w_up";
    } else if (op == "ffn_down") {
      mapped_op = "w_down";
    } else {
      return "";
    }

    return "layers." + layer_str + "." + mapped_op;
  }

  return "";
}

// ---------------------------------------------------------------------------
// Convert GGUF shape to our row-major Tensor view.
// GGUF uses ne0 as the fastest-changing dimension. For a 2D weight advertised
// as [in_features, out_features], the raw bytes are already laid out as
// out_features rows of in_features values. Our Tensor wants [out, in].
// ---------------------------------------------------------------------------

static std::vector<int> MapShape(const std::vector<uint64_t>& gguf_shape) {
  std::vector<int> shape;
  shape.reserve(gguf_shape.size());
  if (gguf_shape.size() == 2) {
    shape.push_back(static_cast<int>(gguf_shape[1]));
    shape.push_back(static_cast<int>(gguf_shape[0]));
    return shape;
  }
  for (uint64_t dim : gguf_shape) {
    shape.push_back(static_cast<int>(dim));
  }
  return shape;
}

// ---------------------------------------------------------------------------
// Load quantized tensor from raw bytes (F32, Q8_0, Q4_0, Q4_1)
// ---------------------------------------------------------------------------

// gguf中tensor数据存储格式有Q8_0和F32两种
// std::vector<uint8_t> raw，只是用统一的
// 逐字节方式把文件内容搬进内存，并没有改变原始编码
    // GGUF F32 字节 → raw逐字节读取 → memcpy 到 vector<float>
    // GGUF Q8_0 字节 → raw → memcpy 到 vector<BlockQ80>
static QuantizedTensor LoadQuantizedTensor(const std::vector<uint8_t>& raw,
                                           const std::vector<int>& shape,
                                           int ggml_type) {
  size_t num_elements = 1;
  for (int d : shape) {
    num_elements *= static_cast<size_t>(d);
  }

  QuantizedTensor q;
  q.shape = shape;

  // 根据tensorInfo存储的数据类型ggml_type，
  // 按照对应的数据格式将数据从raw放入QuantizedTensor的对应的数组
  switch (ggml_type) {
    case kGgmlTypeF32: {
      q.type = QuantType::kF32;
      size_t expected_bytes = num_elements * sizeof(float);
      if (raw.size() != expected_bytes) {
        throw std::runtime_error("F32 tensor size mismatch: expected " +
                                 std::to_string(expected_bytes) +
                                 " bytes, got " + std::to_string(raw.size()));
      }
      q.f32_data.resize(num_elements);
      std::memcpy(q.f32_data.data(), raw.data(), expected_bytes);
      break;
    }
    case kGgmlTypeQ80: {
      q.type = QuantType::kQ80;
      size_t n_blocks = (num_elements + kQ80BlockSize - 1) / kQ80BlockSize;
      size_t expected_bytes = n_blocks * sizeof(BlockQ80);
      if (raw.size() != expected_bytes) {
        throw std::runtime_error("Q8_0 tensor size mismatch: expected " +
                                 std::to_string(expected_bytes) +
                                 " bytes, got " + std::to_string(raw.size()));
      }
      q.q8_0_data.resize(n_blocks);
      // GGUF 中 Q8_0 的二进制布局，和 BlockQ80 布局完全一致，可以直接拷贝
      std::memcpy(q.q8_0_data.data(), raw.data(), expected_bytes);
      break;
    }
    // case kGgmlTypeQ40: {
    //   q.type = QuantType::kQ40;
    //   size_t n_blocks = (num_elements + kQ40BlockSize - 1) / kQ40BlockSize;
    //   size_t expected_bytes = n_blocks * sizeof(BlockQ40);
    //   if (raw.size() != expected_bytes) {
    //     throw std::runtime_error("Q4_0 tensor size mismatch: expected " +
    //                              std::to_string(expected_bytes) +
    //                              " bytes, got " + std::to_string(raw.size()));
    //   }
    //   q.q4_0_data.resize(n_blocks);
    //   std::memcpy(q.q4_0_data.data(), raw.data(), expected_bytes);
    //   break;
    // }
    // case kGgmlTypeQ41: {
    //   q.type = QuantType::kQ41;
    //   size_t n_blocks = (num_elements + kQ41BlockSize - 1) / kQ41BlockSize;
    //   size_t expected_bytes = n_blocks * sizeof(BlockQ41);
    //   if (raw.size() != expected_bytes) {
    //     throw std::runtime_error("Q4_1 tensor size mismatch: expected " +
    //                              std::to_string(expected_bytes) +
    //                              " bytes, got " + std::to_string(raw.size()));
    //   }
    //   q.q4_1_data.resize(n_blocks);
    //   std::memcpy(q.q4_1_data.data(), raw.data(), expected_bytes);
    //   break;
    // }
    default:
      throw std::runtime_error("Unsupported GGUF tensor type: " +
                               std::to_string(ggml_type));
  }

  return q;
}

// ---------------------------------------------------------------------------
// Shape validation helpers
// ---------------------------------------------------------------------------

static void RequireShape(const Tensor& tensor, const std::vector<int>& expected,
                         const std::string& name) {
  if (tensor.shape != expected) {
    throw std::runtime_error(name + ": shape mismatch");
  }
}

static void RequireShape(const QuantizedTensor& tensor,
                         const std::vector<int>& expected,
                         const std::string& name) {
  tensor.AssertShape(expected, name.c_str());
}

static void RequireOptionalShape(const Tensor& tensor,
                                 const std::vector<int>& expected,
                                 const std::string& name) {
  if (tensor.data.empty()) {
    return;
  }
  RequireShape(tensor, expected, name);
}

static void ValidateLoadedShapes(const MiniLlamaModel& model) {
  const ModelConfig& c = model.config;
  const int q_dim = c.n_heads * c.head_dim;
  const int kv_dim = c.n_kv_heads * c.head_dim;

  RequireShape(model.token_embedding, {c.vocab_size, c.dim},
               "GGUF token_embedding");
  RequireShape(model.final_norm, {c.dim}, "GGUF final_norm");
  RequireShape(model.lm_head, {c.vocab_size, c.dim}, "GGUF lm_head");

  if (model.layers.size() != static_cast<size_t>(c.n_layers)) {
    throw std::runtime_error("GGUF layer count mismatch: expected " +
                             std::to_string(c.n_layers) + ", got " +
                             std::to_string(model.layers.size()));
  }

  for (int layer = 0; layer < c.n_layers; ++layer) {
    const LayerWeights& lw = model.layers[layer];
    const std::string prefix = "GGUF layers." + std::to_string(layer) + ".";
    RequireShape(lw.attention_norm, {c.dim}, prefix + "attention_norm");
    RequireShape(lw.wq, {q_dim, c.dim}, prefix + "wq");
    RequireShape(lw.wk, {kv_dim, c.dim}, prefix + "wk");
    RequireShape(lw.wv, {kv_dim, c.dim}, prefix + "wv");
    RequireOptionalShape(lw.bq, {q_dim}, prefix + "bq");
    RequireOptionalShape(lw.bk, {kv_dim}, prefix + "bk");
    RequireOptionalShape(lw.bv, {kv_dim}, prefix + "bv");
    RequireShape(lw.wo, {c.dim, q_dim}, prefix + "wo");
    RequireShape(lw.ffn_norm, {c.dim}, prefix + "ffn_norm");
    RequireShape(lw.w_gate, {c.hidden_dim, c.dim}, prefix + "w_gate");
    RequireShape(lw.w_up, {c.hidden_dim, c.dim}, prefix + "w_up");
    RequireShape(lw.w_down, {c.dim, c.hidden_dim}, prefix + "w_down");
  }
}

// ---------------------------------------------------------------------------
// Build ModelConfig from GGUF metadata
// ---------------------------------------------------------------------------

ModelConfig GgufModelLoader::BuildConfig() const {
  ModelConfig config;

  auto get_u32 = [&](const std::string& key, int fallback) -> int {
    int64_t v = 0;
    if (reader_.GetMetadataInt(key, v)) {
      return static_cast<int>(v);
    }
    return fallback;
  };

  auto get_f32 = [&](const std::string& key, float fallback) -> float {
    double v = 0.0;
    if (reader_.GetMetadataFloat(key, v)) {
      return static_cast<float>(v);
    }
    return fallback;
  };

  // Try qwen2.* keys first, then llama.* keys as fallback
  std::string arch;
  if (reader_.GetMetadataString("general.architecture", arch) &&
      arch == "qwen2") {
    config.rope_type = RopeType::kNeox;
  }

  config.n_layers =
      get_u32("qwen2.block_count", get_u32("llama.block_count", 0));
  config.dim =
      get_u32("qwen2.embedding_length", get_u32("llama.embedding_length", 0));
  config.hidden_dim = get_u32("qwen2.feed_forward_length",
                              get_u32("llama.feed_forward_length", 0));
  config.n_heads = get_u32("qwen2.attention.head_count",
                           get_u32("llama.attention.head_count", 0));
  config.n_kv_heads = get_u32("qwen2.attention.head_count_kv",
                              get_u32("llama.attention.head_count_kv", 0));
  config.max_seq_len =
      get_u32("qwen2.context_length", get_u32("llama.context_length", 0));
  config.rope_theta = get_f32("qwen2.rope.freq_base",
                              get_f32("llama.rope.freq_base", 10000.0f));
  config.rms_norm_eps =
      get_f32("qwen2.attention.layer_norm_rms_epsilon",
              get_f32("llama.attention.layer_norm_rms_epsilon", 1e-5f));

  config.vocab_size =
      get_u32("qwen2.vocab_size", get_u32("llama.vocab_size", 0));

  // Standard GGUF models normally derive vocab size from the token array.
  std::vector<std::string> tokens;
  if (reader_.GetMetadataStringArray("tokenizer.ggml.tokens", tokens)) {
    config.vocab_size = static_cast<int>(tokens.size());
  }

  // Validate
  if (config.n_layers <= 0 || config.dim <= 0 || config.hidden_dim <= 0 ||
      config.n_heads <= 0 || config.n_kv_heads <= 0 ||
      config.max_seq_len <= 0 || config.vocab_size <= 0) {
    throw std::runtime_error(
        "Invalid model config from GGUF: dim=" + std::to_string(config.dim) +
        " layers=" + std::to_string(config.n_layers) +
        " heads=" + std::to_string(config.n_heads) +
        " vocab=" + std::to_string(config.vocab_size));
  }
  if (config.dim % config.n_heads != 0) {
    throw std::runtime_error("dim not divisible by n_heads");
  }
  if (config.n_heads % config.n_kv_heads != 0) {
    throw std::runtime_error("n_heads not divisible by n_kv_heads");
  }

  config.head_dim = config.dim / config.n_heads;

  std::cout << "GGUF config: vocab=" << config.vocab_size
            << " dim=" << config.dim << " layers=" << config.n_layers
            << " heads=" << config.n_heads << " kv_heads=" << config.n_kv_heads
            << " head_dim=" << config.head_dim
            << " hidden_dim=" << config.hidden_dim
            << " max_seq_len=" << config.max_seq_len
            << " rope_theta=" << config.rope_theta
            << " rms_norm_eps=" << config.rms_norm_eps << std::endl;

  return config;
}

// ---------------------------------------------------------------------------
// Tensor index and tensor loading
// ---------------------------------------------------------------------------

void GgufModelLoader::BuildTensorMap() {
  tensor_map_.clear();
  const std::vector<GgufTensorInfo>& tensors = reader_.GetTensors();
  tensor_map_.reserve(tensors.size());

  for (const GgufTensorInfo& tensor : tensors) {
    const std::string mapped_name = MapTensorName(tensor.name);
    if (mapped_name.empty()) {
      continue;
    }

    const auto inserted = tensor_map_.emplace(mapped_name, &tensor);
    if (!inserted.second) {
      throw std::runtime_error("Duplicate mapped tensor: " + mapped_name);
    }
  }
}

QuantizedTensor GgufModelLoader::LoadWeight(const std::string& name) const {
  const auto it = tensor_map_.find(name);
  if (it == tensor_map_.end()) {
    throw std::runtime_error("Missing tensor: " + name);
  }
  const GgufTensorInfo& info = *it->second;
  std::vector<uint8_t> raw;
  if (!reader_.ReadTensorRaw(info, raw)) {
    throw std::runtime_error("Failed to read tensor: " + name);
  }

  return LoadQuantizedTensor(raw, MapShape(info.shape), info.type);
}

Tensor GgufModelLoader::LoadTensorF32(const std::string& name) const {
  const QuantizedTensor weight = LoadWeight(name);
  switch (weight.type) {
    case QuantType::kF32:
      return ToTensor(weight);
    case QuantType::kQ80:
      return DequantizeFromQ80(weight.q8_0_data, weight.shape);
  }
  throw std::runtime_error("Cannot convert tensor to F32: " + name);
}

Tensor GgufModelLoader::LoadOptionalTensorF32(
    const std::string& name) const {
  if (tensor_map_.find(name) == tensor_map_.end()) {
    return Tensor();
  }
  return LoadTensorF32(name);
}

void GgufModelLoader::LoadWeights(MiniLlamaModel& model) const {
  const ModelConfig& config = model.config;

  model.token_embedding = LoadTensorF32("token_embedding");
  model.layers.resize(config.n_layers);

  for (int layer = 0; layer < config.n_layers; ++layer) {
    LayerWeights& weights = model.layers[layer];
    const std::string prefix = "layers." + std::to_string(layer) + ".";

    weights.attention_norm = LoadTensorF32(prefix + "attention_norm");
    weights.wq = LoadWeight(prefix + "wq");
    weights.wk = LoadWeight(prefix + "wk");
    weights.wv = LoadWeight(prefix + "wv");
    weights.bq = LoadOptionalTensorF32(prefix + "bq");
    weights.bk = LoadOptionalTensorF32(prefix + "bk");
    weights.bv = LoadOptionalTensorF32(prefix + "bv");
    weights.wo = LoadWeight(prefix + "wo");
    weights.ffn_norm = LoadTensorF32(prefix + "ffn_norm");
    weights.w_gate = LoadWeight(prefix + "w_gate");
    weights.w_up = LoadWeight(prefix + "w_up");
    weights.w_down = LoadWeight(prefix + "w_down");
  }

  model.final_norm = LoadTensorF32("final_norm");
  if (tensor_map_.find("lm_head") != tensor_map_.end()) {
    model.lm_head = LoadWeight("lm_head");
  } else {
    model.lm_head = ToQuantizedTensor(model.token_embedding);
  }
}

// ---------------------------------------------------------------------------
// Public load entry point
// ---------------------------------------------------------------------------

MiniLlamaModel GgufModelLoader::Load() {
  MiniLlamaModel model;
  tensor_map_.clear();

  if (reader_.GetTensors().empty()) {
    model.load_error =
        "GGUF reader is not loaded or does not contain any tensors";
    if (!reader_.load_error.empty()) {
      model.load_error += ": " + reader_.load_error;
    }
    return model;
  }

  try {
    model.config = BuildConfig();
    BuildTensorMap();
    LoadWeights(model);
    ValidateLoadedShapes(model);
  } catch (const std::exception& e) {
    model.load_error = "Failed to load GGUF model: " + std::string(e.what());
    return model;
  }

  model.loaded = true;
  std::cout << "GGUF model loaded successfully" << std::endl;
  return model;
}

}  // namespace mini_llama
