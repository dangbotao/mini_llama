#include "mini_llama/loader/loader.h"

#include <unordered_map>
#include <map>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace mini_llama {

MiniLlamaModel LoadFailure(const std::string& message) {
    MiniLlamaModel model;
    model.loaded = false;
    std::cerr << "Failed to load model: " << message << std::endl;
    return model;
}

// // 将 shape 格式化为错误信息使用的 "[d0, d1, ...]" 字符串。
// static std::string ShapeToString(const std::vector<int>& shape) {
//   std::string result = "[";
//   for (size_t i = 0; i < shape.size(); ++i) {
//     if (i > 0) {
//       result += ", ";
//     }
//     result += std::to_string(shape[i]);
//   }
//   result += "]";
//   return result;
// }

// // ---------------- 验证Manifest的完整性和正确性 ------------------
// /**
//  * @brief 安全计算 float32 张量所需的存储字节数。
//  *
//  * 将 shape 中的各维度相乘得到元素总数，再乘以 sizeof(float)。
//  * 计算过程会检查非正维度以及 std::size_t 乘法溢出。
//  *
//  * @param info 待计算的张量元数据。
//  * @return 张量数据所需的字节数。
//  * @throws std::runtime_error 任一维度不为正数，或尺寸计算发生溢出。
//  * @pre info.shape 非空，且 info.dtype 为 "float32"。
//  *
//  * @note 本函数只根据 shape 计算字节数，不会检查 info.dtype
//  *       或比较 info.byte_size。
//  */
// static size_t CheckedTensorByteSize(const TensorInfo& info) {
//   size_t num_elements = 1;
//   for (int dim : info.shape) {
//     if (dim <= 0) {
//       throw std::runtime_error("invalid tensor shape for " + info.name);
//     }
//     const size_t d = static_cast<size_t>(dim);
//     if (num_elements > std::numeric_limits<size_t>::max() / d) {
//       throw std::runtime_error("tensor size overflow for " + info.name);
//     }
//     num_elements *= d;
//   }
//   if (num_elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
//     throw std::runtime_error("tensor byte size overflow for " + info.name);
//   }
//   return num_elements * sizeof(float);
// }

// /**
//  * @brief 根据模型配置生成运行时必需张量的名称和期望形状。
//  *
//  * 结果包含 token_embedding、final_norm、lm_head，以及每个
//  * Transformer 层的 9 个必需张量。二维 Linear 权重统一使用
//  * [out_features, in_features] 布局。
//  *
//  * @param config 用于推导张量名称和形状的模型配置。
//  * @return 按模型装配顺序排列的 (tensor_name, expected_shape) 列表。
//  * @pre config 中相关整数参数为正数，且 head 数与 head_dim 的乘积
//  *      可以由 int 表示。
//  *
//  * @note 返回值只描述运行时必需的张量，manifest 可以包含额外张量。
//  */
// static std::vector<std::pair<std::string, std::vector<int>>>
// ExpectedTensorShapes(const ModelConfig& config) {
//   // 二维 Linear 权重统一使用 [out_features, in_features] 布局。
//   std::vector<std::pair<std::string, std::vector<int>>> expected = {
//       {"token_embedding", {config.vocab_size, config.dim}},
//       {"final_norm", {config.dim}},
//       {"lm_head", {config.vocab_size, config.dim}},
//   };

//   for (int layer = 0; layer < config.n_layers; ++layer) {
//     const std::string prefix = "layers." + std::to_string(layer) + ".";
//     expected.push_back({prefix + "attention_norm", {config.dim}});
//     expected.push_back(
//         {prefix + "wq", {config.n_heads * config.head_dim, config.dim}});
//     expected.push_back(
//         {prefix + "wk", {config.n_kv_heads * config.head_dim, config.dim}});
//     expected.push_back(
//         {prefix + "wv", {config.n_kv_heads * config.head_dim, config.dim}});
//     expected.push_back(
//         {prefix + "wo", {config.dim, config.n_heads * config.head_dim}});
//     expected.push_back({prefix + "ffn_norm", {config.dim}});
//     expected.push_back({prefix + "w_gate", {config.hidden_dim, config.dim}});
//     expected.push_back({prefix + "w_up", {config.hidden_dim, config.dim}});
//     expected.push_back({prefix + "w_down", {config.dim, config.hidden_dim}});
//   }

//   return expected;
// }

// /**
//  * @brief 校验模型清单中的张量元数据tensors及其与模型配置的一致性。
//  *
//  * 校验内容包括：张量列表非空；dtype、shape 和 byte_size 合法；
//  * offset 区间计算不溢出；张量名称唯一；数据区间互不重叠；
//  * 以及所有运行时必需张量都存在，且 shape 与 config 推导结果一致。
//  *
//  * @param manifest 待校验的模型清单。
//  * @throws std::runtime_error manifest 不满足上述任一约束。
//  * @pre manifest.config 已通过模型配置合法性校验。
//  *
//  * @note 本函数不读取 model.bin，因此不检查文件是否存在、实际长度或内容。
//  *       张量区间之间允许存在空洞，manifest 也可以包含额外张量。
//  */
// static void ValidateManifest(const ModelManifest& manifest) {
//   if (manifest.tensors.empty()) {
//     throw std::runtime_error("model.json contains no tensors");
//   }

//   std::map<std::string, TensorInfo> tensor_map;
//   std::vector<TensorInfo> by_offset;
//   by_offset.reserve(manifest.tensors.size());

//   // 1. 逐个tensor校验 dtype、shape、字节数、区间溢出和名称唯一性。
//   for (const auto& info : manifest.tensors) {
//     if (info.dtype != "float32") {
//       throw std::runtime_error("unsupported dtype '" + info.dtype +
//                                "' for tensor: " + info.name);
//     }
//     if (info.shape.empty()) {
//       throw std::runtime_error("tensor shape must not be empty: " + info.name);
//     }
//     size_t expected_bytes = CheckedTensorByteSize(info);
//     if (expected_bytes != info.byte_size) {
//       throw std::runtime_error("tensor byte_size mismatch for " + info.name +
//                                ": expected " + std::to_string(expected_bytes) +
//                                ", manifest says " +
//                                std::to_string(info.byte_size));
//     }
//     if (info.offset > std::numeric_limits<size_t>::max() - info.byte_size) {
//       throw std::runtime_error("tensor offset overflow for " + info.name);
//     }
//     auto inserted = tensor_map.emplace(info.name, info);
//     if (!inserted.second) {
//       throw std::runtime_error("duplicate tensor name: " + info.name);
//     }
//     by_offset.push_back(info);
//   }

//   // 2. 检查张量位置是否存在重叠
//   // 张量之间允许存在未使用的间隙，不允许重叠，重叠则报错
//   std::sort(by_offset.begin(), by_offset.end(),
//             [](const TensorInfo& a, const TensorInfo& b) {
//               return a.offset < b.offset;
//             });
//   for (size_t i = 1; i < by_offset.size(); ++i) {
//     size_t previous_end = by_offset[i - 1].offset + by_offset[i - 1].byte_size;
//     if (by_offset[i].offset < previous_end) {
//       throw std::runtime_error(
//           "tensor ranges overlap: " + by_offset[i - 1].name + " and " +
//           by_offset[i].name);
//     }
//   }

//   // 3.确认模型运行必需的张量全部存在，且 shape 与配置严格一致。
//   for (const auto& expected : ExpectedTensorShapes(manifest.config)) {
//     auto it = tensor_map.find(expected.first);
//     if (it == tensor_map.end()) {
//       throw std::runtime_error("missing required tensor: " + expected.first);
//     }
//     if (it->second.shape != expected.second) {
//       throw std::runtime_error("tensor shape mismatch for " + expected.first +
//                                ": expected " + ShapeToString(expected.second) +
//                                ", manifest says " +
//                                ShapeToString(it->second.shape));
//     }
//   }
// }
// // ---------------------------------------------

/**
 * @brief 按照权重名称找到配置信息TensorInfo，
 * 从二进制文件中找到对应权重位置和长度，加载并返回向量
 *
 * @param wf 权重文件流
 * @param weights_map 权重名称和配置映射表
 * @param name 权重名称
 * @return 返回存储权重信息的向量
 */
Tensor LoadTensorByName(std::ifstream& wf,
    const std::unordered_map<std::string, TensorInfo>& weights_map,
    const std::string& name) {
    // 1. 查找张量是否存在
    auto it = weights_map.find(name);
    if (it == weights_map.end()) {
        throw std::runtime_error("missing required tensor: " + name);
    }

      const TensorInfo& info = it->second;

    // 2. 检查数值是否太大，避免强制转换溢出
    if (info.offset >
        static_cast<size_t>(std::numeric_limits<std::streamoff>::max())) {
        throw std::runtime_error("tensor offset exceeds stream limit for " + name);
    }
    if (info.byte_size >
        static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("tensor byte size exceeds stream limit for " +
                                name);
    }

    // 3. 依据offset，跳到二进制文件中张量所在的位置
    wf.seekg(static_cast<std::streamoff>(info.offset), std::ios::beg); // seekg：设置读取位置
    if (!wf.good()) {
        throw std::runtime_error("failed to seek to offset for tensor: " + name);
    }

    // 4. 读取原始字节
    Tensor output(info.shape, 0.0f);
    // output.data.data() 是 float*，而 read() 要求接收 char* 字节缓冲区，
    // reinterpret_cast<char*>(...)：把这块 float 内存当作字节缓冲区写入，不会执行数值转换或量化
    wf.read(reinterpret_cast<char*>(output.data.data()),
        static_cast<std::streamsize>(info.byte_size));

    // 5. 检查是否完整读取
    if (static_cast<size_t>(wf.gcount()) != info.byte_size) {
        // read 可能因文件截断而短读，必须核对最后一次读取的实际字节数。
        throw std::runtime_error("weights file ended while reading tensor: " +
                                name);
    }

    return output;
}

/**
 * @brief 从json配置文件和bin二进制权重文件加载模型。
    * 1. 首先，将模型配置和结构信息从json文件加载到装配表manifest，mainfest保存配置信息和结构信息，
    * 结构信息包括权重形状，数据类型，在二进制文件的偏移量，以及所占用的字节数等，
    * 2. 随后建立权重名称与权重结构信息的映射，通过名称逐一从二进制文件中找到对应的权重，存入模型，
    * 3. 然后验证二进制文件的长度与从结构中推导出的文件期望长度，两者一致则认为模型加载成功
 * @param manifest 已构造的模型清单
 * @param weights_path 模型权重文件路径
 * @return 加载的MiniLlamaModel
 */
MiniLlamaModel LoadModel(const ModelManifest& manifest,
                         const std::string& weights_path) {
    try {
        ValidateManifest(manifest);
    } catch (const std::exception& e) {
        return LoadFailure(e.what());
    }

    std::cout << "Loaded config: vocab=" << manifest.config.vocab_size
            << " dim=" << manifest.config.dim
            << " layers=" << manifest.config.n_layers
            << " heads=" << manifest.config.n_heads
            << " kv_heads=" << manifest.config.n_kv_heads
            << " head_dim=" << manifest.config.head_dim
            << " hidden_dim=" << manifest.config.hidden_dim << std::endl;
    // ------------------------------------------------------------------------

    // Create MiniLlamaModel
    MiniLlamaModel model;
    model.config = manifest.config;

    // --------------------- Load weights from binary file -----------------------
    // read weights from model.bin
    std::ifstream wf(weights_path, std::ios::binary); // models/tiny/model.bin
    if (!wf.is_open()) {
        return LoadFailure("failed to open weights: " + weights_path);
    }

    // build name and TensorTnfo map
    std::unordered_map<std::string, TensorInfo> weights_map;
    for (const auto& tensor : manifest.tensors) {
        weights_map[tensor.name] = tensor;
    }

    try {
        // Load embedding weights
        model.token_embedding = LoadTensorByName(wf, weights_map, "token_embedding");

        // Load weights by Transformer layers
        model.layers.resize(manifest.config.n_layers);
        for (int layer = 0; layer < model.config.n_layers; ++layer) {

            LayerWeights& lw = model.layers[layer];
            const std::string prefix = "layers." + std::to_string(layer) + ".";

            lw.attention_norm = LoadTensorByName(wf, weights_map, prefix + "attention_norm");
            lw.wq = ToQuantizedTensor(
                LoadTensorByName(wf, weights_map, prefix + "wq"));
            lw.wk = ToQuantizedTensor(
                LoadTensorByName(wf, weights_map, prefix + "wk"));
            lw.wv = ToQuantizedTensor(
                LoadTensorByName(wf, weights_map, prefix + "wv"));
            lw.wo = ToQuantizedTensor(
                LoadTensorByName(wf, weights_map, prefix + "wo"));
            lw.ffn_norm = LoadTensorByName(wf, weights_map, prefix + "ffn_norm");
            lw.w_gate = ToQuantizedTensor(
                LoadTensorByName(wf, weights_map, prefix + "w_gate"));
            lw.w_up = ToQuantizedTensor(
                LoadTensorByName(wf, weights_map, prefix + "w_up"));
            lw.w_down = ToQuantizedTensor(
                LoadTensorByName(wf, weights_map, prefix + "w_down"));
        }

        // After transformer layers
        model.final_norm = LoadTensorByName(wf, weights_map, "final_norm");
        model.lm_head = ToQuantizedTensor(
            LoadTensorByName(wf, weights_map, "lm_head"));
        // --------------------------------------------------------------------------
    } catch (const std::exception& e) {
        return LoadFailure(e.what());
    }

    // ------------------------ 通过文件长度校验模型加载是否成功 -----------------------
    // 获取文件实际大小
    wf.seekg(0, std::ios::end); // 把文件读取位置移动到文件末尾
    std::streamoff file_size = wf.tellg(); // 返回当前位置，即文件总字节数
    if (file_size < 0) {
        return LoadFailure("failed to determine weights file size");
    }

    // 通过json配置信息，计算预期文件大小
    size_t expected_end = 0;
    for (const auto& info : manifest.tensors) {
        size_t tensor_end = info.offset + info.byte_size;
        if (tensor_end > expected_end) { // 找到最大的end
        expected_end = tensor_end;
        }
    }

    // 对比实际大小与预期大小：不相等则报错
    if (static_cast<size_t>(file_size) != expected_end) {
        if (static_cast<size_t>(file_size) > expected_end) { // 实际文件过长
        return LoadFailure("weights file has trailing bytes: file size=" +
                            std::to_string(file_size) +
                            ", expected=" + std::to_string(expected_end));
        } else {
        return LoadFailure( // 实际文件过短：文件被截断
            "weights file is too short: file size=" + std::to_string(file_size) +
            ", expected=" + std::to_string(expected_end));
        }
    }
    // -------------------------------------------------------------------------------

    model.loaded = true;
    std::cout << "Model loaded successfully from " << weights_path << std::endl;
    return model;
}

MiniLlamaModel LoadModel(const std::string& json_path,
                         const std::string& weights_path) {
    try {
        return LoadModel(ParseManifest(json_path), weights_path);
    } catch (const std::exception& e) {
        return LoadFailure(e.what());
    }
}



}
