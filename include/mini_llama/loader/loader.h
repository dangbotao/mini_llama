#pragma once

#include "mini_llama/model.h"
#include "mini_llama/loader/parse_json.h"

#include <vector>
#include <string>

namespace mini_llama {

// // 直接从manifest加载模型
MiniLlamaModel LoadModel(const ModelManifest& manifest,
                         const std::string& weights_path);

MiniLlamaModel LoadModel(const std::string& json_path,
                         const std::string& weights_path);

}