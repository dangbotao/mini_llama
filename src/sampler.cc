#include "mini_llama/sampler.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mini_llama/operations.h"

namespace mini_llama {

namespace {

// 小于该温度时按贪心采样处理
constexpr float kGreedyTemperature = 1e-6f;

void ValidateLogits(const Tensor& logits) {
  if (logits.size() == 0) {
    throw std::runtime_error("sampler: cannot sample from empty logits");
  }
  for (float value : logits.data) {
    if (!std::isfinite(value)) {
      throw std::runtime_error("sampler: logits must be finite");
    }
  }
}

void ValidateTemperature(float temperature) {
  if (!std::isfinite(temperature) || temperature < 0.0f) {
    throw std::runtime_error(
        "sampler: temperature must be finite and non-negative");
  }
}

void ValidateTopK(int top_k) {
  if (top_k < 0) {
    throw std::runtime_error("sampler: top_k must be non-negative");
  }
}

}

// 初始化随机数种子，验证params的合法性
MiniSampler::MiniSampler(const SamplingParams& params)
    : params_(params),
      rng_(params.seed == 0 ? std::random_device{}() : params.seed) {
    ValidateTemperature(params.temperature);
    ValidateTopK(params.top_k);
}

// 找到logit最大的索引，即为对应的token_id
int MiniSampler::SampleGreedy(const Tensor& logits) {
  return ArgMax(logits);
}

// 从候选项中按温度采样，pair 保存 (logit, token_id)。
int SampleFromCandidates(const std::vector<std::pair<float, int>>& candidates,
                        float temperature, std::mt19937& rng) {
  // 极低温度直接选择最大值。
  if (temperature < kGreedyTemperature) {
    int best_id = candidates[0].second;
    float best_val = candidates[0].first;
    for (size_t i = 1; i < candidates.size(); ++i) {
      if (candidates[i].first > best_val) {
        best_id = candidates[i].second;
        best_val = candidates[i].first;
      }
    }
    return best_id;
  }

  // 找到最大值max_val，用于减去最大值
  float max_val = candidates[0].first;
  for (size_t i = 1; i < candidates.size(); ++i) {
    if (candidates[i].first > max_val) {
      max_val = candidates[i].first;
    }
  }

  // 概率权重：probs[i] = exp((logit(i) - max_logit) / temperature)
  // 归一化概率：probability(i) = probs[i] / sum
  std::vector<float> probs(candidates.size(), 0.0f);
  float sum = 0.0f;
  for (size_t i = 0; i < candidates.size(); ++i) {
    // 减去最大值，避免 Softmax 指数溢出。
    // temperature 控制概率分布：
      // - 温度较低：放大 logits 的差距，更容易选择最大值。
      // - 温度较高：缩小 logits 的差距，采样更加随机。
    probs[i] = std::exp((candidates[i].first - max_val) / temperature);
    sum += probs[i];
  }

  // 使用累计概率区间cumulative进行随机采样。
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float r = dist(rng); // 生成一个 [0, 1) 范围内的均匀随机数
  float cumulative = 0.0f; // 累积概率
  for (size_t i = 0; i < probs.size(); ++i) {
    cumulative += probs[i] / sum;
    // 遍历候选 token 时不断累加概率，当累计概率第一次大于或等于随机数 r
    // 就选择当前 token
    if (cumulative >= r) {
      return candidates[i].second;
    }
  }

  // 处理浮点累计误差，兜底
  return candidates.back().second; // 累积概率可能<1，此时若随机数大于累积概率，直接使用末尾token

}

int MiniSampler::SampleTemperature(const Tensor& logits, float temperature) {
  ValidateLogits(logits);
  ValidateTemperature(temperature);
  if (temperature < kGreedyTemperature) {
    return SampleGreedy(logits);
  }

  // 不过滤候选，整个词表都参与采样。
  std::vector<std::pair<float, int>> candidates; // (score, token_id)
  candidates.reserve(logits.data.size());
  for (size_t i = 0; i < logits.data.size(); ++i) {
    candidates.emplace_back(logits.data[i], static_cast<int>(i));
  }
  return SampleFromCandidates(candidates, temperature, rng_);
}

int MiniSampler::SampleTopK(const Tensor& logits, float temperature, int top_k) {
  ValidateTopK(top_k);
  if (top_k == 0) {
    throw std::runtime_error(
        "sampler: top_k must be positive when calling SampleTopK");
  }
  if (top_k == 1) {
    return SampleGreedy(logits);
  }
  if (top_k > static_cast<int>(logits.data.size())) {
    // top_k 不超过词表大小。
    top_k = static_cast<int>(logits.data.size());
  }

  // get (score, token_id) pair, the largest k scores
  std::vector<std::pair<float, int>> candidates;
  candidates.reserve(logits.data.size());
  for (size_t i = 0; i < logits.data.size(); ++i) {
    candidates.emplace_back(logits[i], static_cast<int>(i)); // (score, token_id)
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { // 返回 true：表示 a 应排在 b 前面
              // 如果 logit 相同，token id 较小的排在前面
              // logit 不同，logit 较大的排在前面。
              if (a.first == b.first) {
                return a.second < b.second;
              }
              return a.first > b.first;
            });
  candidates.resize(top_k);

  return SampleFromCandidates(candidates, temperature, rng_);
};

int MiniSampler::Sample(const Tensor& logits) {
  ValidateLogits(logits);
  // Greedy
  if (params_.temperature < kGreedyTemperature || params_.top_k == 1) {
    return SampleGreedy(logits);
  }
  if (params_.top_k > 1) {
    return SampleTopK(logits, params_.temperature, params_.top_k);
  }
  // top_k == 0
  return SampleTemperature(logits, params_.temperature);
}

}