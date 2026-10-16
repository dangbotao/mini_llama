#pragma once
#include <random>

#include "mini_llama/tensor.h"

namespace mini_llama {

/// @brief 采样参数，默认使用贪心采样。
struct SamplingParams {
  /// 温度；小于 1e-6 时使用贪心采样，必须为有限非负数。
  float temperature = 0.0f;
  /// 候选 token 数；0 表示关闭 Top-k，1 表示贪心采样, 大于1采用top_K采样
  int top_k = 0;
  /// 随机种子；0 表示使用随机种子，非 0 可复现采样序列。
  unsigned int seed = 0;
};

// T = 1.0：保持原始 logits 的 Softmax 分布，不放大也不缩小差异。
// 0 < T < 1.0：低温，分布更尖锐，更容易选中最大 logit。
// T > 1.0：高温，分布更平坦，随机性更强。
// T 接近 0：接近贪心采样。

class MiniSampler {
public:
    /// @brief 使用 SamplingParams给采样参数赋值,初始化随机数种子
    /// @param params 采样参数
    explicit MiniSampler(const SamplingParams& params);

    /// @brief 根据参数自动选择贪心、温度或 Top-k 采样。
    /// @param logits 模型输出，按一维词表分数处理。
    /// @return 采样得到的 token id。
    /// @throws std::runtime_error 输入或参数非法时抛出。
    int Sample(const Tensor& logits);

private:
    /// @brief 返回 logit 最大的 token，logit相同时返回小索引。
    /// @param logits 模型输出 [vocab_size]
    /// @return 最大值对应的 token id
    /// @throws std::runtime_error logits 为空或包含非有限值时抛出。
    int SampleGreedy(const Tensor& logits);

    /**
     * @brief 按温度缩放后的 Softmax 概率分布随机采样 token。
     *
     * 首先使用 temperature 缩放 logits，然后取对数计算归一化概率，
     * 实际计算时减去最大的logit,防止数值溢出：
     *
     *   p_i = exp((logit_i - max_logit) / temperature) / sum(p)
     *
     * 较低的 temperature 会使概率分布更尖锐，采样结果更倾向于
     * 高 logit 的 token；较高的 temperature 会使分布更平坦，
     * 从而增加采样的随机性。
     *
     * 随后在 [0, 1) 上生成均匀随机数，然后遍历概率分布，依次累加各 token 的概率；
     * 当累计概率首次大于或等于该随机数时，返回对应的 token id。
     *
     * 当 temperature 小于贪心阈值时，直接返回 logit 最大的 token。
     *
     * @param logits 一维词表分数。
     * @param temperature 采样温度，必须为有限非负数。
     * @return 采样得到的 token id。
     * @throws std::runtime_error logits 或 temperature 非法时抛出。
     */
    int SampleTemperature(const Tensor& logits, float temperature);

    /**
     * @brief 在 logit 最高的 Top-k 个候选 token 中按温度随机采样。
     *
     * 首先按照 logit 从高到低排序，只保留前 top_k 个候选；
     * 当 logit 相同时，token id 较小的候选排在前面。
     * 未进入 Top-k 的 token 采样概率为 0。
     * 随后仅在 Top-k 候选中计算温度缩放后的 Softmax 概率
     */
    int SampleTopK(const Tensor& logits, float temperature, int top_k);

private:
    SamplingParams params_;
    std::mt19937 rng_;
};

}