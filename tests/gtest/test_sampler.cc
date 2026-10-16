#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

#include "mini_llama/sampler.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

// TestSamplerGreedy：验证贪心采样选择最大 logit，并在最大值并列时返回较小的 token id。

// TestSamplerTemperatureZeroIsGreedy：验证 temperature 为 0 时始终退化为贪心采样。

// TestSamplerTopKOneIsGreedy：验证 top_k == 1 时等价于贪心采样。

// TestSamplerTemperatureReproducible：验证普通温度采样使用相同 seed 时能产生相同的 token 序列。

// TestSamplerTopKReproducible：验证 Top-k 采样使用相同 seed 时能产生相同的 token 序列。

// TestSamplerDifferentSeedsDiffer：验证不同 seed 通常会产生不同的采样序列。

// TestSamplerTopKLimitsCandidates：验证 Top-k 采样只会从 logit 最高的 K 个 token 中选择结果。

// TestSamplerDefaultParamsAreGreedy：验证默认 SamplingParams 使用贪心采样。

// TestSamplerRejectsNegativeTopK：验证构造采样器时会拒绝负数 top_k。

// TestSamplerRejectsNanTemperature：验证构造采样器时会拒绝值为 NaN 的 temperature。

// TestSamplerRejectsEmptyLogits：验证采样器会拒绝空 logits。

// TestSamplerRejectsNonfiniteLogits：验证采样器会拒绝包含无穷值等非有限数值的 logits。

// TestSamplerTopKLargerThanVocabIsClamped：验证 top_k 大于词表大小时会被限制为词表大小，并返回合法 token id。

// TestSamplerTopKTiesKeepExactK：验证 Top-k 边界存在相同 logit 时仍只保留准确的 K 个候选，并优先保留 token id 较小的候选。

// Greedy
TEST(SamplerTest, TestSamplerGreedy) {
    Tensor logits({5}, 0.0f);
    logits[0] = 1.0f;
    logits[1] = 5.0f;
    logits[2] = 3.0f;
    logits[3] = 5.0f;
    logits[4] = 2.0f;

    SamplingParams params;
    params.seed = 42;
    MiniSampler sampler(params);

    // 最大值并列时应返回较小的 token id。
    EXPECT_EQ(sampler.Sample(logits), 1);
}

// SampleTemperature：temp极小时等价于Greedy
TEST(SamplerTest, TestSamplerTemperatureZeroIsGreedy) {
    Tensor logits({5}, 0.0f);
    logits[0] = 0.1f;
    logits[1] = 2.0f;
    logits[2] = 1.0f;
    logits[3] = 0.5f;
    logits[4] = 0.2f;

    SamplingParams params;
    params.temperature = 0.0f;
    params.top_k = 0;
    params.seed = 42;
    MiniSampler sampler(params);

    for (int i = 0; i < 20; ++i) {
        EXPECT_EQ(sampler.Sample(logits), 1);
    }
}

TEST(SamplerTest, TestSamplerTopKOneIsGreedy) {
    Tensor logits({5}, 0.0f);
    logits[0] = 0.1f;
    logits[1] = 2.0f;
    logits[2] = 1.0f;
    logits[3] = 0.5f;
    logits[4] = 0.2f;

    SamplingParams params;
    params.temperature = 1.0f;
    params.top_k = 1; // top_k为1时等价于Greedy
    params.seed = 42;
    MiniSampler sampler(params);

    for (int i = 0; i < 20; ++i) {
        EXPECT_EQ(sampler.Sample(logits), 1);
    }
}

TEST(SamplerTest, TestSamplerTemperatureReproducible) {
    Tensor logits({10}, 0.0f);
    for (int i = 0; i < 10; ++i) {
        logits[i] = static_cast<float>(i) * 0.5f;
    }

    SamplingParams params;
    params.temperature = 0.8f;
    params.top_k = 0;
    params.seed = 123;
    MiniSampler sampler1(params);
    MiniSampler sampler2(params);

    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(sampler1.Sample(logits), sampler2.Sample(logits));
    }
}

// 验证 Top-k 采样使用相同 seed 时能产生相同的 token 序列
TEST(SamplerTest, TestSamplerTopKReproducible) {
    Tensor logits({10}, 0.0f);
    for (int i = 0; i < 10; ++i) {
        logits[i] = static_cast<float>(i) * 0.5f;
    }

    SamplingParams params;
    params.temperature = 0.8f;
    params.top_k = 3;
    params.seed = 456;
    MiniSampler sampler1(params);
    MiniSampler sampler2(params);

    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(sampler1.Sample(logits), sampler2.Sample(logits));
    }
}

// 使用不同随机种子创建的两个采样器，不应该总是产生几乎完全相同的采样序列
TEST(SamplerTest, TestSamplerDifferentSeedsDiffer) {
    Tensor logits({10}, 0.0f);
    for (int i = 0; i < 10; ++i) {
        logits[i] = static_cast<float>(i) * 0.3f;
    }

    SamplingParams params1;
    params1.temperature = 1.0f;
    params1.seed = 1;
    SamplingParams params2 = params1;
    params2.seed = 2;
    MiniSampler sampler1(params1);
    MiniSampler sampler2(params2);

    int same_count = 0;
    for (int i = 0; i < 20; ++i) {
        if (sampler1.Sample(logits) == sampler2.Sample(logits)) {
            ++same_count;
        }
    }

    EXPECT_LT(same_count, 18); // 采样到完全相等的值的次数小于18
}

// 验证 Top-k 采样只会从 logit 最高的 K 个 token 中选择结果
TEST(SamplerTest, TestSamplerTopKLimitsCandidates) {
    Tensor logits({10}, 0.0f);
    for (int i = 0; i < 10; ++i) {
        logits[i] = static_cast<float>(i);
    }

    SamplingParams params;
    params.temperature = 1.0f; // 于标准、中等温度
    params.top_k = 2;
    params.seed = 789;
    MiniSampler sampler(params);

    for (int i = 0; i < 50; ++i) {
        const int idx = sampler.Sample(logits);
        EXPECT_TRUE(idx == 8 || idx == 9);
    }
}

// 验证默认 SamplingParams 使用贪心采样
TEST(SamplerTest, TestSamplerDefaultParamsAreGreedy) {
    Tensor logits({5}, 0.0f);
    logits[0] = 0.1f;
    logits[1] = 2.0f;
    logits[2] = 1.0f;
    logits[3] = 0.5f;
    logits[4] = 0.2f;

    SamplingParams params;
    MiniSampler sampler(params);

    EXPECT_EQ(sampler.Sample(logits), 1);
}

// top_k不为负
TEST(SamplerTest, TestSamplerRejectsNegativeTopK) {
    SamplingParams params;
    params.temperature = 1.0f;
    params.top_k = -1;

    EXPECT_THROW(
        {
            MiniSampler sampler(params);
        },
        std::runtime_error);
}

TEST(SamplerTest, TestSamplerRejectsNanTemperature) {
    SamplingParams params;
    params.temperature = std::numeric_limits<float>::quiet_NaN();

    EXPECT_THROW(
        {
            MiniSampler sampler(params);
        },
        std::runtime_error);
}

TEST(SamplerTest, TestSamplerRejectsEmptyLogits) {
    Tensor logits;
    SamplingParams params;
    MiniSampler sampler(params);

    EXPECT_THROW(sampler.Sample(logits), std::runtime_error);
}

TEST(SamplerTest, TestSamplerRejectsNonfiniteLogits) {
    Tensor logits({3}, 0.0f);
    logits[0] = 0.1f;
    logits[1] = std::numeric_limits<float>::infinity();
    logits[2] = 0.3f;

    SamplingParams params;
    params.temperature = 1.0f;
    MiniSampler sampler(params);

    EXPECT_THROW(sampler.Sample(logits), std::runtime_error);
}

TEST(SamplerTest, TestSamplerTopKLargerThanVocabIsClamped) {
    Tensor logits({4}, 0.0f);
    logits[0] = 0.0f;
    logits[1] = 1.0f;
    logits[2] = 2.0f;
    logits[3] = 3.0f;

    SamplingParams params;
    params.temperature = 1.0f;
    params.top_k = 100;
    params.seed = 42;
    MiniSampler sampler(params);

    for (int i = 0; i < 20; ++i) {
        const int idx = sampler.Sample(logits);
        EXPECT_GE(idx, 0);
        EXPECT_LT(idx, 4);
    }
}

TEST(SamplerTest, TestSamplerTopKTiesKeepExactK) {
    Tensor logits({5}, 0.0f);
    logits[0] = 5.0f;
    logits[1] = 5.0f;
    logits[2] = 5.0f;
    logits[3] = 1.0f;
    logits[4] = 0.0f;

    SamplingParams params;
    params.temperature = 1.0f;
    params.top_k = 2;
    params.seed = 321;
    MiniSampler sampler(params);

    for (int i = 0; i < 40; ++i) {
        const int idx = sampler.Sample(logits);
        EXPECT_TRUE(idx == 0 || idx == 1);
    }
}

}  // namespace mini_llama
