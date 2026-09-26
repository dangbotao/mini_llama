#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "mini_llama/tensor.h"
#include "mini_llama/operations.h"

namespace mini_llama {

TEST(OpsTest, TestSoftMax) {
    Tensor input = MakeTensor1D(3);
    input[0] = 1000.0f;
    input[1] = 1001.0f;
    input[2] = 1002.0f;

    const Tensor output = SoftMax(input);

    ASSERT_EQ(output.shape, input.shape);
    ASSERT_EQ(output.size(), 3);

    // softmax([1000, 1001, 1002])
    // 等价于 softmax([-2, -1, 0])
    EXPECT_NEAR(output[0], 0.09003057f, 1e-6f);
    EXPECT_NEAR(output[1], 0.24472847f, 1e-6f);
    EXPECT_NEAR(output[2], 0.66524096f, 1e-6f);

    float probability_sum = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_GE(output[i], 0.0f);
        EXPECT_LE(output[i], 1.0f);
        probability_sum += output[i];
    }

    EXPECT_NEAR(probability_sum, 1.0f, 1e-6f);
}

TEST(OpsTest, TestRmsNorm) {
    Tensor input = MakeTensor1D(4);
    input[0] = 1.0f;
    input[1] = 2.0f;
    input[2] = 3.0f;
    input[3] = 4.0f;

    Tensor weight = MakeTensor1D(4);
    weight[0] = 1.0f;
    weight[1] = 0.5f;
    weight[2] = -1.0f;
    weight[3] = 2.0f;

    constexpr float eps = 1e-5f;
    const Tensor output = RmsNorm(input, weight, eps);

    ASSERT_EQ(output.shape, input.shape);
    ASSERT_EQ(output.size(), input.size());

    // mean(x^2) = (1 + 4 + 9 + 16) / 4 = 7.5
    const float scale = 1.0f / std::sqrt(7.5f + eps);
    EXPECT_NEAR(output[0], 1.0f * scale, 1e-6f);
    EXPECT_NEAR(output[1], 1.0f * scale, 1e-6f);
    EXPECT_NEAR(output[2], -3.0f * scale, 1e-6f);
    EXPECT_NEAR(output[3], 8.0f * scale, 1e-6f);
}

TEST(OpsTest, TestSilu) {
    Tensor input = MakeTensor1D(5);
    input[0] = -2.0f;
    input[1] = -1.0f;
    input[2] = 0.0f;
    input[3] = 1.0f;
    input[4] = 2.0f;

    const Tensor output = Silu(input);

    ASSERT_EQ(output.shape, input.shape);
    ASSERT_EQ(output.size(), input.size());

    // SiLU(x) = x / (1 + exp(-x)).
    EXPECT_NEAR(output[0], -0.23840584f, 1e-6f);
    EXPECT_NEAR(output[1], -0.26894142f, 1e-6f);
    EXPECT_NEAR(output[2], 0.0f, 1e-6f);
    EXPECT_NEAR(output[3], 0.73105858f, 1e-6f);
    EXPECT_NEAR(output[4], 1.76159416f, 1e-6f);
}

TEST(OpsTest, TestSwiGLU) {
    Tensor gate = MakeTensor1D(4);
    gate[0] = -2.0f;
    gate[1] = 0.0f;
    gate[2] = 2.0f;
    gate[3] = 1.0f;

    Tensor up = MakeTensor1D(4);
    up[0] = 3.0f;
    up[1] = -1.0f;
    up[2] = 0.5f;
    up[3] = -2.0f;

    const Tensor output = SwiGLU(gate, up);

    ASSERT_EQ(output.shape, gate.shape);
    ASSERT_EQ(output.size(), gate.size());

    // SwiGLU(gate, up) = SiLU(gate) * up, element by element.
    EXPECT_NEAR(output[0], -0.71521753f, 1e-6f);
    EXPECT_NEAR(output[1], 0.0f, 1e-6f);
    EXPECT_NEAR(output[2], 0.88079708f, 1e-6f);
    EXPECT_NEAR(output[3], -1.46211716f, 1e-6f);

    Tensor gate_2d = MakeTensor2D(2, 2);
    gate_2d[0] = -1.0f;
    gate_2d[1] = 0.0f;
    gate_2d[2] = 1.0f;
    gate_2d[3] = 2.0f;

    Tensor up_2d = MakeTensor2D(2, 2);
    up_2d[0] = 2.0f;
    up_2d[1] = 3.0f;
    up_2d[2] = -1.0f;
    up_2d[3] = 0.5f;

    const Tensor output_2d = SwiGLU(gate_2d, up_2d);

    ASSERT_EQ(output_2d.shape, (std::vector<int>{2, 2}));
    ASSERT_EQ(output_2d.size(), gate_2d.size());
    EXPECT_NEAR(output_2d[0], -0.53788284f, 1e-6f);
    EXPECT_NEAR(output_2d[1], 0.0f, 1e-6f);
    EXPECT_NEAR(output_2d[2], -0.73105858f, 1e-6f);
    EXPECT_NEAR(output_2d[3], 0.88079708f, 1e-6f);
}

TEST(OpsTest, TestRope) {
    constexpr int pos = 2;
    constexpr float theta = 10000.0f;

    Tensor input_q = MakeTensor2D(2, 4);
    input_q.data = {1.0f, 2.0f, 3.0f, 4.0f,
                    -1.0f, 0.5f, 2.0f, -3.0f};
    Tensor input_k = MakeTensor2D(1, 4);
    input_k.data = {0.25f, -0.75f, 1.5f, 2.5f};

    // 独立计算期望的旋转结果。
    auto MakeExpected = [](const Tensor& input, int position,
                           float base, RopeType type) {
        Tensor expected = input;
        const int heads = input.shape[0];
        const int head_dim = input.shape[1];

        for (int head = 0; head < heads; ++head) {
            for (int pair = 0; pair < head_dim / 2; ++pair) {
                int first = 0;
                int second = 0;
                if (type == RopeType::kNormal) {
                    first = pair * 2;
                    second = first + 1;
                } else {
                    first = pair;
                    second = pair + head_dim / 2;
                }

                const float freq = 1.0f / std::pow(
                    base, static_cast<float>(2 * pair) / head_dim);
                const float cos_value = std::cos(position * freq);
                const float sin_value = std::sin(position * freq);
                const int offset = head * head_dim;
                const float x0 = input[offset + first];
                const float x1 = input[offset + second];

                expected[offset + first] =
                    x0 * cos_value - x1 * sin_value;
                expected[offset + second] =
                    x0 * sin_value + x1 * cos_value;
            }
        }
        return expected;
    };

    auto ExpectTensorNear = [](const Tensor& actual,
                               const Tensor& expected) {
        ASSERT_EQ(actual.shape, expected.shape);
        ASSERT_EQ(actual.size(), expected.size());
        for (size_t i = 0; i < actual.size(); ++i) {
            EXPECT_NEAR(actual[i], expected[i], 1e-6f)
                << "flat index: " << i;
        }
    };

    // Normal 模式旋转相邻维度。
    const Tensor expected_normal_q =
        MakeExpected(input_q, pos, theta, RopeType::kNormal);
    const Tensor expected_normal_k =
        MakeExpected(input_k, pos, theta, RopeType::kNormal);
    Tensor normal_q = input_q;
    Tensor normal_k = input_k;
    Rope(normal_q, normal_k, pos, theta, RopeType::kNormal);

    ExpectTensorNear(normal_q, expected_normal_q);
    ExpectTensorNear(normal_k, expected_normal_k);

    // NeoX 模式旋转前后两个半区。
    const Tensor expected_neox_q =
        MakeExpected(input_q, pos, theta, RopeType::kNeox);
    const Tensor expected_neox_k =
        MakeExpected(input_k, pos, theta, RopeType::kNeox);
    Tensor neox_q = input_q;
    Tensor neox_k = input_k;
    Rope(neox_q, neox_k, pos, theta, RopeType::kNeox);

    ExpectTensorNear(neox_q, expected_neox_q);
    ExpectTensorNear(neox_k, expected_neox_k);

    // 位置pos 为 0 时不发生旋转。
    Tensor q_at_zero = MakeTensor2D(1, 4);
    q_at_zero.data = {1.0f, -2.0f, 3.0f, -4.0f};
    Tensor k_at_zero = MakeTensor2D(1, 4);
    k_at_zero.data = {-0.5f, 1.5f, -2.5f, 3.5f};
    const Tensor original_q = q_at_zero;
    const Tensor original_k = k_at_zero;

    Rope(q_at_zero, k_at_zero, 0, theta, RopeType::kNormal);

    EXPECT_EQ(q_at_zero.data, original_q.data);
    EXPECT_EQ(k_at_zero.data, original_k.data);

    // 非法形状、位置和 theta 必须抛出异常。
    Tensor one_dimensional = MakeTensor1D(4);
    EXPECT_THROW(
        Rope(one_dimensional, k_at_zero, 0, theta, RopeType::kNormal),
        std::runtime_error);
    EXPECT_THROW(
        Rope(q_at_zero, one_dimensional, 0, theta, RopeType::kNormal),
        std::runtime_error);
    EXPECT_THROW(
        Rope(q_at_zero, k_at_zero, -1, theta, RopeType::kNormal),
        std::out_of_range);
    EXPECT_THROW(
        Rope(q_at_zero, k_at_zero, 0, 0.0f, RopeType::kNormal),
        std::runtime_error);
    EXPECT_THROW(
        Rope(q_at_zero, k_at_zero, 0,
             std::numeric_limits<float>::infinity(), RopeType::kNormal),
        std::runtime_error);
}

}
