
#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/batch.h"
#include "mini_llama/chat.h"
#include "mini_llama/context.h"
#include "mini_llama/forward.h"
#include "mini_llama/gguf.h"
#include "mini_llama/gguf_loader.h"
#include "mini_llama/gguf_tokenizer.h"
#include "mini_llama/model.h"
#include "mini_llama/prompt_builder.h"
#include "mini_llama/sampler.h"
#include "mini_llama/terminal.h"
#include "mini_llama/thread_pool.h"
#include "mini_llama/tokenizer.h"

namespace mini_llama {

struct LoadedModel {
    MiniLlamaModel model;                  // 模型结构和权重
    std::unique_ptr<ITokenizer> tokenizer; // 分词器
    std::string chat_template;             // 聊天模板
};

static LoadedModel LoadModelAndTokenizer(const std::string& gguf_path) {
  LoadedModel result;

  GgufReader reader;
  if (!reader.Load(gguf_path)) {
    result.model.load_error = "Failed to read GGUF: " + reader.load_error;
    return result;
  }

  GgufModelLoader model_loader(reader);
  result.model = model_loader.Load();

  if (!result.model.loaded) {
    return result;
  }

  result.tokenizer = CreateGgufTokenizer(reader);

  if (!result.tokenizer) {
    result.model.loaded = false;
    result.model.load_error = "Failed to load tokenizer from GGUF metadata";
    return result;
  }

  if (result.tokenizer->vocab_size() != result.model.config.vocab_size) {
    result.model.loaded = false;
    result.model.load_error =
        "GGUF model/tokenizer vocabulary size mismatch";
    result.tokenizer.reset();
    return result;
  }

  if (!reader.GetMetadataString("tokenizer.chat_template",
                                result.chat_template)) {
    std::string architecture;
    if (reader.GetMetadataString("general.architecture", architecture) &&
        architecture == "qwen2") {
      result.chat_template = "qwen2";
    }
  }

  return result;
}

// 把模型生成的 token 实时解码并流式输出到终端，每次只输出新增的token
static void StreamDecodedText(const ITokenizer& tokenizer,
                              const std::vector<int>& generated_tokens,
                              std::string& streamed_reply,
                              const Terminal& term) {
  std::string current_reply = tokenizer.Decode(generated_tokens);
  if (current_reply.size() <= streamed_reply.size()) {
    return;
  }
 // substr(pos) 表示：从字符串下标 pos 开始，一直截取到字符串末尾
  term.PrintTokenText(current_reply.substr(streamed_reply.size()));
  term.Flush();
  streamed_reply = current_reply;
}

static int RunChat(const std::string& gguf_path) {

    int max_response_each_round = 8;

    // Use all available logical CPUs for matrix and quantized linear kernels.
    SetThreadCount(0);

    const LoadedModel& lm = LoadModelAndTokenizer(gguf_path);
    const MiniLlamaModel& model = lm.model;
    const std::unique_ptr<ITokenizer>& tokenizer = lm.tokenizer;

    if (!lm.model.loaded) {
        std::cerr << "Failed to load model: " << lm.model.load_error << "\n";
        return 1;
    }
    if (!lm.tokenizer) {
        std::cerr << "Failed to load tokenizer.\n";
        return 1;
    }
    // --------- Create PromptBuilder, Terminal and ChatSession ----------------

    PromptBuilder builder; // 按照模型要求对prompt进行格式化
    if (!lm.chat_template.empty()) {
        builder.SetChatTemplate(lm.chat_template);
    }
    Terminal term;
    ChatSession session;

    MiniLlamaContext ctx(&model);
    SamplingParams sampler_param = {};
    MiniSampler sampler(sampler_param);

    term.PrintMessage("Type /help for commands, /exit to quit.");

    while (true) {
        term.PrintUserPrompt();
        std::string input = term.ReadLine();
        // 输入为空且获取到结束信号（Ctrl+D)
        if (input.empty() && std::cin.eof()) {
            break;
        }
        if (input.empty()) {
            continue;
        }

        if (input == "/help") {
            term.PrintHelp();
            continue;
        }
        if (input == "/exit") {
            break;
        }
        if (input == "/clear") {
            session.Clear();
            ctx = MiniLlamaContext(&model);
            term.PrintMessage("Chat history cleared.");
            continue;
        }
        if (input.front() == '/') {
            term.PrintMessage("Unknown command: " + input);
            continue;
        }

        auto candidate_messages = session.GetMessages();
        candidate_messages.push_back({"user", input});

        try {
            std::string prompt = builder.Build(candidate_messages);
            std::vector<int> tokens = tokenizer->Encode(prompt);

            if (tokens.empty()) {
                term.PrintMessage("Tokenizer produced an empty prompt.");
                continue;
            }
            if (tokens.size() >=
                static_cast<std::size_t>(model.config.max_seq_len)) {
                term.PrintMessage(
                    "Prompt is too long. Use /clear or enter a shorter message.");
                continue;
            }

            // 当前每次用户提问时，程序都会把完整的历史对话重新拼成 Prompt，并重新执行一次 Prefill 推理
            // 后续优化为复用历史 Prompt 对应的 KV Cache，只计算本轮新增内容，以提高推理速度
            // The baseline implementation recomputes the complete prompt for
            // every turn. Prefix/KV reuse can be added after this path is
            // covered by integration tests.
            ctx = MiniLlamaContext(&model);
            MiniBatch prefill_batch =
                MiniBatch::BuildMultiTokensBatch(tokens);

            Tensor logits = ForwardBatch(ctx, model, prefill_batch);

            int max_response_all =
                model.config.max_seq_len - static_cast<int>(tokens.size());
            int response_len =
                std::min(max_response_all, max_response_each_round);

            std::vector<int> generated_tokens;
            std::string streamed_reply;

            term.PrintAssistantPrefix();
            for (int i = 0; i < response_len; ++i) {
                int next_token = sampler.Sample(logits);
                if (next_token == tokenizer->eos_id()) {
                    break;
                }
                tokens.push_back(next_token);
                generated_tokens.push_back(next_token);

                StreamDecodedText(
                    *tokenizer, generated_tokens, streamed_reply, term);

                MiniBatch decode_batch =
                    MiniBatch::BuildSingleTokenBatch(
                        next_token, static_cast<int>(tokens.size() - 1));

                logits = ForwardBatch(ctx, model, decode_batch);
            }

            std::string assistant_reply = tokenizer->Decode(generated_tokens);
            // Commit the turn only after prompt construction, inference and
            // decoding have all completed successfully.
            session.AddMessage("user", input);
            session.AddMessage("assistant", assistant_reply);
            term.NewLine();
        } catch (const std::exception& e) {
            term.NewLine();
            term.PrintMessage("Inference error: " + std::string(e.what()));
            ctx = MiniLlamaContext(&model);
            continue;
        }
    }

    term.PrintMessage("Goodbye.");
    return 0;
}

}  // namespace mini_llama

int main(int argc, char** argv) {
    if (argc > 2) {
      std::cerr << "Usage: " << argv[0] << " [model.gguf]\n";
      return 1;
    }
    const std::string gguf_path =
        argc == 2 ? argv[1] :
                    "models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf";
    return mini_llama::RunChat(gguf_path);
}
