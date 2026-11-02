#include "mini_llama/thread_pool.h"

#include <stdexcept>
#include <thread>
#include <algorithm>
#include <vector>

namespace mini_llama {

static int g_thread_count = 0;

void SetThreadCount(int n) {
    if (n < 0) {
        throw std::runtime_error("thread count can not be negative");
    }
    g_thread_count = (n == 0) ? std::thread::hardware_concurrency() : n;
    g_thread_count = (g_thread_count > 0) ? g_thread_count : 4;
}

int GetThreadCount() {
    return g_thread_count;
}

void ParallelFor(int task_nums, const std::function<void(int, int)>& fn) {
    if (task_nums <= 0) return;
    int thread_count = GetThreadCount();
    if (thread_count == 0) {
        thread_count = 1;
    }
    thread_count = std::min(task_nums, thread_count);

    int tasks_per_thread = task_nums / thread_count;
    int remain_tasks = task_nums % thread_count;

    std::vector<std::thread> thread_vector;
    thread_vector.reserve(thread_count);

    int task_start = 0, task_end = 0;
    for (int i = 0; i < thread_count; ++i) {
        task_end = task_start + tasks_per_thread;
        if (i < remain_tasks) {
            task_end += 1;
        }
        thread_vector.emplace_back(fn, task_start, task_end);
        task_start = task_end;
    }

    for (std::thread& worker : thread_vector) {
        worker.join();
    }
}

}  // namespace mini_llama
