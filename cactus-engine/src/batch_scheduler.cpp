#include "../cactus_engine.h"
#include "engine.h"
#include "utils.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using cactus::engine::Model;

namespace {

struct BatchRequest {
    uint64_t id;
    std::vector<uint32_t> prompt;
    size_t max_tokens;
    cactus_token_callback callback;
    void* user_data;
};

class BatchScheduler {
public:
    BatchScheduler(Model* model, size_t max_slots)
        : model_(model), max_slots_(max_slots > 0 ? max_slots : 1) {
        worker_ = std::thread([this] { run(); });
    }
    ~BatchScheduler() { stop(); }

    uint64_t submit(std::vector<uint32_t> prompt, size_t max_tokens,
                    cactus_token_callback callback, void* user_data) {
        uint64_t id = next_id_.fetch_add(1);
        {
            std::lock_guard<std::mutex> lock(mu_);
            queue_.push_back({id, std::move(prompt), max_tokens, callback, user_data});
        }
        cv_.notify_one();
        return id;
    }

    void cancel(uint64_t id) {
        std::lock_guard<std::mutex> lock(mu_);
        cancelled_.insert(id);
    }

    void stop() {
        if (stopped_.exchange(true)) return;
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_flag_ = true;
        }
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

private:
    struct Active {
        BatchRequest req;
        size_t fed = 0;
        size_t generated = 0;
        uint32_t last = 0;
        bool finished = false;
    };

    static void deliver_done(std::vector<Active>& active) {
        for (auto& a : active)
            if (a.req.callback) a.req.callback(nullptr, 0, a.req.user_data);
    }

    void run() {
        auto* tokenizer = model_->get_tokenizer();
        const size_t cap = std::min(max_slots_, model_->batch_slot_capacity());
        const std::vector<uint32_t> stop_ids = model_->batch_stop_token_ids();
        auto is_stop = [&](uint32_t t) {
            for (uint32_t s : stop_ids) if (s == t) return true;
            return false;
        };

        std::vector<Active> active;
        while (true) {
            std::vector<BatchRequest> admitted;
            {
                std::unique_lock<std::mutex> lock(mu_);
                if (stop_flag_) break;
                if (active.empty() && queue_.empty()) {
                    cv_.wait(lock, [this] { return stop_flag_ || !queue_.empty(); });
                    if (stop_flag_) break;
                }
                for (auto& a : active)
                    if (cancelled_.count(a.req.id)) a.finished = true;
                while (!queue_.empty() && active.size() + admitted.size() < cap) {
                    BatchRequest req = std::move(queue_.front());
                    queue_.pop_front();
                    if (cancelled_.count(req.id) || req.prompt.empty()) continue;
                    admitted.push_back(std::move(req));
                }
            }
            for (auto& req : admitted) {
                model_->reset_decode_slot(active.size());
                active.push_back(Active{std::move(req), 0, 0, 0, false});
            }
            if (active.empty()) continue;

            const size_t K = active.size();
            std::vector<uint32_t> tokens(K);
            std::vector<size_t> positions(K);
            for (size_t i = 0; i < K; ++i) {
                positions[i] = active[i].fed;
                tokens[i] = (active[i].fed < active[i].req.prompt.size())
                                ? active[i].req.prompt[active[i].fed]
                                : active[i].last;
            }
            std::vector<uint32_t> sampled = model_->decode_step_batch(tokens, positions);
            if (sampled.size() != K) {
                deliver_done(active);
                active.clear();
                continue;
            }

            for (size_t i = 0; i < K; ++i) {
                Active& a = active[i];
                ++a.fed;
                if (a.finished || a.fed < a.req.prompt.size()) continue;
                uint32_t tok = sampled[i];
                a.last = tok;
                if (is_stop(tok)) { a.finished = true; continue; }
                if (a.req.callback) {
                    std::string piece = tokenizer ? tokenizer->decode({tok}) : std::string();
                    a.req.callback(piece.c_str(), tok, a.req.user_data);
                }
                if (++a.generated >= a.req.max_tokens) a.finished = true;
            }

            for (size_t i = 0; i < active.size();) {
                if (!active[i].finished) { ++i; continue; }
                if (active[i].req.callback) active[i].req.callback(nullptr, 0, active[i].req.user_data);
                size_t last = active.size() - 1;
                if (i != last) {
                    model_->move_decode_slot(i, last);
                    active[i] = std::move(active[last]);
                }
                active.pop_back();
            }
        }
        deliver_done(active);
    }

    Model* model_;
    size_t max_slots_;
    std::deque<BatchRequest> queue_;
    std::unordered_set<uint64_t> cancelled_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_flag_ = false;
    std::atomic<bool> stopped_{false};
    std::atomic<uint64_t> next_id_{1};
    std::thread worker_;
};

std::mutex g_scheduler_mutex;
std::unordered_map<cactus_model_t, std::unique_ptr<BatchScheduler>> g_schedulers;

Model* model_from_handle(cactus_model_t model) {
    return model ? static_cast<CactusModelHandle*>(model)->model.get() : nullptr;
}

}

extern "C" {

int cactus_batch_start(cactus_model_t model, size_t max_slots) {
    Model* m = model_from_handle(model);
    if (!m) return -1;
    std::lock_guard<std::mutex> lock(g_scheduler_mutex);
    if (g_schedulers.count(model)) return 0;
    g_schedulers[model] = std::make_unique<BatchScheduler>(m, max_slots);
    return 0;
}

uint64_t cactus_submit(cactus_model_t model, const uint32_t* prompt_tokens, size_t num_tokens,
                       size_t max_tokens, cactus_token_callback callback, void* user_data) {
    if (!prompt_tokens || num_tokens == 0) return 0;
    std::lock_guard<std::mutex> lock(g_scheduler_mutex);
    auto it = g_schedulers.find(model);
    if (it == g_schedulers.end()) return 0;
    std::vector<uint32_t> prompt(prompt_tokens, prompt_tokens + num_tokens);
    return it->second->submit(std::move(prompt), max_tokens ? max_tokens : 64, callback, user_data);
}

int cactus_cancel(cactus_model_t model, uint64_t request_id) {
    std::lock_guard<std::mutex> lock(g_scheduler_mutex);
    auto it = g_schedulers.find(model);
    if (it == g_schedulers.end()) return -1;
    it->second->cancel(request_id);
    return 0;
}

void cactus_batch_stop(cactus_model_t model) {
    std::unique_ptr<BatchScheduler> scheduler;
    {
        std::lock_guard<std::mutex> lock(g_scheduler_mutex);
        auto it = g_schedulers.find(model);
        if (it == g_schedulers.end()) return;
        scheduler = std::move(it->second);
        g_schedulers.erase(it);
    }
    scheduler->stop();
}

}
