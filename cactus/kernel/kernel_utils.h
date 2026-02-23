#ifndef KERNEL_UTILS_H
#define KERNEL_UTILS_H

#include <arm_neon.h>
#if defined(__APPLE__)
#include <TargetConditionals.h>
#include <sys/sysctl.h>
#endif
#if defined(__ANDROID__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif
#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>
#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <unistd.h>
#include <unordered_map>
#include <chrono>
#include <string>
#include <cstdio>

constexpr size_t NEON_VECTOR_SIZE = 16;
constexpr size_t STREAMING_STORE_THRESHOLD = 32768;

inline void stream_store_f16x8(__fp16* dst, float16x8_t val) {
#if defined(__aarch64__)
    float16x4_t lo = vget_low_f16(val);
    float16x4_t hi = vget_high_f16(val);
    __asm__ __volatile__(
        "stnp %d0, %d1, [%2]"
        :
        : "w"(lo), "w"(hi), "r"(dst)
        : "memory"
    );
#else
    vst1q_f16(dst, val);
#endif
}


inline float32x4_t fast_exp_f32x4(float32x4_t x) {
    const float32x4_t log2e = vdupq_n_f32(1.4426950408889634f);
    const float32x4_t ln2 = vdupq_n_f32(0.6931471805599453f);

    const float32x4_t c0 = vdupq_n_f32(1.0f);
    const float32x4_t c1 = vdupq_n_f32(0.6931471805599453f); 
    const float32x4_t c2 = vdupq_n_f32(0.2402265069591007f);  
    const float32x4_t c3 = vdupq_n_f32(0.05550410866482158f);
    const float32x4_t c4 = vdupq_n_f32(0.009618129842071803f); 

    x = vmaxq_f32(x, vdupq_n_f32(-87.0f));
    x = vminq_f32(x, vdupq_n_f32(87.0f));

    float32x4_t z = vmulq_f32(x, log2e);

    int32x4_t zi = vcvtq_s32_f32(z);
    float32x4_t zf = vsubq_f32(z, vcvtq_f32_s32(zi));

    uint32x4_t neg_mask = vcltq_f32(zf, vdupq_n_f32(0.0f));
    zi = vsubq_s32(zi, vandq_s32(vreinterpretq_s32_u32(neg_mask), vdupq_n_s32(1)));
    zf = vaddq_f32(zf, vreinterpretq_f32_u32(vandq_u32(neg_mask, vreinterpretq_u32_f32(vdupq_n_f32(1.0f)))));

    float32x4_t zf_ln2 = vmulq_f32(zf, ln2);
    float32x4_t p = c4;
    p = vfmaq_f32(c3, p, zf_ln2);
    p = vfmaq_f32(c2, p, zf_ln2);
    p = vfmaq_f32(c1, p, zf_ln2);
    p = vfmaq_f32(c0, p, zf_ln2);

    int32x4_t exp_bits = vshlq_n_s32(vaddq_s32(zi, vdupq_n_s32(127)), 23);
    float32x4_t scale = vreinterpretq_f32_s32(exp_bits);

    return vmulq_f32(p, scale);
}

inline float32x4_t fast_tanh_f32x4(float32x4_t x) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t neg_one = vdupq_n_f32(-1.0f);

    uint32x4_t pos_sat = vcgtq_f32(x, vdupq_n_f32(4.5f));
    uint32x4_t neg_sat = vcltq_f32(x, vdupq_n_f32(-4.5f));

    const float32x4_t c27 = vdupq_n_f32(27.0f);
    const float32x4_t c9 = vdupq_n_f32(9.0f);

    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t num = vaddq_f32(c27, x2);   
    float32x4_t den = vfmaq_f32(c27, c9, x2);  

    float32x4_t result = vmulq_f32(x, vdivq_f32(num, den));

    result = vbslq_f32(pos_sat, one, result);
    result = vbslq_f32(neg_sat, neg_one, result);

    return result;
}

inline void unpack_int4_as_int8x16x2(const uint8_t* ptr, int8x16_t& high_decoded, int8x16_t& low_decoded) {
    int8x16_t packed = vreinterpretq_s8_u8(vld1q_u8(ptr));
    high_decoded = vshrq_n_s8(packed, 4);
    low_decoded = vshrq_n_s8(vshlq_n_s8(packed, 4), 4);
}

namespace CactusThreading {

    class ThreadPool {
    private:
        static constexpr size_t MAX_WORKERS = 16;

        std::vector<std::thread> workers;
        std::deque<std::function<void()>> tasks;

        std::mutex mutex;
        std::condition_variable work_available;
        std::condition_variable work_done;

        bool stop{false};
        std::atomic<size_t> pending_tasks{0};
        size_t num_workers_;

        void worker_thread() {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    work_available.wait(lock, [this] {
                        return stop || !tasks.empty();
                    });

                    if (stop && tasks.empty()) {
                        return;
                    }

                    task = std::move(tasks.front());
                    tasks.pop_front();
                }

                task();

                if (pending_tasks.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    std::lock_guard<std::mutex> lock(mutex);
                    work_done.notify_one();
                }
            }
        }

    public:
        explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency())
            : stop(false), pending_tasks(0) {
            num_workers_ = std::min(num_threads, MAX_WORKERS);
            if (num_workers_ == 0) num_workers_ = 1;
            workers.reserve(num_workers_);
            for (size_t i = 0; i < num_workers_; ++i) {
                workers.emplace_back(&ThreadPool::worker_thread, this);
            }
        }

        ~ThreadPool() {
            {
                std::lock_guard<std::mutex> lock(mutex);
                stop = true;
            }
            work_available.notify_all();
            for (auto& worker : workers) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
        }

        template<typename F>
        auto enqueue(F&& f) -> std::future<decltype(f())> {
            using return_type = decltype(f());

            auto task = std::make_shared<std::packaged_task<return_type()>>(
                std::forward<F>(f)
            );

            std::future<return_type> res = task->get_future();

            {
                std::lock_guard<std::mutex> lock(mutex);
                pending_tasks.fetch_add(1, std::memory_order_relaxed);
                tasks.emplace_back([task](){ (*task)(); });
            }
            work_available.notify_one();

            return res;
        }

        template<typename F>
        void enqueue_batch(size_t total_work, F task_func) {
            if (total_work == 0) return;

            const size_t num_tasks = std::min(num_workers_, total_work);
            const size_t per_worker = total_work / num_tasks;
            const size_t remainder = total_work % num_tasks;

            {
                std::lock_guard<std::mutex> lock(mutex);
                pending_tasks.fetch_add(num_tasks, std::memory_order_relaxed);

                for (size_t w = 0; w < num_tasks; ++w) {
                    size_t start = w * per_worker + std::min(w, remainder);
                    size_t end = start + per_worker + (w < remainder ? 1 : 0);
                    tasks.emplace_back([=]() { task_func(start, end); });
                }
            }
            work_available.notify_all();
        }

        void wait_all() {
            std::unique_lock<std::mutex> lock(mutex);
            work_done.wait(lock, [this] {
                return pending_tasks.load(std::memory_order_acquire) == 0;
            });
        }

        template<typename F>
        void enqueue_n_threads(size_t total_work, size_t num_threads, F task_func) {
            if (total_work == 0 || num_threads == 0) return;

            num_threads = std::min(num_threads, std::min(num_workers_, total_work));
            const size_t per_thread = total_work / num_threads;
            const size_t remainder = total_work % num_threads;

            {
                std::lock_guard<std::mutex> lock(mutex);
                pending_tasks.fetch_add(num_threads, std::memory_order_relaxed);

                for (size_t t = 0; t < num_threads; ++t) {
                    size_t start = t * per_thread + std::min(t, remainder);
                    size_t end = start + per_thread + (t < remainder ? 1 : 0);
                    tasks.emplace_back([=]() { task_func(start, end); });
                }
            }
            work_available.notify_all();
        }

        size_t num_workers() const { return num_workers_; }
    };

    inline ThreadPool& get_thread_pool() {
        static ThreadPool pool;
        return pool;
    }
    
    struct ParallelConfig {
        size_t min_work_gate;  
        size_t work_per_thread; 

        constexpr ParallelConfig(size_t gate, size_t per_thread)
            : min_work_gate(gate), work_per_thread(per_thread) {}
    };

    inline size_t get_optimal_thread_count(size_t total_work, ParallelConfig config) {
        if (total_work < config.min_work_gate) return 1;

        size_t pool_size = get_thread_pool().num_workers();
        size_t num_threads = (total_work + config.work_per_thread - 1) / config.work_per_thread;
        return std::min(pool_size, std::max(static_cast<size_t>(1), num_threads));
    }

    struct Thresholds {
        #if defined(__ANDROID__)
        static constexpr ParallelConfig ATTENTION{64, 32};
        static constexpr ParallelConfig ELEMENT_WISE{5000, 2500};
        static constexpr ParallelConfig AXIS_REDUCE{1000, 500};
        static constexpr ParallelConfig ALL_REDUCE{10000, 5000};
        static constexpr ParallelConfig SCALAR_BASIC{30000, 15000};
        static constexpr ParallelConfig SCALAR_EXPENSIVE{10000, 5000};
        #else // Apple
        static constexpr ParallelConfig ATTENTION{32, 16};
        static constexpr ParallelConfig ELEMENT_WISE{5000, 2500};
        static constexpr ParallelConfig AXIS_REDUCE{1000, 500};
        static constexpr ParallelConfig ALL_REDUCE{10000, 5000};
        static constexpr ParallelConfig SCALAR_BASIC{5000, 2500};
        static constexpr ParallelConfig SCALAR_EXPENSIVE{2500, 1250};
        #endif
    };

    struct GemmThreading {
        #if defined(__ANDROID__)
        static size_t get_num_threads(size_t M, size_t pool_size) {
            if (M <= 1) return 1;
            return pool_size;
        }
        static size_t get_gemv_threads(size_t /*N_blocks*/, size_t /*pool_size*/) {
            return 1; 
        }
        #elif defined(__APPLE__) && TARGET_OS_IPHONE
        static constexpr size_t GEMV_MIN_N_BLOCKS = 512; 
        static size_t get_num_threads(size_t M, size_t pool_size) {
            if (M <= 1) return std::min(pool_size, static_cast<size_t>(2));
            return pool_size;
        }
        static size_t get_gemv_threads(size_t N_blocks, size_t pool_size) {
            if (N_blocks < GEMV_MIN_N_BLOCKS) return 1;
            return std::min(pool_size, static_cast<size_t>(2));
        }
        #else 
        static constexpr size_t GEMV_MIN_N_BLOCKS = 256;  
        static size_t get_num_threads(size_t M, size_t pool_size) {
            if (M <= 1) return std::min(pool_size, static_cast<size_t>(4));
            return pool_size;
        }
        static size_t get_gemv_threads(size_t N_blocks, size_t pool_size) {
            if (N_blocks < GEMV_MIN_N_BLOCKS) return 1;
            if (N_blocks < 512) return std::min(pool_size, static_cast<size_t>(2));
            return std::min(pool_size, static_cast<size_t>(4));
        }
        #endif
    };

    inline size_t& get_gemm_thread_override() {
        static size_t override_threads = 0; 
        return override_threads;
    }

    inline void set_gemm_threads(size_t num_threads) {
        get_gemm_thread_override() = num_threads;
    }

    inline void reset_gemm_threads() {
        get_gemm_thread_override() = 0;
    }
    
    class TaskHandle {
    private:
        std::vector<std::future<void>> futures_;
        bool auto_wait_;
        
    public:
        TaskHandle(bool auto_wait = true) : auto_wait_(auto_wait) {}
        
        ~TaskHandle() {
            if (auto_wait_) {
                wait();
            }
        }
        
        TaskHandle(TaskHandle&&) = default;
        TaskHandle& operator=(TaskHandle&&) = default;
        TaskHandle(const TaskHandle&) = delete;
        TaskHandle& operator=(const TaskHandle&) = delete;
        
        void add_future(std::future<void>&& f) {
            futures_.push_back(std::move(f));
        }
        
        void wait() {
            for (auto& f : futures_) {
                if (f.valid()) {
                    f.wait();
                }
            }
            futures_.clear();
        }
        
        bool is_ready() const {
            for (const auto& f : futures_) {
                if (f.valid() && f.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                    return false;
                }
            }
            return true;
        }
        
        size_t task_count() const { return futures_.size(); }
    };
    
    template<typename WorkFunc>
    TaskHandle parallel_for(size_t total_work, ParallelConfig config, WorkFunc work_func, bool wait = true) {
        const size_t num_threads = get_optimal_thread_count(total_work, config);
        TaskHandle handle(!wait);

        if (num_threads == 1) {
            if (wait) {
                work_func(0, total_work);
                return handle;
            }
            auto& pool = get_thread_pool();
            handle.add_future(pool.enqueue([work_func, total_work]() {
                work_func(0, total_work);
            }));
            return handle;
        }

        auto& pool = get_thread_pool();
        const size_t work_per_thread = total_work / num_threads;

        for (size_t t = 0; t < num_threads; ++t) {
            handle.add_future(pool.enqueue([work_func, t, num_threads, work_per_thread, total_work]() {
                const size_t start_idx = t * work_per_thread;
                const size_t end_idx = (t == num_threads - 1) ? total_work : (t + 1) * work_per_thread;
                work_func(start_idx, end_idx);
            }));
        }

        if (wait) {
            handle.wait();
        }
        return handle;
    }

    template<typename WorkFunc>
    void parallel_for_2d(size_t outer_size, size_t inner_size, ParallelConfig config, WorkFunc work_func) {
        const size_t total_work = outer_size * inner_size;
        parallel_for(total_work, config, [&](size_t start_idx, size_t end_idx) {
            for (size_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
                const size_t outer = work_idx / inner_size;
                const size_t inner = work_idx % inner_size;
                work_func(outer, inner);
            }
        });
    }

    template<typename WorkFunc, typename ResultType, typename CombineFunc>
    ResultType parallel_reduce(size_t total_work, ParallelConfig config,
                              WorkFunc work_func, ResultType init_value, CombineFunc combine_func) {
        const size_t num_threads = get_optimal_thread_count(total_work, config);
        
        if (num_threads == 1) {
            return work_func(0, total_work);
        }
        
        auto& pool = get_thread_pool();
        std::vector<std::future<ResultType>> futures;
        std::vector<ResultType> partial_results(num_threads, init_value);
        const size_t work_per_thread = total_work / num_threads;
        
        for (size_t t = 0; t < num_threads; ++t) {
            futures.push_back(pool.enqueue([work_func, t, num_threads, work_per_thread, total_work]() -> ResultType {
                const size_t start_idx = t * work_per_thread;
                const size_t end_idx = (t == num_threads - 1) ? total_work : (t + 1) * work_per_thread;
                return work_func(start_idx, end_idx);
            }));
        }
        
        ResultType result = init_value;
        for (auto& future : futures) {
            result = combine_func(result, future.get());
        }
        return result;
    }

    template<typename WorkFunc>
    void parallel_gemm_tiles(size_t M, size_t total_tiles, WorkFunc work_func) {
        auto& pool = get_thread_pool();

        size_t override = get_gemm_thread_override();
        size_t num_threads = (override > 0) ? override : GemmThreading::get_num_threads(M, pool.num_workers());
        num_threads = std::min(num_threads, total_tiles);

        if (num_threads <= 1) {
            work_func(0, total_tiles);
            return;
        }

        pool.enqueue_n_threads(total_tiles, num_threads, work_func);
        pool.wait_all();
    }

}


#endif // KERNEL_UTILS_H 