#ifndef KERNEL_UTILS_H
#define KERNEL_UTILS_H

#include <arm_neon.h>
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

inline int8_t clamp_to_int8(float value) {
    int32_t clamped = static_cast<int32_t>(roundf(value));
    return static_cast<int8_t>(std::max(-128, std::min(127, clamped)));
}

inline int8_t clamp_to_int8(int32_t value) {
    return static_cast<int8_t>(std::max(-128, std::min(127, value)));
}

#if defined(__ARM_FEATURE_DOTPROD)
inline int32x4_t accum_i8mm(int32x4_t acc, int8x16_t a, int8x16_t b) {
    return vdotq_s32(acc, a, b);
}
#else
inline int32x4_t accum_i8mm(int32x4_t acc, int8x16_t a, int8x16_t b) {
    int16x8_t prod_low = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    int32x4_t acc_high = vpaddlq_s16(vmull_s8(vget_high_s8(a), vget_high_s8(b)));
    return vaddq_s32(vaddq_s32(acc, vpaddlq_s16(prod_low)), acc_high);
}
#endif

inline float16x8_t accum_f16_dot(float16x8_t acc, float16x8_t a_low, float16x8_t a_high, 
                                 float16x8_t b_low, float16x8_t b_high) {
    acc = vfmaq_f16(acc, a_low, b_low);
    return vfmaq_f16(acc, a_high, b_high);
}

inline float32x4_t accum_f32_dot(float32x4_t acc, float32x4_t a_low, float32x4_t a_high, 
                                  float32x4_t b_low, float32x4_t b_high) {
    acc = vfmaq_f32(acc, a_low, b_low);
    return vfmaq_f32(acc, a_high, b_high);
}

namespace CactusThreading {
    
    class ThreadPool {
    private:
        std::vector<std::thread> workers;
        std::queue<std::function<void()>> tasks;
        std::mutex queue_mutex;
        std::condition_variable condition;
        std::atomic<bool> stop{false};
        std::atomic<size_t> active_workers{0};
        std::condition_variable finish_condition;
        
        void worker_thread() {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    condition.wait(lock, [this] { return stop || !tasks.empty(); });
                    
                    if (stop && tasks.empty()) return;
                    
                    task = std::move(tasks.front());
                    tasks.pop();
                    active_workers++;
                }
                
                task();
                
                active_workers--;
                finish_condition.notify_all();
            }
        }
        
    public:
        explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency()) {
            workers.reserve(num_threads);
            for (size_t i = 0; i < num_threads; ++i) {
                workers.emplace_back(&ThreadPool::worker_thread, this);
            }
        }
        
        ~ThreadPool() {
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                stop = true;
            }
            condition.notify_all();
            for (auto& worker : workers) {
                worker.join();
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
                std::unique_lock<std::mutex> lock(queue_mutex);
                if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");
                
                tasks.emplace([task](){ (*task)(); });
            }
            condition.notify_one();
            return res;
        }
        
        void wait_all() {
            std::unique_lock<std::mutex> lock(queue_mutex);
            finish_condition.wait(lock, [this] { 
                return tasks.empty() && active_workers == 0; 
            });
        }
        
        size_t num_workers() const { return workers.size(); }
    };
    
    inline ThreadPool& get_thread_pool() {
        static ThreadPool pool;
        return pool;
    }
    
    inline size_t get_optimal_thread_count(size_t total_work, size_t min_work_per_thread) {
        if (total_work < min_work_per_thread) return 1;
        size_t pool_size = get_thread_pool().num_workers();
        return std::min(pool_size, 
                       std::max(static_cast<size_t>(1), total_work / min_work_per_thread));
    }
    
    struct Thresholds {

        #if defined(__ANDROID__)
        static constexpr size_t ELEMENT_WISE = 5000;
        static constexpr size_t AXIS_REDUCE = 1000;
        static constexpr size_t ALL_REDUCE = 10000;
        static constexpr size_t SCALAR_BASIC = 30000;
        static constexpr size_t SCALAR_EXPENSIVE = 10000;
        static constexpr size_t ATTENTION = 512;
        static constexpr size_t GEMM_TILED = 20000; 
        static constexpr size_t GEMM_SMALL = 64 * 64 * 64;
        static constexpr size_t GEMM_MEDIUM = 256 * 256 * 256;
        static constexpr size_t GEMM_TILE_M = 64;
        static constexpr size_t GEMM_TILE_N = 64;
        static constexpr size_t GEMM_TILE_M_SMALL = 32;
        static constexpr size_t GEMM_TILE_N_SMALL = 32;
        #else // iOS
        static constexpr size_t ELEMENT_WISE = 5000;
        static constexpr size_t AXIS_REDUCE = 1000;
        static constexpr size_t ALL_REDUCE = 10000;
        static constexpr size_t SCALAR_BASIC = 5000;
        static constexpr size_t SCALAR_EXPENSIVE = 2500;
        static constexpr size_t ATTENTION = 4;
        static constexpr size_t GEMM_TILED = 4;  
        static constexpr size_t GEMM_SMALL = 64 * 64 * 64;
        static constexpr size_t GEMM_MEDIUM = 256 * 256 * 256;
        static constexpr size_t GEMM_TILE_M = 64;
        static constexpr size_t GEMM_TILE_N = 64;
        static constexpr size_t GEMM_TILE_M_SMALL = 32;
        static constexpr size_t GEMM_TILE_N_SMALL = 32;
        #endif
        static constexpr size_t L2_CACHE_SIZE = 256 * 1024;
    };
    
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
    TaskHandle parallel_for(size_t total_work, size_t threshold, WorkFunc work_func, bool wait = true) {
        const size_t num_threads = get_optimal_thread_count(total_work, threshold);
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
    void parallel_for_2d(size_t outer_size, size_t inner_size, size_t threshold, WorkFunc work_func) {
        const size_t total_work = outer_size * inner_size;
        parallel_for(total_work, threshold, [&](size_t start_idx, size_t end_idx) {
            for (size_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
                const size_t outer = work_idx / inner_size;
                const size_t inner = work_idx % inner_size;
                work_func(outer, inner);
            }
        });
    }
    
    template<typename WorkFunc, typename ResultType, typename CombineFunc>
    ResultType parallel_reduce(size_t total_work, size_t threshold, 
                              WorkFunc work_func, ResultType init_value, CombineFunc combine_func) {
        const size_t num_threads = get_optimal_thread_count(total_work, threshold);
        
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
    
    inline size_t compute_gemm_parallelism(size_t M, size_t K, size_t N, size_t element_size) {
        size_t total_ops = M * K * N;
        
        if (total_ops < Thresholds::GEMM_SMALL) return 1;
        
        if (total_ops < Thresholds::GEMM_MEDIUM) {
            return std::min(static_cast<size_t>(2), get_thread_pool().num_workers());
        }
        
        size_t bytes_accessed = (M * K + K * N + M * N) * element_size;
        size_t cache_tiles = (bytes_accessed + Thresholds::L2_CACHE_SIZE - 1) / Thresholds::L2_CACHE_SIZE;
        
        size_t compute_threads = std::sqrt(static_cast<double>(total_ops) / Thresholds::GEMM_SMALL);
        size_t memory_threads = cache_tiles;
        
        size_t optimal = std::min(compute_threads, memory_threads);
        return std::min(optimal, get_thread_pool().num_workers());
    }
    
    template<typename WorkFunc>
    void parallel_for_2d_tiled(size_t rows, size_t cols, size_t tile_rows, size_t tile_cols, WorkFunc work_func) {
        size_t num_row_tiles = (rows + tile_rows - 1) / tile_rows;
        size_t num_col_tiles = (cols + tile_cols - 1) / tile_cols;
        size_t total_tiles = num_row_tiles * num_col_tiles;

        parallel_for(total_tiles, Thresholds::GEMM_TILED, [=](size_t start_tile, size_t end_tile) {
            for (size_t tile_idx = start_tile; tile_idx < end_tile; ++tile_idx) {
                size_t tile_row = tile_idx / num_col_tiles;
                size_t tile_col = tile_idx % num_col_tiles;
                
                size_t row_start = tile_row * tile_rows;
                size_t row_end = std::min(row_start + tile_rows, rows);
                size_t col_start = tile_col * tile_cols;
                size_t col_end = std::min(col_start + tile_cols, cols);
                
                work_func(row_start, row_end, col_start, col_end);
            }
        });
    }
}

#endif // KERNEL_UTILS_H 