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
#include <sched.h>
#include <fstream>
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

inline bool cpu_has_i8mm() {
#if defined(__aarch64__)
    static std::once_flag once;
    static bool has = false;

    std::call_once(once, []() {
#if defined(__APPLE__)
    has = true;
#elif defined(__ANDROID__)
    unsigned long hwcap2 = getauxval(AT_HWCAP2);
    #ifndef HWCAP2_I8MM
    #define HWCAP2_I8MM (1 << 13)
    #endif
    has = (hwcap2 & HWCAP2_I8MM) != 0;
#endif
    });

    return has;
#else
    return false;
#endif
}

inline bool cpu_has_sme2() {
#if defined(__aarch64__)
	static std::once_flag once;
	static bool has = false;
	
	std::call_once(once, []() {

#if defined(__APPLE__)
	int ret = 0;
	size_t size = sizeof(ret);
	if (sysctlbyname("hw.optional.arm.FEAT_SME2", &ret, &size, nullptr, 0) == 0) {
		has = ret == 1;
	}

#elif defined(__ANDROID__)
	unsigned long hwcap2 = getauxval(AT_HWCAP2);
#ifdef HWCAP2_SME2
	has = (hwcap2 & HWCAP2_SME2) != 0;
#endif

#endif
	});
	
	return has;
#else
	return false;
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

// Cephes-style 13/6 rational tanh approximation (same coefficients as Eigen).
// Constants are stored as static splatted arrays so the compiler emits a single
// pc-relative `ldr q` per load.
alignas(16) inline constexpr float kFastTanhAlpha[7][4] = {
    { 4.89352455891786e-03f, 4.89352455891786e-03f, 4.89352455891786e-03f, 4.89352455891786e-03f },
    { 6.37261928875436e-04f, 6.37261928875436e-04f, 6.37261928875436e-04f, 6.37261928875436e-04f },
    { 1.48572235717979e-05f, 1.48572235717979e-05f, 1.48572235717979e-05f, 1.48572235717979e-05f },
    { 5.12229709037114e-08f, 5.12229709037114e-08f, 5.12229709037114e-08f, 5.12229709037114e-08f },
    {-8.60467152213735e-11f,-8.60467152213735e-11f,-8.60467152213735e-11f,-8.60467152213735e-11f },
    { 2.00018790482477e-13f, 2.00018790482477e-13f, 2.00018790482477e-13f, 2.00018790482477e-13f },
    {-2.76076847742355e-16f,-2.76076847742355e-16f,-2.76076847742355e-16f,-2.76076847742355e-16f },
};
alignas(16) inline constexpr float kFastTanhBeta[4][4] = {
    { 4.89352518554385e-03f, 4.89352518554385e-03f, 4.89352518554385e-03f, 4.89352518554385e-03f },
    { 2.26843463243900e-03f, 2.26843463243900e-03f, 2.26843463243900e-03f, 2.26843463243900e-03f },
    { 1.18534705686654e-04f, 1.18534705686654e-04f, 1.18534705686654e-04f, 1.18534705686654e-04f },
    { 1.19825839466702e-06f, 1.19825839466702e-06f, 1.19825839466702e-06f, 1.19825839466702e-06f },
};
alignas(16) inline constexpr float kFastTanhClampHi[4] = { 9.0f, 9.0f, 9.0f, 9.0f };
alignas(16) inline constexpr float kFastTanhClampLo[4] = {-9.0f,-9.0f,-9.0f,-9.0f };

inline float32x4_t fast_tanh_f32x4(float32x4_t x) {
    x = vmaxq_f32(vld1q_f32(kFastTanhClampLo), vminq_f32(vld1q_f32(kFastTanhClampHi), x));
    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t p = vfmaq_f32(vld1q_f32(kFastTanhAlpha[5]), vld1q_f32(kFastTanhAlpha[6]), x2);
    p = vfmaq_f32(vld1q_f32(kFastTanhAlpha[4]), p, x2);
    p = vfmaq_f32(vld1q_f32(kFastTanhAlpha[3]), p, x2);
    p = vfmaq_f32(vld1q_f32(kFastTanhAlpha[2]), p, x2);
    p = vfmaq_f32(vld1q_f32(kFastTanhAlpha[1]), p, x2);
    p = vfmaq_f32(vld1q_f32(kFastTanhAlpha[0]), p, x2);
    p = vmulq_f32(p, x);
    float32x4_t q = vfmaq_f32(vld1q_f32(kFastTanhBeta[2]), vld1q_f32(kFastTanhBeta[3]), x2);
    q = vfmaq_f32(vld1q_f32(kFastTanhBeta[1]), q, x2);
    q = vfmaq_f32(vld1q_f32(kFastTanhBeta[0]), q, x2);
    return vdivq_f32(p, q);
}

constexpr size_t SIMD_F16_WIDTH = 8;

inline size_t simd_align(size_t count, size_t width = SIMD_F16_WIDTH) {
    return (count / width) * width;
}

inline void f16x8_split_f32(float16x8_t v, float32x4_t& lo, float32x4_t& hi) {
    lo = vcvt_f32_f16(vget_low_f16(v));
    hi = vcvt_f32_f16(vget_high_f16(v));
}

inline float16x8_t f32_merge_f16(float32x4_t lo, float32x4_t hi) {
    return vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi));
}

inline float32x4_t fast_sigmoid_f32x4(float32x4_t x) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    return vdivq_f32(one, vaddq_f32(one, fast_exp_f32x4(vnegq_f32(x))));
}

template<typename F32x4Op>
inline float16x8_t apply_f32_op_on_f16x8(float16x8_t v, F32x4Op op) {
    float32x4_t lo, hi;
    f16x8_split_f32(v, lo, hi);
    return f32_merge_f16(op(lo), op(hi));
}

inline void unpack_int4_as_int8x16x2(const uint8_t* ptr, int8x16_t& high_decoded, int8x16_t& low_decoded) {
    int8x16_t packed = vreinterpretq_s8_u8(vld1q_u8(ptr));
    high_decoded = vshrq_n_s8(packed, 4);
    low_decoded = vshrq_n_s8(vshlq_n_s8(packed, 4), 4);
}

namespace CactusThreading {

#if defined(__ANDROID__)
    struct CoreTopology {
        std::vector<int> performance_cores;  
        std::vector<int> all_cores;

        static CoreTopology& get() {
            static CoreTopology topo = detect();
            return topo;
        }

    private:
        static int read_sysfs_int(const char* path) {
            std::ifstream f(path);
            if (!f.is_open()) return -1;
            int val = -1;
            f >> val;
            return val;
        }

        static CoreTopology detect() {
            CoreTopology topo;
            constexpr int MAX_CPUS = 16;
            std::vector<std::pair<int, int>> core_caps; 

            for (int i = 0; i < MAX_CPUS; ++i) {
                char path[128];

                snprintf(path, sizeof(path),
                         "/sys/devices/system/cpu/cpu%d/cpu_capacity", i);
                int cap = read_sysfs_int(path);
                if (cap > 0) {
                    core_caps.push_back({i, cap});
                    topo.all_cores.push_back(i);
                    continue;
                }

                snprintf(path, sizeof(path),
                         "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
                int freq = read_sysfs_int(path);
                if (freq > 0) {
                    core_caps.push_back({i, freq});
                    topo.all_cores.push_back(i);
                }
            }

            if (core_caps.empty()) return topo;

            int max_cap = 0;
            for (auto& [id, cap] : core_caps) {
                max_cap = std::max(max_cap, cap);
            }

            int threshold = static_cast<int>(max_cap * 0.70);
            for (auto& [id, cap] : core_caps) {
                if (cap >= threshold) {
                    topo.performance_cores.push_back(id);
                }
            }

            return topo;
        }
    };

    inline bool pin_current_thread_to_cores(const std::vector<int>& cores) {
        if (cores.empty()) return false;
        cpu_set_t mask;
        CPU_ZERO(&mask);
        for (int core : cores) {
            CPU_SET(core, &mask);
        }
        return sched_setaffinity(0, sizeof(mask), &mask) == 0;
    }
#endif

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

#if defined(__ANDROID__)
            auto& topo = CoreTopology::get();
            if (!topo.performance_cores.empty()) {
                num_workers_ = std::min(num_workers_, topo.performance_cores.size());
            }
#endif

            workers.reserve(num_workers_);
            for (size_t i = 0; i < num_workers_; ++i) {
                workers.emplace_back([this]() {
#if defined(__ANDROID__)
                    auto& perf = CoreTopology::get().performance_cores;
                    if (!perf.empty()) {
                        pin_current_thread_to_cores(perf);
                    }
#endif
                    worker_thread();
                });
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
            return std::min(pool_size, static_cast<size_t>(3));
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
            return std::min(pool_size, static_cast<size_t>(5));
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

template<typename SimdOp, typename ScalarOp>
void elementwise_op_f16(const __fp16* input, __fp16* output, size_t num_elements,
                        bool use_streaming, CactusThreading::ParallelConfig config,
                        SimdOp simd_op, ScalarOp scalar_op, size_t unroll = 4) {
    CactusThreading::parallel_for(num_elements, config,
        [&](size_t start, size_t end) {
            const size_t n = end - start;
            const size_t vec_end = start + simd_align(n);

            if (use_streaming && unroll >= 4) {
                const size_t unrolled_end = start + simd_align(n, SIMD_F16_WIDTH * 4);
                for (size_t i = start; i < unrolled_end; i += SIMD_F16_WIDTH * 4) {
                    __builtin_prefetch(&input[i + 256], 0, 0);
                    float16x8_t v0 = simd_op(vld1q_f16(&input[i]));
                    float16x8_t v1 = simd_op(vld1q_f16(&input[i + 8]));
                    float16x8_t v2 = simd_op(vld1q_f16(&input[i + 16]));
                    float16x8_t v3 = simd_op(vld1q_f16(&input[i + 24]));
                    stream_store_f16x8(&output[i], v0);
                    stream_store_f16x8(&output[i + 8], v1);
                    stream_store_f16x8(&output[i + 16], v2);
                    stream_store_f16x8(&output[i + 24], v3);
                }
                for (size_t i = unrolled_end; i < vec_end; i += SIMD_F16_WIDTH) {
                    stream_store_f16x8(&output[i], simd_op(vld1q_f16(&input[i])));
                }
            } else if (use_streaming && unroll >= 2) {
                const size_t unrolled_end = start + simd_align(n, SIMD_F16_WIDTH * 2);
                for (size_t i = start; i < unrolled_end; i += SIMD_F16_WIDTH * 2) {
                    __builtin_prefetch(&input[i + 128], 0, 0);
                    float16x8_t v0 = simd_op(vld1q_f16(&input[i]));
                    float16x8_t v1 = simd_op(vld1q_f16(&input[i + 8]));
                    stream_store_f16x8(&output[i], v0);
                    stream_store_f16x8(&output[i + 8], v1);
                }
                for (size_t i = unrolled_end; i < vec_end; i += SIMD_F16_WIDTH) {
                    stream_store_f16x8(&output[i], simd_op(vld1q_f16(&input[i])));
                }
            } else {
                for (size_t i = start; i < vec_end; i += SIMD_F16_WIDTH) {
                    vst1q_f16(&output[i], simd_op(vld1q_f16(&input[i])));
                }
            }
            for (size_t i = vec_end; i < end; ++i) {
                output[i] = scalar_op(input[i]);
            }
        });
}

#endif // KERNEL_UTILS_H
