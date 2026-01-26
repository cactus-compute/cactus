#ifndef KERNEL_UTILS_H
#define KERNEL_UTILS_H

#include <arm_neon.h>
#if defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#include <TargetConditionals.h>
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

#if defined(__ARM_FEATURE_DOTPROD)
inline int32x4_t accum_dot(int32x4_t acc, int8x16_t a, int8x16_t b) {
    return vdotq_s32(acc, a, b);
}
#else
inline int32x4_t accum_dot(int32x4_t acc, int8x16_t a, int8x16_t b) {
    int16x8_t prod_low = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    int32x4_t acc_high = vpaddlq_s16(vmull_s8(vget_high_s8(a), vget_high_s8(b)));
    return vaddq_s32(vaddq_s32(acc, vpaddlq_s16(prod_low)), acc_high);
}
#endif

// I8MM support: runtime detection on Android, compile-time on Apple
#if defined(__ANDROID__) && defined(__aarch64__)

inline bool cactus_has_i8mm() {
    static int8_t supported = -1;
    if (supported == -1) {
        unsigned long hwcaps = getauxval(AT_HWCAP2);
        supported = (hwcaps & HWCAP2_I8MM) ? 1 : 0;
    }
    return supported;
}

__attribute__((target("arch=armv8.2-a+i8mm")))
inline int32x4_t accum_matmul(int32x4_t acc, int8x16_t a, int8x16_t b) {
    return vmmlaq_s32(acc, a, b);
}

#elif defined(__APPLE__) && defined(__aarch64__)

inline bool cactus_has_i8mm() {
    static int8_t supported = -1;
    if (supported == -1) {
        int v = 0; 
        size_t sz = sizeof(v);
        supported = sysctlbyname("hw.optional.arm.FEAT_I8MM", &v, &sz, nullptr, 0) == 0 && v;
    }
    return supported;
}

__attribute__((target("i8mm")))
inline int32x4_t accum_matmul(int32x4_t acc, int8x16_t a, int8x16_t b) {
    return vmmlaq_s32(acc, a, b);
}

#else

inline bool cactus_has_i8mm() {
    return false;
}

#endif

inline float16x8_t accum_f16_dot(float16x8_t acc, float16x8_t a_low, float16x8_t a_high,
                                 float16x8_t b_low, float16x8_t b_high) {
    // Convert to FP32, perform FMA, convert back to FP16 for ARMv8.0 compatibility
    float32x4_t acc_lo = vcvt_f32_f16(vget_low_f16(acc));
    float32x4_t acc_hi = vcvt_f32_f16(vget_high_f16(acc));
    
    float32x4_t a_low_lo = vcvt_f32_f16(vget_low_f16(a_low));
    float32x4_t a_low_hi = vcvt_f32_f16(vget_high_f16(a_low));
    float32x4_t b_low_lo = vcvt_f32_f16(vget_low_f16(b_low));
    float32x4_t b_low_hi = vcvt_f32_f16(vget_high_f16(b_low));
    
    acc_lo = vfmaq_f32(acc_lo, a_low_lo, b_low_lo);
    acc_hi = vfmaq_f32(acc_hi, a_low_hi, b_low_hi);
    
    float32x4_t a_high_lo = vcvt_f32_f16(vget_low_f16(a_high));
    float32x4_t a_high_hi = vcvt_f32_f16(vget_high_f16(a_high));
    float32x4_t b_high_lo = vcvt_f32_f16(vget_low_f16(b_high));
    float32x4_t b_high_hi = vcvt_f32_f16(vget_high_f16(b_high));
    
    acc_lo = vfmaq_f32(acc_lo, a_high_lo, b_high_lo);
    acc_hi = vfmaq_f32(acc_hi, a_high_hi, b_high_hi);
    
    return vcombine_f16(vcvt_f16_f32(acc_lo), vcvt_f16_f32(acc_hi));
}

// ARMv8.0-compatible FP16 vector operations using FP32 conversions
inline float16x8_t vaddq_f16_compat(float16x8_t a, float16x8_t b) {
    float32x4_t a_lo = vcvt_f32_f16(vget_low_f16(a));
    float32x4_t a_hi = vcvt_f32_f16(vget_high_f16(a));
    float32x4_t b_lo = vcvt_f32_f16(vget_low_f16(b));
    float32x4_t b_hi = vcvt_f32_f16(vget_high_f16(b));
    return vcombine_f16(vcvt_f16_f32(vaddq_f32(a_lo, b_lo)), vcvt_f16_f32(vaddq_f32(a_hi, b_hi)));
}

inline float16x8_t vsubq_f16_compat(float16x8_t a, float16x8_t b) {
    float32x4_t a_lo = vcvt_f32_f16(vget_low_f16(a));
    float32x4_t a_hi = vcvt_f32_f16(vget_high_f16(a));
    float32x4_t b_lo = vcvt_f32_f16(vget_low_f16(b));
    float32x4_t b_hi = vcvt_f32_f16(vget_high_f16(b));
    return vcombine_f16(vcvt_f16_f32(vsubq_f32(a_lo, b_lo)), vcvt_f16_f32(vsubq_f32(a_hi, b_hi)));
}

inline float16x8_t vmulq_f16_compat(float16x8_t a, float16x8_t b) {
    float32x4_t a_lo = vcvt_f32_f16(vget_low_f16(a));
    float32x4_t a_hi = vcvt_f32_f16(vget_high_f16(a));
    float32x4_t b_lo = vcvt_f32_f16(vget_low_f16(b));
    float32x4_t b_hi = vcvt_f32_f16(vget_high_f16(b));
    return vcombine_f16(vcvt_f16_f32(vmulq_f32(a_lo, b_lo)), vcvt_f16_f32(vmulq_f32(a_hi, b_hi)));
}

inline float16x8_t vdivq_f16_compat(float16x8_t a, float16x8_t b) {
    float32x4_t a_lo = vcvt_f32_f16(vget_low_f16(a));
    float32x4_t a_hi = vcvt_f32_f16(vget_high_f16(a));
    float32x4_t b_lo = vcvt_f32_f16(vget_low_f16(b));
    float32x4_t b_hi = vcvt_f32_f16(vget_high_f16(b));
    return vcombine_f16(vcvt_f16_f32(vdivq_f32(a_lo, b_lo)), vcvt_f16_f32(vdivq_f32(a_hi, b_hi)));
}

inline float16x8_t vfmaq_f16_compat(float16x8_t acc, float16x8_t a, float16x8_t b) {
    float32x4_t acc_lo = vcvt_f32_f16(vget_low_f16(acc));
    float32x4_t acc_hi = vcvt_f32_f16(vget_high_f16(acc));
    float32x4_t a_lo = vcvt_f32_f16(vget_low_f16(a));
    float32x4_t a_hi = vcvt_f32_f16(vget_high_f16(a));
    float32x4_t b_lo = vcvt_f32_f16(vget_low_f16(b));
    float32x4_t b_hi = vcvt_f32_f16(vget_high_f16(b));
    return vcombine_f16(vcvt_f16_f32(vfmaq_f32(acc_lo, a_lo, b_lo)), vcvt_f16_f32(vfmaq_f32(acc_hi, a_hi, b_hi)));
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

inline int8x16_t unpack_int4_lo(uint8x16_t packed) {
    uint8x16_t lo = vandq_u8(packed, vdupq_n_u8(0x0F));
    uint8x16_t sign_mask = vcgtq_u8(lo, vdupq_n_u8(7));
    uint8x16_t correction = vandq_u8(sign_mask, vdupq_n_u8(16));
    return vreinterpretq_s8_u8(vsubq_u8(lo, correction));
}

inline int8x16_t unpack_int4_hi(uint8x16_t packed) {
    uint8x16_t hi = vshrq_n_u8(packed, 4);
    uint8x16_t sign_mask = vcgtq_u8(hi, vdupq_n_u8(7));
    uint8x16_t correction = vandq_u8(sign_mask, vdupq_n_u8(16));
    return vreinterpretq_s8_u8(vsubq_u8(hi, correction));
}

inline void unpack_int4_to_int8x32(uint8x16_t packed, int8x16_t& out_lo, int8x16_t& out_hi) {
    int8x16_t lo_nibbles = unpack_int4_lo(packed);
    int8x16_t hi_nibbles = unpack_int4_hi(packed);
    int8x16x2_t interleaved = vzipq_s8(lo_nibbles, hi_nibbles);
    out_lo = interleaved.val[0];
    out_hi = interleaved.val[1];
}

inline int32x4_t int4_dot_asm(int32x4_t acc, uint8x16_t packed, int8x16_t a_lo, int8x16_t a_hi) {
#if defined(__aarch64__)
    int8x16_t b_lo, b_hi;

    __asm__ __volatile__ (
        "movi   v16.16b, #0x0F          \n"  // low nibble mask
        "movi   v17.16b, #7             \n"  // sign threshold
        "movi   v18.16b, #16            \n"  // sign correction

        "and    %[b_lo].16b, %[packed].16b, v16.16b  \n"

        "ushr   %[b_hi].16b, %[packed].16b, #4      \n"

        "cmgt   v19.16b, %[b_lo].16b, v17.16b       \n"
        "and    v19.16b, v19.16b, v18.16b           \n"
        "sub    %[b_lo].16b, %[b_lo].16b, v19.16b   \n"

        "cmgt   v20.16b, %[b_hi].16b, v17.16b       \n"
        "and    v20.16b, v20.16b, v18.16b           \n"
        "sub    %[b_hi].16b, %[b_hi].16b, v20.16b   \n"

        "zip1   v21.16b, %[b_lo].16b, %[b_hi].16b   \n"
        "zip2   v22.16b, %[b_lo].16b, %[b_hi].16b   \n"

        ".arch armv8.2-a+dotprod                    \n"
        "sdot   %[acc].4s, %[a_lo].16b, v21.16b     \n"
        "sdot   %[acc].4s, %[a_hi].16b, v22.16b     \n"

        : [acc] "+w"(acc), [b_lo] "=w"(b_lo), [b_hi] "=w"(b_hi)
        : [packed] "w"(packed), [a_lo] "w"(a_lo), [a_hi] "w"(a_hi)
        : "v16", "v17", "v18", "v19", "v20", "v21", "v22"
    );

    return acc;
#else
    int8x16_t b_lo, b_hi;
    unpack_int4_to_int8x32(packed, b_lo, b_hi);
    acc = accum_dot(acc, a_lo, b_lo);
    acc = accum_dot(acc, a_hi, b_hi);
    return acc;
#endif
}

inline int32_t int4_dot_m1_asm(const int8_t* a_ptr, const uint8_t* b_packed, size_t group_size) {
#if defined(__aarch64__)
    int32x4_t acc = vdupq_n_s32(0);

    for (size_t k = 0; k < group_size; k += 64) {
        uint8x16_t p0 = vld1q_u8(b_packed + k/2);
        uint8x16_t p1 = vld1q_u8(b_packed + k/2 + 16);

        int8x16_t a0 = vld1q_s8(a_ptr + k);
        int8x16_t a1 = vld1q_s8(a_ptr + k + 16);
        int8x16_t a2 = vld1q_s8(a_ptr + k + 32);
        int8x16_t a3 = vld1q_s8(a_ptr + k + 48);

        acc = int4_dot_asm(acc, p0, a0, a1);
        acc = int4_dot_asm(acc, p1, a2, a3);
    }

    return vaddvq_s32(acc);
#else
    int32x4_t acc = vdupq_n_s32(0);
    for (size_t k = 0; k < group_size; k += 32) {
        uint8x16_t packed = vld1q_u8(b_packed + k/2);
        int8x16_t b_lo, b_hi;
        unpack_int4_to_int8x32(packed, b_lo, b_hi);
        acc = accum_dot(acc, vld1q_s8(a_ptr + k), b_lo);
        acc = accum_dot(acc, vld1q_s8(a_ptr + k + 16), b_hi);
    }
    return vaddvq_s32(acc);
#endif
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
        #elif defined(__APPLE__) && TARGET_OS_IPHONE
        static size_t get_num_threads(size_t M, size_t pool_size) {
            if (M <= 1) return std::min(pool_size, static_cast<size_t>(2)); 
            return pool_size; 
        }
        #else // Mac
        static size_t get_num_threads(size_t M, size_t pool_size) {
            if (M <= 1) return std::min(pool_size, static_cast<size_t>(4));
            return pool_size; 
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