#ifndef KERNEL_H
#define KERNEL_H

// This mock header is forced via -include to effectively "shadow" the real kernel.h
// by defining the header guard KERNEL_H.

#include <cstddef>
#include <cstdint>

// Ensure __fp16 is defined (using our types mock or defining it here)
#if defined(__x86_64__) || defined(_M_X64)
    typedef float __fp16;
#endif

// Signature for the function we are verifying
double cactus_sum_all_f16(const __fp16* data, size_t num_elements);

// Mock CactusThreading which is used in kernel_reduce.cpp
namespace CactusThreading {
    enum class Thresholds { ALL_REDUCE, AXIS_REDUCE };

    // Simple serial fallback for verification
    template<typename F, typename I, typename R>
    auto parallel_reduce(size_t num_elements, Thresholds t, F func, I init, R reduce_op) {
        // Just call the function for the whole range for testing logic
        return func(0, num_elements);
    }
    
    template<typename WorkFunc>
    void parallel_for_2d(size_t outer_size, size_t inner_size, Thresholds config, WorkFunc work_func) {
        for (size_t i = 0; i < outer_size * inner_size; ++i) {
             size_t outer = i / inner_size;
             size_t inner = i % inner_size;
             work_func(outer, inner);
        }
    }
}

#endif
