#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>

// Mock Dependencies
#include "cactus/kernel/kernel.h" // We need the signature

// We need to define __hexagon__ to trigger the logic in the kernel file
// But we can't define it *inside* the file if it's already pre-processed?
// No, when we compile this, we will pass -D__hexagon__ to the compiler.

// Forward declaration if we don't link everything
// Actually, we will TEXTUALLY INCLUDE the kernel file to force it to use our mocks.
// This is a "Unity Build" trick to test internal logic without linking.
#define __hexagon__ 1

// Include the mocks first
#include "tests/mock_sdk/hexagon_types.h"
#include "tests/mock_sdk/hexagon_protos.h"



// Now include the source file directly!
// We assume we are running from root 'c:\cactus'
#include "cactus/kernel/kernel_reduce.cpp"

int main() {
    std::cout << "Starting Hexagon HVX Logic Verification..." << std::endl;
    std::cout << "sizeof(__fp16): " << sizeof(__fp16) << std::endl;
    std::cout << "sizeof(HVX_Vector): " << sizeof(HVX_Vector) << std::endl;

    // Test Case: Summing 128 elements.
    // Each element is '1'.
    
    size_t num_elements = 128;
    std::vector<uint16_t> data(num_elements, 1);
    
    // Cast to __fp16* (which we treat as void* or uint16_t* in mock)
    // kernel expects const __fp16*
    const __fp16* input_ptr = (const __fp16*)data.data();
    
    // Call the kernel
    double result = cactus_sum_all_f16(input_ptr, num_elements);
    
    std::cout << "Sum Result: " << result << std::endl;
    std::cout << "Expected: " << 128.0 << std::endl;
    
    if (result == 128.0) {
        std::cout << "TEST PASSED: C++ Compilation and Logic Verified!" << std::endl;
        return 0;
    } else {
        std::cout << "TEST FAILED: Result mismatch." << std::endl;
        return 1;
    }
}
