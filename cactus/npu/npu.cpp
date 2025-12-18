
#include "npu.h"

namespace cactus {
namespace npu {

__attribute__((weak))
std::unique_ptr<NPUEncoder> create_encoder() {
    return nullptr;
}

__attribute__((weak))
std::unique_ptr<NPUPrefill> create_prefill() {
    return nullptr;
}

__attribute__((weak))
bool is_npu_available() {
    return false;
}

} // namespace npu
} // namespace cactus