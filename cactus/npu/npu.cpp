// NPU stub implementation for non-Apple platforms
// On Apple platforms, npu_ane.mm provides the real implementation

#include "npu.h"

namespace cactus {
namespace npu {

std::unique_ptr<NPUEncoder> create_encoder() {
    return nullptr;
}

std::unique_ptr<NPUPrefill> create_prefill() {
    return nullptr;
}

bool is_npu_available() {
    return false;
}

} // namespace npu
} // namespace cactus
