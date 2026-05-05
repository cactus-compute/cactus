#include "engine.h"

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
