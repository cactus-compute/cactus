#include "npu.h"
#ifdef CACTUS_HAS_QNN_DIRECT
#include "npu_qnn_direct.h"
#endif
#ifdef CACTUS_HAS_FASTRPC
#include "npu_fastrpc.h"
#endif

namespace cactus {
namespace npu {

std::unique_ptr<NPUEncoder> create_encoder() {
#ifdef CACTUS_HAS_QNN_DIRECT
    return std::make_unique<QNNDirectEncoder>();
#else
    return nullptr;
#endif
}

std::unique_ptr<NPUPrefill> create_prefill() {
#ifdef CACTUS_HAS_FASTRPC
    return std::make_unique<FastRPCPrefill>();
#elif defined(CACTUS_HAS_QNN_DIRECT)
    return std::make_unique<QNNDirectPrefill>();
#else
    return nullptr;
#endif
}

bool is_npu_available() {
#if defined(CACTUS_HAS_FASTRPC) || defined(CACTUS_HAS_QNN_DIRECT)
    return true;
#else
    return false;
#endif
}

} // namespace npu
} // namespace cactus
