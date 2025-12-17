#include "npu.h"
#include <dlfcn.h>
#include <cstdlib>

namespace cactus {
namespace npu {

typedef std::unique_ptr<NPUEncoder> (*create_encoder_fn)();
typedef bool (*is_npu_available_fn)();
typedef std::unique_ptr<NPUPrefill> (*create_prefill_fn)();

static void* g_cactus_util_handle = nullptr;
static bool g_attempted_load = false;

static void try_load_cactus_util() {
    if (g_attempted_load) {
        return;
    }
    g_attempted_load = true;

    const char* env_path = std::getenv("CACTUS_UTIL_PATH");
    if (env_path) {
        void* handle = dlopen(env_path, RTLD_NOW | RTLD_GLOBAL);
        if (handle) {
            g_cactus_util_handle = handle;
            return;
        }
    }

#if defined(__APPLE__)
    const char* lib_names[] = {
        "@rpath/cactus_util.framework/cactus_util",
        "cactus_util",
    };
#else
    const char* lib_names[] = {
        "libcactus_util.so",
        "cactus_util"
    };
#endif

    for (const char* lib_name : lib_names) {
        void* handle = dlopen(lib_name, RTLD_NOW | RTLD_GLOBAL);
        if (handle) {
            g_cactus_util_handle = handle;
            return;
        }
    }
}

template<typename T>
T get_runtime_symbol(const char* symbol_name) {
    if (!g_attempted_load) {
        try_load_cactus_util();
    }

    if (g_cactus_util_handle) {
        void* sym = dlsym(g_cactus_util_handle, symbol_name);
        if (sym) {
            return reinterpret_cast<T>(sym);
        }
    }

    return nullptr;
}

__attribute__((weak, visibility("default")))
std::unique_ptr<NPUEncoder> create_encoder() {
    auto strong_fn = get_runtime_symbol<create_encoder_fn>("_ZN6cactus3npu14create_encoderEv");
    if (strong_fn) {
        return strong_fn();
    }
    return nullptr;
}

__attribute__((weak, visibility("default")))
bool is_npu_available() {
    auto strong_fn = get_runtime_symbol<is_npu_available_fn>("_ZN6cactus3npu16is_npu_availableEv");
    if (strong_fn) {
        return strong_fn();
    }
    return false;
}

__attribute__((weak, visibility("default")))
std::unique_ptr<NPUPrefill> create_prefill() {
    auto strong_fn = get_runtime_symbol<create_prefill_fn>("_ZN6cactus3npu14create_prefillEv");
    if (strong_fn) {
        return strong_fn();
    }
    return nullptr;
}

} // namespace npu
} // namespace cactus
