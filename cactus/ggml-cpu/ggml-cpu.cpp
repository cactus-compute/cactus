#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-aarch64.h"
#include "ggml-cpu-traits.h"
#include "ggml-impl.h"
#include "amx/amx.h"

#include <cctype>
#include <string>
#include <vector>

#ifdef LM_GGML_USE_CPU_HBM
#    include "ggml-cpu-hbm.h"
#endif

#ifdef LM_GGML_USE_CPU_KLEIDIAI
#    include "kleidiai/kleidiai.h"
#endif

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <unistd.h>
#endif

#if defined(__APPLE__)
#    include <sys/sysctl.h>
#    include <sys/types.h>
#endif

// ggml-backend interface

std::vector<lm_ggml_backend_buffer_type_t>& lm_ggml_backend_cpu_get_extra_buffers_type() {
    static std::vector<lm_ggml_backend_buffer_type_t> bufts = []() {
        std::vector<lm_ggml_backend_buffer_type_t> bufts;

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        if (lm_ggml_backend_amx_buffer_type()) {
            bufts.push_back(lm_ggml_backend_amx_buffer_type());
        }
#endif

#ifdef LM_GGML_USE_CPU_KLEIDIAI
        if (lm_ggml_backend_cpu_kleidiai_buffer_type()) {
            bufts.push_back(lm_ggml_backend_cpu_kleidiai_buffer_type());
        }
#endif

#ifdef LM_GGML_USE_CPU_AARCH64
        if (lm_ggml_backend_cpu_aarch64_buffer_type()) {
            bufts.push_back(lm_ggml_backend_cpu_aarch64_buffer_type());
        }
#endif

        bufts.push_back(NULL);

        return bufts;
    }();

    return bufts;
}

static lm_ggml_backend_buffer_type_t * lm_ggml_backend_cpu_device_get_extra_buffers_type(lm_ggml_backend_dev_t device) {
    return lm_ggml_backend_cpu_get_extra_buffers_type().data();

    LM_GGML_UNUSED(device);
}

static bool lm_ggml_backend_cpu_is_extra_buffer_type(lm_ggml_backend_buffer_type_t buft) {
    for (auto * extra : lm_ggml_backend_cpu_get_extra_buffers_type()) {
        if (extra && extra == buft) {
            return true;
        }
    }
    return false;
}

// CPU backend - backend (stream)

struct lm_ggml_backend_cpu_context {
    int                 n_threads;
    lm_ggml_threadpool_t   threadpool;

    uint8_t *           work_data;
    size_t              work_size;

    lm_ggml_abort_callback abort_callback;
    void *              abort_callback_data;
};

static const char * lm_ggml_backend_cpu_get_name(lm_ggml_backend_t backend) {
    return "CPU";

    LM_GGML_UNUSED(backend);
}

static void lm_ggml_backend_cpu_free(lm_ggml_backend_t backend) {
    struct lm_ggml_backend_cpu_context * cpu_ctx = (struct lm_ggml_backend_cpu_context *)backend->context;
    delete[] cpu_ctx->work_data;
    delete cpu_ctx;
    delete backend;
}

struct lm_ggml_backend_plan_cpu {
    struct lm_ggml_cplan cplan;
    struct lm_ggml_cgraph cgraph;
};

static lm_ggml_backend_graph_plan_t lm_ggml_backend_cpu_graph_plan_create(lm_ggml_backend_t backend, const struct lm_ggml_cgraph * cgraph) {
    struct lm_ggml_backend_cpu_context * cpu_ctx = (struct lm_ggml_backend_cpu_context *)backend->context;

    struct lm_ggml_backend_plan_cpu * cpu_plan = new lm_ggml_backend_plan_cpu;

    cpu_plan->cplan = lm_ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);
    cpu_plan->cgraph = *cgraph; // FIXME: deep copy

    if (cpu_plan->cplan.work_size > 0) {
        cpu_plan->cplan.work_data = new uint8_t[cpu_plan->cplan.work_size];
        if (cpu_plan->cplan.work_data == NULL) {
            delete cpu_plan;
            return NULL;
        }
    }

    cpu_plan->cplan.abort_callback      = cpu_ctx->abort_callback;
    cpu_plan->cplan.abort_callback_data = cpu_ctx->abort_callback_data;

    return cpu_plan;
}

static void lm_ggml_backend_cpu_graph_plan_free(lm_ggml_backend_t backend, lm_ggml_backend_graph_plan_t plan) {
    struct lm_ggml_backend_plan_cpu * cpu_plan = (struct lm_ggml_backend_plan_cpu *)plan;

    delete[] cpu_plan->cplan.work_data;
    delete cpu_plan;

    LM_GGML_UNUSED(backend);
}

static enum lm_ggml_status lm_ggml_backend_cpu_graph_plan_compute(lm_ggml_backend_t backend, lm_ggml_backend_graph_plan_t plan) {
    struct lm_ggml_backend_plan_cpu * cpu_plan = (struct lm_ggml_backend_plan_cpu *)plan;

    return lm_ggml_graph_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    LM_GGML_UNUSED(backend);
}

static enum lm_ggml_status lm_ggml_backend_cpu_graph_compute(lm_ggml_backend_t backend, struct lm_ggml_cgraph * cgraph) {
    struct lm_ggml_backend_cpu_context * cpu_ctx = (struct lm_ggml_backend_cpu_context *)backend->context;

    struct lm_ggml_cplan cplan = lm_ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);

    if (cpu_ctx->work_size < cplan.work_size) {
        delete[] cpu_ctx->work_data;
        cpu_ctx->work_data = new uint8_t[cplan.work_size];
        if (cpu_ctx->work_data == NULL) {
            cpu_ctx->work_size = 0;
            return LM_GGML_STATUS_ALLOC_FAILED;
        }
        cpu_ctx->work_size = cplan.work_size;
    }
    cplan.work_data = (uint8_t *)cpu_ctx->work_data;

    cplan.abort_callback      = cpu_ctx->abort_callback;
    cplan.abort_callback_data = cpu_ctx->abort_callback_data;

    return lm_ggml_graph_compute(cgraph, &cplan);
}

static const struct lm_ggml_backend_i lm_ggml_backend_cpu_i = {
    /* .get_name                = */ lm_ggml_backend_cpu_get_name,
    /* .free                    = */ lm_ggml_backend_cpu_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ lm_ggml_backend_cpu_graph_plan_create,
    /* .graph_plan_free         = */ lm_ggml_backend_cpu_graph_plan_free,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ lm_ggml_backend_cpu_graph_plan_compute,
    /* .graph_compute           = */ lm_ggml_backend_cpu_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};

static lm_ggml_guid_t lm_ggml_backend_cpu_guid(void) {
    static lm_ggml_guid guid = { 0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89 };
    return &guid;
}

lm_ggml_backend_t lm_ggml_backend_cpu_init(void) {
    // initialize CPU backend now to avoid slowing the first graph computation
    lm_ggml_cpu_init();

    struct lm_ggml_backend_cpu_context * ctx = new lm_ggml_backend_cpu_context;
    if (ctx == NULL) {
        return NULL;
    }

    ctx->n_threads           = LM_GGML_DEFAULT_N_THREADS;
    ctx->threadpool          = NULL;
    ctx->work_data           = NULL;
    ctx->work_size           = 0;
    ctx->abort_callback      = NULL;
    ctx->abort_callback_data = NULL;

    lm_ggml_backend_t cpu_backend = new lm_ggml_backend {
        /* .guid      = */ lm_ggml_backend_cpu_guid(),
        /* .interface = */ lm_ggml_backend_cpu_i,
        /* .device    = */ lm_ggml_backend_reg_dev_get(lm_ggml_backend_cpu_reg(), 0),
        /* .context   = */ ctx,
    };

    if (cpu_backend == NULL) {
        delete ctx;
        return NULL;
    }

    return cpu_backend;
}

bool lm_ggml_backend_is_cpu(lm_ggml_backend_t backend) {
    return backend != NULL && lm_ggml_guid_matches(backend->guid, lm_ggml_backend_cpu_guid());
}

void lm_ggml_backend_cpu_set_n_threads(lm_ggml_backend_t backend_cpu, int n_threads) {
    LM_GGML_ASSERT(lm_ggml_backend_is_cpu(backend_cpu));

    struct lm_ggml_backend_cpu_context * ctx = (struct lm_ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_threads = n_threads;
}

void lm_ggml_backend_cpu_set_threadpool(lm_ggml_backend_t backend_cpu, lm_ggml_threadpool_t threadpool) {
    LM_GGML_ASSERT(lm_ggml_backend_is_cpu(backend_cpu));

    struct lm_ggml_backend_cpu_context * ctx = (struct lm_ggml_backend_cpu_context *)backend_cpu->context;

    if (ctx->threadpool && ctx->threadpool != threadpool) {
        // already had a different threadpool, pause/suspend it before switching
        lm_ggml_threadpool_pause(ctx->threadpool);
    }
    ctx->threadpool = threadpool;
}

void lm_ggml_backend_cpu_set_abort_callback(lm_ggml_backend_t backend_cpu, lm_ggml_abort_callback abort_callback, void * abort_callback_data) {
    LM_GGML_ASSERT(lm_ggml_backend_is_cpu(backend_cpu));

    struct lm_ggml_backend_cpu_context * ctx = (struct lm_ggml_backend_cpu_context *)backend_cpu->context;
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = abort_callback_data;
}

// CPU backend - device

struct lm_ggml_backend_cpu_device_context {
    std::string description = "CPU";

    lm_ggml_backend_cpu_device_context() {
#ifdef __APPLE__
        size_t len = 0;
        if (!sysctlbyname("machdep.cpu.brand_string", NULL, &len, NULL, 0)) {
            description.resize(len);
            sysctlbyname("machdep.cpu.brand_string", &description[0], &len, NULL, 0); // NOLINT
        }
#elif defined(__linux__)
        FILE * f = fopen("/proc/cpuinfo", "r");
        if (f) {
            char buf[1024];
            while (fgets(buf, sizeof(buf), f)) {
                if (strncmp(buf, "model name", 10) == 0) {
                    char * p = strchr(buf, ':');
                    if (p) {
                        p++;
                        while (std::isspace(*p)) {
                            p++;
                        }
                        while (std::isspace(p[strlen(p) - 1])) {
                            p[strlen(p) - 1] = '\0';
                        }
                        description = p;
                        break;
                    }
                }
            }
            fclose(f);
        }
#elif defined(_WIN32)
        HKEY hKey;
        if (RegOpenKeyEx(HKEY_LOCAL_MACHINE,
                        TEXT("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
                        0,
                        KEY_READ,
                        &hKey) == ERROR_SUCCESS) {
            DWORD cpu_brand_size = 0;
            if (RegQueryValueExA(hKey,
                                "ProcessorNameString",
                                NULL,
                                NULL,
                                NULL,
                                &cpu_brand_size) == ERROR_SUCCESS) {
                description.resize(cpu_brand_size);
                if (RegQueryValueExA(hKey,
                                    "ProcessorNameString",
                                    NULL,
                                    NULL,
                                    (LPBYTE)&description[0], // NOLINT
                                    &cpu_brand_size) == ERROR_SUCCESS) {
                    if (description.find('\0') != std::string::npos) {
                        description.resize(description.find('\0'));
                    }
                }
            }
            RegCloseKey(hKey);
        }
#endif
    }
};

static const char * lm_ggml_backend_cpu_device_get_name(lm_ggml_backend_dev_t dev) {
    return "CPU";

    LM_GGML_UNUSED(dev);
}

static const char * lm_ggml_backend_cpu_device_get_description(lm_ggml_backend_dev_t dev) {
    struct lm_ggml_backend_cpu_device_context * ctx = (struct lm_ggml_backend_cpu_device_context *)dev->context;

    return ctx->description.c_str();
}

static void lm_ggml_backend_cpu_device_get_memory(lm_ggml_backend_dev_t dev, size_t * free, size_t * total) {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    *total = status.ullTotalPhys;
    *free = status.ullAvailPhys;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    *total = pages * page_size;
    *free = *total;
#endif

    LM_GGML_UNUSED(dev);
}

static enum lm_ggml_backend_dev_type lm_ggml_backend_cpu_device_get_type(lm_ggml_backend_dev_t dev) {
    return LM_GGML_BACKEND_DEVICE_TYPE_CPU;

    LM_GGML_UNUSED(dev);
}

static void lm_ggml_backend_cpu_device_get_props(lm_ggml_backend_dev_t dev, struct lm_ggml_backend_dev_props * props) {
    props->name        = lm_ggml_backend_cpu_device_get_name(dev);
    props->description = lm_ggml_backend_cpu_device_get_description(dev);
    props->type        = lm_ggml_backend_cpu_device_get_type(dev);
    lm_ggml_backend_cpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static lm_ggml_backend_t lm_ggml_backend_cpu_device_init_backend(lm_ggml_backend_dev_t dev, const char * params) {
    return lm_ggml_backend_cpu_init();

    LM_GGML_UNUSED(dev);
    LM_GGML_UNUSED(params);
}

static lm_ggml_backend_buffer_type_t lm_ggml_backend_cpu_device_get_buffer_type(lm_ggml_backend_dev_t dev) {
    return lm_ggml_backend_cpu_buffer_type();

    LM_GGML_UNUSED(dev);
}

static lm_ggml_backend_buffer_t lm_ggml_backend_cpu_device_buffer_from_host_ptr(lm_ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return lm_ggml_backend_cpu_buffer_from_ptr(ptr, size);

    LM_GGML_UNUSED(dev);
    LM_GGML_UNUSED(max_tensor_size);
}

static bool lm_ggml_backend_cpu_device_supports_op(lm_ggml_backend_dev_t dev, const struct lm_ggml_tensor * op) {
    const struct lm_ggml_tensor * src0 = op->src[0];
    const struct lm_ggml_tensor * src1 = op->src[1];

    if (op->op == LM_GGML_OP_NONE || op->op == LM_GGML_OP_RESHAPE || op->op == LM_GGML_OP_VIEW || op->op == LM_GGML_OP_PERMUTE || op->op == LM_GGML_OP_TRANSPOSE) {
        return true;
    }

    // extra_buffer_op?
    for (auto extra : lm_ggml_backend_cpu_get_extra_buffers_type()) {
        if (extra) {
            auto buf_extra = (ggml::cpu::extra_buffer_type*) extra->context;
            if (buf_extra && buf_extra->supports_op(dev, op)) {
                return true;
            }
        }
    }

    // the other case need host buffer.
    for (int i = 0; i < LM_GGML_MAX_SRC; i++) {
        if (op->src[i] && op->src[i]->buffer && !lm_ggml_backend_buft_is_host(op->src[i]->buffer->buft)) {
            return false;
        }
    }

    switch (op->op) {
        case LM_GGML_OP_CPY:
            return
                op->type != LM_GGML_TYPE_IQ3_XXS &&
                op->type != LM_GGML_TYPE_IQ3_S   &&
                op->type != LM_GGML_TYPE_IQ2_XXS &&
                op->type != LM_GGML_TYPE_IQ2_XS  &&
                op->type != LM_GGML_TYPE_IQ2_S   &&
                op->type != LM_GGML_TYPE_IQ1_S   &&
                op->type != LM_GGML_TYPE_IQ1_M; // missing type_traits.from_float
        case LM_GGML_OP_MUL_MAT:
            return src1->type == LM_GGML_TYPE_F32 || src1->type == lm_ggml_get_type_traits_cpu(src0->type)->vec_dot_type;
        case LM_GGML_OP_SOFT_MAX_BACK: {
            if (op->src[0]->type != LM_GGML_TYPE_F32 || op->src[1]->type != LM_GGML_TYPE_F32) {
                return false;
            }
            float max_bias = 0.0f;

            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));

            return max_bias == 0.0f;
        }
        case LM_GGML_OP_IM2COL_BACK:
            return src0->type == LM_GGML_TYPE_F32 && src1->type == LM_GGML_TYPE_F32;
        case LM_GGML_OP_GET_ROWS_BACK:
            return src0->type == LM_GGML_TYPE_F32 || src0->type == LM_GGML_TYPE_F16;
        case LM_GGML_OP_OUT_PROD:
            return (src0->type == LM_GGML_TYPE_F32 || (lm_ggml_is_quantized(src0->type) && src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3])) &&
                src1->type == LM_GGML_TYPE_F32 && op->type == LM_GGML_TYPE_F32;
        default:
            return true;
    }
}

static bool lm_ggml_backend_cpu_device_supports_buft(lm_ggml_backend_dev_t dev, lm_ggml_backend_buffer_type_t buft) {
    return lm_ggml_backend_buft_is_host(buft) || lm_ggml_backend_cpu_is_extra_buffer_type(buft);
    LM_GGML_UNUSED(dev);
}

static const struct lm_ggml_backend_device_i lm_ggml_backend_cpu_device_i = {
    /* .get_name             = */ lm_ggml_backend_cpu_device_get_name,
    /* .get_description      = */ lm_ggml_backend_cpu_device_get_description,
    /* .get_memory           = */ lm_ggml_backend_cpu_device_get_memory,
    /* .get_type             = */ lm_ggml_backend_cpu_device_get_type,
    /* .get_props            = */ lm_ggml_backend_cpu_device_get_props,
    /* .init_backend         = */ lm_ggml_backend_cpu_device_init_backend,
    /* .get_buffer_type      = */ lm_ggml_backend_cpu_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ lm_ggml_backend_cpu_device_buffer_from_host_ptr,
    /* .supports_op          = */ lm_ggml_backend_cpu_device_supports_op,
    /* .supports_buft        = */ lm_ggml_backend_cpu_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// CPU backend - backend (reg)

static const char * lm_ggml_backend_cpu_reg_get_name(lm_ggml_backend_reg_t reg) {
    return "CPU";

    LM_GGML_UNUSED(reg);
}

static size_t lm_ggml_backend_cpu_reg_get_device_count(lm_ggml_backend_reg_t reg) {
    return 1;

    LM_GGML_UNUSED(reg);
}

static lm_ggml_backend_dev_t lm_ggml_backend_cpu_reg_get_device(lm_ggml_backend_reg_t reg, size_t index) {
    LM_GGML_ASSERT(index == 0);

    static lm_ggml_backend_cpu_device_context ctx;
    static lm_ggml_backend_device lm_ggml_backend_cpu_device = {
        /* .iface   = */ lm_ggml_backend_cpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &ctx,
    };

    return &lm_ggml_backend_cpu_device;
}

// This is intended to replace the the lm_ggml_cpu_has_* functions when loading the CPU backend dynamically,
// and additionally to allow other backends to expose their own list of features that applications can query using the same API
static lm_ggml_backend_feature * lm_ggml_backend_cpu_get_features(lm_ggml_backend_reg_t reg) {
    static std::vector<lm_ggml_backend_feature> features = []() {
        lm_ggml_cpu_init();

        std::vector<lm_ggml_backend_feature> features;
        if (lm_ggml_cpu_has_sse3()) {
            features.push_back({ "SSE3", "1" });
        }
        if (lm_ggml_cpu_has_ssse3()) {
            features.push_back({ "SSSE3", "1" });
        }
        if (lm_ggml_cpu_has_avx()) {
            features.push_back({ "AVX", "1" });
        }
        if (lm_ggml_cpu_has_avx_vnni()) {
            features.push_back({ "AVX_VNNI", "1" });
        }
        if (lm_ggml_cpu_has_avx2()) {
            features.push_back({ "AVX2", "1" });
        }
        if (lm_ggml_cpu_has_f16c()) {
            features.push_back({ "F16C", "1" });
        }
        if (lm_ggml_cpu_has_fma()) {
            features.push_back({ "FMA", "1" });
        }
        if (lm_ggml_cpu_has_bmi2()) {
            features.push_back({ "BMI2", "1" });
        }
        if (lm_ggml_cpu_has_avx512()) {
            features.push_back({ "AVX512", "1" });
        }
        if (lm_ggml_cpu_has_avx512_vbmi()) {
            features.push_back({ "AVX512_VBMI", "1" });
        }
        if (lm_ggml_cpu_has_avx512_vnni()) {
            features.push_back({ "AVX512_VNNI", "1" });
        }
        if (lm_ggml_cpu_has_avx512_bf16()) {
            features.push_back({ "AVX512_BF16", "1" });
        }
        if (lm_ggml_cpu_has_amx_int8()) {
            features.push_back({ "AMX_INT8", "1" });
        }
        if (lm_ggml_cpu_has_neon()) {
            features.push_back({ "NEON", "1" });
        }
        if (lm_ggml_cpu_has_arm_fma()) {
            features.push_back({ "ARM_FMA", "1" });
        }
        if (lm_ggml_cpu_has_fp16_va()) {
            features.push_back({ "FP16_VA", "1" });
        }
        if (lm_ggml_cpu_has_matmul_int8()) {
            features.push_back({ "MATMUL_INT8", "1" });
        }
        if (lm_ggml_cpu_has_sve()) {
            features.push_back({ "SVE", "1" });
        }
        if (lm_ggml_cpu_has_dotprod()) {
            features.push_back({ "DOTPROD", "1" });
        }
        if (lm_ggml_cpu_get_sve_cnt() > 0) {
            static std::string sve_cnt = std::to_string(lm_ggml_cpu_get_sve_cnt());
            features.push_back({ "SVE_CNT", sve_cnt.c_str() });
        }
        if (lm_ggml_cpu_has_sme()) {
            features.push_back({ "SME", "1" });
        }
        if (lm_ggml_cpu_has_riscv_v()) {
            features.push_back({ "RISCV_V", "1" });
        }
        if (lm_ggml_cpu_has_vsx()) {
            features.push_back({ "VSX", "1" });
        }
        if (lm_ggml_cpu_has_vxe()) {
            features.push_back({ "VXE", "1" });
        }
        if (lm_ggml_cpu_has_wasm_simd()) {
            features.push_back({ "WASM_SIMD", "1" });
        }
        if (lm_ggml_cpu_has_llamafile()) {
            features.push_back({ "LLAMAFILE", "1" });
        }
    #ifdef LM_GGML_USE_ACCELERATE
        features.push_back({ "ACCELERATE", "1" });
    #endif
    #ifdef LM_GGML_USE_CPU_HBM
        features.push_back({ "CPU_HBM", "1" });
    #endif
    #ifdef LM_GGML_USE_OPENMP
        features.push_back({ "OPENMP", "1" });
    #endif
    #ifdef LM_GGML_USE_CPU_KLEIDIAI
        features.push_back({ "KLEIDIAI", "1" });
    #endif
    #ifdef LM_GGML_USE_CPU_AARCH64
        features.push_back({ "AARCH64_REPACK", "1" });
    #endif

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    LM_GGML_UNUSED(reg);
}

static void * lm_ggml_backend_cpu_get_proc_address(lm_ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "lm_ggml_backend_set_n_threads") == 0) {
        lm_ggml_backend_set_n_threads_t fct = lm_ggml_backend_cpu_set_n_threads;
        return (void *)fct;
    }
    if (strcmp(name, "lm_ggml_backend_dev_get_extra_bufts") == 0) {
        lm_ggml_backend_dev_get_extra_bufts_t fct = lm_ggml_backend_cpu_device_get_extra_buffers_type;
        return (void *)fct;
    }
    if (strcmp(name, "lm_ggml_backend_get_features") == 0) {
        return (void *)lm_ggml_backend_cpu_get_features;
    }
    if (strcmp(name, "lm_ggml_backend_set_abort_callback") == 0) {
        return (void *)lm_ggml_backend_cpu_set_abort_callback;
    }
    if (strcmp(name, "lm_ggml_backend_cpu_numa_init") == 0) {
        return (void *)lm_ggml_numa_init;
    }
    if (strcmp(name, "lm_ggml_backend_cpu_is_numa") == 0) {
        return (void *)lm_ggml_is_numa;
    }

    // threadpool - TODO:  move to ggml-base
    if (strcmp(name, "lm_ggml_threadpool_new") == 0) {
        return (void *)lm_ggml_threadpool_new;
    }
    if (strcmp(name, "lm_ggml_threadpool_free") == 0) {
        return (void *)lm_ggml_threadpool_free;
    }
    if (strcmp(name, "lm_ggml_backend_cpu_set_threadpool") == 0) {
        return (void *)lm_ggml_backend_cpu_set_threadpool;
    }

    return NULL;

    LM_GGML_UNUSED(reg);
}

static const struct lm_ggml_backend_reg_i lm_ggml_backend_cpu_reg_i = {
    /* .get_name         = */ lm_ggml_backend_cpu_reg_get_name,
    /* .get_device_count = */ lm_ggml_backend_cpu_reg_get_device_count,
    /* .get_device       = */ lm_ggml_backend_cpu_reg_get_device,
    /* .get_proc_address = */ lm_ggml_backend_cpu_get_proc_address,
};

lm_ggml_backend_reg_t lm_ggml_backend_cpu_reg(void) {
    // init CPU feature detection
    lm_ggml_cpu_init();

    static struct lm_ggml_backend_reg lm_ggml_backend_cpu_reg = {
        /* .api_version = */ LM_GGML_BACKEND_API_VERSION,
        /* .iface       = */ lm_ggml_backend_cpu_reg_i,
        /* .context     = */ NULL,
    };

    return &lm_ggml_backend_cpu_reg;
}

LM_GGML_BACKEND_DL_IMPL(lm_ggml_backend_cpu_reg)
