// End-to-end test: build a QNN graph that runs our CactusMatMul op on HTP,
// compare output against CPU reference. Proves the op package actually
// executes on the DSP (not just loads).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <windows.h>
#undef interface

#include "QnnInterface.h"
#include "QnnBackend.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnTensor.h"
#include "QnnLog.h"
#include "HTP/QnnHtpDevice.h"

typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(
    const QnnInterface_t*** providerList, uint32_t* numProviders);

static void qnn_log_cb(const char* fmt, QnnLog_Level_t /*lvl*/, uint64_t /*ts*/, va_list args) {
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
}

static bool file_exists(const std::string& p) {
    DWORD a = GetFileAttributesA(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

// FP32 -> FP16 (IEEE 754 half-precision) bit conversion.
static uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 31) & 0x1;
    int32_t  exp  = ((x >> 23) & 0xff) - 127;
    uint32_t man  = x & 0x7fffff;
    if (exp > 15) return uint16_t((sign << 15) | 0x7c00);            // inf
    if (exp < -14) return uint16_t(sign << 15);                       // zero/subnorm -> 0
    uint16_t hexp = uint16_t(exp + 15);
    uint16_t hman = uint16_t(man >> 13);
    return uint16_t((sign << 15) | (hexp << 10) | hman);
}

static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t man  = h & 0x3ff;
    uint32_t x;
    if (exp == 0) {
        x = sign << 31;
    } else if (exp == 0x1f) {
        x = (sign << 31) | (0xff << 23) | (man << 13);
    } else {
        x = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

int main(int argc, char** argv) {
    const char* sdk_root = std::getenv("QNN_SDK_ROOT");
    if (!sdk_root) sdk_root = "C:\\Qualcomm\\AIStack\\QAIRT\\2.31.0.250130";

    const char* pkg_path = (argc > 1) ? argv[1]
        : "C:\\Users\\justi\\GitRepos\\qualcomm-npu\\third_party\\cactus\\cactus\\npu\\udo\\CactusMatMulPackage\\build\\hexagon-v73\\libQnnCactusMatMulPackage.so";

    // Test shapes
    const uint32_t M = 4, K = 8, N = 16;

    std::string lib_dir = std::string(sdk_root) + "\\lib\\aarch64-windows-msvc";
    std::string qnn_htp = lib_dir + "\\QnnHtp.dll";

    fprintf(stderr, "[test] QnnHtp.dll=%s\n", qnn_htp.c_str());
    fprintf(stderr, "[test] pkg=%s (%s)\n", pkg_path, file_exists(pkg_path) ? "ok" : "MISSING");

    SetDllDirectoryA(lib_dir.c_str());
    for (const char* name : {"QnnSystem.dll", "QnnHtpV73Stub.dll", "QnnHtpPrepare.dll"}) {
        LoadLibraryA((lib_dir + "\\" + name).c_str());
    }
    HMODULE dll = LoadLibraryA(qnn_htp.c_str());
    if (!dll) { fprintf(stderr, "FAIL: LoadLibrary QnnHtp\n"); return 1; }

    auto fn = reinterpret_cast<QnnInterfaceGetProvidersFn_t>(
        reinterpret_cast<void*>(GetProcAddress(dll, "QnnInterface_getProviders")));
    const QnnInterface_t** providers = nullptr;
    uint32_t num = 0;
    fn(&providers, &num);
    auto qnn = providers[0]->QNN_INTERFACE_VER_NAME;

    Qnn_LogHandle_t log_h = nullptr;
    qnn.logCreate(qnn_log_cb, QNN_LOG_LEVEL_ERROR, &log_h);
    Qnn_BackendHandle_t backend_h = nullptr;
    qnn.backendCreate(log_h, nullptr, &backend_h);
    fprintf(stderr, "[test] backend created\n");

    Qnn_DeviceHandle_t device_h = nullptr;
    QnnHtpDevice_CustomConfig_t htp_cfg{};
    htp_cfg.option = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
    htp_cfg.socModel = 60;  // SC8380XP
    QnnDevice_Config_t dev_cfg = QNN_DEVICE_CONFIG_INIT;
    dev_cfg.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
    dev_cfg.customConfig = (QnnDevice_CustomConfig_t)&htp_cfg;
    const QnnDevice_Config_t* dev_cfgs[] = {&dev_cfg, nullptr};
    if (qnn.deviceCreate(log_h, dev_cfgs, &device_h) != QNN_SUCCESS) {
        qnn.deviceCreate(log_h, nullptr, &device_h);  // fallback
    }

    Qnn_ContextHandle_t ctx = nullptr;
    if (qnn.contextCreate(backend_h, device_h, nullptr, &ctx) != QNN_SUCCESS) {
        fprintf(stderr, "FAIL: contextCreate\n"); return 1;
    }
    fprintf(stderr, "[test] context created\n");

    // Register op package
    if (qnn.backendRegisterOpPackage(backend_h, pkg_path,
                                     "CactusMatMulPackageInterfaceProvider",
                                     "HTP") != QNN_SUCCESS) {
        fprintf(stderr, "FAIL: registerOpPackage\n"); return 1;
    }
    fprintf(stderr, "[test] op package registered\n");

    // Create graph
    Qnn_GraphHandle_t graph = nullptr;
    if (qnn.graphCreate(ctx, "matmul_graph", nullptr, &graph) != QNN_SUCCESS) {
        fprintf(stderr, "FAIL: graphCreate\n"); return 1;
    }

    auto mk_tensor = [&](const char* name, Qnn_TensorType_t ttype, uint32_t* dims, uint32_t rank) {
        Qnn_Tensor_t t = QNN_TENSOR_INIT;
        t.version = QNN_TENSOR_VERSION_1;
        t.v1.id = 0;
        t.v1.name = name;
        t.v1.type = ttype;
        t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
        t.v1.dataType = QNN_DATATYPE_FLOAT_16;
        t.v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
        t.v1.rank = rank;
        t.v1.dimensions = dims;
        t.v1.memType = QNN_TENSORMEMTYPE_RAW;
        t.v1.clientBuf = {nullptr, 0};
        return t;
    };

    uint32_t a_dims[4] = {1, 1, M, K};
    uint32_t b_dims[4] = {1, 1, K, N};
    uint32_t c_dims[4] = {1, 1, M, N};
    Qnn_Tensor_t a_t = mk_tensor("A", QNN_TENSOR_TYPE_APP_WRITE, a_dims, 4);
    Qnn_Tensor_t b_t = mk_tensor("B", QNN_TENSOR_TYPE_APP_WRITE, b_dims, 4);
    Qnn_Tensor_t c_t = mk_tensor("C", QNN_TENSOR_TYPE_APP_READ,  c_dims, 4);

    if (qnn.tensorCreateGraphTensor(graph, &a_t) != QNN_SUCCESS ||
        qnn.tensorCreateGraphTensor(graph, &b_t) != QNN_SUCCESS ||
        qnn.tensorCreateGraphTensor(graph, &c_t) != QNN_SUCCESS) {
        fprintf(stderr, "FAIL: tensorCreateGraphTensor\n"); return 1;
    }
    fprintf(stderr, "[test] tensors created a.id=%u b.id=%u c.id=%u\n",
            a_t.v1.id, b_t.v1.id, c_t.v1.id);

    // Add the CactusMatMul node
    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
    op.version = QNN_OPCONFIG_VERSION_1;
    op.v1.name = "matmul_node";
    op.v1.packageName = "CactusMatMulPackage";
    op.v1.typeName = "CactusMatMul";
    Qnn_Tensor_t ins[2] = {a_t, b_t};
    Qnn_Tensor_t outs[1] = {c_t};
    op.v1.numOfInputs = 2; op.v1.inputTensors = ins;
    op.v1.numOfOutputs = 1; op.v1.outputTensors = outs;
    op.v1.numOfParams = 0; op.v1.params = nullptr;

    Qnn_ErrorHandle_t rc = qnn.graphAddNode(graph, op);
    if (rc != QNN_SUCCESS) {
        fprintf(stderr, "FAIL: graphAddNode rc=0x%llx\n", (unsigned long long)rc); return 1;
    }
    fprintf(stderr, "[test] node added\n");

    rc = qnn.graphFinalize(graph, nullptr, nullptr);
    if (rc != QNN_SUCCESS) {
        fprintf(stderr, "FAIL: graphFinalize rc=0x%llx\n", (unsigned long long)rc); return 1;
    }
    fprintf(stderr, "[test] graph finalized\n");

    // Prepare input data: fill A and B with random FP32 -> FP16.
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint16_t> A_h(M*K), B_h(K*N), C_h(M*N, 0), C_ref(M*N, 0);
    std::vector<float>    A_f(M*K), B_f(K*N);
    for (auto& x : A_f) x = dist(rng);
    for (auto& x : B_f) x = dist(rng);
    for (size_t i = 0; i < A_f.size(); i++) A_h[i] = fp32_to_fp16(A_f[i]);
    for (size_t i = 0; i < B_f.size(); i++) B_h[i] = fp32_to_fp16(B_f[i]);

    // CPU reference matmul (FP32 accumulate, then round to FP16).
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t n = 0; n < N; ++n) {
            float s = 0.0f;
            for (uint32_t k = 0; k < K; ++k)
                s += fp16_to_fp32(A_h[m*K+k]) * fp16_to_fp32(B_h[k*N+n]);
            C_ref[m*N+n] = fp32_to_fp16(s);
        }

    a_t.v1.clientBuf = {A_h.data(), (uint32_t)(A_h.size()*2)};
    b_t.v1.clientBuf = {B_h.data(), (uint32_t)(B_h.size()*2)};
    c_t.v1.clientBuf = {C_h.data(), (uint32_t)(C_h.size()*2)};

    Qnn_Tensor_t exec_ins[2]  = {a_t, b_t};
    Qnn_Tensor_t exec_outs[1] = {c_t};
    rc = qnn.graphExecute(graph, exec_ins, 2, exec_outs, 1, nullptr, nullptr);
    if (rc != QNN_SUCCESS) {
        fprintf(stderr, "FAIL: graphExecute rc=0x%llx\n", (unsigned long long)rc); return 1;
    }
    fprintf(stderr, "[test] graph executed\n");

    // Compare NPU output vs CPU reference.
    float max_abs = 0, mean_abs = 0;
    int bad = 0;
    for (size_t i = 0; i < C_h.size(); ++i) {
        float npu = fp16_to_fp32(C_h[i]);
        float cpu = fp16_to_fp32(C_ref[i]);
        float e = std::fabs(npu - cpu);
        max_abs = std::max(max_abs, e);
        mean_abs += e;
        if (e > 0.05f) bad++;
    }
    mean_abs /= C_h.size();
    fprintf(stderr, "[test] output: max_abs_err=%.4f mean_abs_err=%.4f bad=%d/%zu\n",
            max_abs, mean_abs, bad, C_h.size());

    // Print first row for eyeball
    fprintf(stderr, "[test] C_npu row0 = ");
    for (uint32_t j = 0; j < N; j++) fprintf(stderr, "%.3f ", fp16_to_fp32(C_h[j]));
    fprintf(stderr, "\n[test] C_cpu row0 = ");
    for (uint32_t j = 0; j < N; j++) fprintf(stderr, "%.3f ", fp16_to_fp32(C_ref[j]));
    fprintf(stderr, "\n");

    if (bad == 0) fprintf(stderr, "PASS: matmul output matches CPU reference\n");
    else          fprintf(stderr, "FAIL: %d elements differ by > 0.05\n", bad);

    qnn.contextFree(ctx, nullptr);
    qnn.backendFree(backend_h);
    qnn.logFree(log_h);
    return bad == 0 ? 0 : 1;
}
