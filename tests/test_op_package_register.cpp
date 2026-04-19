// Minimal QNN op package registration test.
// Proves that a user-built HTP op package can be loaded by Qualcomm's signed
// QnnHtp.dll runtime — our signing-wall workaround for direct FastRPC skels.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <windows.h>
#undef interface

#include "QnnInterface.h"
#include "QnnBackend.h"
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

int main(int argc, char** argv) {
    const char* sdk_root = std::getenv("QNN_SDK_ROOT");
    if (!sdk_root) sdk_root = "C:\\Qualcomm\\AIStack\\QAIRT\\2.31.0.250130";

    const char* pkg_path = (argc > 1) ? argv[1]
        : "C:\\Users\\justi\\GitRepos\\qualcomm-npu\\third_party\\cactus\\cactus\\npu\\udo\\CactusSquarePackage\\build\\hexagon-v73\\libQnnCactusSquarePackage.so";

    std::string lib_dir = std::string(sdk_root) + "\\lib\\aarch64-windows-msvc";
    std::string qnn_htp = lib_dir + "\\QnnHtp.dll";

    fprintf(stderr, "[test] QNN_SDK_ROOT=%s\n", sdk_root);
    fprintf(stderr, "[test] QnnHtp.dll=%s (%s)\n", qnn_htp.c_str(),
            file_exists(qnn_htp) ? "exists" : "MISSING");
    fprintf(stderr, "[test] pkg=%s (%s)\n", pkg_path,
            file_exists(pkg_path) ? "exists" : "MISSING");

    SetDllDirectoryA(lib_dir.c_str());
    const char* companions[] = {
        "QnnSystem.dll",
        "QnnHtpV73Stub.dll",
        "QnnHtpPrepare.dll",
        nullptr
    };
    for (int i = 0; companions[i]; i++) {
        std::string full = lib_dir + "\\" + companions[i];
        HMODULE h = LoadLibraryA(full.c_str());
        fprintf(stderr, "[test] pre-load %s: %s\n", companions[i], h ? "ok" : "fail");
    }

    HMODULE dll = LoadLibraryA(qnn_htp.c_str());
    if (!dll) { fprintf(stderr, "FAIL: LoadLibrary QnnHtp.dll (err %lu)\n", GetLastError()); return 1; }
    fprintf(stderr, "[test] QnnHtp.dll loaded\n");

    auto fn = reinterpret_cast<QnnInterfaceGetProvidersFn_t>(
        reinterpret_cast<void*>(GetProcAddress(dll, "QnnInterface_getProviders")));
    if (!fn) { fprintf(stderr, "FAIL: QnnInterface_getProviders not found\n"); return 1; }

    const QnnInterface_t** providers = nullptr;
    uint32_t num = 0;
    if (fn(&providers, &num) != QNN_SUCCESS || num == 0) {
        fprintf(stderr, "FAIL: getProviders\n"); return 1;
    }
    auto qnn = providers[0]->QNN_INTERFACE_VER_NAME;
    fprintf(stderr, "[test] provider ok (backendId=%u numProviders=%u)\n", providers[0]->backendId, num);

    Qnn_LogHandle_t log_h = nullptr;
    if (qnn.logCreate) qnn.logCreate(qnn_log_cb, QNN_LOG_LEVEL_VERBOSE, &log_h);

    Qnn_BackendHandle_t backend_h = nullptr;
    if (!qnn.backendCreate || qnn.backendCreate(log_h, nullptr, &backend_h) != QNN_SUCCESS) {
        fprintf(stderr, "FAIL: backendCreate\n"); return 1;
    }
    fprintf(stderr, "[test] backend created\n");

    // Register our op package.
    // API: backendRegisterOpPackage(backend, packagePath, interfaceProvider, target)
    // target = "HTP" per QNN docs.
    const char* iface_provider = "CactusSquarePackageInterfaceProvider";
    const char* target = "HTP";
    if (!qnn.backendRegisterOpPackage) {
        fprintf(stderr, "FAIL: backendRegisterOpPackage not available\n"); return 1;
    }
    Qnn_ErrorHandle_t rc = qnn.backendRegisterOpPackage(
        backend_h, pkg_path, iface_provider, target);
    if (rc == QNN_SUCCESS) {
        fprintf(stderr, "PASS: op package registered (path=%s iface=%s target=%s)\n",
                pkg_path, iface_provider, target);
    } else {
        fprintf(stderr, "FAIL: backendRegisterOpPackage rc=0x%llx\n", (unsigned long long)rc);
    }

    if (qnn.backendFree) qnn.backendFree(backend_h);
    if (qnn.logFree) qnn.logFree(log_h);
    return rc == QNN_SUCCESS ? 0 : 1;
}
