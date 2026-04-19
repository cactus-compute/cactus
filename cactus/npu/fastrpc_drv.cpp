#include "fastrpc_drv.h"

#include <cstring>
#include <string>
#include <vector>
#include <filesystem>
#include <memory>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Platform DLL helpers
// ---------------------------------------------------------------------------
#ifdef _WIN32
#  include <windows.h>
#  include <winsvc.h>

using DllHandle = HMODULE;
static DllHandle dll_load(const std::string& path) {
    DWORD old = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old | SEM_FAILCRITICALERRORS);
    DllHandle h = LoadLibraryA(path.c_str());
    SetErrorMode(old);
    return h;
}
static void* dll_sym(DllHandle h, const char* sym) {
    return (void*)GetProcAddress(h, sym);
}

// Resolve the driver store path for the qcnspmcdm service (same as htp-drv.cpp).
static std::string get_cdsprpc_path() {
    std::wstring svc = L"qcnspmcdm";
    std::string result;

    SC_HANDLE scm = OpenSCManagerW(nullptr, nullptr, STANDARD_RIGHTS_READ);
    if (!scm) return result;

    SC_HANDLE svc_h = OpenServiceW(scm, svc.c_str(), SERVICE_QUERY_CONFIG);
    if (!svc_h) { CloseServiceHandle(scm); return result; }

    DWORD buf_sz = 0;
    QueryServiceConfigW(svc_h, nullptr, 0, &buf_sz);
    std::vector<char> buf(buf_sz);
    LPQUERY_SERVICE_CONFIGW cfg = reinterpret_cast<LPQUERY_SERVICE_CONFIGW>(buf.data());
    if (!QueryServiceConfigW(svc_h, cfg, buf_sz, &buf_sz)) {
        CloseServiceHandle(svc_h); CloseServiceHandle(scm); return result;
    }

    // lpBinaryPathName is like \SystemRoot\System32\DriverStore\...\qcadsprpc.sys
    std::wstring drv(cfg->lpBinaryPathName);
    CloseServiceHandle(svc_h); CloseServiceHandle(scm);

    // Strip filename, keep directory
    size_t last_bs = drv.rfind(L'\\');
    if (last_bs != std::wstring::npos) drv = drv.substr(0, last_bs);

    // Replace \SystemRoot with actual Windows path
    const std::wstring placeholder = L"\\SystemRoot";
    if (drv.compare(0, placeholder.size(), placeholder) == 0) {
        DWORD n = GetEnvironmentVariableW(L"windir", nullptr, 0);
        if (n) {
            std::vector<wchar_t> windir(n + 1);
            GetEnvironmentVariableW(L"windir", windir.data(), n + 1);
            drv.replace(0, placeholder.size(), std::wstring(windir.data()));
        }
    }

    // Convert wide to UTF-8
    int bytes = WideCharToMultiByte(CP_UTF8, 0, drv.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (bytes > 0) {
        result.resize(bytes - 1);
        WideCharToMultiByte(CP_UTF8, 0, drv.c_str(), -1, result.data(), bytes, nullptr, nullptr);
    }
    return result;
}

#else
#  include <dlfcn.h>
using DllHandle = void*;
static DllHandle dll_load(const std::string& path) {
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
}
static void* dll_sym(DllHandle h, const char* sym) { return dlsym(h, sym); }
static std::string get_cdsprpc_path() { return ""; }
#endif

// ---------------------------------------------------------------------------
// Function pointer tables
// ---------------------------------------------------------------------------
typedef void*  (*rpcmem_alloc_fn)(int, uint32_t, int);
typedef void*  (*rpcmem_alloc2_fn)(int, uint32_t, size_t);
typedef void   (*rpcmem_free_fn)(void*);
typedef int    (*rpcmem_to_fd_fn)(void*);

typedef int    (*fastrpc_mmap_fn)(int, int, void*, int, size_t, enum fastrpc_map_flags);
typedef int    (*fastrpc_munmap_fn)(int, int, void*, size_t);

typedef AEEResult (*dspqueue_create_fn)(int, uint32_t, uint32_t, uint32_t,
                                        dspqueue_callback_t, dspqueue_callback_t, void*,
                                        dspqueue_t*);
typedef AEEResult (*dspqueue_close_fn)(dspqueue_t);
typedef AEEResult (*dspqueue_export_fn)(dspqueue_t, uint64_t*);
typedef AEEResult (*dspqueue_write_fn)(dspqueue_t, uint32_t, uint32_t,
                                       struct dspqueue_buffer*, uint32_t, const uint8_t*, uint32_t);
typedef AEEResult (*dspqueue_read_fn)(dspqueue_t, uint32_t*, uint32_t, uint32_t*,
                                      struct dspqueue_buffer*, uint32_t, uint32_t*,
                                      uint8_t*, uint32_t);

typedef int (*rh64_open_fn)(const char*, remote_handle64*);
typedef int (*rh64_close_fn)(remote_handle64);
typedef int (*rh64_invoke_fn)(remote_handle64, uint32_t, struct remote_arg*);
typedef int (*rh_control_fn)(uint32_t, void*, uint32_t);
typedef int (*rh64_control_fn)(remote_handle64, uint32_t, void*, uint32_t);
typedef int (*sess_control_fn)(uint32_t, void*, uint32_t);

static rpcmem_alloc_fn    g_rpcmem_alloc    = nullptr;
static rpcmem_alloc2_fn   g_rpcmem_alloc2   = nullptr;
static rpcmem_free_fn     g_rpcmem_free     = nullptr;
static rpcmem_to_fd_fn    g_rpcmem_to_fd    = nullptr;
static fastrpc_mmap_fn    g_fastrpc_mmap    = nullptr;
static fastrpc_munmap_fn  g_fastrpc_munmap  = nullptr;
static dspqueue_create_fn g_dspqueue_create = nullptr;
static dspqueue_close_fn  g_dspqueue_close  = nullptr;
static dspqueue_export_fn g_dspqueue_export = nullptr;
static dspqueue_write_fn  g_dspqueue_write  = nullptr;
static dspqueue_read_fn   g_dspqueue_read   = nullptr;
static rh64_open_fn       g_rh64_open       = nullptr;
static rh64_close_fn      g_rh64_close      = nullptr;
static rh64_invoke_fn     g_rh64_invoke     = nullptr;
static rh_control_fn      g_rh_control      = nullptr;
static rh64_control_fn    g_rh64_control    = nullptr;
static sess_control_fn    g_sess_control    = nullptr;

static bool g_initialized = false;
static DllHandle g_dll = nullptr;

#define LOAD_SYM(type, var, name, required) \
    do { \
        var = (type)dll_sym(g_dll, #name); \
        if (required && !var) { \
            fprintf(stderr, "[FastRPC] failed to load symbol: %s\n", #name); \
            return AEE_EUNABLETOLOAD; \
        } \
    } while(0)

AEEResult fastrpc_drv_init(void) {
    if (g_initialized) return AEE_SUCCESS;

#ifdef _WIN32
    // Try candidate paths in order:
    // 1. Path from qcnspmcdm service registry entry
    // 2. All driver store folders containing libcdsprpc.dll (newest first via glob)
    // 3. System PATH fallback
    std::vector<std::string> candidates;
    {
        std::string svc_dir = get_cdsprpc_path();
        if (!svc_dir.empty()) candidates.push_back(svc_dir + "\\libcdsprpc.dll");
    }
    // Enumerate driver store folders
    {
        std::string ds = "C:\\Windows\\System32\\DriverStore\\FileRepository";
        WIN32_FIND_DATAA fd;
        HANDLE hf = FindFirstFileA((ds + "\\qcnspmcdm*.inf_*").c_str(), &fd);
        if (hf != INVALID_HANDLE_VALUE) {
            // Collect with write time; sort newest-install first by DLL modification time
            std::vector<std::pair<FILETIME, std::string>> folders;
            do {
                if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
                    std::string p = ds + "\\" + fd.cFileName + "\\libcdsprpc.dll";
                    // Get the DLL's own write time so we pick the most recently installed driver
                    WIN32_FILE_ATTRIBUTE_DATA fad = {};
                    FILETIME ft = fd.ftLastWriteTime; // fallback: folder write time
                    if (GetFileAttributesExA(p.c_str(), GetFileExInfoStandard, &fad))
                        ft = fad.ftLastWriteTime;
                    folders.push_back({ft, p});
                }
            } while (FindNextFileA(hf, &fd));
            FindClose(hf);
            // Sort descending by write time (newest first)
            std::sort(folders.begin(), folders.end(),
                [](const auto& a, const auto& b) {
                    return CompareFileTime(&a.first, &b.first) > 0;
                });
            for (auto& [ft, p] : folders) candidates.push_back(p);
        }
    }
    candidates.push_back("libcdsprpc.dll"); // last resort: PATH

    std::string dll_path;
    for (auto& c : candidates) {
        if (c == "libcdsprpc.dll" || fs::exists(c)) {
            dll_path = c;
            break;
        }
    }
#else
    std::string dll_path = "libcdsprpc.so";
#endif

    fprintf(stderr, "[FastRPC] loading %s\n", dll_path.c_str());
    g_dll = dll_load(dll_path);
    if (!g_dll) {
        fprintf(stderr, "[FastRPC] failed to load %s\n", dll_path.c_str());
        return AEE_EUNABLETOLOAD;
    }

    LOAD_SYM(rpcmem_alloc_fn,    g_rpcmem_alloc,    rpcmem_alloc,              true);
    LOAD_SYM(rpcmem_alloc2_fn,   g_rpcmem_alloc2,   rpcmem_alloc2,             false);
    LOAD_SYM(rpcmem_free_fn,     g_rpcmem_free,     rpcmem_free,               true);
    LOAD_SYM(rpcmem_to_fd_fn,    g_rpcmem_to_fd,    rpcmem_to_fd,              true);
    LOAD_SYM(fastrpc_mmap_fn,    g_fastrpc_mmap,    fastrpc_mmap,              true);
    LOAD_SYM(fastrpc_munmap_fn,  g_fastrpc_munmap,  fastrpc_munmap,            true);
    LOAD_SYM(dspqueue_create_fn, g_dspqueue_create, dspqueue_create,           true);
    LOAD_SYM(dspqueue_close_fn,  g_dspqueue_close,  dspqueue_close,            true);
    LOAD_SYM(dspqueue_export_fn, g_dspqueue_export, dspqueue_export,           true);
    LOAD_SYM(dspqueue_write_fn,  g_dspqueue_write,  dspqueue_write,            true);
    LOAD_SYM(dspqueue_read_fn,   g_dspqueue_read,   dspqueue_read,             true);
    LOAD_SYM(rh64_open_fn,       g_rh64_open,       remote_handle64_open,      true);
    LOAD_SYM(rh64_close_fn,      g_rh64_close,      remote_handle64_close,     true);
    LOAD_SYM(rh64_invoke_fn,     g_rh64_invoke,     remote_handle64_invoke,    true);
    LOAD_SYM(rh_control_fn,      g_rh_control,      remote_handle_control,     true);
    LOAD_SYM(rh64_control_fn,    g_rh64_control,    remote_handle64_control,   false);
    LOAD_SYM(sess_control_fn,    g_sess_control,    remote_session_control,    false);

    g_initialized = true;
    fprintf(stderr, "[FastRPC] driver loaded\n");
    return AEE_SUCCESS;
}

int fastrpc_drv_get_arch(int* arch_out) {
    if (!g_rh_control) return -1;

    // Dump all capability attribute IDs to find the arch version
    fprintf(stderr, "[FastRPC] probing CDSP capabilities (domain=3):\n");
    for (uint32_t attr = 0; attr <= 15; attr++) {
        struct remote_dsp_capability cap;
        cap.domain       = 3;
        cap.attribute_ID = attr;
        cap.capability   = 0;
        int err = g_rh_control(DSPRPC_GET_DSP_INFO, &cap, sizeof(cap));
        fprintf(stderr, "[FastRPC]   attr[%2u] err=0x%x cap=0x%x (%u)\n",
                attr, err, cap.capability, cap.capability);
    }

    struct remote_dsp_capability cap;
    cap.domain       = 3;
    cap.attribute_ID = ARCH_VER;
    cap.capability   = 0;
    int err = g_rh_control(DSPRPC_GET_DSP_INFO, &cap, sizeof(cap));
    if (err != AEE_SUCCESS) return err;

    // Low byte of capability is a hex-coded arch identifier (0x73 for v73, etc.)
    switch (cap.capability & 0xff) {
        case 0x68: *arch_out = 68; return 0;
        case 0x69: *arch_out = 69; return 0;
        case 0x73: *arch_out = 73; return 0;
        case 0x75: *arch_out = 75; return 0;
        case 0x79: *arch_out = 79; return 0;
        case 0x81: *arch_out = 81; return 0;
    }
    fprintf(stderr, "[FastRPC] get_arch: unrecognized capability 0x%x, defaulting to v73\n", cap.capability);
    *arch_out = 73;
    return 0;
}

// ---------------------------------------------------------------------------
// Public API forwarding
// ---------------------------------------------------------------------------
void* rpcmem_alloc(int heapid, uint32_t flags, int size) {
    if (g_rpcmem_alloc2) return g_rpcmem_alloc2(heapid, flags, (size_t)size);
    return g_rpcmem_alloc ? g_rpcmem_alloc(heapid, flags, size) : nullptr;
}
void rpcmem_free(void* p)       { if (g_rpcmem_free)    g_rpcmem_free(p); }
int  rpcmem_to_fd(void* p)      { return g_rpcmem_to_fd ? g_rpcmem_to_fd(p) : -1; }

int fastrpc_mmap(int domain, int fd, void* addr, int offset, size_t len,
                 enum fastrpc_map_flags flags) {
    return g_fastrpc_mmap ? g_fastrpc_mmap(domain, fd, addr, offset, len, flags) : -1;
}
int fastrpc_munmap(int domain, int fd, void* addr, size_t len) {
    return g_fastrpc_munmap ? g_fastrpc_munmap(domain, fd, addr, len) : -1;
}

AEEResult dspqueue_create(int domain, uint32_t flags,
                          uint32_t req_sz, uint32_t resp_sz,
                          dspqueue_callback_t pcb, dspqueue_callback_t ecb,
                          void* ctx, dspqueue_t* q) {
    return g_dspqueue_create ? g_dspqueue_create(domain, flags, req_sz, resp_sz, pcb, ecb, ctx, q) : AEE_EUNABLETOLOAD;
}
AEEResult dspqueue_close(dspqueue_t q)  { return g_dspqueue_close ? g_dspqueue_close(q) : AEE_EUNABLETOLOAD; }
AEEResult dspqueue_export(dspqueue_t q, uint64_t* id) { return g_dspqueue_export ? g_dspqueue_export(q, id) : AEE_EUNABLETOLOAD; }
AEEResult dspqueue_write(dspqueue_t q, uint32_t flags, uint32_t nb,
                         struct dspqueue_buffer* bufs, uint32_t ml, const uint8_t* msg, uint32_t to) {
    return g_dspqueue_write ? g_dspqueue_write(q, flags, nb, bufs, ml, msg, to) : AEE_EUNABLETOLOAD;
}
AEEResult dspqueue_read(dspqueue_t q, uint32_t* fl, uint32_t mb, uint32_t* nb,
                        struct dspqueue_buffer* bufs, uint32_t mml, uint32_t* ml, uint8_t* msg, uint32_t to) {
    return g_dspqueue_read ? g_dspqueue_read(q, fl, mb, nb, bufs, mml, ml, msg, to) : AEE_EUNABLETOLOAD;
}

int remote_handle64_open(const char* uri, remote_handle64* h) {
    return g_rh64_open ? g_rh64_open(uri, h) : AEE_EUNABLETOLOAD;
}
int remote_handle64_close(remote_handle64 h) {
    return g_rh64_close ? g_rh64_close(h) : AEE_EUNABLETOLOAD;
}
int remote_handle64_invoke(remote_handle64 h, uint32_t sc, struct remote_arg* pra) {
    return g_rh64_invoke ? g_rh64_invoke(h, sc, pra) : AEE_EUNABLETOLOAD;
}
int remote_handle_control(uint32_t req, void* data, uint32_t len) {
    return g_rh_control ? g_rh_control(req, data, len) : AEE_EUNSUPPORTEDAPI;
}
int remote_handle64_control(remote_handle64 h, uint32_t req, void* data, uint32_t len) {
    return g_rh64_control ? g_rh64_control(h, req, data, len) : AEE_EUNSUPPORTEDAPI;
}
int remote_session_control(uint32_t req, void* data, uint32_t len) {
    return g_sess_control ? g_sess_control(req, data, len) : AEE_EUNSUPPORTEDAPI;
}
