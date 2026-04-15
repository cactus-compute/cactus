#include "npu_qnn_direct.h"

#ifdef CACTUS_HAS_QNN_DIRECT

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  #ifdef interface
  #undef interface
  #endif
#else
  #include <dlfcn.h>
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  #include <unistd.h>
  #include <dirent.h>
  #include <limits.h>
  #include <libgen.h>
#endif

#include <QNN/QnnInterface.h>
#include <QNN/QnnBackend.h>
#include <QNN/QnnContext.h>
#include <QNN/QnnGraph.h>
#include <QNN/QnnTensor.h>
#include <QNN/QnnOpDef.h>
#include <QNN/QnnLog.h>
#include <QNN/HTP/QnnHtpGraph.h>
#include <QNN/HTP/QnnHtpContext.h>
#include <QNN/HTP/QnnHtpDevice.h>
#include <QNN/QnnDevice.h>
#include <QNN/QnnTypes.h>

#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <fstream>
#include <stdexcept>
#include <chrono>
#include <thread>
#include <atomic>

static inline int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(
    const QnnInterface_t*** providerList, uint32_t* numProviders);

static void qnn_log_cb(const char* fmt, QnnLog_Level_t level,
                       uint64_t /*ts*/, va_list args) {
    if (level <= QNN_LOG_LEVEL_ERROR) {
        fprintf(stderr, "[QNN] ");
        vfprintf(stderr, fmt, args);
        fprintf(stderr, "\n");
        fflush(stderr);
    }
}

static bool file_exists(const std::string& p) {
#ifdef _WIN32
    DWORD a = GetFileAttributesA(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st;
    return stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
#endif
}

static std::string find_qnn_htp_lib() {
#ifdef _WIN32
    auto probe = [](const std::string& dir, bool require_prepare) -> std::string {
        std::string htp  = dir + "\\QnnHtp.dll";
        if (!file_exists(htp)) return "";
        if (!require_prepare) return htp;
        if (file_exists(dir + "\\QnnHtpPrepare.dll")) return htp;
        if (file_exists(dir + "\\HTP\\QnnHtpPrepareDrv.dll")) return htp;
        return "";
    };

    auto search_python_ort = [&]() -> std::string {
        char lad_buf[MAX_PATH] = {}, appd_buf[MAX_PATH] = {};
        GetEnvironmentVariableA("LOCALAPPDATA", lad_buf, MAX_PATH);
        GetEnvironmentVariableA("APPDATA",      appd_buf, MAX_PATH);
        const char* roots[] = { *lad_buf ? lad_buf : nullptr,
                                 *appd_buf ? appd_buf : nullptr, nullptr };
        WIN32_FIND_DATAA fd;
        for (int ri = 0; roots[ri]; ri++) {
            std::string pat1 = std::string(roots[ri]) + "\\Python\\*";
            HANDLE h = FindFirstFileA(pat1.c_str(), &fd);
            if (h != INVALID_HANDLE_VALUE) {
                do {
                    if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) continue;
                    std::string d = std::string(roots[ri]) + "\\Python\\" + fd.cFileName
                                  + "\\Lib\\site-packages\\onnxruntime\\capi";
                    std::string r = probe(d, true);
                    if (!r.empty()) { FindClose(h); return r; }
                    d = std::string(roots[ri]) + "\\Python\\" + fd.cFileName
                      + "\\site-packages\\onnxruntime\\capi";
                    r = probe(d, true);
                    if (!r.empty()) { FindClose(h); return r; }
                } while (FindNextFileA(h, &fd));
                FindClose(h);
            }
        }
        return "";
    };

    auto search_driver_store = [&]() -> std::string {
        const char* ds = "C:\\Windows\\System32\\DriverStore\\FileRepository";
        WIN32_FIND_DATAA fd;
        std::string pat = std::string(ds) + "\\qcnspmcdm8380*";
        HANDLE h = FindFirstFileA(pat.c_str(), &fd);
        if (h == INVALID_HANDLE_VALUE) return "";
        std::string result;
        do {
            if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
                std::string r = probe(std::string(ds) + "\\" + fd.cFileName, false);
                if (!r.empty()) result = r;
            }
        } while (FindNextFileA(h, &fd));
        FindClose(h);
        return result;
    };

    auto posix_to_win = [](const std::string& s) -> std::string {
        if (s.size() >= 3 && s[0] == '/' && s[2] == '/' &&
            ((s[1] >= 'a' && s[1] <= 'z') || (s[1] >= 'A' && s[1] <= 'Z'))) {
            std::string w;
            w += (char)(s[1] & ~0x20);
            w += ':';
            w += '\\';
            for (size_t i = 3; i < s.size(); i++)
                w += (s[i] == '/') ? '\\' : s[i];
            return w;
        }
        std::string w = s;
        for (char& c : w) if (c == '/') c = '\\';
        return w;
    };

    auto search_path_env = [&]() -> std::string {
        const char* path_env = getenv("PATH");
        if (!path_env) return "";
        char sep = strchr(path_env, ';') ? ';' : ':';
        const char* p = path_env;
        while (*p) {
            const char* delim = strchr(p, sep);
            std::string dir(p, delim ? (size_t)(delim - p) : strlen(p));
            if (!dir.empty()) {
                std::string wdir = posix_to_win(dir);
                std::string r2 = probe(wdir, true);
                if (!r2.empty()) return r2;
            }
            if (!delim) break;
            p = delim + 1;
        }
        return "";
    };

    static const char* known_ort_paths[] = {
        "C:\\Users\\justi\\AppData\\Local\\Python\\pythoncore-3.10-64\\Lib\\site-packages\\onnxruntime\\capi\\QnnHtp.dll",
        nullptr
    };
    for (int i = 0; known_ort_paths[i]; i++) {
        if (file_exists(known_ort_paths[i])) {
            fprintf(stderr, "[QNN Direct] using hardcoded ORT QnnHtp.dll: %s\n", known_ort_paths[i]);
            return known_ort_paths[i];
        }
    }

    std::string r = search_path_env();
    if (!r.empty()) {
        fprintf(stderr, "[QNN Direct] using ORT-bundled QnnHtp.dll from PATH: %s\n", r.c_str());
        return r;
    }
    r = search_python_ort();
    if (!r.empty()) {
        fprintf(stderr, "[QNN Direct] using ORT-bundled QnnHtp.dll (has QnnHtpPrepare.dll): %s\n", r.c_str());
        return r;
    }
    r = search_driver_store();
    if (!r.empty()) {
        fprintf(stderr, "[QNN Direct] using DriverStore QnnHtp.dll (fallback): %s\n", r.c_str());
        return r;
    }
    auto search_exe_relative = [&]() -> std::string {
        char exe_path[MAX_PATH] = {};
        if (!GetModuleFileNameA(NULL, exe_path, MAX_PATH)) return "";
        std::string cur = exe_path;
        auto slash = cur.find_last_of("\\/");
        if (slash != std::string::npos) cur = cur.substr(0, slash);
        for (int depth = 0; depth < 6; depth++) {
            std::string candidate = cur + "\\libs\\ort\\bin";
            std::string r2 = probe(candidate, true);
            if (!r2.empty()) return r2;
            r2 = probe(cur + "\\cactus\\libs\\ort\\bin", true);
            if (!r2.empty()) return r2;
            auto up = cur.find_last_of("\\/");
            if (up == std::string::npos || up == 0) break;
            cur = cur.substr(0, up);
        }
        return "";
    };
    r = search_exe_relative();
    if (!r.empty()) {
        fprintf(stderr, "[QNN Direct] using exe-relative ORT QnnHtp.dll: %s\n", r.c_str());
        return r;
    }
    fprintf(stderr, "[QNN Direct] search_python_ort missed — LOCALAPPDATA=%s\n",
            getenv("LOCALAPPDATA") ? getenv("LOCALAPPDATA") : "(null)");
    return "QnnHtp.dll";

#else // Linux

    auto probe = [](const std::string& dir, bool require_prepare) -> std::string {
        std::string htp = dir + "/libQnnHtp.so";
        if (!file_exists(htp)) return "";
        if (!require_prepare) return htp;
        if (file_exists(dir + "/libQnnHtpPrepare.so")) return htp;
        return "";
    };

    auto search_ld_library_path = [&]() -> std::string {
        const char* ldpath = getenv("LD_LIBRARY_PATH");
        if (!ldpath) return "";
        const char* p = ldpath;
        while (*p) {
            const char* delim = strchr(p, ':');
            std::string dir(p, delim ? (size_t)(delim - p) : strlen(p));
            if (!dir.empty()) {
                std::string r = probe(dir, true);
                if (!r.empty()) return r;
            }
            if (!delim) break;
            p = delim + 1;
        }
        for (p = ldpath; *p; ) {
            const char* delim = strchr(p, ':');
            std::string dir(p, delim ? (size_t)(delim - p) : strlen(p));
            if (!dir.empty()) {
                std::string r = probe(dir, false);
                if (!r.empty()) return r;
            }
            if (!delim) break;
            p = delim + 1;
        }
        return "";
    };

    auto search_qairt_sdk = [&]() -> std::string {
        const char* sdk_dirs[] = {
            "/opt/qcom/aistack/qairt",
            "/usr/local/lib",
            "/usr/lib",
            "/usr/lib/aarch64-linux-gnu",
            nullptr
        };
        for (int i = 0; sdk_dirs[i]; i++) {
            std::string r = probe(sdk_dirs[i], true);
            if (!r.empty()) return r;
            r = probe(sdk_dirs[i], false);
            if (!r.empty()) return r;
        }
        const char* qairt_root = "/opt/qcom/aistack/qairt";
        DIR* d = opendir(qairt_root);
        if (d) {
            struct dirent* ent;
            while ((ent = readdir(d)) != nullptr) {
                if (ent->d_name[0] == '.') continue;
                static const char* lib_subdirs[] = {
                    "lib/aarch64-oe-linux-gcc11.2",
                    "lib/aarch64-ubuntu-gcc9.4",
                    "lib/aarch64-oe-linux-gcc9.3",
                    nullptr
                };
                for (int j = 0; lib_subdirs[j]; j++) {
                    std::string candidate = std::string(qairt_root) + "/" + ent->d_name + "/" + lib_subdirs[j];
                    std::string r = probe(candidate, true);
                    if (!r.empty()) { closedir(d); return r; }
                }
            }
            closedir(d);
        }
        return "";
    };

    auto search_exe_relative = [&]() -> std::string {
        char exe_path[PATH_MAX] = {};
        ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
        if (len <= 0) return "";
        exe_path[len] = '\0';
        std::string cur = exe_path;
        auto slash = cur.find_last_of('/');
        if (slash != std::string::npos) cur = cur.substr(0, slash);
        for (int depth = 0; depth < 6; depth++) {
            std::string r = probe(cur + "/lib", true);
            if (!r.empty()) return r;
            r = probe(cur + "/libs/qnn", true);
            if (!r.empty()) return r;
            auto up = cur.find_last_of('/');
            if (up == std::string::npos || up == 0) break;
            cur = cur.substr(0, up);
        }
        return "";
    };

    std::string r = search_ld_library_path();
    if (!r.empty()) {
        fprintf(stderr, "[QNN Direct] using libQnnHtp.so from LD_LIBRARY_PATH: %s\n", r.c_str());
        return r;
    }
    r = search_qairt_sdk();
    if (!r.empty()) {
        fprintf(stderr, "[QNN Direct] using libQnnHtp.so from QAIRT SDK: %s\n", r.c_str());
        return r;
    }
    r = search_exe_relative();
    if (!r.empty()) {
        fprintf(stderr, "[QNN Direct] using exe-relative libQnnHtp.so: %s\n", r.c_str());
        return r;
    }
    fprintf(stderr, "[QNN Direct] libQnnHtp.so not found — set LD_LIBRARY_PATH to the QAIRT lib directory\n");
    return "libQnnHtp.so";
#endif
}

// ---- CACT weight file reader ----

static constexpr uint32_t CACT_MAGIC = 0x54434143;
static constexpr size_t   CACT_HEADER_SIZE = 84;

struct CactFile {
#ifdef _WIN32
    HANDLE hf = INVALID_HANDLE_VALUE;
    HANDLE hm = nullptr;
#else
    int fd = -1;
#endif
    void*  base = nullptr;
    size_t file_size = 0;

    uint32_t flags       = 0;
    bool     is_interleaved = false;
    uint32_t precision   = 0;   // 0=INT8 1=FP16 2=FP32 3=INT4
    std::vector<uint32_t> shape;
    uint32_t group_size  = 0;
    uint32_t num_groups  = 0;
    uint64_t byte_size   = 0;
    uint64_t scales_bytes = 0;
    uint64_t original_N  = 0;
    size_t   data_offset = 0;
    size_t   scales_offset = 0;

    ~CactFile() {
#ifdef _WIN32
        if (base)  UnmapViewOfFile(base);
        if (hm)    CloseHandle(hm);
        if (hf != INVALID_HANDLE_VALUE) CloseHandle(hf);
#else
        if (base && base != MAP_FAILED) munmap(base, file_size);
        if (fd >= 0) close(fd);
#endif
    }

    const void* data_ptr() const {
        return static_cast<const char*>(base) + data_offset;
    }
    const __fp16* scales_ptr() const {
        if (!scales_bytes) return nullptr;
        return reinterpret_cast<const __fp16*>(static_cast<const char*>(base) + scales_offset);
    }
};

static inline size_t align_up(size_t v, size_t a) {
    size_t r = v % a; return r == 0 ? v : v + (a - r);
}

static bool open_cact(const std::string& path, CactFile& f) {
#ifdef _WIN32
    f.hf = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                       nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f.hf == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "[QNN Direct] cannot open: %s\n", path.c_str());
        return false;
    }
    LARGE_INTEGER sz;
    GetFileSizeEx(f.hf, &sz);
    f.file_size = (size_t)sz.QuadPart;
    f.hm = CreateFileMappingA(f.hf, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!f.hm) return false;
    f.base = MapViewOfFile(f.hm, FILE_MAP_READ, 0, 0, 0);
    if (!f.base) return false;
#else
    f.fd = open(path.c_str(), O_RDONLY);
    if (f.fd < 0) {
        fprintf(stderr, "[QNN Direct] cannot open: %s\n", path.c_str());
        return false;
    }
    struct stat st;
    if (fstat(f.fd, &st) != 0) return false;
    f.file_size = (size_t)st.st_size;
    f.base = mmap(nullptr, f.file_size, PROT_READ, MAP_PRIVATE, f.fd, 0);
    if (f.base == MAP_FAILED) { f.base = nullptr; return false; }
#endif

    if (f.file_size < CACT_HEADER_SIZE) return false;
    const char* p = static_cast<const char*>(f.base);
    size_t off = 0;

    uint32_t magic = *reinterpret_cast<const uint32_t*>(p + off); off += 4;
    if (magic != CACT_MAGIC) { fprintf(stderr, "[QNN Direct] bad magic: %s\n", path.c_str()); return false; }

    f.flags = *reinterpret_cast<const uint32_t*>(p + off); off += 4;
    f.is_interleaved = (f.flags & 8) != 0;
    uint32_t alignment = *reinterpret_cast<const uint32_t*>(p + off); off += 4;
    if (alignment == 0) alignment = 1;
    uint32_t ndim = *reinterpret_cast<const uint32_t*>(p + off); off += 4;

    f.shape.clear();
    for (uint32_t i = 0; i < 4; i++) {
        uint64_t d = *reinterpret_cast<const uint64_t*>(p + off); off += 8;
        if (i < ndim && d > 0) f.shape.push_back((uint32_t)d);
    }

    f.precision    = *reinterpret_cast<const uint32_t*>(p + off); off += 4;
    f.byte_size    = *reinterpret_cast<const uint64_t*>(p + off); off += 8;
    f.scales_bytes = *reinterpret_cast<const uint64_t*>(p + off); off += 8;
    f.group_size   = *reinterpret_cast<const uint32_t*>(p + off); off += 4;
    f.num_groups   = *reinterpret_cast<const uint32_t*>(p + off); off += 4;
    f.original_N = *reinterpret_cast<const uint64_t*>(p + off); off += 8;

    size_t aligned_hdr = align_up(CACT_HEADER_SIZE, alignment);
    if (f.scales_bytes > 0) {
        f.scales_offset = aligned_hdr;
        f.data_offset   = align_up(f.scales_offset + f.scales_bytes, alignment);
    } else {
        f.scales_offset = 0;
        f.data_offset   = aligned_hdr;
    }
    return true;
}

[[maybe_unused]] static float fp16_to_fp32_val(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0)       bits = sign;
    else if (exp == 31) bits = sign | 0x7F800000u | (mant << 13);
    else                bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    float v; memcpy(&v, &bits, 4); return v;
}

static uint16_t float_to_fp16(float val) {
    if (val == 0.0f) return 0;
    uint32_t bits;
    memcpy(&bits, &val, 4);
    int exp = (int)((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = (bits >> 13) & 0x3FF;
    uint32_t sign = (bits >> 31) << 15;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mantissa);
}

// Returns FP16 vector (uint16_t). For INT4, dequantizes. For FP16, copies.
static std::vector<uint16_t> cact_to_fp16(const CactFile& f) {
    if (f.precision == 1) {
        size_t num_elements = 1;
        for (auto d : f.shape) num_elements *= d;
        std::vector<uint16_t> out(num_elements);
        const uint16_t* src = static_cast<const uint16_t*>(f.data_ptr());
        memcpy(out.data(), src, num_elements * 2);
        return out;
    }

    if (f.precision == 0) {
        const int8_t* packed  = static_cast<const int8_t*>(f.data_ptr());
        const __fp16* scales  = f.scales_ptr();
        int gs = (int)f.group_size;
        size_t N = (f.is_interleaved && f.original_N > 0) ? (size_t)f.original_N
                                                           : (size_t)f.shape[0];
        size_t K = f.shape.size() >= 2 ? (size_t)f.shape[1] : 1;
        int num_groups = (gs > 0) ? (int)(K / gs) : 1;

        std::vector<uint16_t> out(N * K);

        if (f.is_interleaved) {
            for (size_t n = 0; n < N; n++) {
                int n_block = (int)(n / 4);
                int n_inner = (int)(n % 4);
                for (size_t k = 0; k < K; k++) {
                    size_t boff = ((size_t)n_block * (K / 4) + k / 4) * 16
                                  + (size_t)n_inner * 4 + k % 4;
                    float scale = 1.0f;
                    if (scales && gs > 0) {
                        int g = (int)(k / gs);
                        scale = (float)scales[(size_t)(n_block * num_groups + g) * 4 + n_inner];
                    }
                    out[n * K + k] = float_to_fp16((float)packed[boff] * scale);
                }
            }
        } else {
            size_t num_elements = N * K;
            for (size_t i = 0; i < num_elements; i++) {
                float scale = (scales && gs > 0) ? (float)scales[i / gs] : 1.0f;
                out[i] = float_to_fp16((float)packed[i] * scale);
            }
        }
        return out;
    }

    if (f.precision == 3) {
        // INT4 — two signed 4-bit values packed per byte (low nibble = even element index)
        const uint8_t* packed = static_cast<const uint8_t*>(f.data_ptr());
        const __fp16* scales  = f.scales_ptr();
        int gs = (int)f.group_size;
        size_t N = (f.is_interleaved && f.original_N > 0) ? (size_t)f.original_N
                                                           : (size_t)f.shape[0];
        size_t K = f.shape.size() >= 2 ? (size_t)f.shape[1] : 1;
        int num_groups = (gs > 0) ? (int)(K / gs) : 1;
        std::vector<uint16_t> out(N * K);

        auto get_nibble = [&](size_t byte_idx, bool high) -> int8_t {
            uint8_t b = packed[byte_idx];
            if (!high) return (int8_t)((b & 0xF) << 4) >> 4;  // sign extend low nibble
            else       return (int8_t)(b) >> 4;                // sign extend high nibble
        };

        if (f.is_interleaved) {
            // Layout matches cactus_gemv_int4 / cactus_gemm_int4 kernel:
            // byte[(n_block * K + k_group * gs) * 2 + k_chunk * 16 + n_inner * 4 + k_inner]
            // low nibble = W[n, k] when (k_within % 8) < 4, high nibble when >= 4
            // where k_chunk = (k % gs) / 8, k_inner = k % 4
            for (size_t n = 0; n < N; n++) {
                int n_block = (int)(n / 4);
                int n_inner = (int)(n % 4);
                for (size_t k = 0; k < K; k++) {
                    float scale = 1.0f;
                    if (scales && gs > 0) {
                        int g = (int)(k / gs);
                        scale = (float)scales[(size_t)(n_block * num_groups + g) * 4 + n_inner];
                    }
                    size_t k_group  = (size_t)(k / gs);
                    size_t k_within = (size_t)(k % gs);
                    size_t k_chunk  = k_within / 8;
                    size_t k_inner  = k_within % 4;
                    bool   high_nib = (k_within % 8) >= 4;
                    size_t boff     = ((size_t)n_block * K + k_group * (size_t)gs) * 2
                                      + k_chunk * 16 + (size_t)n_inner * 4 + k_inner;
                    out[n * K + k] = float_to_fp16((float)get_nibble(boff, high_nib) * scale);
                }
            }
        } else {
            size_t num_elements = N * K;
            for (size_t i = 0; i < num_elements; i++) {
                float scale = (scales && gs > 0) ? (float)scales[i / gs] : 1.0f;
                out[i] = float_to_fp16((float)get_nibble(i / 2, i & 1) * scale);
            }
        }
        return out;
    }

    fprintf(stderr, "[QNN Direct] unsupported precision %u in cact_to_fp16\n", f.precision);
    size_t num_elements = 1;
    for (auto d : f.shape) num_elements *= d;
    return std::vector<uint16_t>(num_elements, 0);
}

// ---- QTensor wrapper ----

struct QTensor {
    std::string        name;
    std::vector<uint32_t> dims;
    std::vector<uint16_t> static_fp16;  // kept alive for static tensors
    std::vector<int32_t>  static_i32;   // for param tensors (axes, ranges, perm)
    std::vector<uint32_t> static_u32;   // for uint32 param tensors
    Qnn_Tensor_t t = QNN_TENSOR_INIT;
};

// ---- Parakeet per-segment build/exec state ----
struct PSegState {
    Qnn_GraphHandle_t graph_handle = nullptr;
    std::vector<std::unique_ptr<QTensor>> tensor_store;
    std::unordered_map<std::string, QTensor*> tensors;
    std::vector<std::unique_ptr<CactFile>> weight_files;
    int op_idx = 0, tensor_idx = 0;
    uint32_t exec_in_id = 0, exec_out_id = 0;
    std::vector<Qnn_Tensor_t> exec_inputs, exec_outputs;
    std::vector<uint16_t> in_buf, out_buf;
    std::string graph_name;
};

namespace cactus {
namespace npu {

struct QNNDirectPrefill::Impl {
    // DLL handles
    void* dll_handle     = nullptr;
    void* sys_dll_handle = nullptr;
    QNN_INTERFACE_VER_TYPE qnn = QNN_INTERFACE_VER_TYPE_INIT;

    // QNN handles
    Qnn_LogHandle_t     log_handle     = nullptr;
    Qnn_BackendHandle_t backend_handle = nullptr;
    Qnn_DeviceHandle_t  device_handle  = nullptr;
    Qnn_ContextHandle_t context_handle = nullptr;
    Qnn_GraphHandle_t   graph_handle   = nullptr;

    // Model params (read from config.txt)
    int chunk_size   = 64;
    int hidden_dim   = 896;
    int num_layers   = 24;
    int num_kv_heads = 2;
    int head_dim     = 64;
    int num_heads    = 14;
    int ffn_dim      = 4864;
    float layer_norm_eps = 1e-6f;
    float rope_theta     = 1000000.0f;

    bool loaded = false;

    // All QTensor objects; unique_ptr ensures stable addresses for dims/data pointers
    std::vector<std::unique_ptr<QTensor>> tensor_store;
    std::unordered_map<std::string, QTensor*> tensors;

    // I/O buffers for graph execution
    std::vector<uint16_t> emb_buf;     // [chunk * hidden] FP16
    std::vector<uint16_t> cos_buf;     // [chunk * head_dim/2] FP16
    std::vector<uint16_t> sin_buf;     // [chunk * head_dim/2] FP16
    std::vector<std::vector<uint16_t>> k_out_bufs;  // [num_layers][chunk * kv_heads * head_dim]
    std::vector<std::vector<uint16_t>> v_out_bufs;

    // QNN tensor structs for execute — we keep pointers to the original registered QTensor
    // objects so we can update clientBuf.data before each graphExecute call.
    // This ensures ALL tensor fields (name, id, dimensions pointer) match exactly what QNN registered.
    QTensor* exec_emb_qt  = nullptr;
    QTensor* exec_cos_qt  = nullptr;
    QTensor* exec_sin_qt  = nullptr;
    std::vector<QTensor*> exec_k_qts;
    std::vector<QTensor*> exec_v_qts;

    // Flat arrays of Qnn_Tensor_t (shallow copies updated before each execute)
    std::vector<Qnn_Tensor_t> exec_inputs;   // [emb, cos, sin]
    std::vector<Qnn_Tensor_t> exec_outputs;  // [k_0, v_0, k_1, v_1, ...]

    // I/O tensor IDs (for cache/sidecar)
    uint32_t exec_emb_id = 0, exec_cos_id = 0, exec_sin_id = 0;
    std::vector<uint32_t> exec_k_ids;
    std::vector<uint32_t> exec_v_ids;

    std::string model_folder_path;

    // Weight files kept alive until after graphFinalize
    std::vector<std::unique_ptr<CactFile>> weight_files;

    // ---- DLL / backend ----
    bool load_dll(const std::string& path);
    bool init_interface();
    bool init_backend();
    void teardown();

    // ---- Config ----
    bool parse_config(const std::string& model_folder);

    // ---- Graph building ----
    bool build_graph(const std::string& model_folder, const std::string& cache_path = "");
    bool load_from_cache(const std::string& path);
    bool save_to_cache(const std::string& path);
    bool save_id_sidecar(const std::string& path);
    bool load_id_sidecar(const std::string& path);
    void setup_exec_tensors();

    // ---- Tensor helpers ----
    QTensor* make_tensor(const std::string& name, Qnn_TensorType_t type,
                         Qnn_DataType_t dtype, const std::vector<uint32_t>& shape,
                         const void* data = nullptr, size_t data_bytes = 0);
    QTensor* make_static_fp16(const std::string& name, const std::vector<uint32_t>& shape,
                               std::vector<uint16_t> data);
    QTensor* make_static_i32(const std::string& name, const std::vector<uint32_t>& shape,
                              std::vector<int32_t> data);
    QTensor* make_static_u32(const std::string& name, const std::vector<uint32_t>& shape,
                              std::vector<uint32_t> data);
    QTensor* make_native(const std::string& name, const std::vector<uint32_t>& shape);
    QTensor* make_output(const std::string& name, const std::vector<uint32_t>& shape);
    QTensor* load_weight_tensor(const std::string& name, const std::string& path);

    // ---- Op helpers ----
    int op_idx = 0;
    std::string op_name(const std::string& type) {
        return type + "_" + std::to_string(op_idx++);
    }

    Qnn_ErrorHandle_t add_op(const char* pkg, const char* type_name, const std::string& name,
                              const std::vector<QTensor*>& ins, const std::vector<QTensor*>& outs,
                              const std::vector<Qnn_Param_t>& params);

    // ---- Composite ops ----
    QTensor* op_rms_norm(QTensor* x, QTensor* w, const std::string& out_name, bool prescale = false);
    QTensor* op_unary(QTensor* x, uint32_t operation, const std::string& out_name);
    QTensor* op_reduce_mean_last(QTensor* x, const std::string& out_name);
    QTensor* op_matmul_T(QTensor* x, QTensor* w, const std::string& out_name);
    QTensor* op_add(QTensor* a, QTensor* b, const std::string& out_name);
    QTensor* op_mul(QTensor* a, QTensor* b, const std::string& out_name);
    QTensor* op_sigmoid(QTensor* x, const std::string& out_name);
    QTensor* op_silu(QTensor* x, const std::string& prefix);
    QTensor* op_softmax(QTensor* x, int axis, const std::string& out_name);
    QTensor* op_reshape(QTensor* x, const std::vector<uint32_t>& new_shape, const std::string& out_name);
    QTensor* op_transpose(QTensor* x, const std::vector<uint32_t>& perm, const std::string& out_name);
    QTensor* op_concat(const std::vector<QTensor*>& xs, int axis, const std::string& out_name);
    QTensor* op_tile(QTensor* x, const std::vector<uint32_t>& multiples, const std::string& out_name);
    QTensor* op_strided_slice(QTensor* x, const std::vector<std::vector<int32_t>>& ranges,
                               int begin_mask, int end_mask, const std::string& out_name);

    QTensor* op_rope(QTensor* x_3d, QTensor* cos_in, QTensor* sin_in,
                     int heads, const std::string& prefix);
    QTensor* op_attention(QTensor* q_3d, QTensor* k_3d, QTensor* v_3d,
                          QTensor* causal_mask, float scale, const std::string& prefix);

    void build_layer(const std::string& folder, int li,
                     QTensor*& hidden, QTensor* cos_in, QTensor* sin_in,
                     QTensor* causal_mask,
                     QTensor*& k_out, QTensor*& v_out);
};

// ---- DLL loading (unchanged) ----

bool QNNDirectPrefill::Impl::load_dll(const std::string& path) {
    size_t sep = path.rfind('/');
#ifdef _WIN32
    { size_t bs = path.rfind('\\'); if (bs != std::string::npos && (sep == std::string::npos || bs > sep)) sep = bs; }
#endif
    std::string dir = (sep != std::string::npos) ? path.substr(0, sep) : ".";

#ifdef _WIN32
    static const char* companions_root[] = { "QnnSystem.dll", "QnnHtpV73Stub.dll", nullptr };
    static const char* companions_htp[]  = { "QnnHtpPrepareDrv.dll", nullptr };
    auto preload = [&](const std::string& full_path) {
        if (!file_exists(full_path)) return;
        HMODULE h = LoadLibraryA(full_path.c_str());
        const char* name = full_path.c_str() + full_path.rfind('\\') + 1;
        if (h && strstr(name, "QnnSystem")) sys_dll_handle = (void*)h;
        fprintf(stderr, "[QNN Direct] pre-load %s: %s\n", name, h ? "ok" : "failed");
    };
    for (int i = 0; companions_root[i]; i++)
        preload(dir + "\\" + companions_root[i]);
    std::string htp_dir = dir + "\\HTP";
    for (int i = 0; companions_htp[i]; i++)
        preload(htp_dir + "\\" + companions_htp[i]);
    dll_handle = (void*)LoadLibraryA(path.c_str());
    if (!dll_handle) {
        fprintf(stderr, "[QNN Direct] failed to load %s (err %lu)\n",
                path.c_str(), GetLastError());
        return false;
    }
#else
    static const char* companions[] = {
        "libQnnSystem.so",
        "libQnnHtpV75Stub.so", "libQnnHtpV73Stub.so", "libQnnHtpV79Stub.so",
        "libQnnHtpPrepare.so",
        nullptr
    };
    for (int i = 0; companions[i]; i++) {
        std::string full = dir + "/" + companions[i];
        if (!file_exists(full)) continue;
        void* h = dlopen(full.c_str(), RTLD_NOW | RTLD_GLOBAL);
        const char* name = companions[i];
        if (h && strstr(name, "QnnSystem")) sys_dll_handle = h;
        fprintf(stderr, "[QNN Direct] pre-load %s: %s\n", name, h ? "ok" : dlerror());
    }
    dll_handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!dll_handle) {
        fprintf(stderr, "[QNN Direct] failed to load %s: %s\n", path.c_str(), dlerror());
        return false;
    }
#endif
    fprintf(stderr, "[QNN Direct] loaded %s\n", path.c_str());
    return true;
}

bool QNNDirectPrefill::Impl::init_interface() {
#ifdef _WIN32
    auto fn = reinterpret_cast<QnnInterfaceGetProvidersFn_t>(
        reinterpret_cast<void*>(GetProcAddress(
            (HMODULE)dll_handle, "QnnInterface_getProviders")));
#else
    auto fn = reinterpret_cast<QnnInterfaceGetProvidersFn_t>(
        dlsym(dll_handle, "QnnInterface_getProviders"));
#endif
    if (!fn) { fprintf(stderr, "[QNN Direct] QnnInterface_getProviders not found\n"); return false; }
    const QnnInterface_t** providers = nullptr;
    uint32_t num = 0;
    if (fn(&providers, &num) != QNN_SUCCESS || num == 0) {
        fprintf(stderr, "[QNN Direct] getProviders failed\n"); return false;
    }
    qnn = providers[0]->QNN_INTERFACE_VER_NAME;
    fprintf(stderr, "[QNN Direct] interface ok (backend %u)\n", providers[0]->backendId);
    return true;
}

bool QNNDirectPrefill::Impl::init_backend() {
    if (qnn.logCreate)
        qnn.logCreate(qnn_log_cb, QNN_LOG_LEVEL_ERROR, &log_handle);

    if (!qnn.backendCreate) return false;
    if (qnn.backendCreate(log_handle, nullptr, &backend_handle) != QNN_SUCCESS) {
        fprintf(stderr, "[QNN Direct] backendCreate failed\n"); return false;
    }
    fprintf(stderr, "[QNN Direct] backend created\n");

    if (qnn.deviceCreate) {
        QnnHtpDevice_CustomConfig_t htp_soc_cfg;
        htp_soc_cfg.option   = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
#ifdef _WIN32
        htp_soc_cfg.socModel = 60;  // QNN_SOC_MODEL_SC8380XP
#else
        htp_soc_cfg.socModel = 0;   // QNN_SOC_MODEL_UNKNOWN — auto-detect
#endif
        QnnDevice_Config_t dev_cfg = QNN_DEVICE_CONFIG_INIT;
        dev_cfg.option       = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
        dev_cfg.customConfig = (QnnDevice_CustomConfig_t)&htp_soc_cfg;
        const QnnDevice_Config_t* dev_cfgs[] = {&dev_cfg, nullptr};
        Qnn_ErrorHandle_t de = qnn.deviceCreate(log_handle, dev_cfgs, &device_handle);
        if (de != QNN_SUCCESS) {
            fprintf(stderr, "[QNN Direct] deviceCreate with SoC config failed (%lld), retrying auto\n", (long long)de);
            de = qnn.deviceCreate(log_handle, nullptr, &device_handle);
        }
    }

    if (!qnn.contextCreate) return false;
    if (qnn.contextCreate(backend_handle, device_handle, nullptr, &context_handle) != QNN_SUCCESS) {
        fprintf(stderr, "[QNN Direct] contextCreate failed\n"); return false;
    }
    fprintf(stderr, "[QNN Direct] context created\n");
    return true;
}

void QNNDirectPrefill::Impl::teardown() {
    if (context_handle && qnn.contextFree) qnn.contextFree(context_handle, nullptr);
    if (device_handle && qnn.deviceFree) qnn.deviceFree(device_handle);
    if (backend_handle && qnn.backendFree) qnn.backendFree(backend_handle);
    if (log_handle && qnn.logFree) qnn.logFree(log_handle);
#ifdef _WIN32
    if (dll_handle) FreeLibrary((HMODULE)dll_handle);
    if (sys_dll_handle) FreeLibrary((HMODULE)sys_dll_handle);
#else
    if (dll_handle) dlclose(dll_handle);
    if (sys_dll_handle) dlclose(sys_dll_handle);
#endif
    loaded = false;
}

// ---- Config parsing ----

bool QNNDirectPrefill::Impl::parse_config(const std::string& folder) {
    std::string path = folder + "/config.txt";
    std::ifstream f(path);
    if (!f.is_open()) { fprintf(stderr, "[QNN Direct] cannot open config.txt: %s\n", path.c_str()); return false; }
    std::string line;
    while (std::getline(f, line)) {
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        if (key == "hidden_dim")            hidden_dim   = std::stoi(val);
        else if (key == "num_layers")       num_layers   = std::stoi(val);
        else if (key == "attention_heads")  num_heads    = std::stoi(val);
        else if (key == "attention_kv_heads") num_kv_heads = std::stoi(val);
        else if (key == "attention_head_dim") head_dim   = std::stoi(val);
        else if (key == "ffn_intermediate_dim") ffn_dim  = std::stoi(val);
        else if (key == "rope_theta")       rope_theta   = std::stof(val);
        else if (key == "layer_norm_eps")   layer_norm_eps = std::stof(val);
    }
    fprintf(stderr, "[QNN Direct] config: hidden=%d layers=%d heads=%d kv_heads=%d head_dim=%d ffn=%d\n",
            hidden_dim, num_layers, num_heads, num_kv_heads, head_dim, ffn_dim);
    return true;
}

// ---- Tensor helpers ----

QTensor* QNNDirectPrefill::Impl::make_tensor(
        const std::string& name, Qnn_TensorType_t type, Qnn_DataType_t dtype,
        const std::vector<uint32_t>& shape, const void* data, size_t data_bytes) {
    auto qt = std::make_unique<QTensor>();
    qt->name = name;
    qt->dims = shape;

    qt->t.version = QNN_TENSOR_VERSION_1;
    qt->t.v1.id   = 0;
    qt->t.v1.name = qt->name.c_str();
    qt->t.v1.type = type;
    qt->t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    qt->t.v1.dataType   = dtype;
    qt->t.v1.quantizeParams.encodingDefinition  = QNN_DEFINITION_UNDEFINED;
    qt->t.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    qt->t.v1.rank       = (uint32_t)shape.size();
    qt->t.v1.dimensions = qt->dims.data();
    qt->t.v1.memType    = QNN_TENSORMEMTYPE_RAW;
    qt->t.v1.clientBuf.data     = (type == QNN_TENSOR_TYPE_STATIC) ? const_cast<void*>(data) : nullptr;
    size_t sz = data_bytes;
    if (sz == 0) {
        sz = 1;
        for (auto d : shape) sz *= d;
        if (dtype == QNN_DATATYPE_FLOAT_16 || dtype == QNN_DATATYPE_INT_16) sz *= 2;
        else if (dtype == QNN_DATATYPE_FLOAT_32 || dtype == QNN_DATATYPE_INT_32) sz *= 4;
        else if (dtype == QNN_DATATYPE_INT_8 || dtype == QNN_DATATYPE_BOOL_8) sz *= 1;
    }
    qt->t.v1.clientBuf.dataSize = (uint32_t)sz;

    QTensor* raw = qt.get();
    tensors[name] = raw;
    tensor_store.push_back(std::move(qt));
    return raw;
}

QTensor* QNNDirectPrefill::Impl::make_static_fp16(
        const std::string& name, const std::vector<uint32_t>& shape, std::vector<uint16_t> data) {
    auto qt = std::make_unique<QTensor>();
    qt->name = name;
    qt->dims = shape;
    qt->static_fp16 = std::move(data);

    qt->t.version = QNN_TENSOR_VERSION_1;
    qt->t.v1.id   = 0;
    qt->t.v1.name = qt->name.c_str();
    qt->t.v1.type = QNN_TENSOR_TYPE_STATIC;
    qt->t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    qt->t.v1.dataType   = QNN_DATATYPE_FLOAT_16;
    qt->t.v1.quantizeParams.encodingDefinition  = QNN_DEFINITION_UNDEFINED;
    qt->t.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    qt->t.v1.rank       = (uint32_t)shape.size();
    qt->t.v1.dimensions = qt->dims.data();
    qt->t.v1.memType    = QNN_TENSORMEMTYPE_RAW;
    qt->t.v1.clientBuf.data     = qt->static_fp16.data();
    qt->t.v1.clientBuf.dataSize = (uint32_t)(qt->static_fp16.size() * 2);

    QTensor* raw = qt.get();
    tensors[name] = raw;
    tensor_store.push_back(std::move(qt));
    return raw;
}

QTensor* QNNDirectPrefill::Impl::make_static_i32(
        const std::string& name, const std::vector<uint32_t>& shape, std::vector<int32_t> data) {
    auto qt = std::make_unique<QTensor>();
    qt->name = name;
    qt->dims = shape;
    qt->static_i32 = std::move(data);

    qt->t.version = QNN_TENSOR_VERSION_1;
    qt->t.v1.id   = 0;
    qt->t.v1.name = qt->name.c_str();
    qt->t.v1.type = QNN_TENSOR_TYPE_STATIC;
    qt->t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    qt->t.v1.dataType   = QNN_DATATYPE_INT_32;
    qt->t.v1.quantizeParams.encodingDefinition  = QNN_DEFINITION_UNDEFINED;
    qt->t.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    qt->t.v1.rank       = (uint32_t)shape.size();
    qt->t.v1.dimensions = qt->dims.data();
    qt->t.v1.memType    = QNN_TENSORMEMTYPE_RAW;
    qt->t.v1.clientBuf.data     = qt->static_i32.data();
    qt->t.v1.clientBuf.dataSize = (uint32_t)(qt->static_i32.size() * 4);

    QTensor* raw = qt.get();
    tensors[name] = raw;
    tensor_store.push_back(std::move(qt));
    return raw;
}

QTensor* QNNDirectPrefill::Impl::make_static_u32(
        const std::string& name, const std::vector<uint32_t>& shape, std::vector<uint32_t> data) {
    auto qt = std::make_unique<QTensor>();
    qt->name = name;
    qt->dims = shape;
    qt->static_u32 = std::move(data);

    qt->t.version = QNN_TENSOR_VERSION_1;
    qt->t.v1.id   = 0;
    qt->t.v1.name = qt->name.c_str();
    qt->t.v1.type = QNN_TENSOR_TYPE_STATIC;
    qt->t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    qt->t.v1.dataType   = QNN_DATATYPE_UINT_32;
    qt->t.v1.quantizeParams.encodingDefinition  = QNN_DEFINITION_UNDEFINED;
    qt->t.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    qt->t.v1.rank       = (uint32_t)shape.size();
    qt->t.v1.dimensions = qt->dims.data();
    qt->t.v1.memType    = QNN_TENSORMEMTYPE_RAW;
    qt->t.v1.clientBuf.data     = qt->static_u32.data();
    qt->t.v1.clientBuf.dataSize = (uint32_t)(qt->static_u32.size() * 4);

    QTensor* raw = qt.get();
    tensors[name] = raw;
    tensor_store.push_back(std::move(qt));
    return raw;
}

QTensor* QNNDirectPrefill::Impl::make_native(const std::string& name, const std::vector<uint32_t>& shape) {
    return make_tensor(name, QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, shape);
}

QTensor* QNNDirectPrefill::Impl::make_output(const std::string& name, const std::vector<uint32_t>& shape) {
    return make_tensor(name, QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_16, shape);
}

QTensor* QNNDirectPrefill::Impl::load_weight_tensor(const std::string& name, const std::string& path) {
    auto cf = std::make_unique<CactFile>();
    if (!open_cact(path, *cf)) {
        fprintf(stderr, "[QNN Direct] failed to load weight: %s\n", path.c_str());
        // Return a zero tensor as fallback
        std::vector<uint32_t> shape = {1};
        return make_static_fp16(name, shape, {0});
    }

    std::vector<uint32_t> shape(cf->shape.begin(), cf->shape.end());
    if (cf->is_interleaved && cf->original_N > 0 && !shape.empty())
        shape[0] = (uint32_t)cf->original_N;
    std::vector<uint16_t> fp16_data = cact_to_fp16(*cf);
    weight_files.push_back(std::move(cf));
    return make_static_fp16(name, shape, std::move(fp16_data));
}

// ---- Op building ----

Qnn_ErrorHandle_t QNNDirectPrefill::Impl::add_op(
        const char* pkg, const char* type_name, const std::string& name,
        const std::vector<QTensor*>& ins, const std::vector<QTensor*>& outs,
        const std::vector<Qnn_Param_t>& params) {

    // Register all tensors first
    auto reg = [&](QTensor* qt) {
        if (qt->t.v1.id == 0 && qt->t.v1.type != QNN_TENSOR_TYPE_UNDEFINED) {
            qnn.tensorCreateGraphTensor(graph_handle, &qt->t);
        }
    };
    for (auto* qt : ins)  reg(qt);
    for (auto* qt : outs) reg(qt);

    std::vector<Qnn_Tensor_t> in_t, out_t;
    for (auto* qt : ins)  in_t.push_back(qt->t);
    for (auto* qt : outs) out_t.push_back(qt->t);

    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
    op.version = QNN_OPCONFIG_VERSION_1;
    op.v1.name        = name.c_str();
    op.v1.packageName = pkg;
    op.v1.typeName    = type_name;
    op.v1.numOfParams  = (uint32_t)params.size();
    op.v1.params       = params.empty() ? nullptr : const_cast<Qnn_Param_t*>(params.data());
    op.v1.numOfInputs  = (uint32_t)in_t.size();
    op.v1.inputTensors = in_t.data();
    op.v1.numOfOutputs  = (uint32_t)out_t.size();
    op.v1.outputTensors = out_t.data();

    Qnn_ErrorHandle_t err = qnn.graphAddNode(graph_handle, op);
    if (err != QNN_SUCCESS)
        fprintf(stderr, "[QNN Direct] graphAddNode %s/%s failed: %lld\n", type_name, name.c_str(), (long long)err);
    return err;
}

// ---- Helper: scalar param ----
static Qnn_Param_t scalar_i32_param(const char* name, int32_t val) {
    Qnn_Param_t p = QNN_PARAM_INIT;
    p.paramType = QNN_PARAMTYPE_SCALAR;
    p.name = name;
    p.scalarParam.dataType = QNN_DATATYPE_INT_32;
    p.scalarParam.int32Value = val;
    return p;
}
static Qnn_Param_t scalar_u32_param(const char* name, uint32_t val) {
    Qnn_Param_t p = QNN_PARAM_INIT;
    p.paramType = QNN_PARAMTYPE_SCALAR;
    p.name = name;
    p.scalarParam.dataType = QNN_DATATYPE_UINT_32;
    p.scalarParam.uint32Value = val;
    return p;
}
static Qnn_Param_t scalar_bool_param(const char* name, bool val) {
    Qnn_Param_t p = QNN_PARAM_INIT;
    p.paramType = QNN_PARAMTYPE_SCALAR;
    p.name = name;
    p.scalarParam.dataType = QNN_DATATYPE_BOOL_8;
    p.scalarParam.uint8Value = val ? 1 : 0;
    return p;
}
static Qnn_Param_t tensor_param(const char* name, QTensor* qt) {
    Qnn_Param_t p = QNN_PARAM_INIT;
    p.paramType = QNN_PARAMTYPE_TENSOR;
    p.name = name;
    p.tensorParam = qt->t;
    return p;
}

// ---- Composite ops ----

// Unary op helper
QTensor* QNNDirectPrefill::Impl::op_unary(QTensor* x, uint32_t operation, const std::string& out_name) {
    auto* out = make_native(out_name, x->dims);
    add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_ELEMENT_WISE_UNARY, op_name("unary"),
           {x}, {out}, {scalar_u32_param(QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION, operation)});
    return out;
}

// ReduceMean over last axis with keepdims=true
QTensor* QNNDirectPrefill::Impl::op_reduce_mean_last(QTensor* x, const std::string& out_name) {
    int last = (int)x->dims.size() - 1;
    std::vector<uint32_t> out_shape(x->dims.begin(), x->dims.end());
    out_shape[last] = 1;
    auto* out = make_native(out_name, out_shape);
    auto* axes_t = make_static_i32("axes_rm_" + out_name, {1}, {last});
    if (axes_t->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &axes_t->t);
    Qnn_Param_t axes_p = tensor_param(QNN_OP_REDUCE_MEAN_PARAM_AXES, axes_t);
    axes_p.tensorParam = axes_t->t;
    add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_REDUCE_MEAN, op_name("reduce_mean"),
           {x}, {out},
           {axes_p, scalar_bool_param(QNN_OP_REDUCE_MEAN_PARAM_KEEP_DIMS, true)});
    return out;
}

// RMS norm: out = x / sqrt(mean(x^2) + eps) * w (decomposed, avoids HTP hang with native op)
// prescale=true: multiply x by 64 before squaring to lift embedding-level values (~0.001)
// above FP16 subnormal threshold. Safe for embeddings (max ~0.05 → scaled 3.2 → sq 10.24).
// The 64x cancels through scale-invariance: (x*64)*rsqrt(mean((x*64)^2)) = x*rsqrt(mean(x^2))
QTensor* QNNDirectPrefill::Impl::op_rms_norm(QTensor* x, QTensor* w, const std::string& out_name,
                                              bool prescale) {
    uint32_t last_dim = x->dims.back();
    QTensor* xs = x;
    if (prescale) {
        const uint16_t k64_fp16 = 0x5400; // 64.0 in FP16
        auto* scale_t = make_static_fp16(out_name + "_psc", {1u, last_dim},
                                         std::vector<uint16_t>(last_dim, k64_fp16));
        if (scale_t->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &scale_t->t);
        xs = op_mul(x, scale_t, out_name + "_xs");
    }
    auto* x2   = op_mul(xs, xs, out_name + "_sq");
    auto* mean  = op_reduce_mean_last(x2, out_name + "_mean");
    uint16_t eps_fp16;
    { float ev = layer_norm_eps; uint32_t b; memcpy(&b, &ev, 4);
      int e = (int)((b>>23)&0xFF)-127+15; uint32_t m=(b>>13)&0x3FF; uint32_t s=(b>>31)<<15;
      eps_fp16 = (e<=0)?0:(e>=31)?0x7C00:(uint16_t)(s|(e<<10)|m);
      if (eps_fp16 == 0) eps_fp16 = 0x0400; }
    size_t msz = 1; for (auto d : mean->dims) msz *= d;
    auto* eps_t = make_static_fp16(out_name + "_eps", mean->dims,
                                   std::vector<uint16_t>(msz, eps_fp16));
    if (eps_t->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &eps_t->t);
    auto* mean_eps = op_add(mean, eps_t, out_name + "_mean_eps");
    auto* rsqrt    = op_unary(mean_eps, QNN_OP_ELEMENT_WISE_UNARY_OPERATION_RSQRT, out_name + "_rsqrt");
    auto* rsqrt_t  = op_tile(rsqrt, {1u, last_dim}, out_name + "_rsqrt_t");
    auto* normed   = op_mul(xs, rsqrt_t, out_name + "_normed");
    // QNN HTP ElementWiseMultiply requires same rank. If w is 1D [hd], expand to [1, hd]
    // so it broadcasts correctly against normed [cs, hd].
    if (w->dims.size() < normed->dims.size()) {
        std::vector<uint32_t> expanded(normed->dims.size() - w->dims.size(), 1u);
        expanded.insert(expanded.end(), w->dims.begin(), w->dims.end());
        w = op_reshape(w, expanded, out_name + "_w_exp");
    }
    return op_mul(normed, w, out_name);
}

QTensor* QNNDirectPrefill::Impl::op_matmul_T(QTensor* x, QTensor* w, const std::string& out_name) {
    // w is [out_dim, in_dim], compute x @ w^T → [M, out_dim]
    std::vector<uint32_t> out_shape;
    for (size_t i = 0; i + 1 < x->dims.size(); i++) out_shape.push_back(x->dims[i]);
    out_shape.push_back(w->dims[0]);
    auto* out = make_native(out_name, out_shape);

    add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_MAT_MUL, op_name("matmul"),
           {x, w}, {out},
           {scalar_bool_param(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, false),
            scalar_bool_param(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, true)});
    return out;
}

QTensor* QNNDirectPrefill::Impl::op_add(QTensor* a, QTensor* b, const std::string& out_name) {
    auto* out = make_native(out_name, a->dims);
    add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_ELEMENT_WISE_ADD, op_name("add"), {a, b}, {out}, {});
    return out;
}

QTensor* QNNDirectPrefill::Impl::op_mul(QTensor* a, QTensor* b, const std::string& out_name) {
    auto* out = make_native(out_name, a->dims);
    add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_ELEMENT_WISE_MULTIPLY, op_name("mul"), {a, b}, {out}, {});
    return out;
}

QTensor* QNNDirectPrefill::Impl::op_sigmoid(QTensor* x, const std::string& out_name) {
    auto* out = make_native(out_name, x->dims);
    add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_ELEMENT_WISE_NEURON, op_name("sigmoid"),
           {x}, {out},
           {scalar_u32_param(QNN_OP_ELEMENT_WISE_NEURON_PARAM_OPERATION,
                             QNN_OP_ELEMENT_WISE_NEURON_OPERATION_SIGMOID)});
    return out;
}

QTensor* QNNDirectPrefill::Impl::op_silu(QTensor* x, const std::string& prefix) {
    auto* sig = op_sigmoid(x, prefix + "_sig");
    return op_mul(x, sig, prefix + "_silu");
}

QTensor* QNNDirectPrefill::Impl::op_softmax(QTensor* x, int axis, const std::string& out_name) {
    auto* out = make_native(out_name, x->dims);
    add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_SOFTMAX, op_name("softmax"),
           {x}, {out}, {scalar_i32_param(QNN_OP_SOFTMAX_PARAM_AXIS, axis)});
    return out;
}

QTensor* QNNDirectPrefill::Impl::op_reshape(QTensor* x, const std::vector<uint32_t>& shape,
                                              const std::string& out_name) {
    auto* out = make_native(out_name, shape);
    add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_RESHAPE, op_name("reshape"), {x}, {out}, {});
    return out;
}

QTensor* QNNDirectPrefill::Impl::op_transpose(QTensor* x, const std::vector<uint32_t>& perm,
                                               const std::string& out_name) {
    std::vector<uint32_t> out_shape(perm.size());
    for (size_t i = 0; i < perm.size(); i++) out_shape[i] = x->dims[perm[i]];
    auto* out = make_native(out_name, out_shape);

    auto* perm_t = make_static_u32("perm_" + out_name, {(uint32_t)perm.size()},
                                    std::vector<uint32_t>(perm.begin(), perm.end()));
    if (perm_t->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &perm_t->t);
    Qnn_Param_t pp = tensor_param(QNN_OP_TRANSPOSE_PARAM_PERM, perm_t);
    pp.tensorParam = perm_t->t;

    add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_TRANSPOSE, op_name("transpose"),
           {x}, {out}, {pp});
    return out;
}

QTensor* QNNDirectPrefill::Impl::op_concat(const std::vector<QTensor*>& xs, int axis,
                                            const std::string& out_name) {
    std::vector<uint32_t> out_shape = xs[0]->dims;
    out_shape[axis] = 0;
    for (auto* t : xs) out_shape[axis] += t->dims[axis];
    auto* out = make_native(out_name, out_shape);
    add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CONCAT, op_name("concat"),
           xs, {out}, {scalar_i32_param(QNN_OP_CONCAT_PARAM_AXIS, axis)});
    return out;
}

QTensor* QNNDirectPrefill::Impl::op_tile(QTensor* x, const std::vector<uint32_t>& multiples,
                                          const std::string& out_name) {
    std::vector<uint32_t> out_shape(x->dims.size());
    for (size_t i = 0; i < x->dims.size(); i++) out_shape[i] = x->dims[i] * multiples[i];
    auto* out = make_native(out_name, out_shape);

    auto* mul_t = make_static_u32("multiples_" + out_name, {(uint32_t)multiples.size()},
                                   std::vector<uint32_t>(multiples.begin(), multiples.end()));
    if (mul_t->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &mul_t->t);
    Qnn_Param_t mp = tensor_param(QNN_OP_TILE_PARAM_MULTIPLES, mul_t);
    mp.tensorParam = mul_t->t;

    add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_TILE, op_name("tile"),
           {x}, {out}, {mp});
    return out;
}

QTensor* QNNDirectPrefill::Impl::op_strided_slice(QTensor* x,
        const std::vector<std::vector<int32_t>>& ranges,
        int begin_mask, int end_mask, const std::string& out_name) {
    // ranges[i] = {begin, end, stride} for dim i
    int rank = (int)x->dims.size();
    std::vector<uint32_t> out_shape(rank);
    for (int i = 0; i < rank; i++) {
        int b = (begin_mask & (1 << i)) ? 0 : ranges[i][0];
        int e = (end_mask   & (1 << i)) ? (int)x->dims[i] : ranges[i][1];
        int s = ranges[i][2];
        out_shape[i] = (uint32_t)((e - b + s - 1) / s);
    }
    auto* out = make_native(out_name, out_shape);

    std::vector<int32_t> ranges_flat;
    for (auto& r : ranges) { ranges_flat.push_back(r[0]); ranges_flat.push_back(r[1]); ranges_flat.push_back(r[2]); }
    auto* ranges_t = make_static_i32("ranges_" + out_name, {(uint32_t)rank, 3}, ranges_flat);
    if (ranges_t->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &ranges_t->t);
    Qnn_Param_t rp = tensor_param(QNN_OP_STRIDED_SLICE_PARAM_RANGES, ranges_t);
    rp.tensorParam = ranges_t->t;

    add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_STRIDED_SLICE, op_name("slice"),
           {x}, {out},
           {rp,
            scalar_i32_param(QNN_OP_STRIDED_SLICE_PARAM_BEGIN_MASK, begin_mask),
            scalar_i32_param(QNN_OP_STRIDED_SLICE_PARAM_END_MASK, end_mask)});
    return out;
}

// RoPE: x_3d [chunk, heads, head_dim] → output [chunk, heads, head_dim]
// cos_in: [chunk, half_dim], sin_in: [chunk, half_dim]
QTensor* QNNDirectPrefill::Impl::op_rope(QTensor* x_3d, QTensor* cos_in, QTensor* sin_in,
                                          int heads, const std::string& prefix) {
    int cs = chunk_size, hd = head_dim, half = hd / 2;
    // Reshape cos/sin from {cs, half} to {cs, 1, half}, then tile to {cs, heads, half}
    auto* cos_r = op_reshape(cos_in, {(uint32_t)cs, 1, (uint32_t)half}, prefix + "_cos_r");
    auto* sin_r = op_reshape(sin_in, {(uint32_t)cs, 1, (uint32_t)half}, prefix + "_sin_r");
    auto* cos_t = op_tile(cos_r, {1u, (uint32_t)heads, 1u}, prefix + "_cos_t");
    auto* sin_t = op_tile(sin_r, {1u, (uint32_t)heads, 1u}, prefix + "_sin_t");

    // Slice first and second halves of head_dim
    auto* x1 = op_strided_slice(x_3d,
        {{0,0,1},{0,0,1},{0,half,1}}, 0b011, 0b011, prefix + "_x1");  // [:, :, :half]
    auto* x2 = op_strided_slice(x_3d,
        {{0,0,1},{0,0,1},{half,hd,1}}, 0b011, 0b011, prefix + "_x2"); // [:, :, half:]

    // new_x1 = x1 * cos - x2 * sin  (all shapes {cs, heads, half} — exact match)
    auto* x1_cos = op_mul(x1, cos_t, prefix + "_x1cos");
    auto* x2_sin = op_mul(x2, sin_t, prefix + "_x2sin");

    auto* neg_x2_sin = op_unary(x2_sin, QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NEG, prefix + "_negx2sin");
    auto* new_x1 = op_add(x1_cos, neg_x2_sin, prefix + "_new_x1");

    // new_x2 = x2 * cos + x1 * sin
    auto* x2_cos = op_mul(x2, cos_t, prefix + "_x2cos");
    auto* x1_sin = op_mul(x1, sin_t, prefix + "_x1sin");
    auto* new_x2 = op_add(x2_cos, x1_sin, prefix + "_new_x2");

    return op_concat({new_x1, new_x2}, 2, prefix + "_rope");
}

// Attention: q_3d [chunk, heads, hd], k_3d [chunk, kv_heads, hd], v_3d [chunk, kv_heads, hd]
// causal_mask: [chunk, chunk]
QTensor* QNNDirectPrefill::Impl::op_attention(QTensor* q_3d, QTensor* k_3d, QTensor* v_3d,
                                               QTensor* causal_mask, float scale,
                                               const std::string& prefix) {
    uint32_t cs = (uint32_t)chunk_size, nh = (uint32_t)num_heads,
             nkv = (uint32_t)num_kv_heads, hd = (uint32_t)head_dim;

    // Expand K and V for GQA (num_heads > num_kv_heads)
    // Need grouped repeat: [KV0*g, KV1*g, ...], NOT interleaved tile [KV0, KV1, KV0, KV1, ...]
    // Use reshape([cs,nkv,hd]→[cs,nkv,1,hd]) + tile(g on dim2) + reshape([cs,nkv,g,hd]→[cs,nh,hd])
    QTensor* k_tiled = k_3d;
    QTensor* v_tiled = v_3d;
    if (nh != nkv) {
        uint32_t g = nh / nkv;
        auto* k_4d = op_reshape(k_3d, {cs, nkv, 1, hd}, prefix + "_k_4d");
        auto* k_tg  = op_tile(k_4d, {1, 1, g, 1}, prefix + "_k_tg");
        k_tiled = op_reshape(k_tg, {cs, nh, hd}, prefix + "_k_tiled");
        auto* v_4d = op_reshape(v_3d, {cs, nkv, 1, hd}, prefix + "_v_4d");
        auto* v_tg  = op_tile(v_4d, {1, 1, g, 1}, prefix + "_v_tg");
        v_tiled = op_reshape(v_tg, {cs, nh, hd}, prefix + "_v_tiled");
    }

    // Transpose to [heads, chunk, head_dim]
    auto* q_T = op_transpose(q_3d,    {1, 0, 2}, prefix + "_q_T");   // [nh, cs, hd]
    auto* k_T = op_transpose(k_tiled, {1, 0, 2}, prefix + "_k_T");   // [nh, cs, hd]
    auto* v_T = op_transpose(v_tiled, {1, 0, 2}, prefix + "_v_T");   // [nh, cs, hd]

    // Pre-scale Q by 1/sqrt(head_dim) before Q@K^T to prevent FP16 overflow.
    // Q and K values can reach ~60; unscaled dot product: 60*60*64=230400 >> FP16 max 65504.
    float scale_val = scale;
    uint16_t scale_fp16;
    { uint32_t b; memcpy(&b, &scale_val, 4);
      int e = (int)((b>>23)&0xFF)-127+15; uint32_t m=(b>>13)&0x3FF; uint32_t s=(b>>31)<<15;
      scale_fp16 = (e<=0)?0:(e>=31)?0x7C00:(uint16_t)(s|(e<<10)|m); }
    auto* q_scale_t = make_static_fp16(prefix + "_q_scale", {nh, cs, hd},
                                       std::vector<uint16_t>((size_t)nh*cs*hd, scale_fp16));
    auto* q_T_scaled = op_mul(q_T, q_scale_t, prefix + "_q_T_scaled");

    // scores = Q_scaled @ K^T: [nh, cs, cs]
    auto* scores = [&]() -> QTensor* {
        std::vector<uint32_t> out_shape = {nh, cs, cs};
        auto* out = make_native(prefix + "_scores", out_shape);
        add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_MAT_MUL, op_name("attn_qk"),
               {q_T_scaled, k_T}, {out},
               {scalar_bool_param(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, false),
                scalar_bool_param(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, true)});
        return out;
    }();

    // Add causal mask: tile from {1, cs, cs} to {nh, cs, cs}
    auto* mask_t = op_tile(causal_mask, {nh, 1u, 1u}, prefix + "_mask_t");
    auto* scores_masked = op_add(scores, mask_t, prefix + "_scores_masked");

    // Softmax over last dim
    auto* attn_weights = op_softmax(scores_masked, 2, prefix + "_attn_w");

    // Output = attn_weights @ V: [nh, cs, cs] × [nh, cs, hd] → [nh, cs, hd]
    auto* attn_out_T = [&]() -> QTensor* {
        std::vector<uint32_t> out_shape = {nh, cs, hd};
        auto* out = make_native(prefix + "_attn_out_T", out_shape);
        add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_MAT_MUL, op_name("attn_v"),
               {attn_weights, v_T}, {out},
               {scalar_bool_param(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, false),
                scalar_bool_param(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, false)});
        return out;
    }();

    // Transpose back to [chunk, heads, head_dim]
    auto* attn_out_3d = op_transpose(attn_out_T, {1, 0, 2}, prefix + "_attn_out_3d");
    // Reshape to [chunk, heads * head_dim]
    return op_reshape(attn_out_3d, {cs, nh * hd}, prefix + "_attn_out_flat");
}


void QNNDirectPrefill::Impl::build_layer(const std::string& folder, int li,
                                          QTensor*& hidden, QTensor* cos_in, QTensor* sin_in,
                                          QTensor* causal_mask,
                                          QTensor*& k_out, QTensor*& v_out) {
    std::string p = "l" + std::to_string(li) + "_";
    std::string wp = folder + "/layer_" + std::to_string(li) + "_";

    uint32_t cs = (uint32_t)chunk_size;
    uint32_t nh = (uint32_t)num_heads, nkv = (uint32_t)num_kv_heads;
    uint32_t hd = (uint32_t)head_dim;

    // Load norm weights
    auto* in_norm_w    = load_weight_tensor(p + "in_norm_w",    wp + "input_norm.weights");
    auto* post_norm_w  = load_weight_tensor(p + "post_norm_w",  wp + "post_attn_norm.weights");

    // Pre-attention rms_norm (prescale all layers: hidden state ~0.001-0.05 RMS keeps x^2 subnormal)
    auto* normed = op_rms_norm(hidden, in_norm_w, p + "normed", true);

    // Q, K, V projections
    auto* wq = load_weight_tensor(p + "wq", wp + "attn_q.weights");
    auto* wk = load_weight_tensor(p + "wk", wp + "attn_k.weights");
    auto* wv = load_weight_tensor(p + "wv", wp + "attn_v.weights");

    auto* q_proj = op_matmul_T(normed, wq, p + "q_proj");
    auto* k_proj = op_matmul_T(normed, wk, p + "k_proj");
    auto* v_proj = op_matmul_T(normed, wv, p + "v_proj");

    // Add biases if present
    auto try_bias = [&](QTensor* proj, const std::string& bias_path, const std::string& name) -> QTensor* {
        if (file_exists(bias_path)) {
            auto* b = load_weight_tensor(p + name + "_b", bias_path);
            return op_add(proj, b, p + name + "_biased");
        }
        return proj;
    };
    q_proj = try_bias(q_proj, wp + "attn_q_bias.weights", "q");
    k_proj = try_bias(k_proj, wp + "attn_k_bias.weights", "k");
    v_proj = try_bias(v_proj, wp + "attn_v_bias.weights", "v");

    // Reshape to 3D for RoPE
    auto* q_3d = op_reshape(q_proj, {cs, nh, hd}, p + "q_3d");
    auto* k_3d = op_reshape(k_proj, {cs, nkv, hd}, p + "k_3d");
    auto* v_3d = op_reshape(v_proj, {cs, nkv, hd}, p + "v_3d");

    // Apply RoPE to Q and K
    auto* q_rope = op_rope(q_3d, cos_in, sin_in, (int)nh, p + "q_rope");
    auto* k_rope = op_rope(k_3d, cos_in, sin_in, (int)nkv, p + "k_rope");

    // Output K (with RoPE) and V directly from QNN
    k_out = make_output("out_k_" + std::to_string(li), {cs, nkv, hd});
    add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_RESHAPE, op_name("k_out_copy"),
           {k_rope}, {k_out}, {});
    v_out = make_output("out_v_" + std::to_string(li), {cs, nkv, hd});
    add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_RESHAPE, op_name("v_out_copy"),
           {v_3d}, {v_out}, {});

    // Full attention computation (needed to evolve hidden state)
    float attn_scale = 1.0f / std::sqrt((float)head_dim);
    auto* attn_out_flat = op_attention(q_rope, k_rope, v_3d, causal_mask, attn_scale, p + "attn");

    // O projection
    auto* wo = load_weight_tensor(p + "wo", wp + "attn_output.weights");
    auto* o_proj = op_matmul_T(attn_out_flat, wo, p + "o_proj");
    o_proj = try_bias(o_proj, wp + "attn_output_bias.weights", "o");

    // Residual add (attention)
    auto* h_attn = op_add(hidden, o_proj, p + "h_attn");

    // Post-attention rms_norm (prescale all layers: h_attn is small in early layers)
    auto* normed2 = op_rms_norm(h_attn, post_norm_w, p + "normed2", true);

    // FFN: SwiGLU
    auto* wgate = load_weight_tensor(p + "wgate", wp + "ffn_gate.weights");
    auto* wup   = load_weight_tensor(p + "wup",   wp + "ffn_up.weights");
    auto* wdown = load_weight_tensor(p + "wdown", wp + "ffn_down.weights");

    auto* gate_out = op_matmul_T(normed2, wgate, p + "gate");
    auto* up_out   = op_matmul_T(normed2, wup,   p + "up");

    auto* gate_silu = op_silu(gate_out, p + "gate_silu");
    auto* gated = op_mul(gate_silu, up_out, p + "gated");

    auto* ffn_out = op_matmul_T(gated, wdown, p + "ffn_out");

    // Residual add (FFN)
    hidden = op_add(h_attn, ffn_out, p + "h_out");
}

// ---- Graph building ----

bool QNNDirectPrefill::Impl::build_graph(const std::string& model_folder, const std::string& cache_path) {
    fprintf(stderr, "[QNN Direct] building graph (%d layers, chunk=%d)...\n",
            num_layers, chunk_size);

    if (!qnn.graphCreate) { fprintf(stderr, "[QNN Direct] graphCreate not available\n"); return false; }
    QnnHtpGraph_CustomConfig_t htp_prec_cfg = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
    htp_prec_cfg.option    = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
    htp_prec_cfg.precision = QNN_PRECISION_FLOAT16;
    QnnGraph_Config_t g_prec = QNN_GRAPH_CONFIG_INIT;
    g_prec.option       = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    g_prec.customConfig = &htp_prec_cfg;
    const QnnGraph_Config_t* g_cfgs[] = {&g_prec, nullptr};
    Qnn_ErrorHandle_t err = qnn.graphCreate(context_handle, "prefill", g_cfgs, &graph_handle);
    if (err != QNN_SUCCESS) {
        fprintf(stderr, "[QNN Direct] graphCreate failed: %lld\n", (long long)err);
        return false;
    }
    fprintf(stderr, "[QNN Direct] graph created\n");

    uint32_t cs = (uint32_t)chunk_size, hd = (uint32_t)hidden_dim;
    uint32_t half_dim = (uint32_t)(head_dim / 2);

    // Input tensors
    auto* emb_in  = make_tensor("emb", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16,
                                 {cs, hd});
    auto* cos_in  = make_tensor("rope_cos", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16,
                                 {cs, half_dim});
    auto* sin_in  = make_tensor("rope_sin", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16,
                                 {cs, half_dim});

    // Causal mask: [1, chunk, chunk] FP16 (broadcast over heads dim in scores [nh, cs, cs])
    std::vector<uint16_t> mask_data((size_t)chunk_size * chunk_size);
    uint16_t neg_inf_fp16 = 0xFBFF;  // -65504 in FP16 (most negative normal)
    uint16_t zero_fp16    = 0x0000;
    for (int i = 0; i < chunk_size; i++)
        for (int j = 0; j < chunk_size; j++)
            mask_data[i * chunk_size + j] = (j <= i) ? zero_fp16 : neg_inf_fp16;
    auto* causal_mask = make_static_fp16("causal_mask", {1, cs, cs}, mask_data);

    // Build all transformer layers
    QTensor* hidden = emb_in;
    std::vector<QTensor*> k_outs(num_layers), v_outs(num_layers);
    for (int li = 0; li < num_layers; li++) {
        build_layer(model_folder, li, hidden, cos_in, sin_in, causal_mask, k_outs[li], v_outs[li]);
        fprintf(stderr, "[QNN Direct] layer %d / %d added\n", li + 1, num_layers);
    }

    // Save I/O tensor IDs before finalize (assigned during tensorCreateGraphTensor in add_op)
    exec_emb_id = emb_in->t.v1.id;
    exec_cos_id = cos_in->t.v1.id;
    exec_sin_id = sin_in->t.v1.id;
    exec_k_ids.resize(num_layers);
    exec_v_ids.resize(num_layers);
    for (int i = 0; i < num_layers; i++) {
        exec_k_ids[i] = k_outs[i]->t.v1.id;
        exec_v_ids[i] = v_outs[i]->t.v1.id;
    }
    fprintf(stderr, "[QNN Direct] I/O ids: emb=%u cos=%u sin=%u k0=%u v0=%u\n",
            exec_emb_id, exec_cos_id, exec_sin_id,
            exec_k_ids.empty() ? 0u : exec_k_ids[0],
            exec_v_ids.empty() ? 0u : exec_v_ids[0]);

    // Finalize
    fprintf(stderr, "[QNN Direct] finalizing (may take ~60s first time)...\n");
    fflush(stderr);
    err = qnn.graphFinalize(graph_handle, nullptr, nullptr);
    if (err != QNN_SUCCESS) {
        fprintf(stderr, "[QNN Direct] graphFinalize failed: %lld\n", (long long)err);
        return false;
    }
    fprintf(stderr, "[QNN Direct] graph finalized\n");

    // Refresh graph handle via graphRetrieve in case graphFinalize internally replaced it.
    if (qnn.graphRetrieve) {
        Qnn_GraphHandle_t retrieved = nullptr;
        Qnn_ErrorHandle_t re = qnn.graphRetrieve(context_handle, "prefill", &retrieved);
        if (re == QNN_SUCCESS && retrieved != nullptr) {
            fprintf(stderr, "[QNN Direct] graphRetrieve: old=%p new=%p\n",
                    (void*)graph_handle, (void*)retrieved);
            graph_handle = retrieved;
        } else {
            fprintf(stderr, "[QNN Direct] graphRetrieve after finalize: %lld\n", (long long)re);
        }
    }

    // Free weight files now (QNN has copied the data)
    weight_files.clear();

    // Try best-effort binary serialization for future caching (don't fail if it doesn't work).
    // On the HTP userPD path, graphExecute works directly on the compiled context after graphFinalize.
    // The binary round-trip is optional: only needed to skip re-compilation on next run.
    if (qnn.contextGetBinarySize && qnn.contextGetBinary) {
        Qnn_ContextBinarySize_t bin_size = 0;
        Qnn_ContextBinarySize_t written  = 0;
        if (qnn.contextGetBinarySize(context_handle, &bin_size) == QNN_SUCCESS && bin_size > 0) {
            std::vector<char> bin_buf((size_t)bin_size);
            if (qnn.contextGetBinary(context_handle, bin_buf.data(), bin_size, &written) == QNN_SUCCESS && written > 0) {
                if (!cache_path.empty()) {
                    std::ofstream cf(cache_path, std::ios::binary);
                    if (cf.is_open()) {
                        cf.write(bin_buf.data(), (size_t)written);
                        fprintf(stderr, "[QNN Direct] saved cache: %s (%zu bytes)\n",
                                cache_path.c_str(), (size_t)written);
                    }
                    save_id_sidecar(cache_path + ".ids");
                }
                // Attempt round-trip to validate the binary (and use the deserialized context).
                // If this fails, keep the compiled context and use it directly.
                if (qnn.contextCreateFromBinary && qnn.graphRetrieve) {
                    Qnn_ContextHandle_t new_ctx = nullptr;
                    Qnn_ErrorHandle_t rt = qnn.contextCreateFromBinary(
                        backend_handle, device_handle, nullptr,
                        bin_buf.data(), (size_t)written, &new_ctx, nullptr);
                    if (rt == QNN_SUCCESS) {
                        Qnn_GraphHandle_t new_graph = nullptr;
                        rt = qnn.graphRetrieve(new_ctx, "prefill", &new_graph);
                        if (rt == QNN_SUCCESS) {
                            qnn.contextFree(context_handle, nullptr);
                            context_handle = new_ctx;
                            graph_handle   = new_graph;
                            fprintf(stderr, "[QNN Direct] binary round-trip ok\n");
                        } else {
                            fprintf(stderr, "[QNN Direct] graphRetrieve failed: %lld — using compiled context\n",
                                    (long long)rt);
                            qnn.contextFree(new_ctx, nullptr);
                        }
                    } else {
                        fprintf(stderr, "[QNN Direct] contextCreateFromBinary failed: %lld — using compiled context\n",
                                (long long)rt);
                    }
                }
            }
        }
    }
    fprintf(stderr, "[QNN Direct] using context %p graph %p\n",
            (void*)context_handle, (void*)graph_handle);

    setup_exec_tensors();
    return true;
}

bool QNNDirectPrefill::Impl::load_from_cache(const std::string& path) {
    if (!load_id_sidecar(path + ".ids")) {
        fprintf(stderr, "[QNN Direct] missing/invalid sidecar: %s.ids\n", path.c_str());
        return false;
    }
    fprintf(stderr, "[QNN Direct] loading cache: %s\n", path.c_str());
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) return false;
    size_t size = (size_t)f.tellg();
    f.seekg(0);
    std::vector<char> buf(size);
    f.read(buf.data(), size);
    f.close();

    Qnn_ErrorHandle_t err = qnn.contextCreateFromBinary(
        backend_handle, device_handle, nullptr,
        buf.data(), size, &context_handle, nullptr);
    if (err != QNN_SUCCESS) {
        fprintf(stderr, "[QNN Direct] contextCreateFromBinary failed: %lld\n", (long long)err);
        return false;
    }
    err = qnn.graphRetrieve(context_handle, "prefill", &graph_handle);
    if (err != QNN_SUCCESS) {
        fprintf(stderr, "[QNN Direct] graphRetrieve failed: %lld\n", (long long)err);
        return false;
    }
    setup_exec_tensors();
    fprintf(stderr, "[QNN Direct] loaded from cache\n");
    return true;
}

bool QNNDirectPrefill::Impl::save_to_cache(const std::string& path) {
    if (!qnn.contextGetBinarySize || !qnn.contextGetBinary) return false;
    Qnn_ContextBinarySize_t size = 0;
    if (qnn.contextGetBinarySize(context_handle, &size) != QNN_SUCCESS || size == 0) return false;
    std::vector<char> buf(size);
    Qnn_ContextBinarySize_t written = 0;
    if (qnn.contextGetBinary(context_handle, buf.data(), size, &written) != QNN_SUCCESS) return false;
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f.write(buf.data(), written);
    fprintf(stderr, "[QNN Direct] saved cache: %s (%zu bytes)\n", path.c_str(), (size_t)written);
    save_id_sidecar(path + ".ids");
    return true;
}

bool QNNDirectPrefill::Impl::save_id_sidecar(const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    uint32_t magic = 0x4B564F59u; // "KVOY" — bumped to invalidate h0 debug caches
    uint32_t nl    = (uint32_t)num_layers;
    f.write(reinterpret_cast<const char*>(&magic),       4);
    f.write(reinterpret_cast<const char*>(&exec_emb_id), 4);
    f.write(reinterpret_cast<const char*>(&exec_cos_id), 4);
    f.write(reinterpret_cast<const char*>(&exec_sin_id), 4);
    f.write(reinterpret_cast<const char*>(&nl),          4);
    for (uint32_t id : exec_k_ids) f.write(reinterpret_cast<const char*>(&id), 4);
    for (uint32_t id : exec_v_ids) f.write(reinterpret_cast<const char*>(&id), 4);
    return true;
}

bool QNNDirectPrefill::Impl::load_id_sidecar(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    uint32_t magic = 0, nl = 0;
    f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != 0x4B564F59u) return false;
    f.read(reinterpret_cast<char*>(&exec_emb_id), 4);
    f.read(reinterpret_cast<char*>(&exec_cos_id), 4);
    f.read(reinterpret_cast<char*>(&exec_sin_id), 4);
    f.read(reinterpret_cast<char*>(&nl), 4);
    if ((int)nl != num_layers) { fprintf(stderr, "[QNN Direct] sidecar layer count mismatch\n"); return false; }
    exec_k_ids.resize(nl);
    exec_v_ids.resize(nl);
    for (auto& id : exec_k_ids) f.read(reinterpret_cast<char*>(&id), 4);
    for (auto& id : exec_v_ids) f.read(reinterpret_cast<char*>(&id), 4);
    return true;
}

void QNNDirectPrefill::Impl::setup_exec_tensors() {
    emb_buf.assign((size_t)chunk_size * hidden_dim, 0);
    cos_buf.assign((size_t)chunk_size * (head_dim / 2), 0);
    sin_buf.assign((size_t)chunk_size * (head_dim / 2), 0);
    k_out_bufs.resize(num_layers);
    v_out_bufs.resize(num_layers);
    for (int i = 0; i < num_layers; i++) {
        k_out_bufs[i].assign((size_t)chunk_size * num_kv_heads * head_dim, 0);
        v_out_bufs[i].assign((size_t)chunk_size * num_kv_heads * head_dim, 0);
    }

    // Helper: create a minimal QTensor from a saved ID (used when loading from cache,
    // where the tensors map is empty but exec_*_id fields are populated).
    auto make_from_id = [&](uint32_t id, Qnn_TensorType_t type,
                             const std::vector<uint32_t>& shape) -> QTensor* {
        auto qt = std::make_unique<QTensor>();
        qt->name = "cached_" + std::to_string(id);
        qt->dims = shape;
        qt->t = QNN_TENSOR_INIT;
        qt->t.version          = QNN_TENSOR_VERSION_1;
        qt->t.v1.id            = id;
        qt->t.v1.type          = type;
        qt->t.v1.dataFormat    = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
        qt->t.v1.dataType      = QNN_DATATYPE_FLOAT_16;
        qt->t.v1.quantizeParams = { QNN_DEFINITION_UNDEFINED,
                                    QNN_QUANTIZATION_ENCODING_UNDEFINED, {{0, 0}} };
        qt->t.v1.rank          = (uint32_t)shape.size();
        qt->t.v1.dimensions    = qt->dims.data();
        qt->t.v1.memType       = QNN_TENSORMEMTYPE_RAW;
        qt->t.v1.clientBuf     = { nullptr, 0 };
        QTensor* ptr = qt.get();
        tensor_store.push_back(std::move(qt));
        return ptr;
    };

    // Look up by name (graph-build path), or synthesise from saved ID (cache path).
    auto get_or_make = [&](const std::string& name, uint32_t saved_id,
                            Qnn_TensorType_t type,
                            const std::vector<uint32_t>& shape) -> QTensor* {
        if (tensors.count(name)) return tensors.at(name);
        if (saved_id != 0)       return make_from_id(saved_id, type, shape);
        return nullptr;
    };

    uint32_t cs = (uint32_t)chunk_size, hd = (uint32_t)hidden_dim;
    uint32_t half = (uint32_t)(head_dim / 2);

    uint32_t nkv = (uint32_t)num_kv_heads;
    uint32_t hd_u = (uint32_t)head_dim;

    exec_emb_qt = get_or_make("emb",      exec_emb_id, QNN_TENSOR_TYPE_APP_WRITE, {cs, hd});
    exec_cos_qt = get_or_make("rope_cos", exec_cos_id, QNN_TENSOR_TYPE_APP_WRITE, {cs, half});
    exec_sin_qt = get_or_make("rope_sin", exec_sin_id, QNN_TENSOR_TYPE_APP_WRITE, {cs, half});

    exec_k_qts.resize(num_layers, nullptr);
    exec_v_qts.resize(num_layers, nullptr);
    for (int i = 0; i < num_layers; i++) {
        uint32_t kid = (i < (int)exec_k_ids.size()) ? exec_k_ids[i] : 0;
        uint32_t vid = (i < (int)exec_v_ids.size()) ? exec_v_ids[i] : 0;
        exec_k_qts[i] = get_or_make("out_k_" + std::to_string(i), kid,
                                     QNN_TENSOR_TYPE_APP_READ, {cs, nkv, hd_u});
        exec_v_qts[i] = get_or_make("out_v_" + std::to_string(i), vid,
                                     QNN_TENSOR_TYPE_APP_READ, {cs, nkv, hd_u});
    }

    // Set data pointers
    auto set_buf = [](QTensor* qt, void* data, uint32_t sz) {
        if (qt) { qt->t.v1.clientBuf.data = data; qt->t.v1.clientBuf.dataSize = sz; }
    };
    set_buf(exec_emb_qt, emb_buf.data(), (uint32_t)(emb_buf.size() * 2));
    set_buf(exec_cos_qt, cos_buf.data(), (uint32_t)(cos_buf.size() * 2));
    set_buf(exec_sin_qt, sin_buf.data(), (uint32_t)(sin_buf.size() * 2));
    for (int i = 0; i < num_layers; i++) {
        set_buf(exec_k_qts[i], k_out_bufs[i].data(), (uint32_t)(k_out_bufs[i].size() * 2));
        set_buf(exec_v_qts[i], v_out_bufs[i].data(), (uint32_t)(v_out_bufs[i].size() * 2));
    }

    // Build flat Qnn_Tensor_t arrays — outputs interleaved: [k_0, v_0, k_1, v_1, ...]
    exec_inputs.resize(3);
    exec_inputs[0] = exec_emb_qt ? exec_emb_qt->t : Qnn_Tensor_t{};
    exec_inputs[1] = exec_cos_qt ? exec_cos_qt->t : Qnn_Tensor_t{};
    exec_inputs[2] = exec_sin_qt ? exec_sin_qt->t : Qnn_Tensor_t{};

    exec_outputs.resize(2 * num_layers);
    for (int i = 0; i < num_layers; i++) {
        exec_outputs[2*i]   = exec_k_qts[i] ? exec_k_qts[i]->t : Qnn_Tensor_t{};
        exec_outputs[2*i+1] = exec_v_qts[i] ? exec_v_qts[i]->t : Qnn_Tensor_t{};
    }

    fprintf(stderr, "[QNN Direct] exec tensors: emb=%s cos=%s sin=%s k0=%s v0=%s\n",
            exec_emb_qt ? "ok" : "MISSING",
            exec_cos_qt ? "ok" : "MISSING",
            exec_sin_qt ? "ok" : "MISSING",
            (num_layers > 0 && exec_k_qts[0]) ? "ok" : "MISSING",
            (num_layers > 0 && exec_v_qts[0]) ? "ok" : "MISSING");
}

// ---- Main load function ----

QNNDirectPrefill::QNNDirectPrefill() : impl_(std::make_unique<Impl>()) {}
QNNDirectPrefill::~QNNDirectPrefill() { impl_->teardown(); }

bool QNNDirectPrefill::load(const std::string& model_folder) {
    auto& I = *impl_;
    std::string dll_path = find_qnn_htp_lib();
    if (!I.load_dll(dll_path)) return false;
    if (!I.init_interface()) return false;

    if (!I.parse_config(model_folder)) return false;
    I.model_folder_path = model_folder;

    // Try loading from cache first (skips backend/context creation from scratch)
    std::string cache_path = model_folder + "/qnn_prefill_c" +
                             std::to_string(I.chunk_size) + "_kvout.bin";

    // Init backend (always needed)
    if (!I.init_backend()) return false;

    if (file_exists(cache_path) && file_exists(cache_path + ".ids")) {
        // Free the context created by init_backend, load from binary instead
        if (I.context_handle && I.qnn.contextFree)
            I.qnn.contextFree(I.context_handle, nullptr);
        I.context_handle = nullptr;
        if (I.load_from_cache(cache_path)) {
            I.loaded = true;
            fprintf(stderr, "[QNN Direct] ready (from cache)\n");
            return true;
        }
        // If cache load failed, recreate context and build from scratch
        if (I.qnn.contextCreate)
            I.qnn.contextCreate(I.backend_handle, I.device_handle, nullptr, &I.context_handle);
    }

    // build_graph saves cache during the round-trip (before freeing the compiled context)
    if (!I.build_graph(model_folder, cache_path)) return false;

    I.loaded = true;
    fprintf(stderr, "[QNN Direct] ready\n");
    return true;
}

bool QNNDirectPrefill::is_available() const { return impl_->loaded; }
int  QNNDirectPrefill::get_chunk_size()   const { return impl_->chunk_size; }
int  QNNDirectPrefill::get_hidden_dim()   const { return impl_->hidden_dim; }
int  QNNDirectPrefill::get_num_layers()   const { return impl_->num_layers; }
int  QNNDirectPrefill::get_num_kv_heads() const { return impl_->num_kv_heads; }
int  QNNDirectPrefill::get_head_dim()     const { return impl_->head_dim; }

// ---- Execution ----

NPUPrefillDirectResult QNNDirectPrefill::prefill_chunk_direct(
    const std::vector<__fp16>& embeddings,
    int position_offset,
    const std::string& /*input_name*/)
{
    NPUPrefillDirectResult r{};
    auto& I = *impl_;
    if (!I.loaded || !I.graph_handle) return r;
    fprintf(stderr, "[QNN Direct] prefill_chunk_direct pos=%d emb_size=%zu\n",
            position_offset, embeddings.size());

    // Copy embeddings into input buffer
    size_t emb_sz = (size_t)I.chunk_size * I.hidden_dim;
    memcpy(I.emb_buf.data(), embeddings.data(), std::min(emb_sz, embeddings.size()) * 2);

    // Compute RoPE cos/sin tables for positions [offset, offset+chunk_size)
    int half_dim = I.head_dim / 2;
    for (int pos = 0; pos < I.chunk_size; pos++) {
        int abs_pos = position_offset + pos;
        for (int i = 0; i < half_dim; i++) {
            float theta = (float)abs_pos / std::pow(I.rope_theta, (2.0f * i) / I.head_dim);
            float cv = std::cos(theta), sv = std::sin(theta);
            // Convert to fp16
            auto to_fp16 = [](float v) -> uint16_t {
                if (v == 0.0f) return 0;
                uint32_t b; memcpy(&b, &v, 4);
                int e = (int)((b>>23)&0xFF)-127+15;
                uint32_t m = (b>>13)&0x3FF, s = (b>>31)<<15;
                if (e <= 0) return (uint16_t)s;
                if (e >= 31) return (uint16_t)(s|0x7C00);
                return (uint16_t)(s|(e<<10)|m);
            };
            I.cos_buf[pos * half_dim + i] = to_fp16(cv);
            I.sin_buf[pos * half_dim + i] = to_fp16(sv);
        }
    }

    // Execute
    Qnn_ErrorHandle_t err = I.qnn.graphExecute(
        I.graph_handle,
        I.exec_inputs.data(),  (uint32_t)I.exec_inputs.size(),
        I.exec_outputs.data(), (uint32_t)I.exec_outputs.size(),
        nullptr, nullptr);

    if (err != QNN_SUCCESS) {
        fprintf(stderr, "[QNN Direct] graphExecute failed: %lld  ctx=%p grph=%p\n",
                (long long)err, (void*)I.context_handle, (void*)I.graph_handle);
        fprintf(stderr, "[QNN Direct]   inputs: n=%u ids=", (uint32_t)I.exec_inputs.size());
        for (auto& t : I.exec_inputs)  fprintf(stderr, "%u ", t.v1.id);
        fprintf(stderr, "\n[QNN Direct]   outputs: n=%u ids=", (uint32_t)I.exec_outputs.size());
        for (int i = 0; i < 3 && i < (int)I.exec_outputs.size(); i++) fprintf(stderr, "%u ", I.exec_outputs[i].v1.id);
        fprintf(stderr, "...\n");
        return r;
    }
    r.valid = true;
    r.k_caches.resize(I.num_layers);
    r.v_caches.resize(I.num_layers);
    for (int li = 0; li < I.num_layers; li++) {
        r.k_caches[li] = { reinterpret_cast<const __fp16*>(I.k_out_bufs[li].data()),
                           I.k_out_bufs[li].size() };
        r.v_caches[li] = { reinterpret_cast<const __fp16*>(I.v_out_bufs[li].data()),
                           I.v_out_bufs[li].size() };
    }

    {
        int kv_stride = I.num_kv_heads * I.head_dim;
        for (int ci = 0; ci < I.num_layers; ci++) {
            int li = ci;
            if (li >= I.num_layers) continue;
            const __fp16* k0 = reinterpret_cast<const __fp16*>(I.k_out_bufs[li].data());
            const __fp16* v0 = reinterpret_cast<const __fp16*>(I.v_out_bufs[li].data());
            fprintf(stderr, "[QNN] K[0][%d]:", li);
            for (int i = 0; i < 8; i++) { uint16_t u; memcpy(&u, &k0[i], 2); fprintf(stderr, " %04x", u); }
            fprintf(stderr, "\n");
            fprintf(stderr, "[QNN] K[1][%d]:", li);
            for (int i = 0; i < 8; i++) { uint16_t u; memcpy(&u, &k0[kv_stride + i], 2); fprintf(stderr, " %04x", u); }
            fprintf(stderr, "\n");
            fprintf(stderr, "[QNN] V[0][%d]:", li);
            for (int i = 0; i < 8; i++) { uint16_t u; memcpy(&u, &v0[i], 2); fprintf(stderr, " %04x", u); }
            fprintf(stderr, "\n");
        }
    }

    return r;
}

// ============================================================
// QNNDirectEncoder
// ============================================================

struct QNNDirectEncoder::Impl {
    void* dll_handle     = nullptr;
    void* sys_dll_handle = nullptr;
    QNN_INTERFACE_VER_TYPE qnn = QNN_INTERFACE_VER_TYPE_INIT;

    Qnn_LogHandle_t     log_handle     = nullptr;
    Qnn_BackendHandle_t backend_handle = nullptr;
    Qnn_DeviceHandle_t  device_handle  = nullptr;
    Qnn_ContextHandle_t context_handle = nullptr;
    Qnn_GraphHandle_t   graph_handle   = nullptr;

    std::string model_folder;
    int hidden_dim   = 384;
    int num_layers   = 4;
    int num_heads    = 6;
    int head_dim     = 64;
    int ffn_dim      = 1536;
    int n_mels       = 80;
    int T_mel        = 3000;
    int T_enc        = 1500;
    float ln_eps     = 1e-5f;

    bool loaded      = false;
    bool graph_built = false;

    std::vector<std::unique_ptr<QTensor>> tensor_store;
    std::unordered_map<std::string, QTensor*> tensors;
    std::vector<std::unique_ptr<CactFile>> weight_files;
    int op_idx = 0;

    // I/O buffers
    std::vector<uint16_t> input_buf;   // [1, T_mel, n_mels] FP16 (NTC)
    std::vector<uint16_t> output_buf;  // [T_enc, hidden_dim] FP16

    // I/O tensor ids
    uint32_t exec_in_id  = 0;
    uint32_t exec_out_id = 0;

    QTensor* exec_in_qt  = nullptr;
    QTensor* exec_out_qt = nullptr;
    std::vector<Qnn_Tensor_t> exec_inputs;
    std::vector<Qnn_Tensor_t> exec_outputs;

    // ---- Parakeet members ----
    bool is_parakeet        = false;
    int pk_time_frames      = 3000;
    int pk_num_mel_bins     = 80;
    int pk_num_layers       = 42;
    int pk_hidden_dim       = 1024;
    int pk_num_heads        = 8;
    int pk_head_dim         = 128;
    int pk_conv_channels    = 256;
    int pk_conv_kernel_size = 9;
    float pk_ln_eps         = 1e-5f;
    float pk_bn_eps         = 1e-5f;
    int pk_T_out            = 0;
    int pk_n_segs           = 0;
    std::vector<PSegState> pk_segs;          // 1-indexed: pk_segs[1..pk_n_segs]
    std::vector<Qnn_ContextHandle_t> pk_seg_ctxs;
    PSegState* pk_cur       = nullptr;

    struct PkCpuSub {
        int C = 256, W3 = 0;
        std::vector<float> conv0_w, conv0_b;
        std::vector<float> dw1_w, dw1_b;
        std::vector<float> pw1_w, pw1_b;
        std::vector<float> dw2_w, dw2_b;
        std::vector<float> pw2_w, pw2_b;
        std::vector<float> lin_w, lin_b;
        std::vector<float> tmp0, tmp1, tmp2;
        bool ok = false;
    } pk_cpu_sub;

    // ---- Gemma4-Audio members ----
    bool is_g4a = false;
    int g4a_mel_bins   = 128;
    int g4a_conv0_ch   = 128;
    int g4a_conv1_ch   = 32;
    int g4a_hidden     = 1024;
    int g4a_out_dim    = 1536;
    int g4a_num_layers = 12;
    int g4a_num_heads  = 8;
    int g4a_head_dim   = 128;
    int g4a_chunk_mel  = 48;
    int g4a_chunk_out  = 12;
    float g4a_rms_eps  = 1e-6f;
    float g4a_ln_eps   = 0.001f;
    float g4a_residual = 0.5f;
    float g4a_logit_cap = 50.0f;
    float g4a_k_scale  = 0.0f; // computed in g4a_load
    int g4a_context_left  = 13;
    int g4a_context_right = 0;
    std::vector<uint16_t> g4a_input_buf;
    std::vector<uint16_t> g4a_output_buf;
    uint32_t g4a_exec_in_id  = 0;
    uint32_t g4a_exec_out_id = 0;
    QTensor* g4a_exec_in_qt  = nullptr;
    QTensor* g4a_exec_out_qt = nullptr;
    std::vector<Qnn_Tensor_t> g4a_exec_inputs;
    std::vector<Qnn_Tensor_t> g4a_exec_outputs;

    // ---- helpers shared with Prefill::Impl ----
    std::string op_name(const std::string& type) {
        return type + "_e" + std::to_string(op_idx++);
    }

    QTensor* make_tensor(const std::string& name, Qnn_TensorType_t type,
                         Qnn_DataType_t dtype, const std::vector<uint32_t>& shape,
                         const void* data = nullptr, size_t data_bytes = 0) {
        auto qt = std::make_unique<QTensor>();
        qt->name = name; qt->dims = shape;
        qt->t = QNN_TENSOR_INIT;
        qt->t.version = QNN_TENSOR_VERSION_1;
        qt->t.v1.id   = 0;
        qt->t.v1.name = qt->name.c_str();
        qt->t.v1.type = type;
        qt->t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
        qt->t.v1.dataType   = dtype;
        qt->t.v1.quantizeParams.encodingDefinition  = QNN_DEFINITION_UNDEFINED;
        qt->t.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
        qt->t.v1.rank       = (uint32_t)shape.size();
        qt->t.v1.dimensions = qt->dims.data();
        qt->t.v1.memType    = QNN_TENSORMEMTYPE_RAW;
        qt->t.v1.clientBuf.data = (type == QNN_TENSOR_TYPE_STATIC) ? const_cast<void*>(data) : nullptr;
        size_t sz = data_bytes;
        if (sz == 0) {
            sz = 1; for (auto d : shape) sz *= d;
            if (dtype == QNN_DATATYPE_FLOAT_16 || dtype == QNN_DATATYPE_INT_16) sz *= 2;
            else if (dtype == QNN_DATATYPE_FLOAT_32 || dtype == QNN_DATATYPE_INT_32) sz *= 4;
        }
        qt->t.v1.clientBuf.dataSize = (uint32_t)sz;
        QTensor* raw = qt.get();
        tensors[name] = raw;
        tensor_store.push_back(std::move(qt));
        return raw;
    }

    QTensor* make_static_fp16(const std::string& name, const std::vector<uint32_t>& shape,
                               std::vector<uint16_t> data) {
        auto qt = std::make_unique<QTensor>();
        qt->name = name; qt->dims = shape; qt->static_fp16 = std::move(data);
        qt->t = QNN_TENSOR_INIT;
        qt->t.version = QNN_TENSOR_VERSION_1;
        qt->t.v1.id   = 0; qt->t.v1.name = qt->name.c_str();
        qt->t.v1.type = QNN_TENSOR_TYPE_STATIC;
        qt->t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
        qt->t.v1.dataType   = QNN_DATATYPE_FLOAT_16;
        qt->t.v1.quantizeParams.encodingDefinition  = QNN_DEFINITION_UNDEFINED;
        qt->t.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
        qt->t.v1.rank = (uint32_t)shape.size(); qt->t.v1.dimensions = qt->dims.data();
        qt->t.v1.memType = QNN_TENSORMEMTYPE_RAW;
        qt->t.v1.clientBuf.data = qt->static_fp16.data();
        qt->t.v1.clientBuf.dataSize = (uint32_t)(qt->static_fp16.size() * 2);
        QTensor* raw = qt.get(); tensors[name] = raw;
        tensor_store.push_back(std::move(qt)); return raw;
    }

    QTensor* make_static_i32(const std::string& name, const std::vector<uint32_t>& shape,
                              std::vector<int32_t> data) {
        auto qt = std::make_unique<QTensor>();
        qt->name = name; qt->dims = shape; qt->static_i32 = std::move(data);
        qt->t = QNN_TENSOR_INIT;
        qt->t.version = QNN_TENSOR_VERSION_1; qt->t.v1.id = 0; qt->t.v1.name = qt->name.c_str();
        qt->t.v1.type = QNN_TENSOR_TYPE_STATIC;
        qt->t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
        qt->t.v1.dataType = QNN_DATATYPE_INT_32;
        qt->t.v1.quantizeParams.encodingDefinition  = QNN_DEFINITION_UNDEFINED;
        qt->t.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
        qt->t.v1.rank = (uint32_t)shape.size(); qt->t.v1.dimensions = qt->dims.data();
        qt->t.v1.memType = QNN_TENSORMEMTYPE_RAW;
        qt->t.v1.clientBuf.data = qt->static_i32.data();
        qt->t.v1.clientBuf.dataSize = (uint32_t)(qt->static_i32.size() * 4);
        QTensor* raw = qt.get(); tensors[name] = raw;
        tensor_store.push_back(std::move(qt)); return raw;
    }

    QTensor* make_static_u32(const std::string& name, const std::vector<uint32_t>& shape,
                              std::vector<uint32_t> data) {
        auto qt = std::make_unique<QTensor>();
        qt->name = name; qt->dims = shape; qt->static_u32 = std::move(data);
        qt->t = QNN_TENSOR_INIT;
        qt->t.version = QNN_TENSOR_VERSION_1; qt->t.v1.id = 0; qt->t.v1.name = qt->name.c_str();
        qt->t.v1.type = QNN_TENSOR_TYPE_STATIC;
        qt->t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
        qt->t.v1.dataType = QNN_DATATYPE_UINT_32;
        qt->t.v1.quantizeParams.encodingDefinition  = QNN_DEFINITION_UNDEFINED;
        qt->t.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
        qt->t.v1.rank = (uint32_t)shape.size(); qt->t.v1.dimensions = qt->dims.data();
        qt->t.v1.memType = QNN_TENSORMEMTYPE_RAW;
        qt->t.v1.clientBuf.data = qt->static_u32.data();
        qt->t.v1.clientBuf.dataSize = (uint32_t)(qt->static_u32.size() * 4);
        QTensor* raw = qt.get(); tensors[name] = raw;
        tensor_store.push_back(std::move(qt)); return raw;
    }

    QTensor* make_native(const std::string& name, const std::vector<uint32_t>& shape) {
        return make_tensor(name, QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, shape);
    }
    QTensor* make_output(const std::string& name, const std::vector<uint32_t>& shape) {
        return make_tensor(name, QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_16, shape);
    }

    QTensor* load_weight(const std::string& name, const std::string& path) {
        auto cf = std::make_unique<CactFile>();
        if (!open_cact(path, *cf)) {
            fprintf(stderr, "[QNN Enc] failed to load: %s\n", path.c_str());
            return make_static_fp16(name, {1}, {0});
        }
        std::vector<uint32_t> shape(cf->shape.begin(), cf->shape.end());
        std::vector<uint16_t> fp16 = cact_to_fp16(*cf);
        weight_files.push_back(std::move(cf));
        return make_static_fp16(name, shape, std::move(fp16));
    }


    // Concat two tensors along an axis
    QTensor* op_concat(QTensor* a, QTensor* b, int axis, const std::string& n) {
        std::vector<uint32_t> out_shape = a->dims;
        out_shape[axis] += b->dims[axis];
        auto* out = make_native(n, out_shape);
        add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CONCAT, op_name("concat"),
               {a, b}, {out}, {scalar_i32_param(QNN_OP_CONCAT_PARAM_AXIS, axis)});
        return out;
    }

    // StridedSlice on a 2D tensor [R, C]: extract [r0_begin:r0_end:r0_step, :]
    QTensor* op_slice_2d(QTensor* x, int r0_begin, int r0_end, int r0_step, const std::string& n) {
        uint32_t C = x->dims[1];
        uint32_t T_out = (uint32_t)((r0_end - r0_begin + r0_step - 1) / r0_step);
        auto* out = make_native(n, {T_out, C});
        // ranges: [[r0_begin, r0_end, r0_step], [0, C, 1]]
        auto* rt = make_static_i32("ranges_" + n, {2, 3},
                                   {r0_begin, r0_end, r0_step, 0, (int)C, 1});
        if (rt->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &rt->t);
        Qnn_Param_t rp = tensor_param(QNN_OP_STRIDED_SLICE_PARAM_RANGES, rt);
        rp.tensorParam = rt->t;
        add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_STRIDED_SLICE, op_name("slice"),
               {x}, {out}, {rp});
        return out;
    }

    // Zero-pad a 2D tensor [T, C] along axis 0 only: → [T + pre + post, C]
    QTensor* op_pad_rows(QTensor* x, int pre, int post, const std::string& n) {
        uint32_t T_out = x->dims[0] + (uint32_t)(pre + post);
        uint32_t C     = x->dims[1];
        auto* out = make_native(n, {T_out, C});
        auto* amt = make_static_i32("padamt_" + n, {2, 2}, {pre, post, 0, 0});
        if (amt->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &amt->t);
        Qnn_Param_t scheme_p = scalar_i32_param(QNN_OP_PAD_PARAM_SCHEME, QNN_OP_PAD_SCHEME_CONSTANT);
        Qnn_Param_t amt_p    = tensor_param(QNN_OP_PAD_PARAM_PAD_AMOUNT, amt);
        amt_p.tensorParam    = amt->t;
        add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_PAD, op_name("pad"),
               {x}, {out}, {scheme_p, amt_p});
        return out;
    }

    // Contiguous row slice of a 2D tensor [R, C]: rows [begin, end) → [end-begin, C]
    QTensor* op_slice_rows(QTensor* x, int begin, int end, const std::string& n) {
        uint32_t C    = x->dims[1];
        uint32_t rows = (uint32_t)(end - begin);
        auto* out = make_native(n, {rows, C});
        auto* rt  = make_static_i32("ranges_" + n, {2, 3},
                                    {begin, end, 1, 0, (int)C, 1});
        if (rt->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &rt->t);
        Qnn_Param_t rp = tensor_param(QNN_OP_STRIDED_SLICE_PARAM_RANGES, rt);
        rp.tensorParam = rt->t;
        add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_STRIDED_SLICE, op_name("slice"),
               {x}, {out}, {rp});
        return out;
    }

    // Conv1D via im2col + single MatMul. Uses QNN_OP_PAD and step=1 StridedSlice only.
    // For stride=2: computes stride=1 then downsamples by reshape+slice trick.
    // Input NCT [1, C_in, T_in]. Weight file: [C_out, C_in, k]. Bias file: [C_out].
    QTensor* op_conv1d(QTensor* input, const std::string& weight_path,
                       const std::string& bias_path, int stride, int pad, const std::string& n) {
        auto cf = std::make_unique<CactFile>();
        if (!open_cact(weight_path, *cf)) {
            fprintf(stderr, "[QNN Enc] failed to load conv weight: %s\n", weight_path.c_str());
            return input;
        }
        std::vector<uint16_t> raw = cact_to_fp16(*cf);
        uint32_t C_out = (uint32_t)cf->shape[0];
        uint32_t C_in  = (uint32_t)cf->shape[1];
        uint32_t k     = (uint32_t)cf->shape[2];
        weight_files.push_back(std::move(cf));

        uint32_t T_in    = input->dims[2];
        uint32_t T_out   = (T_in + 2*(uint32_t)pad - k) / (uint32_t)stride + 1;
        uint32_t T_out_s1 = T_in + 2*(uint32_t)pad - k + 1;  // stride-1 output length

        // W_flat [C_out, k*C_in]: W_flat[o, ki*C_in+i] = W[o,i,ki]
        std::vector<uint16_t> wflat(C_out * k * C_in);
        for (uint32_t o = 0; o < C_out; o++)
            for (uint32_t ki = 0; ki < k; ki++)
                for (uint32_t i = 0; i < C_in; i++)
                    wflat[o * k * C_in + ki * C_in + i] = raw[o * C_in * k + i * k + ki];
        auto* W = make_static_fp16(n + "_W", {C_out, k * C_in}, std::move(wflat));

        // Reshape [1, C_in, T_in] → [C_in, T_in] → transpose → [T_in, C_in]
        auto* x_ct = op_reshape(input, {C_in, T_in}, n + "_xct");
        auto* x_tc = op_transpose(x_ct, {1, 0}, n + "_xtc");

        // Zero-pad via QNN_OP_PAD: [T_in, C_in] → [T_in+2*pad, C_in]
        auto* padded = op_pad_rows(x_tc, pad, pad, n + "_padded");

        // im2col: k contiguous row slices concatenated along axis 1
        // col_ki = padded[ki : ki+T_out_s1, :] → [T_out_s1, C_in]
        std::vector<QTensor*> cols(k);
        for (uint32_t ki = 0; ki < k; ki++)
            cols[ki] = op_slice_rows(padded, (int)ki, (int)(ki + T_out_s1),
                                     n + "_col" + std::to_string(ki));
        // Sequential concat: col0, col1, ..., col_{k-1} along axis 1
        QTensor* im2col = cols[0];
        for (uint32_t ki = 1; ki < k; ki++)
            im2col = op_concat(im2col, cols[ki], 1, n + "_im2col" + std::to_string(ki));
        // im2col: [T_out_s1, k*C_in]

        // out_s1 [T_out_s1, C_out] = im2col @ W.T + bias
        auto* out_s1 = op_matmul_T(im2col, W, n + "_mm");
        if (!bias_path.empty() && file_exists(bias_path)) {
            auto* b = load_weight(n + "_b", bias_path);
            if (b->dims.size() == 1) b = op_reshape(b, {1, C_out}, n + "_b2d");
            out_s1 = op_add(out_s1, b, n + "_bias");
        }

        QTensor* out;
        if ((uint32_t)stride == 1) {
            out = out_s1;
        } else {
            // Downsample stride=2: reshape [T_out_s1, C_out] → [T_out, 2, C_out]
            // then take axis-1 index 0 → [T_out, 1, C_out] → [T_out, C_out]
            auto* r   = op_reshape(out_s1, {T_out, 2, C_out}, n + "_ds_r");
            // Slice axis 1 from 0 to 1 (step=1) → [T_out, 1, C_out]
            auto* rt  = make_static_i32("ranges_" + n + "_ds", {3, 3},
                                        {0, (int)T_out, 1,  0, 1, 1,  0, (int)C_out, 1});
            if (rt->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &rt->t);
            Qnn_Param_t rp = tensor_param(QNN_OP_STRIDED_SLICE_PARAM_RANGES, rt);
            rp.tensorParam = rt->t;
            auto* sliced = make_native(n + "_ds_sl", {T_out, 1, C_out});
            add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_STRIDED_SLICE, op_name("slice"),
                   {r}, {sliced}, {rp});
            out = op_reshape(sliced, {T_out, C_out}, n + "_ds_flat");
        }

        // Transpose [T_out, C_out] → [C_out, T_out] → reshape to NCT [1, C_out, T_out]
        auto* out_ct = op_transpose(out, {1, 0}, n + "_tct");
        return op_reshape(out_ct, {1, C_out, T_out}, n);
    }

    Qnn_ErrorHandle_t add_op(const char* pkg, const char* type_name, const std::string& name,
                              const std::vector<QTensor*>& ins, const std::vector<QTensor*>& outs,
                              const std::vector<Qnn_Param_t>& params) {
        auto reg = [&](QTensor* qt) {
            if (qt->t.v1.id == 0 && qt->t.v1.type != QNN_TENSOR_TYPE_UNDEFINED)
                qnn.tensorCreateGraphTensor(graph_handle, &qt->t);
        };
        for (auto* qt : ins)  reg(qt);
        for (auto* qt : outs) reg(qt);
        std::vector<Qnn_Tensor_t> in_t, out_t;
        for (auto* qt : ins)  in_t.push_back(qt->t);
        for (auto* qt : outs) out_t.push_back(qt->t);
        Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
        op.version = QNN_OPCONFIG_VERSION_1;
        op.v1.name = name.c_str(); op.v1.packageName = pkg; op.v1.typeName = type_name;
        op.v1.numOfParams  = (uint32_t)params.size();
        op.v1.params       = params.empty() ? nullptr : const_cast<Qnn_Param_t*>(params.data());
        op.v1.numOfInputs  = (uint32_t)in_t.size(); op.v1.inputTensors  = in_t.data();
        op.v1.numOfOutputs = (uint32_t)out_t.size(); op.v1.outputTensors = out_t.data();
        Qnn_ErrorHandle_t err = qnn.graphAddNode(graph_handle, op);
        if (err != QNN_SUCCESS)
            fprintf(stderr, "[QNN Enc] addNode %s failed: %lld\n", name.c_str(), (long long)err);
        return err;
    }

    QTensor* op_add(QTensor* a, QTensor* b, const std::string& n) {
        auto* out = make_native(n, a->dims);
        add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_ELEMENT_WISE_ADD, op_name("add"), {a,b}, {out}, {});
        return out;
    }
    QTensor* op_mul(QTensor* a, QTensor* b, const std::string& n) {
        auto* out = make_native(n, a->dims);
        add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_ELEMENT_WISE_MULTIPLY, op_name("mul"), {a,b}, {out}, {});
        return out;
    }
    QTensor* op_reshape(QTensor* x, const std::vector<uint32_t>& shape, const std::string& n) {
        auto* out = make_native(n, shape);
        add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_RESHAPE, op_name("reshape"), {x}, {out}, {});
        return out;
    }
    QTensor* op_transpose(QTensor* x, const std::vector<uint32_t>& perm, const std::string& n) {
        std::vector<uint32_t> out_shape(perm.size());
        for (size_t i = 0; i < perm.size(); i++) out_shape[i] = x->dims[perm[i]];
        auto* out = make_native(n, out_shape);
        auto* pt = make_static_u32("perm_" + n, {(uint32_t)perm.size()},
                                   std::vector<uint32_t>(perm.begin(), perm.end()));
        if (pt->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &pt->t);
        Qnn_Param_t pp = tensor_param(QNN_OP_TRANSPOSE_PARAM_PERM, pt);
        pp.tensorParam = pt->t;
        add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_TRANSPOSE, op_name("transpose"), {x}, {out}, {pp});
        return out;
    }
    QTensor* op_matmul_T(QTensor* x, QTensor* w, const std::string& n) {
        std::vector<uint32_t> out_shape;
        for (size_t i = 0; i + 1 < x->dims.size(); i++) out_shape.push_back(x->dims[i]);
        out_shape.push_back(w->dims[0]);
        auto* out = make_native(n, out_shape);
        add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_MAT_MUL, op_name("mm"),
               {x, w}, {out},
               {scalar_bool_param(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, false),
                scalar_bool_param(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, true)});
        return out;
    }
    QTensor* op_softmax(QTensor* x, int axis, const std::string& n) {
        auto* out = make_native(n, x->dims);
        add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_SOFTMAX, op_name("softmax"),
               {x}, {out}, {scalar_i32_param(QNN_OP_SOFTMAX_PARAM_AXIS, axis)});
        return out;
    }
    QTensor* op_gelu(QTensor* x, const std::string& n) {
        auto* out = make_native(n, x->dims);
        add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_GELU, op_name("gelu"), {x}, {out}, {});
        return out;
    }

    // LayerNorm: inputs [x, gamma, beta], param epsilon (float32 scalar) and axes tensor
    QTensor* op_layer_norm(QTensor* x, QTensor* gamma, QTensor* beta, const std::string& n) {
        auto* out = make_native(n, x->dims);
        // axes = {last_dim_index}
        int last_axis = (int)x->dims.size() - 1;
        auto* axes_t = make_static_i32("axes_ln_" + n, {1}, {last_axis});
        if (axes_t->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &axes_t->t);
        Qnn_Param_t axes_p = tensor_param(QNN_OP_LAYER_NORM_PARAM_AXES, axes_t);
        axes_p.tensorParam = axes_t->t;
        // epsilon as float32 scalar
        Qnn_Param_t eps_p = QNN_PARAM_INIT;
        eps_p.paramType = QNN_PARAMTYPE_SCALAR;
        eps_p.name = QNN_OP_LAYER_NORM_PARAM_EPSILON;
        eps_p.scalarParam.dataType  = QNN_DATATYPE_FLOAT_32;
        eps_p.scalarParam.floatValue = ln_eps;
        add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_LAYER_NORM, op_name("ln"),
               {x, gamma, beta}, {out}, {eps_p, axes_p});
        return out;
    }

    // Conv1D implemented via Conv2D (H=T layout: put temporal dim in H for better HTP tiling).
    // Input NCT [1, in_ch, T_in] → reshape to NHWC [1, T_in, 1, in_ch]
    // Filter [out_ch, k, 1, in_ch] (already in QNN Conv2D OHWI format)
    // Output NHWC [1, T_out, 1, out_ch] → reshape to NCT [1, out_ch, T_out]
    QTensor* op_conv1d(QTensor* input, QTensor* filter, QTensor* bias,
                       int stride, int pad, const std::string& n) {
        uint32_t in_ch  = input->dims[1];  // NCT
        uint32_t T_in   = input->dims[2];
        uint32_t out_ch = filter->dims[0]; // [out_ch, k, 1, in_ch]
        uint32_t k      = filter->dims[1];
        uint32_t T_out  = (T_in + 2*(uint32_t)pad - k) / (uint32_t)stride + 1;
        (void)in_ch;

        // Reshape input NCT [1, in_ch, T_in] → NHWC [1, T_in, 1, in_ch]
        auto* in_nhwc = op_reshape(input, {1, T_in, 1, input->dims[1]}, n + "_in_nhwc");

        // stride [H=stride, W=1]
        auto* stride_t = make_static_i32("stride_" + n, {2}, {stride, 1});
        if (stride_t->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &stride_t->t);
        Qnn_Param_t sp = tensor_param(QNN_OP_CONV_2D_PARAM_STRIDE, stride_t);
        sp.tensorParam = stride_t->t;

        // pad_amount [2,2] = {{top=pad,bot=pad},{left=0,right=0}}
        auto* pad_t = make_static_i32("pad_" + n, {2, 2}, {pad, pad, 0, 0});
        if (pad_t->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &pad_t->t);
        Qnn_Param_t pp = tensor_param(QNN_OP_CONV_2D_PARAM_PAD_AMOUNT, pad_t);
        pp.tensorParam = pad_t->t;

        // Bias: 1D [out_ch]; create zero bias if not provided
        QTensor* b = bias;
        if (!b) {
            b = make_static_fp16("zbias_" + n, {out_ch}, std::vector<uint16_t>(out_ch, 0));
        } else if (b->dims.size() != 1) {
            b = op_reshape(b, {out_ch}, "bias1d_" + n);
        }

        // Conv2D output: NHWC [1, T_out, 1, out_ch]
        auto* out_nhwc = make_native(n + "_nhwc", {1, T_out, 1, out_ch});
        add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CONV_2D, op_name("conv2d"),
               {in_nhwc, filter, b}, {out_nhwc}, {sp, pp});

        // Reshape NHWC [1, T_out, 1, out_ch] → NCT [1, out_ch, T_out]
        return op_reshape(out_nhwc, {1, out_ch, T_out}, n);
    }

    // Full (non-causal) attention: q/k/v are [T, heads, head_dim]
    QTensor* op_full_attention(QTensor* q_3d, QTensor* k_3d, QTensor* v_3d,
                               float scale, const std::string& prefix) {
        uint32_t T  = q_3d->dims[0];
        uint32_t nh = q_3d->dims[1];
        uint32_t hd = q_3d->dims[2];
        // Transpose to [heads, T, head_dim]
        auto* q_T = op_transpose(q_3d, {1, 0, 2}, prefix + "_qT");
        auto* k_T = op_transpose(k_3d, {1, 0, 2}, prefix + "_kT");
        auto* v_T = op_transpose(v_3d, {1, 0, 2}, prefix + "_vT");
        // Scale Q — full {nh, T, hd} tensor (QNN HTP requires exact shape for elementwise)
        uint16_t s16 = float_to_fp16(scale);
        auto* scale_t = make_static_fp16(prefix + "_qsc", {nh, T, hd},
                                         std::vector<uint16_t>((size_t)nh*T*hd, s16));
        auto* q_sc = op_mul(q_T, scale_t, prefix + "_qsc_out");
        // scores = Q_scaled @ K^T → [nh, T, T]
        auto* scores = [&]() -> QTensor* {
            auto* out = make_native(prefix + "_scores", {nh, T, T});
            add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_MAT_MUL, op_name("attn_qk"),
                   {q_sc, k_T}, {out},
                   {scalar_bool_param(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, false),
                    scalar_bool_param(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, true)});
            return out;
        }();
        auto* w = op_softmax(scores, 2, prefix + "_attn_w");
        // out = w @ V → [nh, T, hd]
        auto* out_T = [&]() -> QTensor* {
            auto* out = make_native(prefix + "_attn_out_T", {nh, T, hd});
            add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_MAT_MUL, op_name("attn_v"),
                   {w, v_T}, {out},
                   {scalar_bool_param(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, false),
                    scalar_bool_param(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, false)});
            return out;
        }();
        auto* out_3d = op_transpose(out_T, {1, 0, 2}, prefix + "_attn_3d");
        return op_reshape(out_3d, {T, nh * hd}, prefix + "_attn_flat");
    }

    // ================================================================
    // Gemma4-Audio QNN graph helpers
    // ================================================================

    QTensor* g4a_op_unary(QTensor* x, uint32_t operation, const std::string& n) {
        auto* out = make_native(n, x->dims);
        add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_ELEMENT_WISE_UNARY, op_name("unary"),
               {x}, {out}, {scalar_u32_param(QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION, operation)});
        return out;
    }

    QTensor* g4a_op_relu(QTensor* x, const std::string& n) {
        auto* out = make_native(n, x->dims);
        add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_RELU, op_name("relu"), {x}, {out}, {});
        return out;
    }

    QTensor* g4a_op_sigmoid(QTensor* x, const std::string& n) {
        auto* out = make_native(n, x->dims);
        add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_ELEMENT_WISE_NEURON, op_name("sigmoid"),
               {x}, {out},
               {scalar_u32_param(QNN_OP_ELEMENT_WISE_NEURON_PARAM_OPERATION,
                                 QNN_OP_ELEMENT_WISE_NEURON_OPERATION_SIGMOID)});
        return out;
    }

    QTensor* g4a_op_silu(QTensor* x, const std::string& n) {
        return op_mul(x, g4a_op_sigmoid(x, n + "_sig"), n);
    }

    // reduce mean over last axis, keep_dims=true
    QTensor* g4a_op_reduce_mean_last(QTensor* x, const std::string& n) {
        int nd = (int)x->dims.size();
        std::vector<uint32_t> out_shape = x->dims; out_shape.back() = 1;
        auto* out   = make_native(n, out_shape);
        auto* axes  = make_static_i32("axes_" + n, {1}, {nd - 1});
        if (axes->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &axes->t);
        Qnn_Param_t ap = tensor_param(QNN_OP_REDUCE_MEAN_PARAM_AXES, axes);
        ap.tensorParam = axes->t;
        add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_REDUCE_MEAN, op_name("rmean"),
               {x}, {out}, {ap, scalar_bool_param(QNN_OP_REDUCE_MEAN_PARAM_KEEP_DIMS, true)});
        return out;
    }

    // tile on last axis only: {1, ..., 1, mult}
    QTensor* g4a_op_tile_last(QTensor* x, uint32_t mult, const std::string& n) {
        std::vector<uint32_t> out_shape = x->dims; out_shape.back() *= mult;
        auto* out = make_native(n, out_shape);
        std::vector<uint32_t> mults(x->dims.size(), 1u); mults.back() = mult;
        auto* mt = make_static_u32("mt_" + n, {(uint32_t)mults.size()}, mults);
        if (mt->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &mt->t);
        Qnn_Param_t mp = tensor_param(QNN_OP_TILE_PARAM_MULTIPLES, mt);
        mp.tensorParam = mt->t;
        add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_TILE, op_name("tile"), {x}, {out}, {mp});
        return out;
    }

    // RMSNorm: w is [hidden] or broadcastable
    QTensor* g4a_op_rms_norm(QTensor* x, QTensor* w, float eps, const std::string& n) {
        uint32_t hd = x->dims.back();
        auto* x2      = op_mul(x, x, n + "_sq");
        auto* mean    = g4a_op_reduce_mean_last(x2, n + "_mean");
        auto* eps_t   = make_static_fp16(n + "_eps", mean->dims,
                          std::vector<uint16_t>(mean->dims[0] == 1 ? 1 : mean->dims[0],
                                                float_to_fp16(eps)));
        if (eps_t->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &eps_t->t);
        auto* me      = op_add(mean, eps_t, n + "_me");
        auto* rsqrt   = g4a_op_unary(me, QNN_OP_ELEMENT_WISE_UNARY_OPERATION_RSQRT, n + "_rsqrt");
        auto* rsqrt_t = g4a_op_tile_last(rsqrt, hd, n + "_rt");
        auto* normed  = op_mul(x, rsqrt_t, n + "_normed");
        if (w->dims.size() < normed->dims.size()) {
            std::vector<uint32_t> exp_shape(normed->dims.size() - w->dims.size(), 1u);
            exp_shape.insert(exp_shape.end(), w->dims.begin(), w->dims.end());
            w = op_reshape(w, exp_shape, n + "_wexp");
        }
        return op_mul(normed, w, n);
    }

    // LayerNorm with scale-only (no bias): applied along last axis
    QTensor* g4a_op_layer_norm(QTensor* x, QTensor* gamma, float eps, const std::string& n) {
        uint32_t hd = x->dims.back();
        auto* out = make_native(n, x->dims);
        int last_axis = (int)x->dims.size() - 1;
        auto* axes_t = make_static_i32("axes_ln_" + n, {1}, {last_axis});
        if (axes_t->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &axes_t->t);
        Qnn_Param_t axes_p = tensor_param(QNN_OP_LAYER_NORM_PARAM_AXES, axes_t);
        axes_p.tensorParam = axes_t->t;
        Qnn_Param_t eps_p = QNN_PARAM_INIT;
        eps_p.paramType = QNN_PARAMTYPE_SCALAR;
        eps_p.name = QNN_OP_LAYER_NORM_PARAM_EPSILON;
        eps_p.scalarParam.dataType = QNN_DATATYPE_FLOAT_32;
        eps_p.scalarParam.floatValue = eps;
        auto* beta_t = make_static_fp16(n + "_beta", {hd}, std::vector<uint16_t>(hd, 0));
        if (gamma->dims.size() != 1 || gamma->dims[0] != hd)
            gamma = op_reshape(gamma, {hd}, n + "_gamma");
        if (beta_t->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &beta_t->t);
        add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_LAYER_NORM, op_name("ln"),
               {x, gamma, beta_t}, {out}, {eps_p, axes_p});
        return out;
    }

    // GLU: split x [T, 2*H] → {a[T,H], b[T,H]}, return a * silu(b)
    QTensor* g4a_op_glu(QTensor* x, const std::string& n) {
        uint32_t T = x->dims[0], D2 = x->dims[1], H = D2 / 2;
        auto* rt = make_static_i32("r_" + n, {2, 3}, {0, (int)T, 1, 0, (int)H, 1});
        if (rt->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &rt->t);
        auto* rt2 = make_static_i32("r2_" + n, {2, 3}, {0, (int)T, 1, (int)H, (int)D2, 1});
        if (rt2->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &rt2->t);
        Qnn_Param_t rp = tensor_param(QNN_OP_STRIDED_SLICE_PARAM_RANGES, rt);
        rp.tensorParam = rt->t;
        auto* a = make_native(n + "_a", {T, H});
        add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_STRIDED_SLICE, op_name("slice"),
               {x}, {a}, {rp});
        Qnn_Param_t rp2 = tensor_param(QNN_OP_STRIDED_SLICE_PARAM_RANGES, rt2);
        rp2.tensorParam = rt2->t;
        auto* b = make_native(n + "_b", {T, H});
        add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_STRIDED_SLICE, op_name("slice"),
               {x}, {b}, {rp2});
        auto* sig_b = make_native(n + "_sg", b->dims);
        add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_ELEMENT_WISE_NEURON, op_name("sigmoid"),
               {b}, {sig_b},
               {scalar_u32_param(QNN_OP_ELEMENT_WISE_NEURON_PARAM_OPERATION,
                                 QNN_OP_ELEMENT_WISE_NEURON_OPERATION_SIGMOID)});
        return op_mul(a, sig_b, n);
    }

    // Pad a 2D tensor [H, W] symmetrically on all sides → [H+2*pad, W+2*pad]
    QTensor* g4a_op_pad2d(QTensor* x, int pad, const std::string& n) {
        uint32_t H = x->dims[0], W = x->dims[1];
        uint32_t Hp = H + 2*(uint32_t)pad, Wp = W + 2*(uint32_t)pad;
        auto* out = make_native(n, {Hp, Wp});
        auto* amt = make_static_i32("amt_" + n, {2, 2}, {pad, pad, pad, pad});
        if (amt->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &amt->t);
        Qnn_Param_t sc = scalar_i32_param(QNN_OP_PAD_PARAM_SCHEME, QNN_OP_PAD_SCHEME_CONSTANT);
        Qnn_Param_t ap = tensor_param(QNN_OP_PAD_PARAM_PAD_AMOUNT, amt); ap.tensorParam = amt->t;
        add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_PAD, op_name("pad"), {x}, {out}, {sc, ap});
        return out;
    }

    // Strided slice [H_in, W_in] with stride=s in both dims, starting at (rh, rw)
    // → [H_out, W_out] where H_out = ceil((H_in - rh) / s), W_out = ceil((W_in - rw) / s)
    QTensor* g4a_op_stride2d(QTensor* x, int rh, int rw, int s, uint32_t H_out, uint32_t W_out, const std::string& n) {
        auto* out = make_native(n, {H_out, W_out});
        auto* rt = make_static_i32("rng_" + n, {2, 3},
                                   {rh, rh + (int)(H_out * s), s,
                                    rw, rw + (int)(W_out * s), s});
        if (rt->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &rt->t);
        Qnn_Param_t rp = tensor_param(QNN_OP_STRIDED_SLICE_PARAM_RANGES, rt); rp.tensorParam = rt->t;
        add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_STRIDED_SLICE, op_name("ss2d"), {x}, {out}, {rp});
        return out;
    }

    // Conv2D implemented via im2col + MatMul (avoids QNN_OP_CONV_2D shape constraints).
    // Input: [H_in, W_in] (2D, single channel). Weight file: [C_out, C_in=1, kH, kW].
    // Returns: [H_out, W_out, C_out].
    QTensor* g4a_op_conv2d_im2col(const std::string& wpath, QTensor* x,
                                   int stride, int pad, const std::string& n) {
        auto cf = std::make_unique<CactFile>();
        if (!open_cact(wpath, *cf)) {
            fprintf(stderr, "[G4A] conv2d weight missing: %s\n", wpath.c_str()); return x;
        }
        std::vector<uint16_t> raw = cact_to_fp16(*cf);
        uint32_t C_out = (uint32_t)cf->shape[0];
        uint32_t C_in  = (uint32_t)cf->shape[1];  // should be 1 for first conv
        uint32_t kH    = (uint32_t)cf->shape[2];
        uint32_t kW    = (uint32_t)cf->shape[3];
        weight_files.push_back(std::move(cf));

        uint32_t H_in = x->dims[0], W_in = x->dims[1];
        uint32_t H_out = (H_in + 2*(uint32_t)pad - kH) / (uint32_t)stride + 1;
        uint32_t W_out = (W_in + 2*(uint32_t)pad - kW) / (uint32_t)stride + 1;
        uint32_t patches = H_out * W_out;
        uint32_t kk = kH * kW * C_in;

        // W_flat [C_out, kk]: W[o, dh*kW*C_in + dw*C_in + ci] = raw[o, ci, dh, dw]
        std::vector<uint16_t> wflat(C_out * kk);
        for (uint32_t o = 0; o < C_out; o++)
            for (uint32_t ci = 0; ci < C_in; ci++)
                for (uint32_t dh = 0; dh < kH; dh++)
                    for (uint32_t dw = 0; dw < kW; dw++)
                        wflat[o * kk + (dh * kW + dw) * C_in + ci] =
                            raw[((o * C_in + ci) * kH + dh) * kW + dw];
        auto* W = make_static_fp16(n + "_W", {C_out, kk}, std::move(wflat));

        // For C_in > 1: x is [H_in, W_in, C_in] — but we handle the simple C_in=1 case here.
        // Pad to [H_in+2*pad, W_in+2*pad]
        auto* xp = g4a_op_pad2d(x, pad, n + "_pad");

        // Build im2col: for each (dh, dw) offset, extract strided [H_out, W_out] patch
        // and flatten to [patches]. Concat all kk=kH*kW patches along axis=1 → [patches, kk]
        std::vector<QTensor*> cols;
        cols.reserve(kk);
        for (uint32_t dh = 0; dh < kH; dh++) {
            for (uint32_t dw = 0; dw < kW; dw++) {
                auto tag = n + "_c" + std::to_string(dh) + "_" + std::to_string(dw);
                auto* patch = g4a_op_stride2d(xp, (int)dh, (int)dw, stride, H_out, W_out, tag);
                auto* flat  = op_reshape(patch, {patches, 1}, tag + "_f");
                cols.push_back(flat);
            }
        }
        // Concat [patches, 1] × kk → [patches, kk]
        QTensor* im2col = cols[0];
        for (uint32_t i = 1; i < (uint32_t)cols.size(); i++)
            im2col = op_concat(im2col, cols[i], 1, n + "_im" + std::to_string(i));

        // MatMul: [patches, kk] @ [C_out, kk]^T → [patches, C_out]
        auto* out_flat = op_matmul_T(im2col, W, n + "_mm");
        // Reshape to [H_out, W_out, C_out]
        return op_reshape(out_flat, {H_out, W_out, C_out}, n);
    }

    // Causal depthwise conv1d: x [1, T, C], weight file [C, 1, K]
    QTensor* g4a_op_dw_conv1d_causal(QTensor* x, const std::string& wpath, int K, const std::string& n) {
        auto cf = std::make_unique<CactFile>();
        if (!open_cact(wpath, *cf)) {
            fprintf(stderr, "[G4A] dw conv weight missing: %s\n", wpath.c_str()); return x;
        }
        uint32_t C = x->dims[2], T = x->dims[1];
        std::vector<uint16_t> raw = cact_to_fp16(*cf);
        // file: [C, 1, K] → QNN: [K, 1, 1, C] (HWIM)
        std::vector<uint16_t> hwim((uint32_t)K * C);
        for (uint32_t k = 0; k < (uint32_t)K; k++)
            for (uint32_t c = 0; c < C; c++)
                hwim[k * C + c] = raw[c * (uint32_t)K + k];
        auto* W_t = make_static_fp16(n + "_W", {(uint32_t)K, 1, 1, C}, std::move(hwim));
        auto* b_t = make_static_fp16(n + "_b", {C}, std::vector<uint16_t>(C, 0));
        weight_files.push_back(std::move(cf));
        // Causal pad: K-1 on left
        int left_pad = K - 1;
        uint32_t Tp = T + (uint32_t)left_pad;
        // Pad [1, T, 1, C] ← first reshape to NHWC for pad op
        auto* x_nhwc = op_reshape(x, {1, T, 1, C}, n + "_nhwc");
        auto* pt = make_static_i32(n + "_pad", {4, 2}, {0,0, left_pad,0, 0,0, 0,0});
        if (pt->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &pt->t);
        Qnn_Param_t scheme_p = scalar_i32_param(QNN_OP_PAD_PARAM_SCHEME, QNN_OP_PAD_SCHEME_CONSTANT);
        Qnn_Param_t amt_p    = tensor_param(QNN_OP_PAD_PARAM_PAD_AMOUNT, pt);
        amt_p.tensorParam    = pt->t;
        auto* xp = make_native(n + "_xp", {1, Tp, 1, C});
        add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_PAD, op_name("pad"),
               {x_nhwc}, {xp}, {scheme_p, amt_p});
        if (W_t->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &W_t->t);
        if (b_t->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &b_t->t);
        auto* st = make_static_i32(n + "_stride", {2}, {1, 1});
        auto* no_pad = make_static_i32(n + "_nopad", {2, 2}, {0, 0, 0, 0});
        if (st->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &st->t);
        if (no_pad->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &no_pad->t);
        Qnn_Param_t dsp = tensor_param(QNN_OP_DEPTH_WISE_CONV_2D_PARAM_STRIDE, st); dsp.tensorParam = st->t;
        Qnn_Param_t dpp = tensor_param(QNN_OP_DEPTH_WISE_CONV_2D_PARAM_PAD_AMOUNT, no_pad); dpp.tensorParam = no_pad->t;
        auto* out_nhwc = make_native(n + "_out", {1, T, 1, C});
        add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_DEPTH_WISE_CONV_2D, op_name("dwc"),
               {xp, W_t, b_t}, {out_nhwc}, {dsp, dpp});
        return op_reshape(out_nhwc, {1, T, C}, n);
    }

    // FFW sub-block: pre_norm → w1 → silu → w2 → post_norm → * residual_weight → + residual
    QTensor* g4a_build_ffw(QTensor* h, int li, bool is_end, const std::string& lp) {
        std::string tag = is_end ? "end" : "start";
        std::string wp  = model_folder + "/audio_conformer_" + std::to_string(li) + "_ffw_layer_" + tag + "_";
        auto* pre_w  = load_weight(lp+"pre",  wp + "pre_layer_norm.weights");
        auto* post_w = load_weight(lp+"post", wp + "post_layer_norm.weights");
        auto* w1     = load_weight(lp+"w1",   wp + "ffw_layer_1.weights");
        auto* w2     = load_weight(lp+"w2",   wp + "ffw_layer_2.weights");
        uint32_t hd  = (uint32_t)g4a_hidden;

        auto* x = g4a_op_rms_norm(h, pre_w, g4a_rms_eps, lp+"prn");
        x = op_matmul_T(x, w1, lp+"w1o");
        x = g4a_op_silu(x, lp+"silu");
        x = op_matmul_T(x, w2, lp+"w2o");
        x = g4a_op_rms_norm(x, post_w, g4a_rms_eps, lp+"pn");
        // scale by residual_weight
        auto* scale_t = make_static_fp16(lp+"rsc", {1, hd},
            std::vector<uint16_t>(hd, float_to_fp16(g4a_residual)));
        if (scale_t->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &scale_t->t);
        x = op_mul(x, scale_t, lp+"rscm");
        return op_add(h, x, lp+"res");
    }

    // Build static rel_key [num_pos, nh, hd] for relative position bias.
    // position_ids = [max_past, max_past-1, ..., 0] (13 entries for context_left=13)
    // rel_key[p] = relative_k_proj(sinusoidal(position_ids[p]))
    QTensor* g4a_build_rel_key_static(int li) {
        uint32_t nh     = (uint32_t)g4a_num_heads;
        uint32_t hd     = (uint32_t)g4a_head_dim;
        uint32_t hidden = (uint32_t)g4a_hidden;
        int max_past    = (g4a_context_left > 0) ? g4a_context_left - 1 : 0;
        uint32_t num_pos = (uint32_t)(max_past + g4a_context_right + 1);  // = 13

        uint32_t num_ts = hidden / 2;
        float log_ts_inc = std::log(1.0e4f) / std::max((int)num_ts - 1, 1);
        std::vector<float> timing(num_pos * hidden);
        for (uint32_t p = 0; p < num_pos; p++) {
            float pos = (float)max_past - (float)p;  // position_ids[p]
            for (uint32_t i = 0; i < num_ts; i++) {
                float sc = pos * std::expf(-(float)i * log_ts_inc);
                timing[p * hidden + i]          = std::sinf(sc);
                timing[p * hidden + num_ts + i] = std::cosf(sc);
            }
        }

        std::string wpath = model_folder + "/audio_conformer_" + std::to_string(li) +
                            "_attention_attn_relative_position_embedding_pos_proj.weights";
        std::string tname = "g4l" + std::to_string(li) + "_rk";
        CactFile cf;
        if (!open_cact(wpath, cf)) {
            fprintf(stderr, "[G4A] missing rel_pos_proj layer %d\n", li);
            return make_static_fp16(tname, {num_pos, nh, hd},
                                    std::vector<uint16_t>(num_pos * nh * hd, 0));
        }
        auto w_fp16 = cact_to_fp16(cf);

        // sin_emb[p, j] = timing[p] @ W^T → relative_k_proj output
        std::vector<float> sin_emb(num_pos * hidden, 0.f);
        for (uint32_t p = 0; p < num_pos; p++)
            for (uint32_t j = 0; j < hidden; j++) {
                float s = 0.f;
                for (uint32_t k = 0; k < hidden; k++)
                    s += timing[p * hidden + k] * fp16_to_fp32_val(w_fp16[j * hidden + k]);
                sin_emb[p * hidden + j] = s;
            }

        // Pack as [num_pos, nh, hd] (sin_emb reshaped per head)
        std::vector<uint16_t> rk_fp16(num_pos * nh * hd);
        for (uint32_t p = 0; p < num_pos; p++)
            for (uint32_t h = 0; h < nh; h++)
                for (uint32_t d = 0; d < hd; d++)
                    rk_fp16[(p * nh + h) * hd + d] =
                        float_to_fp16(sin_emb[p * hidden + h * hd + d]);
        return make_static_fp16(tname, {num_pos, nh, hd}, std::move(rk_fp16));
    }

    // Pad a [T, nh, hd] key/value tensor with max_past_horizon zeros on the left (axis 0).
    // Returns [Kc, nh, hd] where Kc = max_past_horizon + T = context_size.
    QTensor* g4a_pad_kv(QTensor* kv3, uint32_t max_past, const std::string& n) {
        uint32_t T = kv3->dims[0], nh = kv3->dims[1], hd = kv3->dims[2];
        uint32_t Kc = max_past + T;
        // Reshape to NHWC [1, T, 1, nh*hd] for PAD op, then back
        auto* kv_4d = op_reshape(kv3, {1, T, 1, nh * hd}, n + "_4d");
        auto* pt = make_static_i32(n + "_pa", {4, 2}, {0,0, (int)max_past,0, 0,0, 0,0});
        if (pt->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &pt->t);
        Qnn_Param_t sp = scalar_i32_param(QNN_OP_PAD_PARAM_SCHEME, QNN_OP_PAD_SCHEME_CONSTANT);
        Qnn_Param_t ap = tensor_param(QNN_OP_PAD_PARAM_PAD_AMOUNT, pt);
        ap.tensorParam = pt->t;
        auto* kv_pad = make_native(n + "_p4d", {1, Kc, 1, nh * hd});
        add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_PAD, op_name("kvpad"),
               {kv_4d}, {kv_pad}, {sp, ap});
        return op_reshape(kv_pad, {Kc, nh, hd}, n);
    }

    // Attention: pre_norm → Q/K/V → per-dim-scale → context-padded KV →
    //            matrix_ac + matrix_bd(rel_shift) → logit_cap → causal_mask → softmax → out_proj → post_norm → + residual
    // Keys/values are padded with max_past_horizon zero rows (= previous chunk context, all zeros for chunk 0).
    // Scores shape: [nh, T, Kc] where Kc = max_past_horizon + T = context_size.
    QTensor* g4a_build_attention(QTensor* h, int li, const std::string& lp) {
        std::string wp = model_folder + "/audio_conformer_" + std::to_string(li) + "_attention_";
        auto* pre_w   = load_weight(lp+"atpre",  wp + "pre_attn_norm.weights");
        auto* post_w  = load_weight(lp+"atpost", wp + "post_norm.weights");
        auto* wq = load_weight(lp+"wq", wp + "attn_q_proj.weights");
        auto* wk = load_weight(lp+"wk", wp + "attn_k_proj.weights");
        auto* wv = load_weight(lp+"wv", wp + "attn_v_proj.weights");
        auto* wo = load_weight(lp+"wo", wp + "post.weights");
        auto* pds_w = load_weight(lp+"pds_w", wp + "attn_per_dim_scale.weights");

        uint32_t T      = h->dims[0];
        uint32_t nh     = (uint32_t)g4a_num_heads;
        uint32_t hd     = (uint32_t)g4a_head_dim;
        uint32_t hidden = (uint32_t)g4a_hidden;
        int max_past    = (g4a_context_left > 0) ? g4a_context_left - 1 : 0;
        uint32_t Kc     = (uint32_t)max_past + T;  // context_size = 24

        auto* x = g4a_op_rms_norm(h, pre_w, g4a_rms_eps, lp+"prn");

        auto* q = op_matmul_T(x, wq, lp+"q");
        auto* k = op_matmul_T(x, wk, lp+"k");
        auto* v = op_matmul_T(x, wv, lp+"v");

        // Per-dim scale on Q: softplus(pds_w) * q_scale
        float q_scale = (1.0f / sqrtf((float)hd)) / std::log(2.0f);
        auto* pds_exp = g4a_op_unary(pds_w, QNN_OP_ELEMENT_WISE_UNARY_OPERATION_EXP, lp+"pds_e");
        auto* one_v   = make_static_fp16(lp+"one", {hd}, std::vector<uint16_t>(hd, float_to_fp16(1.0f)));
        if (one_v->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &one_v->t);
        auto* pds_1p  = op_add(pds_exp, one_v, lp+"pds_1p");
        auto* pds_log = g4a_op_unary(pds_1p, QNN_OP_ELEMENT_WISE_UNARY_OPERATION_LOG, lp+"pds_l");
        auto* qsc_v   = make_static_fp16(lp+"qsc", {hd}, std::vector<uint16_t>(hd, float_to_fp16(q_scale)));
        if (qsc_v->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &qsc_v->t);
        auto* pds_sc  = op_mul(pds_log, qsc_v, lp+"pds_sc");
        auto* pds_2d  = op_reshape(pds_sc, {1, hd}, lp+"pds_2d");
        auto* q_flat  = op_reshape(q, {T * nh, hd}, lp+"q_flat");
        auto* q_pds   = op_mul(q_flat, pds_2d, lp+"q_pds");
        q = op_reshape(q_pds, {T, hidden}, lp+"q_rs");

        // K scale
        auto* ksc_v = make_static_fp16(lp+"ksc", {1, hidden},
            std::vector<uint16_t>(hidden, float_to_fp16(g4a_k_scale)));
        if (ksc_v->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &ksc_v->t);
        k = op_mul(k, ksc_v, lp+"k_sc");

        // Reshape Q/K/V to [T, nh, hd]
        auto* q3 = op_reshape(q, {T, nh, hd}, lp+"q3");
        auto* k3 = op_reshape(k, {T, nh, hd}, lp+"k3");
        auto* v3 = op_reshape(v, {T, nh, hd}, lp+"v3");

        // Pad K/V with max_past zeros on left → [Kc, nh, hd]
        auto* k3p = g4a_pad_kv(k3, (uint32_t)max_past, lp+"kp");
        auto* v3p = g4a_pad_kv(v3, (uint32_t)max_past, lp+"vp");

        // Transpose Q → [nh, T, hd], K/V → [nh, Kc, hd]
        auto* q_T  = op_transpose(q3,  {1, 0, 2}, lp+"qT");
        auto* k_T  = op_transpose(k3p, {1, 0, 2}, lp+"kT");
        auto* v_T  = op_transpose(v3p, {1, 0, 2}, lp+"vT");

        // matrix_ac = Q @ K^T → [nh, T, Kc]
        auto* scores = make_native(lp+"sc", {nh, T, Kc});
        add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_MAT_MUL, op_name("qk"),
               {q_T, k_T}, {scores},
               {scalar_bool_param(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, false),
                scalar_bool_param(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, true)});

        // Relative position bias (matrix_bd after rel_shift):
        // rel_key [num_pos=13, nh, hd] static
        // raw_bd = Q [nh, T, hd] @ rel_key^T [nh, hd, 13] → [nh, T, 13]
        // After rel_shift: bd[h, i, j] = raw_bd[h, i, clamp(j-i, 0, 12)] → [nh, T, Kc]
        {
            uint32_t num_pos = (uint32_t)(max_past + g4a_context_right + 1);  // 13
            auto* rk_st  = g4a_build_rel_key_static(li);  // static [num_pos, nh, hd]
            auto* rk_nat = op_reshape(rk_st, {num_pos, nh, hd}, lp+"rk_nat");
            auto* rk_T   = op_transpose(rk_nat, {1, 2, 0}, lp+"rk_T");  // [nh, hd, num_pos]
            // raw_bd: [nh, T, num_pos]
            auto* raw_bd = make_native(lp+"rb_raw", {nh, T, num_pos});
            add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_MAT_MUL, op_name("rbmm"),
                   {q_T, rk_T}, {raw_bd},
                   {scalar_bool_param(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, false),
                    scalar_bool_param(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, false)});
            // Gather with idx[h, i, j] = clamp(j-i, 0, num_pos-1) → [nh, T, Kc]
            std::vector<int32_t> idx_data((size_t)nh * T * Kc);
            int np1 = (int)num_pos - 1;
            for (uint32_t hh = 0; hh < nh; hh++)
                for (uint32_t i = 0; i < T; i++)
                    for (uint32_t j = 0; j < Kc; j++) {
                        int k = std::max(0, std::min(np1, (int)j - (int)i));
                        idx_data[(hh * T + i) * Kc + j] = k;
                    }
            auto* idx_t  = make_static_i32(lp+"rb_idx", {nh, T, Kc}, std::move(idx_data));
            auto* rb_out = make_native(lp+"rb", {nh, T, Kc});
            add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_GATHER_ELEMENTS, op_name("rb_ge"),
                   {raw_bd, idx_t}, {rb_out},
                   {scalar_i32_param(QNN_OP_GATHER_ELEMENTS_PARAM_AXIS, 2)});
            scores = op_add(scores, rb_out, lp+"sc_rb");
        }

        // Logit cap: logit_cap * tanh(scores / logit_cap)
        float inv_cap = 1.0f / g4a_logit_cap;
        size_t n_sc = (size_t)nh * T * Kc;
        auto* inv_cap_t = make_static_fp16(lp+"icp", {nh, T, Kc},
                              std::vector<uint16_t>(n_sc, float_to_fp16(inv_cap)));
        auto* cap_t = make_static_fp16(lp+"cap", {nh, T, Kc},
                          std::vector<uint16_t>(n_sc, float_to_fp16(g4a_logit_cap)));
        if (inv_cap_t->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &inv_cap_t->t);
        if (cap_t->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &cap_t->t);
        auto* sc_s = op_mul(scores, inv_cap_t, lp+"sc_s");
        auto* sc_th = [&]() -> QTensor* {
            auto* o = make_native(lp+"sc_th", sc_s->dims);
            add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_ELEMENT_WISE_NEURON, op_name("tanh"),
                   {sc_s}, {o},
                   {scalar_u32_param(QNN_OP_ELEMENT_WISE_NEURON_PARAM_OPERATION,
                                     QNN_OP_ELEMENT_WISE_NEURON_OPERATION_TANH)});
            return o;
        }();
        auto* sc_cap = op_mul(sc_th, cap_t, lp+"sc_cap");

        // Causal mask [nh, T, Kc]: mask[h,i,j] = -inf if j > max_past + i
        std::vector<uint16_t> mask_data(n_sc, 0);
        const uint16_t neg_inf = 0xFC00;
        for (uint32_t hh = 0; hh < nh; hh++)
            for (uint32_t i = 0; i < T; i++)
                for (uint32_t j = 0; j < Kc; j++)
                    if (j > (uint32_t)max_past + i)
                        mask_data[(hh * T + i) * Kc + j] = neg_inf;
        auto* mask_t = make_static_fp16(lp+"mask", {nh, T, Kc}, std::move(mask_data));
        if (mask_t->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &mask_t->t);
        auto* sc_masked = op_add(sc_cap, mask_t, lp+"sc_m");

        auto* w = op_softmax(sc_masked, 2, lp+"attn_w");

        // out = w [nh, T, Kc] @ v_T [nh, Kc, hd] → [nh, T, hd]
        auto* out_T = make_native(lp+"out_T", {nh, T, hd});
        add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_MAT_MUL, op_name("av"),
               {w, v_T}, {out_T},
               {scalar_bool_param(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, false),
                scalar_bool_param(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, false)});
        auto* out_3d   = op_transpose(out_T, {1, 0, 2}, lp+"out3d");
        auto* out_flat = op_reshape(out_3d, {T, hidden}, lp+"out_flat");

        auto* proj      = op_matmul_T(out_flat, wo, lp+"proj");
        auto* proj_norm = g4a_op_rms_norm(proj, post_w, g4a_rms_eps, lp+"pn");
        return op_add(h, proj_norm, lp+"res");
    }

    // LConv block
    QTensor* g4a_build_lconv1d(QTensor* h, int li, const std::string& lp) {
        std::string wp = model_folder + "/audio_conformer_" + std::to_string(li) + "_lconv1d_";
        auto* pre_w   = load_weight(lp+"lcpre", wp + "pre_layer_norm.weights");
        auto* cnorm_w = load_weight(lp+"cnn",   wp + "conv_norm.weights");
        auto* wstart  = load_weight(lp+"ws",    wp + "linear_start.weights");
        auto* wend    = load_weight(lp+"we",    wp + "linear_end.weights");

        uint32_t T  = h->dims[0];
        uint32_t hd = (uint32_t)g4a_hidden;

        auto* x = g4a_op_rms_norm(h, pre_w, g4a_rms_eps, lp+"prn");
        x = op_matmul_T(x, wstart, lp+"ws_o");   // [T, 2*hidden]
        x = g4a_op_glu(x, lp+"glu");             // [T, hidden]
        // depthwise causal conv1d
        auto* x1d = op_reshape(x, {1, T, hd}, lp+"x1d");
        x1d = g4a_op_dw_conv1d_causal(x1d, wp + "depthwise_conv1d.weights",
                                       g4a_conf_K, lp + "dwc");
        x = op_reshape(x1d, {T, hd}, lp+"xrs");
        x = g4a_op_rms_norm(x, cnorm_w, g4a_rms_eps, lp+"cnorm");
        x = g4a_op_silu(x, lp+"silu");
        x = op_matmul_T(x, wend, lp+"we_o");
        return op_add(x, h, lp+"res");
    }

    // Full conformer block: ffw_start → attn → lconv → ffw_end → block_norm
    QTensor* g4a_build_conformer_block(QTensor* h, int li) {
        std::string lp = "g4l" + std::to_string(li) + "_";
        std::string bp = model_folder + "/audio_conformer_" + std::to_string(li) + "_";
        h = g4a_build_ffw(h, li, false, lp + "fs_");
        h = g4a_build_attention(h, li, lp + "at_");
        h = g4a_build_lconv1d(h, li, lp + "lc_");
        h = g4a_build_ffw(h, li, true, lp + "fe_");
        auto* bnorm_w = load_weight(lp+"bn", bp + "norm.weights");
        return g4a_op_rms_norm(h, bnorm_w, g4a_rms_eps, lp + "bn_out");
    }

    int g4a_conf_K = 5;

    bool g4a_build_graph() {
        QnnHtpGraph_CustomConfig_t htp_cfg = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
        htp_cfg.option    = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
        htp_cfg.precision = QNN_PRECISION_FLOAT16;
        QnnHtpGraph_CustomConfig_t htp_opt = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
        htp_opt.option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
        htp_opt.optimizationOption.type = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
        htp_opt.optimizationOption.floatValue = 3.0f;
        QnnGraph_Config_t gc0 = QNN_GRAPH_CONFIG_INIT; gc0.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM; gc0.customConfig = &htp_cfg;
        QnnGraph_Config_t gc1 = QNN_GRAPH_CONFIG_INIT; gc1.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM; gc1.customConfig = &htp_opt;
        const QnnGraph_Config_t* gcfgs[] = {&gc0, &gc1, nullptr};

        Qnn_ErrorHandle_t err = qnn.graphCreate(context_handle, "g4a_encoder", gcfgs, &graph_handle);
        if (err != QNN_SUCCESS) {
            fprintf(stderr, "[G4A] graphCreate failed: %lld\n", (long long)err); return false;
        }

        uint32_t T_mel = (uint32_t)g4a_chunk_mel;
        uint32_t mel   = (uint32_t)g4a_mel_bins;
        uint32_t T_out = (uint32_t)g4a_chunk_out;
        uint32_t c0    = (uint32_t)g4a_conv0_ch;
        (void)g4a_conv1_ch;
        std::string mf  = model_folder + "/audio_subsample_conv_projection_";

        // Input: [T_mel, mel_bins] FP16
        auto* mel_in = make_tensor("g4a_mel", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16,
                                   {T_mel, mel});
        g4a_exec_in_id = 0;

        // SSCP conv0: [T_mel, mel_bins] → [H1, W1, c0] via im2col+matmul
        auto* x0 = g4a_op_conv2d_im2col(mf + "conv_0_conv.weights", mel_in, 2, 1, "g4a_c0");
        uint32_t H1 = x0->dims[0], W1 = x0->dims[1];  // [H1, W1, c0]

        // Norm0: reshape to [H1*W1, c0], LayerNorm, relu
        auto* norm0_w = load_weight("g4a_n0", mf + "conv_0_norm.weights");
        auto* x0f = op_reshape(x0, {H1 * W1, c0}, "g4a_n0_flat");
        x0f = g4a_op_layer_norm(x0f, norm0_w, g4a_ln_eps, "g4a_n0_ln");
        x0f = g4a_op_relu(x0f, "g4a_n0_relu");
        auto* x0r = op_reshape(x0f, {H1, W1, c0}, "g4a_n0_rs");  // [H1, W1, c0]

        // For conv1, the input has C_in=c0. We need to handle multi-channel input.
        // Treat [H1, W1, c0] as [H1, W1*c0] then use a different conv formulation.
        // Instead: apply conv1 as op_conv1d_c0 treating c0 channels explicitly.
        // Conv1 weight: [c1=32, c0=128, 3, 3] → need im2col over [H1, W1, c0] with c_in=c0
        // Flatten spatial to [H1*W1, c0], then apply patch extraction per (dh,dw).
        QTensor* h = nullptr;
        {
            // Load conv1 weight [c1, c0, kH=3, kW=3]
            auto cf1 = std::make_unique<CactFile>();
            open_cact(mf + "conv_1_conv.weights", *cf1);
            std::vector<uint16_t> raw1 = cact_to_fp16(*cf1);
            uint32_t C1 = (uint32_t)cf1->shape[0]; // c1=32
            uint32_t C0 = (uint32_t)cf1->shape[1]; // c0=128
            uint32_t kH1 = (uint32_t)cf1->shape[2]; // 3
            uint32_t kW1 = (uint32_t)cf1->shape[3]; // 3
            weight_files.push_back(std::move(cf1));
            uint32_t H2 = (H1 + 2 - kH1) / 2 + 1;
            uint32_t W2 = (W1 + 2 - kW1) / 2 + 1;
            uint32_t kk1 = kH1 * kW1;  // 9, per channel
            // W_flat [C1, kk1*C0]: W[o, (dh*kW+dw)*C0 + ci] = raw[o, ci, dh, dw]
            std::vector<uint16_t> wf1(C1 * kk1 * C0);
            for (uint32_t o = 0; o < C1; o++)
                for (uint32_t ci = 0; ci < C0; ci++)
                    for (uint32_t dh = 0; dh < kH1; dh++)
                        for (uint32_t dw = 0; dw < kW1; dw++)
                            wf1[o * kk1 * C0 + (dh * kW1 + dw) * C0 + ci] =
                                raw1[((o * C0 + ci) * kH1 + dh) * kW1 + dw];
            auto* W1t = make_static_fp16("g4a_c1_W", {C1, kk1 * C0}, std::move(wf1));

            // Pad x0r [H1, W1, c0] → [H1+2, W1+2, c0]
            auto* x0p = [&]() -> QTensor* {
                auto* out = make_native("g4a_c1_xp", {H1+2, W1+2, C0});
                auto* amt = make_static_i32("g4a_c1_amt", {3, 2}, {1,1, 1,1, 0,0});
                if (amt->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &amt->t);
                Qnn_Param_t sc = scalar_i32_param(QNN_OP_PAD_PARAM_SCHEME, QNN_OP_PAD_SCHEME_CONSTANT);
                Qnn_Param_t ap = tensor_param(QNN_OP_PAD_PARAM_PAD_AMOUNT, amt); ap.tensorParam = amt->t;
                add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_PAD, op_name("pad"),
                       {x0r}, {out}, {sc, ap});
                return out;
            }();

            // For each (dh, dw), extract every-2nd-row/col → [H2, W2, C0], reshape to [H2*W2, C0]
            // Concat kk1=9 such patches along axis=1 → [H2*W2, 9*C0]
            std::vector<QTensor*> patches1;
            patches1.reserve(kk1);
            for (uint32_t dh = 0; dh < kH1; dh++) {
                for (uint32_t dw = 0; dw < kW1; dw++) {
                    auto tag = "g4a_c1p" + std::to_string(dh) + "_" + std::to_string(dw);
                    // Strided slice [H1+2, W1+2, C0] at offset (dh, dw), stride=2 in H and W
                    auto* rt = make_static_i32("rng_" + tag, {3, 3},
                                               {(int)dh, (int)(dh + H2*2), 2,
                                                (int)dw, (int)(dw + W2*2), 2,
                                                0, (int)C0, 1});
                    if (rt->t.v1.id == 0) qnn.tensorCreateGraphTensor(graph_handle, &rt->t);
                    Qnn_Param_t rp = tensor_param(QNN_OP_STRIDED_SLICE_PARAM_RANGES, rt);
                    rp.tensorParam = rt->t;
                    auto* patch = make_native(tag, {H2, W2, C0});
                    add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_STRIDED_SLICE, op_name("ss3d"),
                           {x0p}, {patch}, {rp});
                    auto* pf = op_reshape(patch, {H2 * W2, C0}, tag + "_f");
                    patches1.push_back(pf);
                }
            }
            QTensor* im2col1 = patches1[0];
            for (uint32_t i = 1; i < (uint32_t)patches1.size(); i++)
                im2col1 = op_concat(im2col1, patches1[i], 1, "g4a_im1_" + std::to_string(i));

            auto* out1 = op_matmul_T(im2col1, W1t, "g4a_c1_mm");  // [H2*W2, C1]
            uint32_t H2_ = H2, W2_ = W2;
            auto* norm1_w = load_weight("g4a_n1", mf + "conv_1_norm.weights");
            auto* x1f = g4a_op_layer_norm(out1, norm1_w, g4a_ln_eps, "g4a_n1_ln");
            x1f = g4a_op_relu(x1f, "g4a_n1_relu");
            // [H2*W2, c1] where H2=12, W2=32, c1=32 → flatten to [H2, W2*c1] = [12, 1024]
            auto* x1r = op_reshape(x1f, {H2_, W2_ * C1}, "g4a_n1_flat");
            auto* iproj_w = load_weight("g4a_ip", mf + "input_proj.weights");
            h = op_matmul_T(x1r, iproj_w, "g4a_proj");  // [T_out, hidden]
        }

        // 12 conformer blocks
        int max_layers = g4a_num_layers;
        if (const char* ml = getenv("CACTUS_ENC_MAX_LAYERS"))
            max_layers = std::min(max_layers, std::stoi(ml));
        fprintf(stderr, "[G4A] building %d conformer layers\n", max_layers);
        for (int li = 0; li < max_layers; li++)
            h = g4a_build_conformer_block(h, li);

        // Output projection: [T_out, hidden] → [T_out, out_dim]
        if (g4a_out_dim > 0 && g4a_out_dim != g4a_hidden) {
            auto* oproj_w = load_weight("g4a_op", model_folder + "/audio_output_proj.weights");
            auto* oproj_b = load_weight("g4a_ob", model_folder + "/audio_output_proj.bias");
            h = op_matmul_T(h, oproj_w, "g4a_oproj");
            if (oproj_b->dims.size() == 1)
                oproj_b = op_reshape(oproj_b, {1, (uint32_t)g4a_out_dim}, "g4a_ob_exp");
            h = op_add(h, oproj_b, "g4a_ob_add");
        }

        auto* g4a_out = make_output("g4a_out", {T_out, (uint32_t)(g4a_out_dim > 0 ? g4a_out_dim : g4a_hidden)});
        add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_RESHAPE, op_name("g4a_out_cp"),
               {h}, {g4a_out}, {});

        g4a_exec_in_id  = mel_in->t.v1.id;
        g4a_exec_out_id = g4a_out->t.v1.id;

        fprintf(stderr, "[G4A] finalizing graph (may take ~60s)...\n"); fflush(stderr);
        err = qnn.graphFinalize(graph_handle, nullptr, nullptr);
        if (err != QNN_SUCCESS) {
            fprintf(stderr, "[G4A] graphFinalize failed: %lld\n", (long long)err); return false;
        }
        fprintf(stderr, "[G4A] finalized\n");

        if (qnn.graphRetrieve) {
            Qnn_GraphHandle_t rg = nullptr;
            if (qnn.graphRetrieve(context_handle, "g4a_encoder", &rg) == QNN_SUCCESS && rg)
                graph_handle = rg;
        }
        weight_files.clear();

        // Save cache
        std::string cache_path = model_folder + "/qnn_g4a_T" + std::to_string(g4a_chunk_mel) +
                                 "_L" + std::to_string(max_layers) + ".bin";
        if (qnn.contextGetBinarySize && qnn.contextGetBinary) {
            Qnn_ContextBinarySize_t bin_sz = 0, written = 0;
            if (qnn.contextGetBinarySize(context_handle, &bin_sz) == QNN_SUCCESS && bin_sz > 0) {
                std::vector<char> bin((size_t)bin_sz);
                if (qnn.contextGetBinary(context_handle, bin.data(), bin_sz, &written) == QNN_SUCCESS && written > 0) {
                    std::ofstream cf(cache_path, std::ios::binary);
                    if (cf.is_open()) { cf.write(bin.data(), (size_t)written); }
                    std::ofstream sf(cache_path + ".ids", std::ios::binary);
                    uint32_t magic = 0x41344751u; // "G4QA"
                    sf.write(reinterpret_cast<char*>(&magic),            4);
                    sf.write(reinterpret_cast<char*>(&g4a_exec_in_id),  4);
                    sf.write(reinterpret_cast<char*>(&g4a_exec_out_id), 4);
                    fprintf(stderr, "[G4A] saved cache: %s\n", cache_path.c_str());
                }
            }
        }
        return true;
    }

    bool g4a_load_from_cache(const std::string& cache_path) {
        std::ifstream sf(cache_path + ".ids", std::ios::binary);
        if (!sf.is_open()) return false;
        uint32_t magic = 0;
        sf.read(reinterpret_cast<char*>(&magic), 4);
        if (magic != 0x41344751u) return false;
        sf.read(reinterpret_cast<char*>(&g4a_exec_in_id),  4);
        sf.read(reinterpret_cast<char*>(&g4a_exec_out_id), 4);

        std::ifstream cf(cache_path, std::ios::binary | std::ios::ate);
        if (!cf.is_open()) return false;
        size_t sz = (size_t)cf.tellg(); cf.seekg(0);
        std::vector<char> buf(sz); cf.read(buf.data(), sz); cf.close();

        Qnn_ErrorHandle_t err = qnn.contextCreateFromBinary(
            backend_handle, device_handle, nullptr,
            buf.data(), sz, &context_handle, nullptr);
        if (err != QNN_SUCCESS) {
            fprintf(stderr, "[G4A] contextCreateFromBinary failed: %lld\n", (long long)err);
            return false;
        }
        if (qnn.graphRetrieve(context_handle, "g4a_encoder", &graph_handle) != QNN_SUCCESS) {
            fprintf(stderr, "[G4A] graphRetrieve failed\n"); return false;
        }
        fprintf(stderr, "[G4A] loaded from cache: %s\n", cache_path.c_str());
        return true;
    }

    void g4a_setup_exec_tensors() {
        int out_dim = g4a_out_dim > 0 ? g4a_out_dim : g4a_hidden;
        g4a_input_buf.assign((size_t)g4a_chunk_mel * g4a_mel_bins, 0);
        g4a_output_buf.assign((size_t)g4a_chunk_out * out_dim, 0);

        auto make_qt = [&](uint32_t id, Qnn_TensorType_t type,
                            const std::vector<uint32_t>& shape) -> QTensor* {
            auto qt = std::make_unique<QTensor>();
            qt->name = "g4a_" + std::to_string(id); qt->dims = shape;
            qt->t = QNN_TENSOR_INIT; qt->t.version = QNN_TENSOR_VERSION_1;
            qt->t.v1.id = id; qt->t.v1.type = type;
            qt->t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
            qt->t.v1.dataType   = QNN_DATATYPE_FLOAT_16;
            qt->t.v1.quantizeParams = { QNN_DEFINITION_UNDEFINED,
                                        QNN_QUANTIZATION_ENCODING_UNDEFINED, {{0,0}} };
            qt->t.v1.rank = (uint32_t)shape.size(); qt->t.v1.dimensions = qt->dims.data();
            qt->t.v1.memType = QNN_TENSORMEMTYPE_RAW;
            qt->t.v1.clientBuf = {nullptr, 0};
            QTensor* p = qt.get(); tensor_store.push_back(std::move(qt)); return p;
        };

        // Look up existing tensor by id if already created
        auto get = [&](const std::string& nm, uint32_t id,
                        Qnn_TensorType_t type, const std::vector<uint32_t>& sh) -> QTensor* {
            if (tensors.count(nm)) return tensors.at(nm);
            return make_qt(id, type, sh);
        };

        g4a_exec_in_qt  = get("g4a_mel",  g4a_exec_in_id,  QNN_TENSOR_TYPE_APP_WRITE,
                               {(uint32_t)g4a_chunk_mel, (uint32_t)g4a_mel_bins});
        g4a_exec_out_qt = get("g4a_out", g4a_exec_out_id, QNN_TENSOR_TYPE_APP_READ,
                               {(uint32_t)g4a_chunk_out, (uint32_t)out_dim});

        if (g4a_exec_in_qt)  { g4a_exec_in_qt->t.v1.clientBuf.data  = g4a_input_buf.data();
                                g4a_exec_in_qt->t.v1.clientBuf.dataSize = (uint32_t)(g4a_input_buf.size() * 2); }
        if (g4a_exec_out_qt) { g4a_exec_out_qt->t.v1.clientBuf.data  = g4a_output_buf.data();
                                g4a_exec_out_qt->t.v1.clientBuf.dataSize = (uint32_t)(g4a_output_buf.size() * 2); }

        g4a_exec_inputs  = { g4a_exec_in_qt  ? g4a_exec_in_qt->t  : Qnn_Tensor_t{} };
        g4a_exec_outputs = { g4a_exec_out_qt ? g4a_exec_out_qt->t : Qnn_Tensor_t{} };
        fprintf(stderr, "[G4A] exec tensors ready: in_id=%u out_id=%u\n",
                g4a_exec_in_id, g4a_exec_out_id);
    }

    bool g4a_load() {
        // read audio config
        {
            std::ifstream f(model_folder + "/config.txt");
            if (!f.is_open()) { fprintf(stderr, "[G4A] cannot open config.txt\n"); return false; }
            std::string line;
            while (std::getline(f, line)) {
                auto eq = line.find('='); if (eq == std::string::npos) continue;
                std::string k = line.substr(0, eq), v = line.substr(eq + 1);
                if      (k == "audio_hidden_dim")          g4a_hidden     = std::stoi(v);
                else if (k == "audio_num_layers")          g4a_num_layers = std::stoi(v);
                else if (k == "audio_num_heads")           g4a_num_heads  = std::stoi(v);
                else if (k == "audio_head_dim")            g4a_head_dim   = std::stoi(v);
                else if (k == "audio_input_feat_size")     g4a_mel_bins   = std::stoi(v);
                else if (k == "audio_output_proj_dims")    g4a_out_dim    = std::stoi(v);
                else if (k == "audio_sscp_conv0_channels") g4a_conv0_ch   = std::stoi(v);
                else if (k == "audio_sscp_conv1_channels") g4a_conv1_ch   = std::stoi(v);
                else if (k == "audio_sscp_conv_eps")       g4a_ln_eps     = std::stof(v);
                else if (k == "audio_rms_norm_eps")        g4a_rms_eps    = std::stof(v);
                else if (k == "audio_logit_cap")           g4a_logit_cap  = std::stof(v);
                else if (k == "audio_residual_weight")     g4a_residual   = std::stof(v);
                else if (k == "audio_conf_conv_kernel_size") g4a_conf_K   = std::stoi(v);
                else if (k == "audio_context_left")          g4a_context_left  = std::stoi(v);
                else if (k == "audio_context_right")         g4a_context_right = std::stoi(v);
            }
            // chunk_mel: 4× chunk_out; chunk_out from config audio_chunk_size
            // re-derive chunk sizes from SSCP stride
            // 2 stride-2 convs: T_mel → T_mel/4 = chunk_out
            g4a_chunk_out = 12; // default; override from config if present
        }
        {
            std::ifstream f(model_folder + "/config.txt");
            std::string line;
            while (std::getline(f, line)) {
                auto eq = line.find('='); if (eq == std::string::npos) continue;
                std::string k = line.substr(0, eq), v = line.substr(eq + 1);
                if (k == "audio_chunk_size") g4a_chunk_out = std::stoi(v);
            }
        }
        g4a_chunk_mel = g4a_chunk_out * 4;
        g4a_k_scale = std::log(1.0f + std::expf(1.0f)) / std::log(2.0f);

        fprintf(stderr, "[G4A] config: hidden=%d layers=%d heads=%d hd=%d out=%d mel=%d chunk_mel=%d chunk_out=%d\n",
                g4a_hidden, g4a_num_layers, g4a_num_heads, g4a_head_dim,
                g4a_out_dim, g4a_mel_bins, g4a_chunk_mel, g4a_chunk_out);

        if (qnn.contextCreate(backend_handle, device_handle, nullptr, &context_handle) != QNN_SUCCESS)
            return false;

        int max_layers_cfg = g4a_num_layers;
        if (const char* ml = getenv("CACTUS_ENC_MAX_LAYERS"))
            max_layers_cfg = std::min(max_layers_cfg, std::stoi(ml));
        std::string cache_path = model_folder + "/qnn_g4a_T" + std::to_string(g4a_chunk_mel) +
                                 "_L" + std::to_string(max_layers_cfg) + ".bin";
        bool from_cache = false;
        if (file_exists(cache_path) && file_exists(cache_path + ".ids")) {
            if (context_handle && qnn.contextFree) { qnn.contextFree(context_handle, nullptr); context_handle = nullptr; }
            from_cache = g4a_load_from_cache(cache_path);
            if (!from_cache && qnn.contextCreate)
                qnn.contextCreate(backend_handle, device_handle, nullptr, &context_handle);
        }

        if (!from_cache && !g4a_build_graph()) return false;
        g4a_setup_exec_tensors();
        loaded = true; graph_built = true;
        return true;
    }

    size_t g4a_encode(const __fp16* input, __fp16* output, const std::vector<int>& shape) {
        if (!graph_built || !graph_handle) return 0;
        size_t in_elems = (size_t)g4a_chunk_mel * g4a_mel_bins;
        if (shape.size() >= 2) in_elems = (size_t)shape[0] * shape[1];
        memcpy(g4a_input_buf.data(), input, in_elems * sizeof(uint16_t));
        if (g4a_exec_in_qt)
            g4a_exec_in_qt->t.v1.clientBuf.data = g4a_input_buf.data();
        if (g4a_exec_out_qt)
            g4a_exec_out_qt->t.v1.clientBuf.data = g4a_output_buf.data();
        Qnn_ErrorHandle_t err = qnn.graphExecute(
            graph_handle,
            g4a_exec_inputs.data(),  (uint32_t)g4a_exec_inputs.size(),
            g4a_exec_outputs.data(), (uint32_t)g4a_exec_outputs.size(),
            nullptr, nullptr);
        if (err != QNN_SUCCESS) {
            fprintf(stderr, "[G4A] graphExecute failed: %lld\n", (long long)err); return 0;
        }
        int out_dim = g4a_out_dim > 0 ? g4a_out_dim : g4a_hidden;
        size_t out_elems = (size_t)g4a_chunk_out * out_dim;
        if (output) memcpy(output, g4a_output_buf.data(), out_elems * 2);
        return out_elems;
    }

    void build_enc_layer(int li, QTensor*& h) {
        std::string p = "el" + std::to_string(li) + "_";
        std::string wp = model_folder + "/encoder.layer_" + std::to_string(li) + "_";
        uint32_t T  = (uint32_t)T_enc;
        uint32_t hd = (uint32_t)hidden_dim;
        uint32_t nh = (uint32_t)num_heads;
        uint32_t hd2 = (uint32_t)head_dim;
        uint32_t fd = (uint32_t)ffn_dim;

        auto* ln1_w = load_weight(p+"ln1_w", wp+"self_attn_norm.weights");
        auto* ln1_b = load_weight(p+"ln1_b", wp+"self_attn_norm.bias");
        // LayerNorm gamma/beta need to be 1D [hidden_dim] for QNN but x is 2D [T, hd]
        // Expand to [1, hd] so they broadcast
        if (ln1_w->dims.size() == 1) {
            ln1_w = op_reshape(ln1_w, {1, hd}, p+"ln1_w_exp");
            ln1_b = op_reshape(ln1_b, {1, hd}, p+"ln1_b_exp");
        }
        auto* ln1 = op_layer_norm(h, ln1_w, ln1_b, p+"ln1");

        auto* wq = load_weight(p+"wq", wp+"self_attn_q.weights");
        auto* wk = load_weight(p+"wk", wp+"self_attn_k.weights");
        auto* wv = load_weight(p+"wv", wp+"self_attn_v.weights");

        auto* q = op_matmul_T(ln1, wq, p+"q");
        if (file_exists(wp+"self_attn_q.bias")) {
            auto* bq = load_weight(p+"bq", wp+"self_attn_q.bias");
            if (bq->dims.size() == 1) bq = op_reshape(bq, {1, hd}, p+"bq_exp");
            q = op_add(q, bq, p+"q_b");
        }
        auto* k = op_matmul_T(ln1, wk, p+"k");
        auto* v = op_matmul_T(ln1, wv, p+"v");
        if (file_exists(wp+"self_attn_v.bias")) {
            auto* bv = load_weight(p+"bv", wp+"self_attn_v.bias");
            if (bv->dims.size() == 1) bv = op_reshape(bv, {1, hd}, p+"bv_exp");
            v = op_add(v, bv, p+"v_b");
        }

        auto* q3 = op_reshape(q, {T, nh, hd2}, p+"q3");
        auto* k3 = op_reshape(k, {T, nh, hd2}, p+"k3");
        auto* v3 = op_reshape(v, {T, nh, hd2}, p+"v3");

        float scale = 1.0f / std::sqrt((float)head_dim);
        auto* attn_out = op_full_attention(q3, k3, v3, scale, p+"attn");

        auto* wo = load_weight(p+"wo", wp+"self_attn_output.weights");
        auto* o_proj = op_matmul_T(attn_out, wo, p+"o");
        if (file_exists(wp+"self_attn_output.bias")) {
            auto* bo = load_weight(p+"bo", wp+"self_attn_output.bias");
            if (bo->dims.size() == 1) bo = op_reshape(bo, {1, hd}, p+"bo_exp");
            o_proj = op_add(o_proj, bo, p+"o_b");
        }
        h = op_add(h, o_proj, p+"h_attn");

        auto* ln2_w = load_weight(p+"ln2_w", wp+"final_norm.weights");
        auto* ln2_b = load_weight(p+"ln2_b", wp+"final_norm.bias");
        if (ln2_w->dims.size() == 1) {
            ln2_w = op_reshape(ln2_w, {1, hd}, p+"ln2_w_exp");
            ln2_b = op_reshape(ln2_b, {1, hd}, p+"ln2_b_exp");
        }
        auto* ln2 = op_layer_norm(h, ln2_w, ln2_b, p+"ln2");

        auto* wfc1 = load_weight(p+"wfc1", wp+"mlp_fc1.weights");
        auto* wfc2 = load_weight(p+"wfc2", wp+"mlp_fc2.weights");
        auto* fc1 = op_matmul_T(ln2, wfc1, p+"fc1");
        if (file_exists(wp+"mlp_fc1.bias")) {
            auto* b1 = load_weight(p+"b1", wp+"mlp_fc1.bias");
            if (b1->dims.size() == 1) b1 = op_reshape(b1, {1, fd}, p+"b1_exp");
            fc1 = op_add(fc1, b1, p+"fc1_b");
        }
        auto* fc1_act = op_gelu(fc1, p+"gelu");
        auto* fc2 = op_matmul_T(fc1_act, wfc2, p+"fc2");
        if (file_exists(wp+"mlp_fc2.bias")) {
            auto* b2 = load_weight(p+"b2", wp+"mlp_fc2.bias");
            if (b2->dims.size() == 1) b2 = op_reshape(b2, {1, hd}, p+"b2_exp");
            fc2 = op_add(fc2, b2, p+"fc2_b");
        }
        h = op_add(h, fc2, p+"h_out");
        fprintf(stderr, "[QNN Enc] encoder layer %d / %d done\n", li + 1, num_layers);
    }

    bool build_graph() {
        QnnHtpGraph_CustomConfig_t htp_cfg = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
        htp_cfg.option    = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
        htp_cfg.precision = QNN_PRECISION_FLOAT16;
        QnnHtpGraph_CustomConfig_t htp_vtcm = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
        htp_vtcm.option         = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE_IN_MB;
        htp_vtcm.vtcmSizeInMB   = 8;
        QnnHtpGraph_CustomConfig_t htp_opt = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
        htp_opt.option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
        htp_opt.optimizationOption.type       = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
        htp_opt.optimizationOption.floatValue = 1.0f;  // 1=light, 3=aggressive (default); use light to avoid tiler hang
        QnnGraph_Config_t g_cfg0 = QNN_GRAPH_CONFIG_INIT;
        g_cfg0.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM; g_cfg0.customConfig = &htp_cfg;
        QnnGraph_Config_t g_cfg1 = QNN_GRAPH_CONFIG_INIT;
        g_cfg1.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM; g_cfg1.customConfig = &htp_vtcm;
        QnnGraph_Config_t g_cfg2 = QNN_GRAPH_CONFIG_INIT;
        g_cfg2.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM; g_cfg2.customConfig = &htp_opt;
        const QnnGraph_Config_t* g_cfgs[] = {&g_cfg0, &g_cfg1, &g_cfg2, nullptr};
        // Allow limiting layers for diagnostics via CACTUS_ENC_MAX_LAYERS env var
        int max_enc_layers = num_layers;
        if (const char* ml_env = getenv("CACTUS_ENC_MAX_LAYERS"))
            max_enc_layers = std::min(max_enc_layers, std::stoi(ml_env));
        if (max_enc_layers < num_layers)
            fprintf(stderr, "[QNN Enc] diagnostic mode: building %d/%d layers\n", max_enc_layers, num_layers);
        Qnn_ErrorHandle_t err = qnn.graphCreate(context_handle, "encoder", g_cfgs, &graph_handle);
        if (err != QNN_SUCCESS) {
            fprintf(stderr, "[QNN Enc] graphCreate failed: %lld\n", (long long)err);
            return false;
        }

        // Input: mel [1, n_mels, T_mel] (NCT format — matches model_whisper.cpp output)
        auto* mel_in = make_tensor("mel", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16,
                                   {1, (uint32_t)n_mels, (uint32_t)T_mel});
        exec_in_id = 0; // will be set after tensorCreateGraphTensor in add_op

        // Conv1: [1, T_mel, 80] → [1, T_mel, 384], stride=1, pad=1, k=3
        // Conv1: mel [1, 80, 3000] → [1, 384, 3000] via matmul decomposition
        auto* conv1_out = op_conv1d(mel_in,
                                    model_folder + "/encoder_conv1_weight.weights",
                                    model_folder + "/encoder_conv1_bias.bias",
                                    1, 1, "conv1");
        auto* conv1_act  = op_gelu(conv1_out, "conv1_gelu");

        // Conv2: [1, 384, 3000] → [1, 384, 1500] via matmul decomposition (stride=2)
        auto* conv2_out  = op_conv1d(conv1_act,
                                     model_folder + "/encoder_conv2_weight.weights",
                                     model_folder + "/encoder_conv2_bias.bias",
                                     2, 1, "conv2");
        auto* conv2_act  = op_gelu(conv2_out, "conv2_gelu");

        // conv2_act is [1, 384, T_enc] (NCT). Transpose to [1, T_enc, 384] then reshape to [T_enc, 384]
        uint32_t T = (uint32_t)T_enc, hd = (uint32_t)hidden_dim;
        auto* conv2_t = op_transpose(conv2_act, {0, 2, 1}, "conv2_t");  // [1, T_enc, 384]
        auto* h = op_reshape(conv2_t, {T, hd}, "h_2d");

        // Positional embeddings [T_enc, 384] (slice first T rows from stored [1500, 384])
        auto* pos_full = load_weight("pos_emb", model_folder + "/encoder_position_embeddings.weights");
        // pos_full shape may be [1500, 384]; we need first T rows
        QTensor* pos;
        if (pos_full->dims[0] == (uint32_t)T_enc) {
            pos = pos_full;
        } else {
            // Slice [0:T_enc, :] via reshape if T_enc matches, else use strided_slice equivalent
            // For simplicity, build a static tensor with just the T_enc rows
            std::vector<uint16_t> pos_slice(T * hd);
            const uint16_t* src = pos_full->static_fp16.data();
            memcpy(pos_slice.data(), src, T * hd * 2);
            pos = make_static_fp16("pos_slice", {T, hd}, std::move(pos_slice));
        }
        h = op_add(h, pos, "h_pos");

        // Transformer encoder layers
        for (int li = 0; li < max_enc_layers; li++)
            build_enc_layer(li, h);

        // Final LayerNorm (skip with CACTUS_ENC_SKIP_NORM=1 for diagnostics)
        bool skip_norm = getenv("CACTUS_ENC_SKIP_NORM") && getenv("CACTUS_ENC_SKIP_NORM")[0] == '1';
        QTensor* h_out;
        if (skip_norm) {
            fprintf(stderr, "[QNN Enc] SKIP_NORM: bypassing final LayerNorm\n"); fflush(stderr);
            h_out = h;
        } else {
            auto* norm_w = load_weight("enc_norm_w", model_folder + "/encoder_norm_weight.weights");
            auto* norm_b = load_weight("enc_norm_b", model_folder + "/encoder_norm_bias.bias");
            if (norm_w->dims.size() == 1) {
                norm_w = op_reshape(norm_w, {1, hd}, "enc_norm_w_exp");
                norm_b = op_reshape(norm_b, {1, hd}, "enc_norm_b_exp");
            }
            h_out = op_layer_norm(h, norm_w, norm_b, "enc_norm");
        }

        // Output tensor [T_enc, hidden_dim]
        auto* enc_out = make_output("enc_out", {T, hd});
        add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_RESHAPE, op_name("out_copy"),
               {h_out}, {enc_out}, {});

        exec_in_id  = mel_in->t.v1.id;
        exec_out_id = enc_out->t.v1.id;

        fprintf(stderr, "[QNN Enc] finalizing encoder graph (may take ~30s)...\n");
        fflush(stderr);
        err = qnn.graphFinalize(graph_handle, nullptr, nullptr);
        if (err != QNN_SUCCESS) {
            fprintf(stderr, "[QNN Enc] graphFinalize failed: %lld\n", (long long)err);
            return false;
        }
        fprintf(stderr, "[QNN Enc] finalized\n");

        if (qnn.graphRetrieve) {
            Qnn_GraphHandle_t rg = nullptr;
            if (qnn.graphRetrieve(context_handle, "encoder", &rg) == QNN_SUCCESS && rg)
                graph_handle = rg;
        }

        weight_files.clear();

        // Save cache
        std::string cache_path = model_folder + "/qnn_encoder_T" + std::to_string(T_mel) + ".bin";
        if (qnn.contextGetBinarySize && qnn.contextGetBinary) {
            Qnn_ContextBinarySize_t bin_sz = 0, written = 0;
            if (qnn.contextGetBinarySize(context_handle, &bin_sz) == QNN_SUCCESS && bin_sz > 0) {
                std::vector<char> bin((size_t)bin_sz);
                if (qnn.contextGetBinary(context_handle, bin.data(), bin_sz, &written) == QNN_SUCCESS && written > 0) {
                    std::ofstream cf(cache_path, std::ios::binary);
                    if (cf.is_open()) {
                        cf.write(bin.data(), (size_t)written);
                        fprintf(stderr, "[QNN Enc] saved cache: %s\n", cache_path.c_str());
                    }
                    // Write sidecar
                    std::ofstream sf(cache_path + ".ids", std::ios::binary);
                    uint32_t magic = 0x454E4351u; // "QCNE"
                    sf.write(reinterpret_cast<char*>(&magic),       4);
                    sf.write(reinterpret_cast<char*>(&exec_in_id),  4);
                    sf.write(reinterpret_cast<char*>(&exec_out_id), 4);
                    // Round-trip validate
                    if (qnn.contextCreateFromBinary && qnn.graphRetrieve) {
                        Qnn_ContextHandle_t nc = nullptr;
                        if (qnn.contextCreateFromBinary(backend_handle, device_handle, nullptr,
                                bin.data(), (size_t)written, &nc, nullptr) == QNN_SUCCESS) {
                            Qnn_GraphHandle_t ng = nullptr;
                            if (qnn.graphRetrieve(nc, "encoder", &ng) == QNN_SUCCESS) {
                                qnn.contextFree(context_handle, nullptr);
                                context_handle = nc; graph_handle = ng;
                                fprintf(stderr, "[QNN Enc] binary round-trip ok\n");
                            } else { qnn.contextFree(nc, nullptr); }
                        }
                    }
                }
            }
        }
        return true;
    }

    bool load_from_cache(const std::string& cache_path) {
        std::string sid = cache_path + ".ids";
        std::ifstream sf(sid, std::ios::binary);
        if (!sf.is_open()) return false;
        uint32_t magic = 0;
        sf.read(reinterpret_cast<char*>(&magic), 4);
        if (magic != 0x454E4351u) return false;
        sf.read(reinterpret_cast<char*>(&exec_in_id),  4);
        sf.read(reinterpret_cast<char*>(&exec_out_id), 4);

        std::ifstream cf(cache_path, std::ios::binary | std::ios::ate);
        if (!cf.is_open()) return false;
        size_t sz = (size_t)cf.tellg(); cf.seekg(0);
        std::vector<char> buf(sz); cf.read(buf.data(), sz); cf.close();

        Qnn_ErrorHandle_t err = qnn.contextCreateFromBinary(
            backend_handle, device_handle, nullptr,
            buf.data(), sz, &context_handle, nullptr);
        if (err != QNN_SUCCESS) {
            fprintf(stderr, "[QNN Enc] contextCreateFromBinary failed: %lld\n", (long long)err);
            return false;
        }
        if (qnn.graphRetrieve(context_handle, "encoder", &graph_handle) != QNN_SUCCESS) {
            fprintf(stderr, "[QNN Enc] graphRetrieve failed\n"); return false;
        }
        fprintf(stderr, "[QNN Enc] loaded from cache: %s\n", cache_path.c_str());
        return true;
    }

    void setup_exec_tensors() {
        fprintf(stderr, "[QNN Enc] setup_exec_tensors: T_mel=%d n_mels=%d T_enc=%d hidden=%d in_id=%u out_id=%u\n",
                T_mel, n_mels, T_enc, hidden_dim, exec_in_id, exec_out_id);
        fflush(stderr);
        input_buf.assign((size_t)T_mel * n_mels, 0);
        output_buf.assign((size_t)T_enc * hidden_dim, 0);

        auto make_from_id = [&](uint32_t id, Qnn_TensorType_t type,
                                 const std::vector<uint32_t>& shape) -> QTensor* {
            auto qt = std::make_unique<QTensor>();
            qt->name = "ec_" + std::to_string(id); qt->dims = shape;
            qt->t = QNN_TENSOR_INIT;
            qt->t.version = QNN_TENSOR_VERSION_1;
            qt->t.v1.id = id; qt->t.v1.type = type;
            qt->t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
            qt->t.v1.dataType   = QNN_DATATYPE_FLOAT_16;
            qt->t.v1.quantizeParams = { QNN_DEFINITION_UNDEFINED,
                                        QNN_QUANTIZATION_ENCODING_UNDEFINED, {{0,0}} };
            qt->t.v1.rank = (uint32_t)shape.size(); qt->t.v1.dimensions = qt->dims.data();
            qt->t.v1.memType = QNN_TENSORMEMTYPE_RAW;
            qt->t.v1.clientBuf = {nullptr, 0};
            QTensor* ptr = qt.get();
            tensor_store.push_back(std::move(qt));
            return ptr;
        };

        auto get = [&](const std::string& name, uint32_t id,
                        Qnn_TensorType_t type, const std::vector<uint32_t>& shape) -> QTensor* {
            if (tensors.count(name)) return tensors.at(name);
            return make_from_id(id, type, shape);
        };

        fprintf(stderr, "[QNN Enc] set1: calling get for mel\n"); fflush(stderr);
        exec_in_qt  = get("mel",     exec_in_id,  QNN_TENSOR_TYPE_APP_WRITE,
                          {1, (uint32_t)n_mels, (uint32_t)T_mel});
        fprintf(stderr, "[QNN Enc] set2: exec_in_qt=%p\n", (void*)exec_in_qt); fflush(stderr);
        exec_out_qt = get("enc_out", exec_out_id, QNN_TENSOR_TYPE_APP_READ,
                          {(uint32_t)T_enc, (uint32_t)hidden_dim});
        fprintf(stderr, "[QNN Enc] set3: exec_out_qt=%p\n", (void*)exec_out_qt); fflush(stderr);

        auto set_buf = [](QTensor* qt, void* data, uint32_t sz) {
            if (qt) { qt->t.v1.clientBuf.data = data; qt->t.v1.clientBuf.dataSize = sz; }
        };
        set_buf(exec_in_qt,  input_buf.data(),  (uint32_t)(input_buf.size()  * 2));
        fprintf(stderr, "[QNN Enc] set4: in buf set\n"); fflush(stderr);
        set_buf(exec_out_qt, output_buf.data(), (uint32_t)(output_buf.size() * 2));
        fprintf(stderr, "[QNN Enc] set5: out buf set\n"); fflush(stderr);

        exec_inputs  = { exec_in_qt  ? exec_in_qt->t  : Qnn_Tensor_t{} };
        fprintf(stderr, "[QNN Enc] set6: exec_inputs assigned\n"); fflush(stderr);
        exec_outputs = { exec_out_qt ? exec_out_qt->t : Qnn_Tensor_t{} };
        fprintf(stderr, "[QNN Enc] exec tensors set: in_qt=%p out_qt=%p\n",
                (void*)exec_in_qt, (void*)exec_out_qt);
        fflush(stderr);
    }

    // ================================================================
    // Parakeet helpers
    // ================================================================

    std::string pk_op_name(const std::string& t) {
        return t + "_pk" + std::to_string(pk_cur->op_idx++);
    }

    QTensor* pk_make_tensor(const std::string& name, Qnn_TensorType_t type,
                             Qnn_DataType_t dtype, const std::vector<uint32_t>& shape,
                             const void* data = nullptr, size_t data_bytes = 0) {
        auto qt = std::make_unique<QTensor>();
        qt->name = name + "_Z" + std::to_string(pk_cur->tensor_idx++);
        qt->dims = shape;
        qt->t = QNN_TENSOR_INIT; qt->t.version = QNN_TENSOR_VERSION_1;
        qt->t.v1.id = 0; qt->t.v1.name = qt->name.c_str(); qt->t.v1.type = type;
        qt->t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER; qt->t.v1.dataType = dtype;
        qt->t.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
        qt->t.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
        qt->t.v1.rank = (uint32_t)shape.size(); qt->t.v1.dimensions = qt->dims.data();
        qt->t.v1.memType = QNN_TENSORMEMTYPE_RAW;
        qt->t.v1.clientBuf.data = (type == QNN_TENSOR_TYPE_STATIC) ? const_cast<void*>(data) : nullptr;
        size_t sz = data_bytes;
        if (!sz) {
            sz = 1; for (auto d : shape) sz *= d;
            if (dtype == QNN_DATATYPE_FLOAT_16 || dtype == QNN_DATATYPE_INT_16) sz *= 2;
            else if (dtype == QNN_DATATYPE_FLOAT_32 || dtype == QNN_DATATYPE_INT_32) sz *= 4;
        }
        qt->t.v1.clientBuf.dataSize = (uint32_t)sz;
        QTensor* raw = qt.get();
        pk_cur->tensors[name] = raw;
        pk_cur->tensor_store.push_back(std::move(qt));
        return raw;
    }

    QTensor* pk_make_static_fp16(const std::string& name, const std::vector<uint32_t>& shape,
                                  std::vector<uint16_t> data) {
        auto qt = std::make_unique<QTensor>();
        qt->name = name; qt->dims = shape; qt->static_fp16 = std::move(data);
        qt->t = QNN_TENSOR_INIT; qt->t.version = QNN_TENSOR_VERSION_1;
        qt->t.v1.id = 0; qt->t.v1.name = qt->name.c_str(); qt->t.v1.type = QNN_TENSOR_TYPE_STATIC;
        qt->t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
        qt->t.v1.dataType = QNN_DATATYPE_FLOAT_16;
        qt->t.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
        qt->t.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
        qt->t.v1.rank = (uint32_t)shape.size(); qt->t.v1.dimensions = qt->dims.data();
        qt->t.v1.memType = QNN_TENSORMEMTYPE_RAW;
        qt->t.v1.clientBuf.data = qt->static_fp16.data();
        qt->t.v1.clientBuf.dataSize = (uint32_t)(qt->static_fp16.size() * 2);
        QTensor* raw = qt.get();
        pk_cur->tensors[name] = raw;
        pk_cur->tensor_store.push_back(std::move(qt));
        return raw;
    }

    QTensor* pk_make_static_i32(const std::string& name, const std::vector<uint32_t>& shape,
                                 std::vector<int32_t> data) {
        auto qt = std::make_unique<QTensor>();
        qt->name = name; qt->dims = shape; qt->static_i32 = std::move(data);
        qt->t = QNN_TENSOR_INIT; qt->t.version = QNN_TENSOR_VERSION_1;
        qt->t.v1.id = 0; qt->t.v1.name = qt->name.c_str(); qt->t.v1.type = QNN_TENSOR_TYPE_STATIC;
        qt->t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
        qt->t.v1.dataType = QNN_DATATYPE_INT_32;
        qt->t.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
        qt->t.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
        qt->t.v1.rank = (uint32_t)shape.size(); qt->t.v1.dimensions = qt->dims.data();
        qt->t.v1.memType = QNN_TENSORMEMTYPE_RAW;
        qt->t.v1.clientBuf.data = qt->static_i32.data();
        qt->t.v1.clientBuf.dataSize = (uint32_t)(qt->static_i32.size() * 4);
        QTensor* raw = qt.get();
        pk_cur->tensors[name] = raw;
        pk_cur->tensor_store.push_back(std::move(qt));
        return raw;
    }

    QTensor* pk_make_static_u32(const std::string& name, const std::vector<uint32_t>& shape,
                                 std::vector<uint32_t> data) {
        auto qt = std::make_unique<QTensor>();
        qt->name = name; qt->dims = shape; qt->static_u32 = std::move(data);
        qt->t = QNN_TENSOR_INIT; qt->t.version = QNN_TENSOR_VERSION_1;
        qt->t.v1.id = 0; qt->t.v1.name = qt->name.c_str(); qt->t.v1.type = QNN_TENSOR_TYPE_STATIC;
        qt->t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
        qt->t.v1.dataType = QNN_DATATYPE_UINT_32;
        qt->t.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
        qt->t.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
        qt->t.v1.rank = (uint32_t)shape.size(); qt->t.v1.dimensions = qt->dims.data();
        qt->t.v1.memType = QNN_TENSORMEMTYPE_RAW;
        qt->t.v1.clientBuf.data = qt->static_u32.data();
        qt->t.v1.clientBuf.dataSize = (uint32_t)(qt->static_u32.size() * 4);
        QTensor* raw = qt.get();
        pk_cur->tensors[name] = raw;
        pk_cur->tensor_store.push_back(std::move(qt));
        return raw;
    }

    QTensor* pk_make_native(const std::string& name, const std::vector<uint32_t>& shape) {
        return pk_make_tensor(name, QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, shape);
    }
    QTensor* pk_make_output(const std::string& name, const std::vector<uint32_t>& shape) {
        return pk_make_tensor(name, QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_16, shape);
    }
    QTensor* pk_make_app_write(const std::string& name, const std::vector<uint32_t>& shape) {
        return pk_make_tensor(name, QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16, shape);
    }

    QTensor* pk_load_weight(const std::string& name, const std::string& path) {
        auto cf = std::make_unique<CactFile>();
        if (!open_cact(path, *cf)) {
            fprintf(stderr, "[QNN Pkrt] failed to load: %s\n", path.c_str());
            return pk_make_static_fp16(name + "_W", {1}, {0});
        }
        std::vector<uint32_t> shape(cf->shape.begin(), cf->shape.end());
        std::vector<uint16_t> fp16 = cact_to_fp16(*cf);
        pk_cur->weight_files.push_back(std::move(cf));
        return pk_make_static_fp16(name + "_W", shape, std::move(fp16));
    }

    static float pk_fp16_to_fp32(uint16_t h) {
        uint32_t sign = (uint32_t)(h >> 15) << 31, exp = (h >> 10) & 0x1F, mant = h & 0x3FF, bits;
        if (exp == 0) bits = sign;
        else if (exp == 31) bits = sign | 0x7F800000u | (mant << 13);
        else bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        float v; memcpy(&v, &bits, 4); return v;
    }

    QTensor* pk_make_static_int8(const std::string& name, const std::vector<uint32_t>& shape,
                                  float scale, int32_t offset, std::vector<int8_t> data) {
        auto qt = std::make_unique<QTensor>();
        qt->name = name; qt->dims = shape;
        qt->t = QNN_TENSOR_INIT; qt->t.version = QNN_TENSOR_VERSION_1;
        qt->t.v1.id = 0; qt->t.v1.name = qt->name.c_str(); qt->t.v1.type = QNN_TENSOR_TYPE_STATIC;
        qt->t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
        qt->t.v1.dataType = QNN_DATATYPE_SFIXED_POINT_8;
        qt->t.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
        qt->t.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
        qt->t.v1.quantizeParams.scaleOffsetEncoding.scale = scale;
        qt->t.v1.quantizeParams.scaleOffsetEncoding.offset = offset;
        qt->t.v1.rank = (uint32_t)shape.size(); qt->t.v1.dimensions = qt->dims.data();
        qt->t.v1.memType = QNN_TENSORMEMTYPE_RAW;
        size_t n = 1; for (auto d : shape) n *= d;
        // Store int8 bytes in static_i32 (4 bytes per int32, contains raw int8 bytes)
        std::vector<int32_t> storage((n + 3) / 4, 0);
        memcpy(storage.data(), data.data(), n);
        qt->static_i32 = std::move(storage);
        qt->t.v1.clientBuf.data = qt->static_i32.data();
        qt->t.v1.clientBuf.dataSize = (uint32_t)n;
        QTensor* raw = qt.get();
        pk_cur->tensors[name] = raw;
        pk_cur->tensor_store.push_back(std::move(qt));
        return raw;
    }

    Qnn_ErrorHandle_t pk_add_op(const char* pkg, const char* type_name, const std::string& name,
                                  const std::vector<QTensor*>& ins, const std::vector<QTensor*>& outs,
                                  const std::vector<Qnn_Param_t>& params) {
        auto reg = [&](QTensor* qt) {
            if (qt->t.v1.id == 0 && qt->t.v1.type != QNN_TENSOR_TYPE_UNDEFINED)
                qnn.tensorCreateGraphTensor(pk_cur->graph_handle, &qt->t);
        };
        for (auto* qt : ins)  reg(qt);
        for (auto* qt : outs) reg(qt);
        std::vector<Qnn_Tensor_t> in_t, out_t;
        for (auto* qt : ins)  in_t.push_back(qt->t);
        for (auto* qt : outs) out_t.push_back(qt->t);
        Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
        op.version = QNN_OPCONFIG_VERSION_1;
        op.v1.name = name.c_str(); op.v1.packageName = pkg; op.v1.typeName = type_name;
        op.v1.numOfParams = (uint32_t)params.size();
        op.v1.params = params.empty() ? nullptr : const_cast<Qnn_Param_t*>(params.data());
        op.v1.numOfInputs = (uint32_t)in_t.size(); op.v1.inputTensors = in_t.data();
        op.v1.numOfOutputs = (uint32_t)out_t.size(); op.v1.outputTensors = out_t.data();
        Qnn_ErrorHandle_t err = qnn.graphAddNode(pk_cur->graph_handle, op);
        if (err != QNN_SUCCESS)
            fprintf(stderr, "[QNN Pkrt] addNode %s failed: %lld\n", name.c_str(), (long long)err);
        return err;
    }

    QTensor* pk_op_dequantize(QTensor* x, const std::string& n) {
        auto* out = pk_make_native(n, x->dims);
        pk_add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_DEQUANTIZE, pk_op_name("dequant"), {x}, {out}, {});
        return out;
    }

    QTensor* pk_load_weight_int8(const std::string& name, const std::string& path) {
        auto cf = std::make_unique<CactFile>();
        if (!open_cact(path, *cf)) {
            fprintf(stderr, "[QNN Pkrt] failed to load: %s\n", path.c_str());
            return pk_make_static_fp16(name + "_W", {1}, {0});
        }
        std::vector<uint32_t> shape(cf->shape.begin(), cf->shape.end());
        while (shape.size() > 2 && shape.back() == 1) shape.pop_back();
        std::vector<uint16_t> fp16 = cact_to_fp16(*cf);
        pk_cur->weight_files.push_back(std::move(cf));
        size_t n = 1; for (auto d : shape) n *= d;
        float max_val = 0.0f;
        for (size_t i = 0; i < n; i++) {
            float v = pk_fp16_to_fp32(fp16[i]);
            float av = v < 0.f ? -v : v;
            if (av > max_val) max_val = av;
        }
        float scale = (max_val > 0.f) ? max_val / 127.0f : 1.0f / 127.0f;
        std::vector<int8_t> i8(n);
        for (size_t i = 0; i < n; i++) {
            float v = pk_fp16_to_fp32(fp16[i]);
            int32_t q = (int32_t)roundf(v / scale);
            if (q > 127) q = 127; if (q < -128) q = -128;
            i8[i] = (int8_t)q;
        }
        auto* wq = pk_make_static_int8(name + "_W", shape, scale, 0, std::move(i8));
        if (!wq->t.v1.id) qnn.tensorCreateGraphTensor(pk_cur->graph_handle, &wq->t);
        return pk_op_dequantize(wq, name + "_deq");
    }

    QTensor* pk_op_add(QTensor* a, QTensor* b, const std::string& n) {
        auto* out = pk_make_native(n, a->dims);
        pk_add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_ELEMENT_WISE_ADD, pk_op_name("add"), {a,b}, {out}, {});
        return out;
    }
    QTensor* pk_op_mul(QTensor* a, QTensor* b, const std::string& n) {
        auto* out = pk_make_native(n, a->dims);
        pk_add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_ELEMENT_WISE_MULTIPLY, pk_op_name("mul"), {a,b}, {out}, {});
        return out;
    }
    QTensor* pk_op_relu(QTensor* x, const std::string& n) {
        auto* out = pk_make_native(n, x->dims);
        pk_add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_RELU, pk_op_name("relu"), {x}, {out}, {});
        return out;
    }
    QTensor* pk_op_sigmoid(QTensor* x, const std::string& n) {
        auto* out = pk_make_native(n, x->dims);
        pk_add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_ELEMENT_WISE_NEURON, pk_op_name("sig"), {x}, {out},
                  {scalar_u32_param(QNN_OP_ELEMENT_WISE_NEURON_PARAM_OPERATION,
                                    QNN_OP_ELEMENT_WISE_NEURON_OPERATION_SIGMOID)});
        return out;
    }
    QTensor* pk_op_silu(QTensor* x, const std::string& n) {
        return pk_op_mul(x, pk_op_sigmoid(x, n + "_sig"), n);
    }
    QTensor* pk_op_softmax(QTensor* x, int axis, const std::string& n) {
        auto* out = pk_make_native(n, x->dims);
        pk_add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_SOFTMAX, pk_op_name("sm"), {x}, {out},
                  {scalar_i32_param(QNN_OP_SOFTMAX_PARAM_AXIS, axis)});
        return out;
    }
    QTensor* pk_op_reshape(QTensor* x, const std::vector<uint32_t>& shape, const std::string& n) {
        auto* out = pk_make_native(n, shape);
        pk_add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_RESHAPE, pk_op_name("rs"), {x}, {out}, {});
        return out;
    }
    QTensor* pk_op_transpose(QTensor* x, const std::vector<uint32_t>& perm, const std::string& n) {
        std::vector<uint32_t> sh(perm.size());
        for (size_t i = 0; i < perm.size(); i++) sh[i] = x->dims[perm[i]];
        auto* out = pk_make_native(n, sh);
        auto* pt = pk_make_static_u32("perm_" + n, {(uint32_t)perm.size()},
                                       std::vector<uint32_t>(perm.begin(), perm.end()));
        if (!pt->t.v1.id) qnn.tensorCreateGraphTensor(pk_cur->graph_handle, &pt->t);
        Qnn_Param_t pp = tensor_param(QNN_OP_TRANSPOSE_PARAM_PERM, pt); pp.tensorParam = pt->t;
        pk_add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_TRANSPOSE, pk_op_name("tr"), {x}, {out}, {pp});
        return out;
    }
    QTensor* pk_op_concat(QTensor* a, QTensor* b, int axis, const std::string& n) {
        std::vector<uint32_t> sh = a->dims; sh[(size_t)axis] += b->dims[(size_t)axis];
        auto* out = pk_make_native(n, sh);
        pk_add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CONCAT, pk_op_name("cat"), {a, b}, {out},
                  {scalar_i32_param(QNN_OP_CONCAT_PARAM_AXIS, axis)});
        return out;
    }
    QTensor* pk_op_matmul_T(QTensor* x, QTensor* w, const std::string& n) {
        std::vector<uint32_t> sh;
        for (size_t i = 0; i + 1 < x->dims.size(); i++) sh.push_back(x->dims[i]);
        sh.push_back(w->dims[0]);
        auto* out = pk_make_native(n, sh);
        pk_add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_MAT_MUL, pk_op_name("mm"), {x, w}, {out},
                  {scalar_bool_param(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, false),
                   scalar_bool_param(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, true)});
        return out;
    }
    QTensor* pk_op_bmm(QTensor* a, QTensor* b, bool tB, const std::string& n) {
        size_t r = a->dims.size();
        std::vector<uint32_t> sh;
        for (size_t i = 0; i + 2 < r; i++) sh.push_back(a->dims[i]);
        sh.push_back(a->dims[r - 2]);
        sh.push_back(tB ? b->dims[r - 2] : b->dims[r - 1]);
        auto* out = pk_make_native(n, sh);
        pk_add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_MAT_MUL, pk_op_name("bmm"), {a, b}, {out},
                  {scalar_bool_param(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, false),
                   scalar_bool_param(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, tB)});
        return out;
    }
    QTensor* pk_op_gather_elements(QTensor* data, QTensor* indices, int axis, const std::string& n) {
        std::vector<uint32_t> sh = data->dims;
        sh[(size_t)axis] = indices->dims[(size_t)axis];
        auto* out = pk_make_native(n, sh);
        Qnn_Param_t ap = scalar_i32_param(QNN_OP_GATHER_ELEMENTS_PARAM_AXIS, axis);
        pk_add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_GATHER_ELEMENTS, pk_op_name("ge"),
                  {data, indices}, {out}, {ap});
        return out;
    }
    QTensor* pk_op_scalar_mul(QTensor* x, float val, const std::string& n) {
        std::vector<uint32_t> sh(x->dims.size(), 1u);
        auto* sc = pk_make_static_fp16(n + "_sc", sh, {float_to_fp16(val)});
        if (!sc->t.v1.id) qnn.tensorCreateGraphTensor(pk_cur->graph_handle, &sc->t);
        return pk_op_mul(x, sc, n);
    }
    QTensor* pk_op_slice_nd(QTensor* x, const std::vector<int32_t>& begins,
                             const std::vector<int32_t>& ends, const std::string& n) {
        int rank = (int)x->dims.size();
        std::vector<uint32_t> sh(rank);
        for (int i = 0; i < rank; i++) sh[i] = (uint32_t)(ends[i] - begins[i]);
        auto* out = pk_make_native(n, sh);
        std::vector<int32_t> ranges;
        for (int i = 0; i < rank; i++) {
            ranges.push_back(begins[i]); ranges.push_back(ends[i]); ranges.push_back(1);
        }
        auto* rt = pk_make_static_i32("ranges_" + n, {(uint32_t)rank, 3}, ranges);
        if (!rt->t.v1.id) qnn.tensorCreateGraphTensor(pk_cur->graph_handle, &rt->t);
        Qnn_Param_t rp = tensor_param(QNN_OP_STRIDED_SLICE_PARAM_RANGES, rt); rp.tensorParam = rt->t;
        pk_add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_STRIDED_SLICE, pk_op_name("slc"), {x}, {out}, {rp});
        return out;
    }
    QTensor* pk_op_glu(QTensor* x, const std::string& n) {
        int rank = (int)x->dims.size();
        uint32_t half = x->dims[rank - 1] / 2;
        std::vector<int32_t> b1(rank, 0), e1(rank);
        for (int i = 0; i < rank; i++) e1[i] = (int)x->dims[i];
        e1[rank - 1] = (int)half;
        auto* first = pk_op_slice_nd(x, b1, e1, n + "_f");
        std::vector<int32_t> b2(rank, 0), e2(rank);
        for (int i = 0; i < rank; i++) e2[i] = (int)x->dims[i];
        b2[rank - 1] = (int)half;
        auto* second = pk_op_slice_nd(x, b2, e2, n + "_s");
        return pk_op_mul(first, pk_op_sigmoid(second, n + "_sig"), n);
    }
    QTensor* pk_op_pad(QTensor* x, const std::vector<std::pair<int,int>>& pad_pairs,
                        const std::string& n) {
        int rank = (int)x->dims.size();
        std::vector<uint32_t> sh = x->dims;
        std::vector<int32_t> amt;
        for (int i = 0; i < rank; i++) {
            amt.push_back(pad_pairs[i].first); amt.push_back(pad_pairs[i].second);
            sh[i] += (uint32_t)(pad_pairs[i].first + pad_pairs[i].second);
        }
        auto* out = pk_make_native(n, sh);
        auto* at = pk_make_static_i32("padamt_" + n, {(uint32_t)rank, 2}, amt);
        if (!at->t.v1.id) qnn.tensorCreateGraphTensor(pk_cur->graph_handle, &at->t);
        Qnn_Param_t sp = scalar_i32_param(QNN_OP_PAD_PARAM_SCHEME, QNN_OP_PAD_SCHEME_CONSTANT);
        Qnn_Param_t ap = tensor_param(QNN_OP_PAD_PARAM_PAD_AMOUNT, at); ap.tensorParam = at->t;
        pk_add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_PAD, pk_op_name("pad"), {x}, {out}, {sp, ap});
        return out;
    }
    QTensor* pk_op_layer_norm_wb(QTensor* x, QTensor* gamma, QTensor* beta, const std::string& n) {
        auto* out = pk_make_native(n, x->dims);
        int last = (int)x->dims.size() - 1;
        auto* at = pk_make_static_i32("axes_ln_" + n, {1}, {last});
        if (!at->t.v1.id) qnn.tensorCreateGraphTensor(pk_cur->graph_handle, &at->t);
        Qnn_Param_t ap = tensor_param(QNN_OP_LAYER_NORM_PARAM_AXES, at); ap.tensorParam = at->t;
        Qnn_Param_t ep = QNN_PARAM_INIT; ep.paramType = QNN_PARAMTYPE_SCALAR;
        ep.name = QNN_OP_LAYER_NORM_PARAM_EPSILON; ep.scalarParam.dataType = QNN_DATATYPE_FLOAT_32;
        ep.scalarParam.floatValue = pk_ln_eps;
        pk_add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_LAYER_NORM, pk_op_name("ln"),
                  {x, gamma, beta}, {out}, {ep, ap});
        return out;
    }
    QTensor* pk_op_depthwise_conv1d_same(QTensor* x, const std::string& wpath,
                                          const std::string& bpath, int K, const std::string& n) {
        auto cf = std::make_unique<CactFile>();
        if (!open_cact(wpath, *cf)) {
            fprintf(stderr, "[QNN Pkrt] dw_conv1d weight missing: %s\n", wpath.c_str()); return x;
        }
        auto raw = cact_to_fp16(*cf);
        uint32_t T = x->dims[1], C = x->dims[2], Kk = (uint32_t)K;
        pk_cur->weight_files.push_back(std::move(cf));
        int pad = K / 2;
        std::vector<uint16_t> w_fp16(Kk * C);
        for (uint32_t k = 0; k < Kk; k++)
            for (uint32_t c = 0; c < C; c++)
                w_fp16[k * C + c] = raw[c * Kk + k];
        auto* W_t = pk_make_static_fp16(n + "_W", {Kk, 1, 1, C}, std::move(w_fp16));
        if (!W_t->t.v1.id) qnn.tensorCreateGraphTensor(pk_cur->graph_handle, &W_t->t);
        std::vector<uint16_t> bias_fp16(C, 0);
        if (file_exists(bpath)) {
            CactFile bf; if (open_cact(bpath, bf)) {
                auto bfp = cact_to_fp16(bf);
                for (uint32_t c = 0; c < C && c < (uint32_t)bfp.size(); c++) bias_fp16[c] = bfp[c];
            }
        }
        auto* b_t = pk_make_static_fp16(n + "_bias", {C}, std::move(bias_fp16));
        if (!b_t->t.v1.id) qnn.tensorCreateGraphTensor(pk_cur->graph_handle, &b_t->t);
        uint32_t Tp = (uint32_t)(T + 2 * pad);
        auto* xp = pk_op_pad(x, {{0,0}, {pad, pad}, {0,0}}, n + "_pad");
        auto* xp_nhwc = pk_op_reshape(xp, {1, Tp, 1, C}, n + "_nhwc");
        auto* stride_t = pk_make_static_i32(n + "_stride", {2}, {1, 1});
        if (!stride_t->t.v1.id) qnn.tensorCreateGraphTensor(pk_cur->graph_handle, &stride_t->t);
        Qnn_Param_t sp = tensor_param(QNN_OP_DEPTH_WISE_CONV_2D_PARAM_STRIDE, stride_t); sp.tensorParam = stride_t->t;
        auto* pad_t = pk_make_static_i32(n + "_pad0", {2, 2}, {0,0,0,0});
        if (!pad_t->t.v1.id) qnn.tensorCreateGraphTensor(pk_cur->graph_handle, &pad_t->t);
        Qnn_Param_t pp = tensor_param(QNN_OP_DEPTH_WISE_CONV_2D_PARAM_PAD_AMOUNT, pad_t); pp.tensorParam = pad_t->t;
        auto* out = pk_make_native(n + "_out", {1, T, 1, C});
        pk_add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_DEPTH_WISE_CONV_2D, pk_op_name("dwc"),
                  {xp_nhwc, W_t, b_t}, {out}, {sp, pp});
        return pk_op_reshape(out, {1, T, C}, n);
    }
    QTensor* pk_op_batchnorm(QTensor* x, const std::string& wpath, const std::string& bpath,
                              const std::string& mean_path, const std::string& var_path,
                              const std::string& n) {
        CactFile fw, fb, fm, fv;
        bool ok_w = open_cact(wpath, fw), ok_b = open_cact(bpath, fb);
        bool ok_m = open_cact(mean_path, fm), ok_v = open_cact(var_path, fv);
        uint32_t C = x->dims[2];
        if (!ok_w || !ok_b || !ok_m || !ok_v) {
            fprintf(stderr, "[QNN Pkrt] batchnorm files missing: %s\n", n.c_str()); return x;
        }
        auto wfp16 = cact_to_fp16(fw), bfp16 = cact_to_fp16(fb);
        auto mfp16 = cact_to_fp16(fm), vfp16 = cact_to_fp16(fv);
        std::vector<uint16_t> scale_fp16(C), shift_fp16(C);
        for (uint32_t c = 0; c < C; c++) {
            float wc = pk_fp16_to_fp32(wfp16[c]), bc = pk_fp16_to_fp32(bfp16[c]);
            float mc = pk_fp16_to_fp32(mfp16[c]), vc = pk_fp16_to_fp32(vfp16[c]);
            float sc = wc / sqrtf(vc + pk_bn_eps);
            float sh = bc - mc * sc;
            scale_fp16[c] = float_to_fp16(sc); shift_fp16[c] = float_to_fp16(sh);
        }
        auto* sc_t = pk_make_static_fp16(n + "_sc", {1,1,C}, std::move(scale_fp16));
        auto* sh_t = pk_make_static_fp16(n + "_sh", {1,1,C}, std::move(shift_fp16));
        if (!sc_t->t.v1.id) qnn.tensorCreateGraphTensor(pk_cur->graph_handle, &sc_t->t);
        if (!sh_t->t.v1.id) qnn.tensorCreateGraphTensor(pk_cur->graph_handle, &sh_t->t);
        return pk_op_add(pk_op_mul(x, sc_t, n + "_xsc"), sh_t, n);
    }

    // ---- Relative position encoding ----
    QTensor* pk_build_rel_key_static(int li) {
        uint32_t T = (uint32_t)pk_T_out, H = (uint32_t)pk_num_heads, D = (uint32_t)pk_head_dim;
        uint32_t hidden = (uint32_t)pk_hidden_dim, R = 2 * T - 1;
        std::vector<float> pos_fp32(R * hidden, 0.f);
        for (uint32_t p = 0; p < R; p++) {
            int rel_pos = (int)(T - 1) - (int)p;
            for (uint32_t i = 0; i < hidden / 2; i++) {
                float inv_freq = 1.f / powf(10000.f, 2.f * (float)i / (float)hidden);
                float angle = (float)rel_pos * inv_freq;
                pos_fp32[p * hidden + 2 * i] = sinf(angle);
                if (2 * i + 1 < hidden) pos_fp32[p * hidden + 2 * i + 1] = cosf(angle);
            }
        }
        std::string wpath = model_folder + "/layer_" + std::to_string(li) + "_self_attn_relative_k.weights";
        CactFile cf;
        if (!open_cact(wpath, cf))
            return pk_make_static_fp16("rk_" + std::to_string(li), {1,R,H,D},
                                        std::vector<uint16_t>(R*H*D, 0));
        auto w_fp16 = cact_to_fp16(cf);
        std::vector<float> rel_key_fp32(R * hidden, 0.f);
        for (uint32_t r = 0; r < R; r++)
            for (uint32_t j = 0; j < hidden; j++) {
                float s = 0.f;
                for (uint32_t k = 0; k < hidden; k++)
                    s += pos_fp32[r * hidden + k] * pk_fp16_to_fp32(w_fp16[j * hidden + k]);
                rel_key_fp32[r * hidden + j] = s;
            }
        std::vector<uint16_t> out_fp16(R * H * D);
        for (uint32_t r = 0; r < R; r++)
            for (uint32_t hh = 0; hh < H; hh++)
                for (uint32_t d = 0; d < D; d++)
                    out_fp16[r * H * D + hh * D + d] =
                        float_to_fp16(rel_key_fp32[r * hidden + hh * D + d]);
        return pk_make_static_fp16("rk_" + std::to_string(li) + "_rk", {1,R,H,D}, std::move(out_fp16));
    }

    QTensor* pk_build_rel_pos_bias(QTensor* qv4, QTensor* rel_key, float scale, const std::string& pfx) {
        uint32_t T = (uint32_t)pk_T_out, H = (uint32_t)pk_num_heads;
        auto* qv_T = pk_op_transpose(qv4, {0,2,1,3}, pfx + "_qvT");
        auto* rk_BHRD = pk_op_transpose(rel_key, {0,2,1,3}, pfx + "_rkT");
        auto* raw = pk_op_bmm(qv_T, rk_BHRD, true, pfx + "_raw");
        raw = pk_op_scalar_mul(raw, scale, pfx + "_rawsc");
        std::vector<int32_t> idx_data((size_t)H * T * T);
        for (uint32_t h = 0; h < H; h++)
            for (uint32_t t = 0; t < T; t++)
                for (uint32_t s = 0; s < T; s++)
                    idx_data[h * T * T + t * T + s] = (int32_t)(t + T - 1 - s);
        auto* idx = pk_make_static_i32(pfx + "_idx", {1,H,T,T}, std::move(idx_data));
        if (!idx->t.v1.id) qnn.tensorCreateGraphTensor(pk_cur->graph_handle, &idx->t);
        return pk_op_gather_elements(raw, idx, 3, pfx + "_pos");
    }

    QTensor* pk_build_attention(QTensor* x, int li, QTensor* rel_key_t) {
        std::string lp = model_folder + "/layer_" + std::to_string(li) + "_";
        uint32_t T = (uint32_t)pk_T_out, H = (uint32_t)pk_num_heads;
        uint32_t D = (uint32_t)pk_head_dim, hidden = (uint32_t)pk_hidden_dim;
        float scale = 1.f / sqrtf((float)D);
        std::string pfx = "l" + std::to_string(li) + "_attn";
        auto* ln_w = pk_load_weight(pfx + "_lnw", lp + "norm_self_attn.weights");
        auto* ln_b = pk_load_weight(pfx + "_lnb", lp + "norm_self_attn.bias");
        auto* residual = x;
        auto* h = pk_op_layer_norm_wb(x, ln_w, ln_b, pfx + "_ln");
        auto* wq = pk_load_weight_int8(pfx + "_wq", lp + "self_attn_q.weights");
        auto* bq = pk_make_static_fp16(pfx + "_bq", {1,hidden}, std::vector<uint16_t>(hidden, 0));
        { CactFile bf; std::string bp = lp + "self_attn_q.bias";
          if (file_exists(bp) && open_cact(bp, bf)) { auto d = cact_to_fp16(bf);
            bq = pk_make_static_fp16(pfx + "_bq", {1,hidden}, std::move(d)); } }
        auto* wk = pk_load_weight_int8(pfx + "_wk", lp + "self_attn_k.weights");
        auto* bk = pk_make_static_fp16(pfx + "_bk", {1,hidden}, std::vector<uint16_t>(hidden, 0));
        { CactFile bf; std::string bp = lp + "self_attn_k.bias";
          if (file_exists(bp) && open_cact(bp, bf)) { auto d = cact_to_fp16(bf);
            bk = pk_make_static_fp16(pfx + "_bk", {1,hidden}, std::move(d)); } }
        auto* wv = pk_load_weight_int8(pfx + "_wv", lp + "self_attn_v.weights");
        auto* bv_w = pk_make_static_fp16(pfx + "_bv", {1,hidden}, std::vector<uint16_t>(hidden, 0));
        { CactFile bf; std::string bp = lp + "self_attn_v.bias";
          if (file_exists(bp) && open_cact(bp, bf)) { auto d = cact_to_fp16(bf);
            bv_w = pk_make_static_fp16(pfx + "_bv", {1,hidden}, std::move(d)); } }
        auto* q = pk_op_add(pk_op_matmul_T(h, wq, pfx + "_q"), bq, pfx + "_q_b");
        auto* k = pk_op_add(pk_op_matmul_T(h, wk, pfx + "_k"), bk, pfx + "_k_b");
        auto* v = pk_op_add(pk_op_matmul_T(h, wv, pfx + "_v"), bv_w, pfx + "_v_b");
        auto* bu_raw = pk_load_weight(pfx + "_bu", lp + "self_attn_bias_u.weights");
        auto* bv_raw = pk_load_weight(pfx + "_bvw", lp + "self_attn_bias_v.weights");
        auto* bu_1d = pk_op_reshape(bu_raw, {1,hidden}, pfx + "_bu1d");
        auto* bv_1d = pk_op_reshape(bv_raw, {1,hidden}, pfx + "_bv1d");
        auto* q_u = pk_op_add(q, bu_1d, pfx + "_qu");
        auto* q_v = pk_op_add(q, bv_1d, pfx + "_qv");
        auto* qu4 = pk_op_reshape(q_u, {1,T,H,D}, pfx + "_qu4");
        auto* qv4 = pk_op_reshape(q_v, {1,T,H,D}, pfx + "_qv4");
        auto* k4  = pk_op_reshape(k,   {1,T,H,D}, pfx + "_k4");
        auto* v4  = pk_op_reshape(v,   {1,T,H,D}, pfx + "_v4");
        auto* qu_BHTD = pk_op_transpose(qu4, {0,2,1,3}, pfx + "_quHTD");
        auto* k_BHTD  = pk_op_transpose(k4,  {0,2,1,3}, pfx + "_kHTD");
        auto* cs = pk_op_bmm(qu_BHTD, k_BHTD, true, pfx + "_cs");
        cs = pk_op_scalar_mul(cs, scale, pfx + "_cs_sc");
        auto* pos_bias = pk_build_rel_pos_bias(qv4, rel_key_t, scale, pfx + "_rpb");
        auto* scores = pk_op_add(cs, pos_bias, pfx + "_scores");
        auto* attn_w = pk_op_softmax(scores, 3, pfx + "_attnw");
        auto* v_BHTD = pk_op_transpose(v4, {0,2,1,3}, pfx + "_vHTD");
        auto* ao = pk_op_bmm(attn_w, v_BHTD, false, pfx + "_ao");
        auto* ao_T = pk_op_transpose(ao, {0,2,1,3}, pfx + "_aoT");
        auto* ao_flat = pk_op_reshape(ao_T, {T,hidden}, pfx + "_ao_flat");
        auto* wo = pk_load_weight_int8(pfx + "_wo", lp + "self_attn_output.weights");
        auto* bo = pk_make_static_fp16(pfx + "_bo", {1,hidden}, std::vector<uint16_t>(hidden, 0));
        { CactFile bf; std::string bp = lp + "self_attn_output.bias";
          if (file_exists(bp) && open_cact(bp, bf)) { auto d = cact_to_fp16(bf);
            bo = pk_make_static_fp16(pfx + "_bo", {1,hidden}, std::move(d)); } }
        auto* out = pk_op_add(pk_op_matmul_T(ao_flat, wo, pfx + "_op"), bo, pfx + "_op_b");
        return pk_op_add(residual, out, pfx + "_out");
    }

    QTensor* pk_build_conv_module(QTensor* x, int li) {
        std::string lp = model_folder + "/layer_" + std::to_string(li) + "_";
        uint32_t T = (uint32_t)pk_T_out, C = (uint32_t)pk_hidden_dim;
        std::string pfx = "l" + std::to_string(li) + "_conv";
        auto* ln_w = pk_load_weight(pfx + "_lnw", lp + "norm_conv.weights");
        auto* ln_b = pk_load_weight(pfx + "_lnb", lp + "norm_conv.bias");
        auto* residual = x;
        auto* h = pk_op_layer_norm_wb(x, ln_w, ln_b, pfx + "_ln");
        auto* wp1 = pk_load_weight_int8(pfx + "_pw1", lp + "conv_pointwise1.weights");
        auto* flat_h = pk_op_reshape(h, {T, C}, pfx + "_flat_h");
        auto* expanded = pk_op_matmul_T(flat_h, wp1, pfx + "_exp");
        { CactFile bf; std::string bp = lp + "conv_pointwise1.bias";
          if (file_exists(bp) && open_cact(bp, bf)) {
            auto d = cact_to_fp16(bf);
            auto* bt = pk_make_static_fp16(pfx + "_pw1b", {1,2*C}, std::move(d));
            if (!bt->t.v1.id) qnn.tensorCreateGraphTensor(pk_cur->graph_handle, &bt->t);
            expanded = pk_op_add(expanded, bt, pfx + "_exp_b");
          }
        }
        auto* gated  = pk_op_glu(expanded, pfx + "_glu");
        auto* gated3 = pk_op_reshape(gated, {1,T,C}, pfx + "_gated3");
        auto* dw_out = pk_op_depthwise_conv1d_same(gated3,
                            lp + "conv_depthwise.weights", lp + "conv_depthwise.bias",
                            pk_conv_kernel_size, pfx + "_dw");
        auto* bn_out = pk_op_batchnorm(dw_out,
                            lp + "conv_batchnorm_weight.weights",
                            lp + "conv_batchnorm_bias.bias",
                            lp + "conv_batchnorm_running_mean.weights",
                            lp + "conv_batchnorm_running_var.weights",
                            pfx + "_bn");
        auto* act_out  = pk_op_silu(bn_out, pfx + "_silu");
        auto* act_flat = pk_op_reshape(act_out, {T,C}, pfx + "_af");
        auto* wp2 = pk_load_weight_int8(pfx + "_pw2", lp + "conv_pointwise2.weights");
        auto* out = pk_op_matmul_T(act_flat, wp2, pfx + "_pw2");
        { CactFile bf; std::string bp = lp + "conv_pointwise2.bias";
          if (file_exists(bp) && open_cact(bp, bf)) {
            auto d = cact_to_fp16(bf);
            auto* bt = pk_make_static_fp16(pfx + "_pw2b", {1,C}, std::move(d));
            if (!bt->t.v1.id) qnn.tensorCreateGraphTensor(pk_cur->graph_handle, &bt->t);
            out = pk_op_add(out, bt, pfx + "_pw2_b");
          }
        }
        return pk_op_add(residual, out, pfx + "_out");
    }

    QTensor* pk_build_ff(QTensor* x, int li, bool is_second) {
        std::string lp = model_folder + "/layer_" + std::to_string(li) + "_";
        std::string tag = is_second ? "ff2" : "ff1";
        std::string pfx = "l" + std::to_string(li) + "_" + tag;
        auto* ln_w = pk_load_weight(pfx + "_lnw", lp + "norm_" + tag + ".weights");
        auto* ln_b = pk_load_weight(pfx + "_lnb", lp + "norm_" + tag + ".bias");
        auto* residual = x;
        auto* h   = pk_op_layer_norm_wb(x, ln_w, ln_b, pfx + "_ln");
        auto* w1  = pk_load_weight_int8(pfx + "_w1", lp + tag + "_linear1.weights");
        auto* out1 = pk_op_matmul_T(h, w1, pfx + "_mm1");
        { CactFile bf; std::string bp = lp + tag + "_linear1.bias";
          if (file_exists(bp) && open_cact(bp, bf)) {
            auto d = cact_to_fp16(bf); uint32_t N = (uint32_t)d.size();
            auto* bt = pk_make_static_fp16(pfx + "_b1", {1,N}, std::move(d));
            if (!bt->t.v1.id) qnn.tensorCreateGraphTensor(pk_cur->graph_handle, &bt->t);
            out1 = pk_op_add(out1, bt, pfx + "_mm1_b");
          }
        }
        auto* act = pk_op_silu(out1, pfx + "_act");
        auto* w2  = pk_load_weight_int8(pfx + "_w2", lp + tag + "_linear2.weights");
        auto* out2 = pk_op_matmul_T(act, w2, pfx + "_mm2");
        { CactFile bf; std::string bp = lp + tag + "_linear2.bias";
          if (file_exists(bp) && open_cact(bp, bf)) {
            auto d = cact_to_fp16(bf); uint32_t N2 = (uint32_t)d.size();
            auto* bt = pk_make_static_fp16(pfx + "_b2", {1,N2}, std::move(d));
            if (!bt->t.v1.id) qnn.tensorCreateGraphTensor(pk_cur->graph_handle, &bt->t);
            out2 = pk_op_add(out2, bt, pfx + "_mm2_b");
          }
        }
        auto* scaled = pk_op_scalar_mul(out2, 0.5f, pfx + "_half");
        return pk_op_add(residual, scaled, pfx + "_out");
    }

    QTensor* pk_build_encoder_block(QTensor* x, int li, QTensor* rel_key_t) {
        std::string lp  = model_folder + "/layer_" + std::to_string(li) + "_";
        std::string pfx = "l" + std::to_string(li);
        x = pk_build_ff(x, li, false);
        x = pk_build_attention(x, li, rel_key_t);
        x = pk_build_conv_module(x, li);
        x = pk_build_ff(x, li, true);
        auto* ln_w = pk_load_weight(pfx + "_outln_w", lp + "norm_out.weights");
        auto* ln_b = pk_load_weight(pfx + "_outln_b", lp + "norm_out.bias");
        return pk_op_layer_norm_wb(x, ln_w, ln_b, pfx + "_outln");
    }

    std::string pk_seg_cache_path(int seg_idx) {
        return model_folder + "/qnn_parakeet_T" + std::to_string(pk_time_frames)
               + "_seg" + std::to_string(seg_idx) + ".bin";
    }

    bool pk_build_segment(int seg_idx, int layer_start, int layer_end) {
        PSegState& s = pk_segs[seg_idx];
        pk_cur = &s;
        fprintf(stderr, "[QNN Pkrt] building seg%d: layers [%d,%d)\n", seg_idx, layer_start, layer_end);
        Qnn_ContextHandle_t build_ctx = nullptr;
        if (qnn.contextCreate(backend_handle, device_handle, nullptr, &build_ctx) != QNN_SUCCESS) {
            fprintf(stderr, "[QNN Pkrt] contextCreate failed for seg%d\n", seg_idx); return false;
        }
        QnnHtpGraph_CustomConfig_t fp16_cfg = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
        fp16_cfg.option    = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
        fp16_cfg.precision = QNN_PRECISION_FLOAT16;
        QnnGraph_Config_t g_cfg1 = QNN_GRAPH_CONFIG_INIT;
        g_cfg1.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM; g_cfg1.customConfig = &fp16_cfg;
        const QnnGraph_Config_t* g_cfgs[] = {&g_cfg1, nullptr};
        if (qnn.graphCreate(build_ctx, s.graph_name.c_str(), g_cfgs, &s.graph_handle) != QNN_SUCCESS) {
            fprintf(stderr, "[QNN Pkrt] graphCreate failed for seg%d\n", seg_idx);
            qnn.contextFree(build_ctx, nullptr); return false;
        }
        auto* seg_in = pk_make_app_write("seg_in", {(uint32_t)pk_T_out, (uint32_t)pk_hidden_dim});
        if (!seg_in->t.v1.id) qnn.tensorCreateGraphTensor(s.graph_handle, &seg_in->t);
        s.exec_in_id = seg_in->t.v1.id;
        QTensor* h = seg_in;
        for (int li = layer_start; li < layer_end; li++) {
            auto* rel_key = pk_build_rel_key_static(li);
            if (!rel_key->t.v1.id) qnn.tensorCreateGraphTensor(s.graph_handle, &rel_key->t);
            h = pk_build_encoder_block(h, li, rel_key);
            fprintf(stderr, "[QNN Pkrt] seg%d layer %d/%d done\n", seg_idx, li + 1, layer_end);
        }
        {
            auto* seg_out = pk_make_output("seg_out", {(uint32_t)pk_T_out, (uint32_t)pk_hidden_dim});
            pk_add_op(QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_RESHAPE, pk_op_name("out_copy"),
                      {h}, {seg_out}, {});
            if (s.tensors.count("seg_out")) s.exec_out_id = s.tensors.at("seg_out")->t.v1.id;
            else s.exec_out_id = seg_out->t.v1.id;
        }
        if (s.tensors.count("seg_in")) s.exec_in_id = s.tensors.at("seg_in")->t.v1.id;
        fprintf(stderr, "[QNN Pkrt] finalizing seg%d...\n", seg_idx);
        if (qnn.graphFinalize(s.graph_handle, nullptr, nullptr) != QNN_SUCCESS) {
            fprintf(stderr, "[QNN Pkrt] graphFinalize failed for seg%d\n", seg_idx);
            qnn.contextFree(build_ctx, nullptr); return false;
        }
        std::string cp = pk_seg_cache_path(seg_idx);
        if (qnn.contextGetBinarySize && qnn.contextGetBinary) {
            Qnn_ContextBinarySize_t bin_sz = 0, written = 0;
            if (qnn.contextGetBinarySize(build_ctx, &bin_sz) == QNN_SUCCESS && bin_sz > 0) {
                std::vector<char> bin((size_t)bin_sz);
                if (qnn.contextGetBinary(build_ctx, bin.data(), bin_sz, &written) == QNN_SUCCESS
                        && written > 0) {
                    std::ofstream cf(cp, std::ios::binary);
                    if (cf.is_open()) {
                        cf.write(bin.data(), (std::streamsize)written);
                        fprintf(stderr, "[QNN Pkrt] saved seg%d: %s (%llu MB)\n", seg_idx, cp.c_str(),
                                (unsigned long long)written / (1024*1024));
                    }
                    std::ofstream sf(cp + ".ids", std::ios::binary);
                    uint32_t magic = 0x52415050u;
                    sf.write(reinterpret_cast<char*>(&magic), 4);
                    sf.write(reinterpret_cast<char*>(&s.exec_in_id), 4);
                    sf.write(reinterpret_cast<char*>(&s.exec_out_id), 4);
                }
            }
        }
        qnn.contextFree(build_ctx, nullptr);
        s.graph_handle = nullptr;
        s.tensor_store.clear(); s.tensors.clear(); s.weight_files.clear();
        return true;
    }

    bool pk_read_seg_ids(int seg_idx) {
        PSegState& s = pk_segs[seg_idx];
        std::ifstream sf(pk_seg_cache_path(seg_idx) + ".ids", std::ios::binary);
        if (!sf.is_open()) return false;
        uint32_t magic = 0; sf.read(reinterpret_cast<char*>(&magic), 4);
        if (magic != 0x52415050u) return false;
        sf.read(reinterpret_cast<char*>(&s.exec_in_id), 4);
        sf.read(reinterpret_cast<char*>(&s.exec_out_id), 4);
        return true;
    }

    void pk_setup_exec_tensors(int seg_idx) {
        PSegState& s = pk_segs[seg_idx];
        pk_cur = &s;
        s.in_buf.assign((size_t)pk_T_out * pk_hidden_dim, 0);
        auto* in_qt = pk_make_tensor("seg_in_exec", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16,
                                     {(uint32_t)pk_T_out, (uint32_t)pk_hidden_dim},
                                     s.in_buf.data(), s.in_buf.size() * 2);
        in_qt->t.v1.id = s.exec_in_id;
        s.exec_inputs = {in_qt->t};
        s.out_buf.assign((size_t)pk_T_out * pk_hidden_dim, 0);
        auto* out_qt = pk_make_tensor("seg_out_exec", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_16,
                                      {(uint32_t)pk_T_out, (uint32_t)pk_hidden_dim},
                                      s.out_buf.data(), s.out_buf.size() * 2);
        out_qt->t.v1.id = s.exec_out_id;
        s.exec_outputs = {out_qt->t};
    }

    // ---- CPU subsampling ----
    static void pk_cpu_conv2d_s2(const float* x, int H, int W, int Ci,
                                   const float* wt, const float* bias, int Co, float* y) {
        int Ho = (H-1)/2+1, Wo = (W-1)/2+1;
        for (int oh = 0; oh < Ho; oh++) for (int ow = 0; ow < Wo; ow++) {
            float* yo = y + (oh * Wo + ow) * Co;
            for (int co = 0; co < Co; co++) yo[co] = bias ? bias[co] : 0.f;
            for (int kh = 0; kh < 3; kh++) for (int kw = 0; kw < 3; kw++) {
                int ih = oh*2+kh-1, iw = ow*2+kw-1;
                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                const float* xi = x + (ih * W + iw) * Ci;
                for (int ci = 0; ci < Ci; ci++) {
                    const float* wp2 = wt + (ci * 9 + kh * 3 + kw) * Co;
                    float xv = xi[ci];
                    for (int co = 0; co < Co; co++) yo[co] += xv * wp2[co];
                }
            }
        }
    }
    static void pk_cpu_dw_conv2d_s2(const float* x, int H, int W, int C,
                                      const float* wt, const float* bias, float* y) {
        int Ho = (H-1)/2+1, Wo = (W-1)/2+1;
        for (int oh = 0; oh < Ho; oh++) for (int ow = 0; ow < Wo; ow++) {
            float* yo = y + (oh * Wo + ow) * C;
            for (int c = 0; c < C; c++) yo[c] = bias ? bias[c] : 0.f;
            for (int kh = 0; kh < 3; kh++) for (int kw = 0; kw < 3; kw++) {
                int ih = oh*2+kh-1, iw = ow*2+kw-1;
                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                const float* xi = x + (ih * W + iw) * C;
                const float* wp2 = wt + (kh*3+kw) * C;
                for (int c = 0; c < C; c++) yo[c] += xi[c] * wp2[c];
            }
        }
    }
    static void pk_cpu_matmul(const float* x, int N, int K,
                               const float* w, const float* b, int M2, float* y) {
        for (int n = 0; n < N; n++) {
            float* yn = y + n * M2;
            for (int m = 0; m < M2; m++) yn[m] = b ? b[m] : 0.f;
            for (int k = 0; k < K; k++) {
                float xv = x[n*K+k]; const float* wk = w + k * M2;
                for (int m = 0; m < M2; m++) yn[m] += xv * wk[m];
            }
        }
    }
    static void pk_cpu_relu(float* x, int n) {
        for (int i = 0; i < n; i++) if (x[i] < 0.f) x[i] = 0.f;
    }

    bool pk_load_cpu_sub() {
        const std::string wp = model_folder + "/";
        auto load_fp32 = [&](const std::string& path, std::vector<float>& out) -> bool {
            CactFile cf; if (!open_cact(path, cf)) return false;
            auto fp16 = cact_to_fp16(cf); out.resize(fp16.size());
            for (size_t i = 0; i < fp16.size(); i++) out[i] = pk_fp16_to_fp32(fp16[i]);
            return true;
        };
        int C = pk_conv_channels, M = pk_num_mel_bins, T = pk_time_frames;
        int Ho1 = (T-1)/2+1, Wo1 = (M-1)/2+1;
        int Ho2 = (Ho1-1)/2+1, Wo2 = (Wo1-1)/2+1;
        int Ho3 = (Ho2-1)/2+1, Wo3 = (Wo2-1)/2+1;
        pk_cpu_sub.C = C; pk_cpu_sub.W3 = Wo3; (void)Ho3;
        { std::vector<float> raw;
          if (!load_fp32(wp + "subsampling_conv0_weight.weights", raw)) return false;
          int Co = C, Ci = 1; pk_cpu_sub.conv0_w.assign(Ci*9*Co, 0.f);
          for (int co = 0; co < Co; co++) for (int ci = 0; ci < Ci; ci++) for (int k = 0; k < 9; k++)
              pk_cpu_sub.conv0_w[ci*9*Co+k*Co+co] = raw[co*Ci*9+ci*9+k]; }
        load_fp32(wp + "subsampling_conv0_bias.bias", pk_cpu_sub.conv0_b);
        { std::vector<float> raw;
          if (!load_fp32(wp + "subsampling_depthwise1_weight.weights", raw)) return false;
          pk_cpu_sub.dw1_w.assign(9*C, 0.f);
          for (int c = 0; c < C; c++) for (int k = 0; k < 9; k++) pk_cpu_sub.dw1_w[k*C+c] = raw[c*9+k]; }
        load_fp32(wp + "subsampling_depthwise1_bias.bias", pk_cpu_sub.dw1_b);
        { std::vector<float> raw;
          if (!load_fp32(wp + "subsampling_pointwise1_weight.weights", raw)) return false;
          pk_cpu_sub.pw1_w.assign(C*C, 0.f);
          for (int co = 0; co < C; co++) for (int ci = 0; ci < C; ci++)
              pk_cpu_sub.pw1_w[ci*C+co] = raw[co*C+ci]; }
        load_fp32(wp + "subsampling_pointwise1_bias.bias", pk_cpu_sub.pw1_b);
        { std::vector<float> raw;
          if (!load_fp32(wp + "subsampling_depthwise2_weight.weights", raw)) return false;
          pk_cpu_sub.dw2_w.assign(9*C, 0.f);
          for (int c = 0; c < C; c++) for (int k = 0; k < 9; k++) pk_cpu_sub.dw2_w[k*C+c] = raw[c*9+k]; }
        load_fp32(wp + "subsampling_depthwise2_bias.bias", pk_cpu_sub.dw2_b);
        { std::vector<float> raw;
          if (!load_fp32(wp + "subsampling_pointwise2_weight.weights", raw)) return false;
          pk_cpu_sub.pw2_w.assign(C*C, 0.f);
          for (int co = 0; co < C; co++) for (int ci = 0; ci < C; ci++)
              pk_cpu_sub.pw2_w[ci*C+co] = raw[co*C+ci]; }
        load_fp32(wp + "subsampling_pointwise2_bias.bias", pk_cpu_sub.pw2_b);
        { int CW3 = C * Wo3, H2 = pk_hidden_dim;
          std::vector<float> raw;
          if (!load_fp32(wp + "subsampling_linear_weight.weights", raw)) return false;
          pk_cpu_sub.lin_w.assign(CW3 * H2, 0.f);
          for (int m = 0; m < H2; m++) for (int k = 0; k < CW3; k++)
              pk_cpu_sub.lin_w[k*H2+m] = raw[m*CW3+k]; }
        load_fp32(wp + "subsampling_linear_bias.bias", pk_cpu_sub.lin_b);
        pk_cpu_sub.tmp0.assign((size_t)Ho1*Wo1*C, 0.f);
        pk_cpu_sub.tmp1.assign((size_t)Ho2*Wo2*C, 0.f);
        pk_cpu_sub.tmp2.assign((size_t)Ho3*Wo3*C, 0.f);
        pk_cpu_sub.ok = true;
        fprintf(stderr, "[QNN Pkrt] CPU subsampling loaded (W3=%d T_out=%d)\n", Wo3, Ho3);
        return true;
    }

    bool pk_cpu_subsample(const uint16_t* mel, size_t mel_sz, uint16_t* out) {
        auto& s = pk_cpu_sub; if (!s.ok) return false;
        int T = pk_time_frames, M = pk_num_mel_bins, C = s.C;
        int Ho1 = (T-1)/2+1, Wo1 = (M-1)/2+1;
        int Ho2 = (Ho1-1)/2+1, Wo2 = (Wo1-1)/2+1;
        int Ho3 = (Ho2-1)/2+1, Wo3 = s.W3, H = pk_hidden_dim;
        (void)Wo1; (void)Wo2;
        std::vector<float> mel_fp32(T * M);
        for (int i = 0; i < T * M; i++)
            mel_fp32[i] = pk_fp16_to_fp32(i < (int)mel_sz ? mel[i] : 0);
        pk_cpu_conv2d_s2(mel_fp32.data(), T, M, 1,
                          s.conv0_w.data(), s.conv0_b.empty() ? nullptr : s.conv0_b.data(),
                          C, s.tmp0.data());
        pk_cpu_relu(s.tmp0.data(), (int)s.tmp0.size());
        pk_cpu_dw_conv2d_s2(s.tmp0.data(), Ho1, Wo1, C,
                             s.dw1_w.data(), s.dw1_b.empty() ? nullptr : s.dw1_b.data(),
                             s.tmp1.data());
        { std::vector<float> pw1_out(s.tmp1.size());
          pk_cpu_matmul(s.tmp1.data(), Ho2*Wo2, C,
                         s.pw1_w.data(), s.pw1_b.empty() ? nullptr : s.pw1_b.data(),
                         C, pw1_out.data());
          s.tmp1.swap(pw1_out); }
        pk_cpu_relu(s.tmp1.data(), (int)s.tmp1.size());
        pk_cpu_dw_conv2d_s2(s.tmp1.data(), Ho2, Wo2, C,
                             s.dw2_w.data(), s.dw2_b.empty() ? nullptr : s.dw2_b.data(),
                             s.tmp2.data());
        { std::vector<float> pw2_out(s.tmp2.size());
          pk_cpu_matmul(s.tmp2.data(), Ho3*Wo3, C,
                         s.pw2_w.data(), s.pw2_b.empty() ? nullptr : s.pw2_b.data(),
                         C, pw2_out.data());
          s.tmp2.swap(pw2_out); }
        pk_cpu_relu(s.tmp2.data(), (int)s.tmp2.size());
        int CW3 = C * Wo3;
        std::vector<float> flat(Ho3 * CW3);
        for (int t = 0; t < Ho3; t++) for (int c = 0; c < C; c++) for (int w = 0; w < Wo3; w++)
            flat[t*CW3 + c*Wo3 + w] = s.tmp2[t*Wo3*C + w*C + c];
        for (int t = 0; t < Ho3; t++) {
            const float* xr = flat.data() + t * CW3;
            for (int m = 0; m < H; m++) {
                float acc = s.lin_b.empty() ? 0.f : s.lin_b[m];
                const float* wc = s.lin_w.data() + m;
                for (int k = 0; k < CW3; k++) acc += xr[k] * wc[k * H];
                out[t * H + m] = float_to_fp16(acc);
            }
        }
        return true;
    }

    bool pk_load() {
        {
            std::string cfg_path = model_folder + "/parakeet_config.txt";
            if (!file_exists(cfg_path)) cfg_path = model_folder + "/config.txt";
            std::ifstream f(cfg_path);
            if (!f.is_open()) { fprintf(stderr, "[QNN Pkrt] cannot open config\n"); return false; }
            std::string line;
            while (std::getline(f, line)) {
                auto eq = line.find('='); if (eq == std::string::npos) continue;
                std::string k = line.substr(0, eq), v = line.substr(eq + 1);
                if      (k == "num_layers")         pk_num_layers       = std::stoi(v);
                else if (k == "hidden_dim")          pk_hidden_dim       = std::stoi(v);
                else if (k == "attention_heads")     pk_num_heads        = std::stoi(v);
                else if (k == "attention_head_dim")  pk_head_dim         = std::stoi(v);
                else if (k == "time_frames")         pk_time_frames      = std::stoi(v);
                else if (k == "conv_kernel_size")    pk_conv_kernel_size = std::stoi(v);
                else if (k == "num_mel_bins")        pk_num_mel_bins     = std::stoi(v);
            }
        }
        const char* t_env = std::getenv("CACTUS_PARAKEET_T");
        if (t_env && t_env[0]) pk_time_frames = std::atoi(t_env);

        fprintf(stderr, "[QNN Pkrt] config: layers=%d hidden=%d heads=%d mel=%d T_in=%d\n",
                pk_num_layers, pk_hidden_dim, pk_num_heads, pk_num_mel_bins, pk_time_frames);

        int h = pk_time_frames;
        h = (h-1)/2+1; h = (h-1)/2+1; h = (h-1)/2+1;
        pk_T_out = h;
        fprintf(stderr, "[QNN Pkrt] T_out=%d\n", pk_T_out);

        int lps = (pk_num_layers + 2) / 3;
        pk_n_segs = (pk_num_layers + lps - 1) / lps;
        pk_segs.resize(pk_n_segs + 1);
        pk_seg_ctxs.resize(pk_n_segs + 1, nullptr);
        for (int s = 1; s <= pk_n_segs; s++)
            pk_segs[s].graph_name = "parakeet_seg" + std::to_string(s);
        auto seg_layer_start = [&](int s) { return (s - 1) * lps; };
        auto seg_layer_end   = [&](int s) { return std::min(s * lps, pk_num_layers); };
        fprintf(stderr, "[QNN Pkrt] lps=%d n_segs=%d\n", lps, pk_n_segs);

        for (int s = 1; s <= pk_n_segs; s++) {
            std::string cp = pk_seg_cache_path(s);
            if (!file_exists(cp) || !file_exists(cp + ".ids")) {
                fprintf(stderr, "[QNN Pkrt] seg%d cache missing, compiling...\n", s);
                if (!pk_build_segment(s, seg_layer_start(s), seg_layer_end(s))) return false;
            }
        }

        if (!pk_load_cpu_sub()) return false;

        for (int s = 1; s <= pk_n_segs; s++) {
            if (!pk_read_seg_ids(s)) {
                fprintf(stderr, "[QNN Pkrt] failed to read IDs seg%d\n", s); return false;
            }
        }
        for (int s = 1; s <= pk_n_segs; s++) pk_setup_exec_tensors(s);

        fprintf(stderr, "[QNN Pkrt] loading %d segment contexts persistently...\n", pk_n_segs);
        int64_t t_load_start = now_ms();

        // Parallel fread: all segments read from disk concurrently
        struct SegBuf { std::vector<char> data; bool ok = false; };
        std::vector<SegBuf> seg_bufs(pk_n_segs + 1);
        {
            std::vector<std::thread> readers;
            for (int s = 1; s <= pk_n_segs; s++) {
                readers.emplace_back([&, s]() {
                    std::string cp = pk_seg_cache_path(s);
                    std::ifstream f(cp, std::ios::binary | std::ios::ate);
                    if (!f.is_open()) return;
                    size_t sz = (size_t)f.tellg(); f.seekg(0);
                    seg_bufs[s].data.resize(sz);
                    f.read(seg_bufs[s].data.data(), (std::streamsize)sz);
                    seg_bufs[s].ok = true;
                    fprintf(stderr, "[QNN Pkrt] seg%d fread: %zu MB\n", s, sz / (1024*1024));
                });
            }
            for (auto& t : readers) t.join();
        }
        fprintf(stderr, "[QNN Pkrt] parallel fread done: %dms\n", (int)(now_ms() - t_load_start));

        std::vector<Qnn_ContextHandle_t> seg_ctxs(pk_n_segs + 1, nullptr);
        std::vector<Qnn_GraphHandle_t>   seg_graphs(pk_n_segs + 1, nullptr);
        std::atomic<bool> ctx_ok{true};
        {
            std::vector<std::thread> workers;
            for (int s = 1; s <= pk_n_segs; s++) {
                workers.emplace_back([&, s]() {
                    if (!seg_bufs[s].ok) { ctx_ok = false; return; }
                    auto& buf = seg_bufs[s].data;
                    Qnn_ContextHandle_t ctx = nullptr;
                    auto err = qnn.contextCreateFromBinary(backend_handle, device_handle, nullptr,
                                                           buf.data(), (uint64_t)buf.size(), &ctx, nullptr);
                    if (err != QNN_SUCCESS || !ctx) {
                        fprintf(stderr, "[QNN Pkrt] contextCreateFromBinary failed seg%d err=%lld\n",
                                s, (long long)err);
                        ctx_ok = false; return;
                    }
                    Qnn_GraphHandle_t gh = nullptr;
                    if (qnn.graphRetrieve(ctx, pk_segs[s].graph_name.c_str(), &gh) != QNN_SUCCESS) {
                        fprintf(stderr, "[QNN Pkrt] graphRetrieve failed seg%d\n", s);
                        qnn.contextFree(ctx, nullptr); ctx_ok = false; return;
                    }
                    seg_ctxs[s]   = ctx;
                    seg_graphs[s] = gh;
                });
            }
            for (auto& t : workers) t.join();
        }
        if (!ctx_ok) return false;
        for (int s = 1; s <= pk_n_segs; s++) {
            pk_seg_ctxs[s]           = seg_ctxs[s];
            pk_segs[s].graph_handle  = seg_graphs[s];
        }
        fprintf(stderr, "[QNN Pkrt] total load: %dms\n", (int)(now_ms()-t_load_start));

        is_parakeet  = true;
        loaded       = true;
        graph_built  = true;
        fprintf(stderr, "[QNN Pkrt] ready: %d layers, T_out=%d, hidden=%d\n",
                pk_num_layers, pk_T_out, pk_hidden_dim);
        return true;
    }

    void teardown() {
        for (int s = 1; s < (int)pk_seg_ctxs.size(); s++) {
            if (pk_seg_ctxs[s] && qnn.contextFree) {
                pk_segs[s].graph_handle = nullptr;
                qnn.contextFree(pk_seg_ctxs[s], nullptr);
                pk_seg_ctxs[s] = nullptr;
            }
        }
        if (context_handle && qnn.contextFree) qnn.contextFree(context_handle, nullptr);
        if (device_handle  && qnn.deviceFree)  qnn.deviceFree(device_handle);
        if (backend_handle && qnn.backendFree)  qnn.backendFree(backend_handle);
        if (log_handle     && qnn.logFree)      qnn.logFree(log_handle);
#ifdef _WIN32
        if (dll_handle) FreeLibrary((HMODULE)dll_handle);
        if (sys_dll_handle) FreeLibrary((HMODULE)sys_dll_handle);
#else
        if (dll_handle) dlclose(dll_handle);
        if (sys_dll_handle) dlclose(sys_dll_handle);
#endif
        loaded = false;
    }
};

// ---- QNNDirectEncoder public API ----

QNNDirectEncoder::QNNDirectEncoder() : impl_(std::make_unique<Impl>()) {}
QNNDirectEncoder::~QNNDirectEncoder() { impl_->teardown(); }

bool QNNDirectEncoder::load(const std::string& model_path) {
    auto& I = *impl_;
    // Strip /model.mlpackage suffix if present
    std::string folder = model_path;
    {
        const std::string suffix = "/model.mlpackage";
        if (folder.size() >= suffix.size() &&
            folder.substr(folder.size() - suffix.size()) == suffix)
            folder = folder.substr(0, folder.size() - suffix.size());
        const std::string suffix2 = "\\model.mlpackage";
        if (folder.size() >= suffix2.size() &&
            folder.substr(folder.size() - suffix2.size()) == suffix2)
            folder = folder.substr(0, folder.size() - suffix2.size());
    }
    I.model_folder = folder;
    fprintf(stderr, "[QNN Enc] loading encoder from: %s\n", folder.c_str());

    // Parse config
    {
        std::ifstream f(folder + "/config.txt");
        if (!f.is_open()) { fprintf(stderr, "[QNN Enc] cannot open config.txt\n"); return false; }
        std::string line;
        while (std::getline(f, line)) {
            auto eq = line.find('='); if (eq == std::string::npos) continue;
            std::string k = line.substr(0, eq), v = line.substr(eq + 1);
            if      (k == "hidden_dim")              I.hidden_dim = std::stoi(v);
            else if (k == "num_layers")              I.num_layers = std::stoi(v);
            else if (k == "attention_heads")         I.num_heads  = std::stoi(v);
            else if (k == "attention_head_dim")      I.head_dim   = std::stoi(v);
            else if (k == "ffn_intermediate_dim") {
                int fd = std::stoi(v);
                if (fd > 0) I.ffn_dim = fd;
                else        I.ffn_dim = 4 * I.hidden_dim;
            }
        }
        if (I.ffn_dim == 0) I.ffn_dim = 4 * I.hidden_dim;
        fprintf(stderr, "[QNN Enc] config: hidden=%d layers=%d heads=%d hd=%d ffn=%d\n",
                I.hidden_dim, I.num_layers, I.num_heads, I.head_dim, I.ffn_dim);
    }

    // Load DLL and init backend
    std::string dll_path = find_qnn_htp_lib();

    {
        size_t sep = dll_path.rfind('/');
#ifdef _WIN32
        { size_t bs = dll_path.rfind('\\'); if (bs != std::string::npos && (sep == std::string::npos || bs > sep)) sep = bs; }
#endif
        std::string dir = (sep != std::string::npos) ? dll_path.substr(0, sep) : ".";

#ifdef _WIN32
        auto preload = [&](const std::string& full_path) {
            if (!file_exists(full_path)) return;
            HMODULE h = LoadLibraryA(full_path.c_str());
            const char* nm = full_path.c_str() + full_path.rfind('\\') + 1;
            if (h && strstr(nm, "QnnSystem")) I.sys_dll_handle = (void*)h;
        };
        static const char* companions_root[] = { "QnnSystem.dll", "QnnHtpV73Stub.dll", nullptr };
        static const char* companions_htp[]  = { "QnnHtpPrepareDrv.dll", nullptr };
        for (int i = 0; companions_root[i]; i++)
            preload(dir + "\\" + companions_root[i]);
        for (int i = 0; companions_htp[i]; i++)
            preload(dir + "\\HTP\\" + companions_htp[i]);
        I.dll_handle = (void*)LoadLibraryA(dll_path.c_str());
#else
        static const char* companions[] = {
            "libQnnSystem.so",
            "libQnnHtpV75Stub.so", "libQnnHtpV73Stub.so", "libQnnHtpV79Stub.so",
            "libQnnHtpPrepare.so",
            nullptr
        };
        for (int i = 0; companions[i]; i++) {
            std::string full = dir + "/" + companions[i];
            if (!file_exists(full)) continue;
            void* h = dlopen(full.c_str(), RTLD_NOW | RTLD_GLOBAL);
            if (h && strstr(companions[i], "QnnSystem")) I.sys_dll_handle = h;
        }
        I.dll_handle = dlopen(dll_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
#endif
        if (!I.dll_handle) {
            fprintf(stderr, "[QNN Enc] failed to load %s\n", dll_path.c_str()); return false;
        }
    }

#ifdef _WIN32
    auto get_providers = reinterpret_cast<QnnInterfaceGetProvidersFn_t>(
        reinterpret_cast<void*>(GetProcAddress((HMODULE)I.dll_handle, "QnnInterface_getProviders")));
#else
    auto get_providers = reinterpret_cast<QnnInterfaceGetProvidersFn_t>(
        dlsym(I.dll_handle, "QnnInterface_getProviders"));
#endif
    if (!get_providers) { fprintf(stderr, "[QNN Enc] getProviders not found\n"); return false; }
    const QnnInterface_t** providers = nullptr; uint32_t np = 0;
    if (get_providers(&providers, &np) != QNN_SUCCESS || np == 0) return false;
    I.qnn = providers[0]->QNN_INTERFACE_VER_NAME;

    if (I.qnn.logCreate) I.qnn.logCreate(qnn_log_cb, QNN_LOG_LEVEL_ERROR, &I.log_handle);
    if (!I.qnn.backendCreate) return false;
    if (I.qnn.backendCreate(I.log_handle, nullptr, &I.backend_handle) != QNN_SUCCESS) return false;
    if (I.qnn.deviceCreate) {
        QnnHtpDevice_CustomConfig_t soc_cfg;
        soc_cfg.option = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
#ifdef _WIN32
        soc_cfg.socModel = 60;  // QNN_SOC_MODEL_SC8380XP
#else
        soc_cfg.socModel = 0;   // QNN_SOC_MODEL_UNKNOWN — auto-detect
#endif
        QnnDevice_Config_t dev_cfg = QNN_DEVICE_CONFIG_INIT;
        dev_cfg.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
        dev_cfg.customConfig = (QnnDevice_CustomConfig_t)&soc_cfg;
        const QnnDevice_Config_t* dcfgs[] = {&dev_cfg, nullptr};
        if (I.qnn.deviceCreate(I.log_handle, dcfgs, &I.device_handle) != QNN_SUCCESS)
            I.qnn.deviceCreate(I.log_handle, nullptr, &I.device_handle);
    }

    if (!I.qnn.contextCreate) return false;

    if (file_exists(folder + "/audio_subsample_conv_projection_conv_0_conv.weights")) {
        I.is_g4a = true;
        if (!I.g4a_load()) return false;
        return true;
    }

    if (file_exists(folder + "/layer_0_ff1_linear1.weights")) {
        if (!I.pk_load()) return false;
        return true;
    }

    if (I.qnn.contextCreate(I.backend_handle, I.device_handle, nullptr, &I.context_handle) != QNN_SUCCESS)
        return false;

    I.loaded = true;
    fprintf(stderr, "[QNN Enc] backend ready (graph will be built on first preallocate)\n");
    return true;
}

bool QNNDirectEncoder::preallocate(const std::vector<int>& input_shape,
                                    const std::string& /*input_name*/,
                                    const std::string& /*output_name*/) {
    auto& I = *impl_;
    if (!I.loaded) return false;
    if (I.is_parakeet) return true;
    if (I.is_g4a) return true;
    if (input_shape.size() >= 3) {
        I.n_mels = input_shape[1];
        I.T_mel  = input_shape[2];
    }
    I.T_enc = I.T_mel / 2;

    std::string cache_path = I.model_folder + "/qnn_encoder_T" + std::to_string(I.T_mel) + ".bin";

    if (file_exists(cache_path) && file_exists(cache_path + ".ids")) {
        if (I.context_handle && I.qnn.contextFree)
            I.qnn.contextFree(I.context_handle, nullptr);
        I.context_handle = nullptr;
        if (I.load_from_cache(cache_path)) {
            fprintf(stderr, "[QNN Enc] before setup_exec_tensors (cache path)\n"); fflush(stderr);
            I.setup_exec_tensors();
            fprintf(stderr, "[QNN Enc] setup_exec_tensors done (cache path)\n"); fflush(stderr);
            I.graph_built = true;
            fprintf(stderr, "[QNN Enc] graph_built=true (cache path), returning\n"); fflush(stderr);
            return true;
        }
        if (I.qnn.contextCreate)
            I.qnn.contextCreate(I.backend_handle, I.device_handle, nullptr, &I.context_handle);
    }

    if (!I.build_graph()) return false;
    fprintf(stderr, "[QNN Enc] build_graph done, calling setup_exec_tensors (in_id=%u out_id=%u)\n",
            I.exec_in_id, I.exec_out_id);
    fflush(stderr);
    I.setup_exec_tensors();
    fprintf(stderr, "[QNN Enc] setup_exec_tensors done\n");
    fflush(stderr);
    I.graph_built = true;
    return true;
}

size_t QNNDirectEncoder::encode(const __fp16* input, __fp16* output,
                                 const std::vector<int>& shape,
                                 const std::string& /*input_name*/,
                                 const std::string& /*output_name*/) {
    auto& I = *impl_;
    if (I.is_g4a) return I.g4a_encode(input, output, shape);
    if (I.is_parakeet) {
        if (!I.loaded || I.pk_n_segs == 0) return 0;
        size_t inter_sz = (size_t)I.pk_T_out * I.pk_hidden_dim;
        const uint16_t* mel = reinterpret_cast<const uint16_t*>(input);
        size_t mel_sz = shape.size() >= 2 ? (size_t)shape[0] * shape[1]
                                           : (size_t)I.pk_time_frames * I.pk_num_mel_bins;
        if (!I.pk_cpu_subsample(mel, mel_sz, I.pk_segs[1].in_buf.data())) return 0;
        for (int s = 1; s <= I.pk_n_segs; s++) {
            if (s > 1) memcpy(I.pk_segs[s].in_buf.data(), I.pk_segs[s-1].out_buf.data(), inter_sz * 2);
            auto& seg = I.pk_segs[s];
            seg.exec_inputs[0].v1.clientBuf.data  = seg.in_buf.data();
            seg.exec_outputs[0].v1.clientBuf.data = seg.out_buf.data();
            Qnn_ErrorHandle_t err = I.qnn.graphExecute(
                seg.graph_handle,
                seg.exec_inputs.data(),  (uint32_t)seg.exec_inputs.size(),
                seg.exec_outputs.data(), (uint32_t)seg.exec_outputs.size(),
                nullptr, nullptr);
            if (err != QNN_SUCCESS) {
                fprintf(stderr, "[QNN Pkrt] graphExecute seg%d failed: %lld\n", s, (long long)err);
                return 0;
            }
        }
        if (output) memcpy(output, I.pk_segs[I.pk_n_segs].out_buf.data(), inter_sz * 2);
        return inter_sz;
    }

    if (!I.graph_built || !I.graph_handle) return 0;

    // Input from model_whisper.cpp is [1, 80, T_mel] (NCT). QNN Conv1D also uses NCT. Copy directly.
    int n_mels  = shape.size() >= 2 ? shape[1] : I.n_mels;
    int T_mel   = shape.size() >= 3 ? shape[2] : I.T_mel;
    size_t total_in = (size_t)n_mels * T_mel;
    memcpy(I.input_buf.data(), input, total_in * sizeof(uint16_t));

    {
        const uint16_t* ip = reinterpret_cast<const uint16_t*>(input);
        fprintf(stderr, "[QNN Enc] in[0..7]: %04x %04x %04x %04x %04x %04x %04x %04x\n",
                ip[0],ip[1],ip[2],ip[3],ip[4],ip[5],ip[6],ip[7]);
        fflush(stderr);
        // Dump mel input to file for Python comparison (always overwrite)
        {
            static int dump_count = 0;
            char mel_path[64];
            snprintf(mel_path, sizeof(mel_path), "C:/temp/mel_input_%d.bin", dump_count++);
            FILE* fp = fopen(mel_path, "wb");
            if (fp) {
                fwrite(input, sizeof(uint16_t), total_in, fp);
                fclose(fp);
                fprintf(stderr, "[QNN Enc] dumped mel to %s (%zu elems)\n", mel_path, total_in);
            }
        }
    }

    fprintf(stderr, "[QNN Enc] calling graphExecute: in_id=%u out_id=%u in_buf=%p out_buf=%p\n",
            I.exec_inputs.empty() ? 0 : I.exec_inputs[0].v1.id,
            I.exec_outputs.empty() ? 0 : I.exec_outputs[0].v1.id,
            I.exec_inputs.empty() ? nullptr : I.exec_inputs[0].v1.clientBuf.data,
            I.exec_outputs.empty() ? nullptr : I.exec_outputs[0].v1.clientBuf.data);
    fflush(stderr);
    Qnn_ErrorHandle_t err = I.qnn.graphExecute(
        I.graph_handle,
        I.exec_inputs.data(),  (uint32_t)I.exec_inputs.size(),
        I.exec_outputs.data(), (uint32_t)I.exec_outputs.size(),
        nullptr, nullptr);
    fprintf(stderr, "[QNN Enc] graphExecute returned: %lld\n", (long long)err);
    fflush(stderr);
    if (err != QNN_SUCCESS) {
        fprintf(stderr, "[QNN Enc] graphExecute failed: %lld\n", (long long)err);
        return 0;
    }

    size_t out_elems = (size_t)I.T_enc * I.hidden_dim;
    {
        const uint16_t* op = I.output_buf.data();
        fprintf(stderr, "[QNN Enc] out[0..15] (fp16 hex): "
                "%04x %04x %04x %04x %04x %04x %04x %04x "
                "%04x %04x %04x %04x %04x %04x %04x %04x\n",
                op[0],op[1],op[2],op[3],op[4],op[5],op[6],op[7],
                op[8],op[9],op[10],op[11],op[12],op[13],op[14],op[15]);
        fflush(stderr);
    }
    if (output)
        memcpy(output, I.output_buf.data(), out_elems * 2);
    return out_elems;
}

bool QNNDirectEncoder::is_available() const {
    if (impl_->is_g4a)      return impl_->loaded && impl_->graph_built;
    if (impl_->is_parakeet) return impl_->loaded && impl_->pk_n_segs > 0;
    return impl_->loaded && impl_->graph_built;
}
std::vector<int> QNNDirectEncoder::get_input_shape() const {
    if (impl_->is_g4a)      return {impl_->g4a_chunk_mel, impl_->g4a_mel_bins};
    if (impl_->is_parakeet) return {impl_->pk_time_frames, impl_->pk_num_mel_bins};
    return {1, impl_->n_mels, impl_->T_mel};
}
std::vector<int> QNNDirectEncoder::get_output_shape() const {
    if (impl_->is_g4a) {
        int od = impl_->g4a_out_dim > 0 ? impl_->g4a_out_dim : impl_->g4a_hidden;
        return {impl_->g4a_chunk_out, od};
    }
    if (impl_->is_parakeet) return {impl_->pk_T_out, impl_->pk_hidden_dim};
    return {impl_->T_enc, impl_->hidden_dim};
}
__fp16* QNNDirectEncoder::get_output_buffer() {
    if (impl_->is_g4a)
        return reinterpret_cast<__fp16*>(impl_->g4a_output_buf.data());
    if (impl_->is_parakeet && impl_->pk_n_segs > 0)
        return reinterpret_cast<__fp16*>(impl_->pk_segs[impl_->pk_n_segs].out_buf.data());
    return reinterpret_cast<__fp16*>(impl_->output_buf.data());
}
size_t QNNDirectEncoder::get_output_buffer_size() const {
    if (impl_->is_g4a) {
        int od = impl_->g4a_out_dim > 0 ? impl_->g4a_out_dim : impl_->g4a_hidden;
        return (size_t)impl_->g4a_chunk_out * od;
    }
    if (impl_->is_parakeet) return (size_t)impl_->pk_T_out * impl_->pk_hidden_dim;
    return impl_->output_buf.size();
}
size_t QNNDirectEncoder::encode_multimodal_input(
        const std::vector<NPUNamedInput>& /*inputs*/,
        __fp16* /*output*/,
        const std::string& /*output_name*/) {
    return 0;
}

} // namespace npu
} // namespace cactus

#endif // CACTUS_HAS_QNN_DIRECT
